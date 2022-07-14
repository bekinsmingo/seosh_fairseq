# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import warnings
from argparse import Namespace
from typing import Any, Callable, Dict, List

import torch
from fairseq import metrics, search, tokenizer, utils
from fairseq.data import Dictionary, FairseqDataset, data_utils, encoders, iterators
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.optim.amp_optimizer import AMPOptimizer
from omegaconf import DictConfig

from pdb import set_trace as Tra

logger = logging.getLogger(__name__)


class StatefulContainer(object):
    def __init__(self):
        self._state = dict()
        self._factories = dict()

    def add_factory(self, name, factory: Callable[[], Any]):
        self._factories[name] = factory

    def merge_state_dict(self, state_dict: Dict[str, Any]):
        self._state.update(state_dict)

    @property
    def state_dict(self) -> Dict[str, Any]:
        return self._state

    def __getattr__(self, name):
        if name not in self._state and name in self._factories:
            self._state[name] = self._factories[name]()

        if name in self._state:
            return self._state[name]

        raise AttributeError(f"Task state has no factory for attribute {name}")


class FairseqTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.

    Tasks have limited statefulness. In particular, state that needs to be
    saved to/loaded from checkpoints needs to be stored in the `self.state`
    :class:`StatefulContainer` object. For example::

        self.state.add_factory("dictionary", self.load_dictionary)
        print(self.state.dictionary)  # calls self.load_dictionary()

    This is necessary so that when loading checkpoints, we can properly
    recreate the task state after initializing the task instance.
    """

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @staticmethod
    def logging_outputs_can_be_summed(criterion) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `aggregate_logging_outputs`.
        Setting this to True will improves distributed training speed.
        """
        return criterion.logging_outputs_can_be_summed()

    def __init__(self, cfg: FairseqDataclass, **kwargs):
        self.cfg = cfg
        self.datasets = dict()
        self.dataset_to_epoch_iter = dict()
        self.state = StatefulContainer()

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (omegaconf.DictConfig): parsed command-line arguments
        """
        return cls(cfg, **kwargs)

    def has_sharded_data(self, split):
        return os.pathsep in getattr(self.cfg, "data", "")

    def load_dataset(
        self,
        split: str,
        combine: bool = False,
        task_cfg: FairseqDataclass = None,
        **kwargs,
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            combine (bool): combines a split segmented into pieces into one dataset
            task_cfg (FairseqDataclass): optional task configuration stored in the checkpoint that can be used
                                         to load datasets
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], FairseqDataset):
            raise TypeError("Datasets are expected to be of type FairseqDataset")
        return self.datasets[split]

    def filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since max_positions={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(ignored[0], dataset.size(ignored[0]), max_positions)
                )
            logger.warning(
                (
                    "{:,} samples have invalid sizes and will be skipped, "
                    "max_positions={}, first few sample ids={}"
                ).format(len(ignored), max_positions, ignored[:10])
            )
        return indices

    def can_reuse_epoch_itr(self, dataset):
        # We can reuse the epoch iterator across epochs as long as the dataset
        # hasn't disabled it. We default to ``False`` here, although in practice
        # this will be ``True`` for most datasets that inherit from
        # ``FairseqDataset`` due to the base implementation there.
        return getattr(dataset, "can_reuse_epoch_itr_across_epochs", False)

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None, # bbs tokens
        max_sentences=None, # batch_size
        max_positions=None, # 
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and not update_epoch_batch_itr
            and self.can_reuse_epoch_itr(dataset)
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
        reuse_dataloader = getattr(self.cfg, "reuse_dataloader", True)

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
            grouped_shuffling=grouped_shuffling,
            reuse_dataloader=reuse_dataloader,
        )

        # import pdb; pdb.set_trace()

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            cfg (FairseqDataclass): configuration object

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models, quantization_utils

        model = models.build_model(cfg, self, from_checkpoint)
        model = quantization_utils.quantize_model_scalar(model, cfg)
        return model

    def build_criterion(self, cfg: DictConfig):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            cfg (omegaconf.DictConfig): configration object

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions

        return criterions.build_criterion(cfg, self)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        # import pdb; pdb.set_trace()

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """


        # tmp_dict = {v: k for k, v in self.source_dictionary.indices.items()}
        # tmp = []

        # for label in sample['net_input']['src_tokens'][0]:
        #     tmp.append(tmp_dict[int(label)])
        # print("".join(list(tmp)))

        # if sample['net_input']['src_tokens'].eq(self.source_dictionary.pad_index).any().item() == True:
        #     import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        
        # print("sample['net_input']['src_tokens'].size()",sample['net_input']['src_tokens'].size())
        # print("sample['nsentences']",sample['nsentences'])
        # print("sample['ntokens']",sample['ntokens'])
        

        '''
        ====== librispeech with character vocab pdb examples ======

        (Pdb) len(self.source_dictionary); self.source_dictionary.bos_index; 
        self.source_dictionary.pad_index; self.source_dictionary.eos_index; 
        self.source_dictionary.unk_index;

        32
        0
        1
        2
        3


        (Pdb) sample.keys()
        dict_keys(['id', 'nsentences', 'ntokens', 'net_input', 'target'])

        (Pdb) len(sample['net_input']['src_tokens'])
        4

        # 8 * 512 로 고정?
        ########################### 1. Transformer (lm) ###########################
        (Pdb) sample['net_input']['src_tokens'][0]                                                                                            
        tensor([12,  5, 12,  4,  6, 11,  7,  6,  4, 12,  5,  5, 17,  5, 14,  4, 10, 13,
                13,  5, 12, 10, 12,  6, 10, 24, 15,  5,  4,  2, 12,  7, 22,  4, 18,  7,
                9, 12,  6,  4,  6, 11,  8, 16,  4, 19,  8,  8,  4,  7,  9,  8,  6, 11,
                5, 13,  4, 20,  8, 13,  4,  6, 11, 22,  4, 24, 13, 10, 14,  5,  4, 19,
                11, 10, 15, 12,  6,  4, 10,  4,  7, 17,  4, 15, 10, 25, 10,  9, 21,  4,
                5, 25,  5, 13,  4,  9,  5,  7, 13,  4,  6, 11,  5,  5,  4, 12,  6, 10,
                15, 15,  4,  2, 24,  7,  9, 21,  4, 24,  7,  9, 21,  4, 10,  6,  4, 19,
                5,  9,  6,  4,  7, 21,  7, 10,  9,  4,  2, 21,  5, 13,  6, 13, 16, 14,
                5,  4, 10,  4, 12, 11,  7, 15, 15,  4, 24,  5,  4, 25,  5, 13, 22,  4,
                21, 15,  7, 14,  4,  6,  8,  4, 11,  7, 25,  5,  4, 22,  8, 16,  4, 26,
                5,  5, 23,  4, 17,  5,  4, 10,  9,  4, 18,  8, 16,  9,  6,  5,  9,  7,
                9, 18,  5,  4,  2, 10, 15, 15, 16, 12,  6, 13,  7,  6, 10,  8,  9,  4,
                11,  5,  9, 13, 22,  4, 10, 13, 25, 10,  9, 21,  4,  7, 12,  4, 24,  5,
                18, 26,  5,  6,  4,  6, 11,  5,  4, 23,  7, 13,  6,  4, 10,  9,  4, 19,
                11, 10, 18, 11,  4, 10, 13, 25, 10,  9, 21,  4, 17,  7, 14,  5,  4, 11,
                10, 12,  4, 15,  7, 12,  6,  4,  7, 23, 23,  5,  7, 13,  7,  9, 18,  5,
                4,  8,  9,  4,  8, 18,  6,  8, 24,  5, 13,  4,  6, 11, 10, 13,  6,  5,
                5,  9,  6, 11,  4,  9, 10,  9,  5,  6,  5,  5,  9,  4,  8,  4, 20, 10,
                25,  5,  4,  6, 11,  5,  4,  9, 10, 21, 11,  6,  4,  8, 20,  4, 11, 10,
                12,  4, 14,  5,  7,  6, 11,  4, 10,  9,  4,  9, 10,  9,  5,  6,  5,  5,
                9,  4,  8,  4,  6, 19,  8,  4,  8,  9,  4,  6, 11,  5,  4, 15,  7, 12,
                6,  4, 23, 13,  8, 25, 10,  9, 18, 10,  7, 15,  4,  6,  8, 16, 13,  4,
                6, 11,  7,  6,  4, 19,  5,  4,  5, 25,  5, 13,  4, 19,  5,  9,  6,  4,
                6,  8, 21,  5,  6, 11,  5, 13,  4, 11,  5,  4, 19,  7, 12,  4, 10, 15,
                15,  4,  7, 21,  7, 10,  9,  4, 24, 16,  6,  4, 11,  5,  4, 14, 10, 14,
                4,  9,  8,  6,  4, 21, 10, 25,  5,  4, 10,  9,  4,  2,  8,  9,  5,  4,
                17, 10, 21, 11,  6,  4, 11,  7, 25,  5,  4,  7, 15, 15,  4, 12,  8, 13,
                6, 12,  4,  8, 20,  4, 12, 16, 12, 23, 10, 18, 10,  8,  9, 12,  4, 24,
                16,  6,  4, 10,  6,  4, 19,  8], device='cuda:0')

        (Pdb) sample['target'][0]  
        tensor([ 5, 12,  4,  6, 11,  7,  6,  4, 12,  5,  5, 17,  5, 14,  4, 10, 13, 13,
                5, 12, 10, 12,  6, 10, 24, 15,  5,  4,  2, 12,  7, 22,  4, 18,  7,  9,
                12,  6,  4,  6, 11,  8, 16,  4, 19,  8,  8,  4,  7,  9,  8,  6, 11,  5,
                13,  4, 20,  8, 13,  4,  6, 11, 22,  4, 24, 13, 10, 14,  5,  4, 19, 11,
                10, 15, 12,  6,  4, 10,  4,  7, 17,  4, 15, 10, 25, 10,  9, 21,  4,  5,
                25,  5, 13,  4,  9,  5,  7, 13,  4,  6, 11,  5,  5,  4, 12,  6, 10, 15,
                15,  4,  2, 24,  7,  9, 21,  4, 24,  7,  9, 21,  4, 10,  6,  4, 19,  5,
                9,  6,  4,  7, 21,  7, 10,  9,  4,  2, 21,  5, 13,  6, 13, 16, 14,  5,
                4, 10,  4, 12, 11,  7, 15, 15,  4, 24,  5,  4, 25,  5, 13, 22,  4, 21,
                15,  7, 14,  4,  6,  8,  4, 11,  7, 25,  5,  4, 22,  8, 16,  4, 26,  5,
                5, 23,  4, 17,  5,  4, 10,  9,  4, 18,  8, 16,  9,  6,  5,  9,  7,  9,
                18,  5,  4,  2, 10, 15, 15, 16, 12,  6, 13,  7,  6, 10,  8,  9,  4, 11,
                5,  9, 13, 22,  4, 10, 13, 25, 10,  9, 21,  4,  7, 12,  4, 24,  5, 18,
                26,  5,  6,  4,  6, 11,  5,  4, 23,  7, 13,  6,  4, 10,  9,  4, 19, 11,
                10, 18, 11,  4, 10, 13, 25, 10,  9, 21,  4, 17,  7, 14,  5,  4, 11, 10,
                12,  4, 15,  7, 12,  6,  4,  7, 23, 23,  5,  7, 13,  7,  9, 18,  5,  4,
                8,  9,  4,  8, 18,  6,  8, 24,  5, 13,  4,  6, 11, 10, 13,  6,  5,  5,
                9,  6, 11,  4,  9, 10,  9,  5,  6,  5,  5,  9,  4,  8,  4, 20, 10, 25,
                5,  4,  6, 11,  5,  4,  9, 10, 21, 11,  6,  4,  8, 20,  4, 11, 10, 12,
                4, 14,  5,  7,  6, 11,  4, 10,  9,  4,  9, 10,  9,  5,  6,  5,  5,  9,
                4,  8,  4,  6, 19,  8,  4,  8,  9,  4,  6, 11,  5,  4, 15,  7, 12,  6,
                4, 23, 13,  8, 25, 10,  9, 18, 10,  7, 15,  4,  6,  8, 16, 13,  4,  6,
                11,  7,  6,  4, 19,  5,  4,  5, 25,  5, 13,  4, 19,  5,  9,  6,  4,  6,
                8, 21,  5,  6, 11,  5, 13,  4, 11,  5,  4, 19,  7, 12,  4, 10, 15, 15,
                4,  7, 21,  7, 10,  9,  4, 24, 16,  6,  4, 11,  5,  4, 14, 10, 14,  4,
                9,  8,  6,  4, 21, 10, 25,  5,  4, 10,  9,  4,  2,  8,  9,  5,  4, 17,
                10, 21, 11,  6,  4, 11,  7, 25,  5,  4,  7, 15, 15,  4, 12,  8, 13,  6,
                12,  4,  8, 20,  4, 12, 16, 12, 23, 10, 18, 10,  8,  9, 12,  4, 24, 16,
                6,  4, 10,  6,  4, 19,  8, 16], device='cuda:0')
        '''

        '''
        ########################### 2. Transformer XL (truncated bptt lm) ###########################

        # B * seq_len -> 15 * 150
        (Pdb) sample['net_input']['src_tokens'][0] 
        tensor([ 4, 11,  7, 14,  4, 14,  8,  9,  5,  4, 10,  6,  4,  7,  6,  4,  9, 10,
                21, 11,  6,  4, 24, 16,  6,  4, 10,  6,  4, 19,  7, 12,  4,  9,  8,  6,
                4, 11,  5,  4, 19,  7, 12,  4, 18,  8, 17, 23,  5, 15, 15,  5, 14,  4,
                6,  8,  4, 13,  5, 17, 10,  9, 14,  4, 11, 10, 17, 12,  5, 15, 20,  4,
                8,  9,  4,  7,  4,  9, 10, 21, 11,  6,  4, 15, 10, 26,  5,  4,  6, 11,
                10, 12,  4,  2, 11,  5,  4, 19,  7, 12,  4, 12, 16, 20, 20, 10, 18, 10,
                5,  9,  6, 15, 22,  4, 16,  9, 15, 10, 26,  5,  4,  7,  9, 22,  6, 11,
                10,  9, 21,  4,  6, 11,  7,  6,  4, 19,  7, 12,  4,  9,  7,  6, 10, 25,
                5,  4,  6,  8,  4, 20], device='cuda:0')

        (Pdb) sample['target'][0]                                                                                                             
        tensor([11,  7, 14,  4, 14,  8,  9,  5,  4, 10,  6,  4,  7,  6,  4,  9, 10, 21,
                11,  6,  4, 24, 16,  6,  4, 10,  6,  4, 19,  7, 12,  4,  9,  8,  6,  4,
                11,  5,  4, 19,  7, 12,  4, 18,  8, 17, 23,  5, 15, 15,  5, 14,  4,  6,
                8,  4, 13,  5, 17, 10,  9, 14,  4, 11, 10, 17, 12,  5, 15, 20,  4,  8,
                9,  4,  7,  4,  9, 10, 21, 11,  6,  4, 15, 10, 26,  5,  4,  6, 11, 10,
                12,  4,  2, 11,  5,  4, 19,  7, 12,  4, 12, 16, 20, 20, 10, 18, 10,  5,
                9,  6, 15, 22,  4, 16,  9, 15, 10, 26,  5,  4,  7,  9, 22,  6, 11, 10,
                9, 21,  4,  6, 11,  7,  6,  4, 19,  7, 12,  4,  9,  7,  6, 10, 25,  5,
                4,  6,  8,  4, 20,  1], device='cuda:0')


        ########################### 3. BBS loader ###########################

        (Pdb) sample['net_input']['src_tokens'].size()
        torch.Size([88, 46]) # max token 4096

        (Pdb) sample['net_input']['src_tokens'][-1]
        tensor([ 2, 10,  4,  7, 12, 26,  5, 14,  4, 20,  8, 13,  4,  9,  8,  6, 11, 10,
                9, 21,  4, 24,  5,  6,  6,  5, 13,  4,  6, 11,  7,  9,  4,  7,  4, 18,
                8, 17, 23, 13,  8, 17, 10, 12,  5,  4], device='cuda:0')
        (Pdb) sample['target'][0]  
        tensor([10, 27, 17,  4, 29, 16, 12,  6,  4,  7,  4, 18,  8, 16,  9,  6, 13, 22,
                17,  7,  9,  4,  7,  9, 14,  4, 11,  5, 27, 12,  4,  7,  4, 17, 10, 15,
                15, 10,  8,  9,  7, 10, 13,  5,  4,  2], device='cuda:0') 

        '''

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def build_dataset_for_inference(
        self, src_tokens: List[torch.Tensor], src_lengths: List[int], **kwargs
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass

    def begin_valid_epoch(self, epoch, model):
        """Hook function called before the start of each validation epoch."""
        pass

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        """[deprecated] Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            "The aggregate_logging_outputs API is deprecated. "
            "Please use the reduce_metrics API instead."
        )
        with metrics.aggregate() as agg:
            self.reduce_metrics(logging_outputs, criterion)
            return agg.get_smoothed_values()

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = FairseqTask.aggregate_logging_outputs
        self_func = getattr(self, "aggregate_logging_outputs").__func__
        if self_func is not base_func:
            utils.deprecation_warning(
                "Tasks should implement the reduce_metrics API. "
                "Falling back to deprecated aggregate_logging_outputs API."
            )
            agg_logging_outputs = self.aggregate_logging_outputs(
                logging_outputs, criterion
            )
            for k, v in agg_logging_outputs.items():
                metrics.log_scalar(k, v)
            return

        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", nsentences, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)

    def state_dict(self):
        if self.state is not None:
            return self.state.state_dict
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.state is not None:
            self.state.merge_state_dict(state_dict)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    def build_tokenizer(self, args):
        """Build the pre-tokenizer for this task."""
        return encoders.build_tokenizer(args)

    def build_bpe(self, args):
        """Build the tokenizer for this task."""
        return encoders.build_bpe(args)

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        tokens = [
            self.source_dictionary.encode_line(
                encode_fn(src_str), add_if_not_exist=False
            ).long()
            for src_str in lines
        ]
        '''
        (Pdb) self.source_dictionary.encode_line('but good night child')
        tensor([32, 33, 34, 35,  2], dtype=torch.int32)

        (Pdb) self.source_dictionary.encode_line('BUT GOOD NIGHT CHILD')                                                                                            
        tensor([36, 37, 38, 39,  2], dtype=torch.int32)

        (Pdb) self.source_dictionary.encode_line('B U T | G O O D | N I G H T | C H I L D')
        tensor([24, 16,  6,  4, 21,  8,  8, 14,  4,  9, 10, 21, 11,  6,  4, 19, 11, 10,
                15, 14,  2], dtype=torch.int32)

        (Pdb) self.source_dictionary.encode_line('B U T G O O D N I G H T C H I L D')
        tensor([24, 16,  6, 21,  8,  8, 14,  9, 10, 21, 11,  6, 19, 11, 10, 15, 14,  2],
            dtype=torch.int32)
        '''

        lengths = [t.numel() for t in tokens]
        return tokens, lengths


class LegacyFairseqTask(FairseqTask):
    def __init__(self, args: Namespace):
        super().__init__(None)
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

    @classmethod
    def setup_task(cls, args: Namespace, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    def has_sharded_data(self, split):
        return os.pathsep in getattr(self.args, "data", "")

    def build_model(self, args: Namespace, from_checkpoint=False):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models, quantization_utils

        # Tra()
        model = models.build_model(args, self, from_checkpoint)
        model = quantization_utils.quantize_model_scalar(model, args)
        return model

    def build_criterion(self, args: Namespace):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions

        return criterions.build_criterion(args, self)

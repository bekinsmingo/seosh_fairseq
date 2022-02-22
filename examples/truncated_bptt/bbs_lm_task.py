# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
)

from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from omegaconf import II


import datetime
import time
from fairseq.data import FairseqDataset, iterators

###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


@dataclass
class LanguageModelingConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    # max_target_positions: Optional[int] = field(
    #     default=None, metadata={"help": "max number of tokens in the target sequence"}
    # )

    max_target_positions: int = II("task.tokens_per_sample")

    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False, metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False, metadata={"help": "boolean to pad to fixed batch size"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


@register_task("bbs_lm_task", dataclass=LanguageModelingConfig)
class BBSLMTask(LegacyFairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args)

        # import pdb
        # pdb.set_trace()

        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary

        if targets is None:
            targets = ["future"]
        self.targets = targets

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        output_dictionary = None
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0

            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))

            # import pdb
            # pdb.set_trace()

            # 여기가 키포인트
            # 아래처럼 dict.txt가 꼭 포함되어 있어야함.

            '''
            .
            |-- fairseq_char_data
            |   |-- dict.txt
            |   |-- preprocess.log
            |   |-- test.bin
            |   |-- test.idx
            |   |-- train.bin
            |   |-- train.idx
            |   |-- valid.bin
            |   `-- valid.idx
            '''

            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )
        return (dictionary, output_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary, output_dictionary = cls.setup_dictionary(args, **kwargs)

        # upgrade old checkpoints
        if getattr(args, "exclude_self_target", False):
            args.self_target = False

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):

        # import pdb
        # pdb.set_trace()

        model = super().build_model(args)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        return model

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        # (Pdb) paths
        # ['/workspace/lexicon_free_librispeech/decoder/fairseq_char_data']

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        # (Pdb) data_path
        # '/workspace/lexicon_free_librispeech/decoder/fairseq_char_data'
        # (Pdb) split_path
        # '/workspace/lexicon_free_librispeech/decoder/fairseq_char_data/valid'

        # each process has its own copy of the raw data (likely to be an np.memmap)
        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(f"Dataset not found: {split} ({split_path})")

        # (Pdb) dataset
        # <fairseq.data.indexed_dataset.MMapIndexedDataset object at 0x7f090d646b80>
        # (Pdb) len(dataset)
        # 2703

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        # (Pdb) dataset
        # <fairseq.data.indexed_dataset.MMapIndexedDataset object at 0x7f090d646b80>
        # (Pdb) len(dataset)
        # 2703

        # import pdb
        # pdb.set_trace()

        # (Pdb) dataset.sizes
        # array([ 91,  65, 174, ..., 131,  63, 119], dtype=int32)
        # (Pdb) len(dataset.sizes)
        # 2703
        # (Pdb) self.args.tokens_per_sample
        # 512
        # (Pdb) self.args.sample_break_mode
        # none

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            # self.args.tokens_per_sample,
            block_size=None,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            # break_mode=self.args.sample_break_mode,
            break_mode='eos',
            include_targets=True,
            use_plasma_view=self.args.use_plasma_view,
            split_path=split_path,
            plasma_path=self.args.plasma_path,
        )

        # (Pdb) dataset
        # <fairseq.data.token_block_dataset.TokenBlockDataset object at 0x7f090d2af250>
        # (Pdb) len(dataset)
        # 574

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = self.args.batch_size_valid if 'valid' in split else self.args.batch_size

        self.datasets[split] = MonolingualDataset(
            dataset=dataset,
            sizes=dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
            fixed_pad_length=fixed_pad_length,
            pad_to_bsz=pad_to_bsz,
        )

        # import pdb
        # pdb.set_trace()


    # def create_batch_sampler_func(
    #         self,
    #         max_positions,
    #         ignore_invalid_inputs,
    #         max_tokens,
    #         max_sentences,
    #         required_batch_size_multiple=1,
    #         seed=1,
    #     ):
    #         def construct_batch_sampler(dataset, epoch):
    #             splits = [
    #                 s for s, _ in self.datasets.items() if self.datasets[s] == dataset
    #             ]
    #             split = splits[0] if len(splits) > 0 else None
    #             # NEW implementation

    #             # import pdb
    #             # pdb.set_trace()

    #             if epoch is not None:
    #                 # initialize the dataset with the correct starting epoch
    #                 dataset.set_epoch(epoch)

    #             # get indices ordered by example size
    #             start_time = time.time()
    #             logger.info(f"start batch sampler: mem usage: {data_utils.get_mem_usage()}")

    #             with data_utils.numpy_seed(seed):
    #                 indices = dataset.ordered_indices()
    #             logger.info(
    #                 f"[{split}] @batch_sampler order indices time: {get_time_gap(start_time, time.time())}"
    #             )
    #             logger.info(f"mem usage: {data_utils.get_mem_usage()}")

    #             # filter examples that are too large
    #             if max_positions is not None:
    #                 my_time = time.time()
    #                 '''
    #                 (Pdb) max_positions
    #                 512
    #                 (Pdb) indices
    #                 array([8452410, 6133604, 7973398, ..., 5962670, 8451083, 1228825])
    #                 (Pdb) len(indices)
    #                 8452411
    #                 '''
    #                 indices = self.filter_indices_by_size(
    #                     indices, dataset, max_positions, ignore_invalid_inputs
    #                 )
    #                 logger.info(
    #                     f"[{split}] @batch_sampler filter_by_size time: {get_time_gap(my_time, time.time())}"
    #                 )
    #                 logger.info(f"mem usage: {data_utils.get_mem_usage()}")

    #             # create mini-batches with given size constraints
    #             my_time = time.time()
    #             '''
    #             (Pdb) max_tokens
    #             4096
    #             (Pdb) max_sentences
    #             None
    #             (Pdb) required_batch_size_multiple
    #             8
    #             '''
    #             batch_sampler = dataset.batch_by_size(
    #                 indices,
    #                 max_tokens=max_tokens,
    #                 max_sentences=max_sentences,
    #                 required_batch_size_multiple=required_batch_size_multiple,
    #             )

    #             # import pdb
    #             # pdb.set_trace()
                
    #             logger.info(
    #                 f"[{split}] @batch_sampler batch_by_size time: {get_time_gap(my_time, time.time())}"
    #             )
    #             logger.info(
    #                 f"[{split}] per epoch batch_sampler set-up time: {get_time_gap(start_time, time.time())}"
    #             )
    #             logger.info(f"mem usage: {data_utils.get_mem_usage()}")

    #             return batch_sampler

    #         return construct_batch_sampler


    # # we need to override get_batch_iterator because we want to reset the epoch iterator each time
    # def get_batch_iterator(
    #     self,
    #     dataset,
    #     max_tokens=None,
    #     max_sentences=None,
    #     max_positions=None,
    #     ignore_invalid_inputs=False,
    #     required_batch_size_multiple=1,
    #     seed=1,
    #     num_shards=1,
    #     shard_id=0,
    #     num_workers=0,
    #     epoch=1,
    #     data_buffer_size=0,
    #     disable_iterator_cache=False,
    #     skip_remainder_batch=False,
    #     grouped_shuffling=False,
    #     update_epoch_batch_itr=False,
    # ):
    #     """
    #     Get an iterator that yields batches of data from the given dataset.

    #     Args:
    #         dataset (~fairseq.data.FairseqDataset): dataset to batch
    #         max_tokens (int, optional): max number of tokens in each batch
    #             (default: None).
    #         max_sentences (int, optional): max number of sentences in each
    #             batch (default: None).
    #         max_positions (optional): max sentence length supported by the
    #             model (default: None).
    #         ignore_invalid_inputs (bool, optional): don't raise Exception for
    #             sentences that are too long (default: False).
    #         required_batch_size_multiple (int, optional): require batch size to
    #             be a multiple of N (default: 1).
    #         seed (int, optional): seed for random number generator for
    #             reproducibility (default: 1).
    #         num_shards (int, optional): shard the data iterator into N
    #             shards (default: 1).
    #         shard_id (int, optional): which shard of the data iterator to
    #             return (default: 0).
    #         num_workers (int, optional): how many subprocesses to use for data
    #             loading. 0 means the data will be loaded in the main process
    #             (default: 0).
    #         epoch (int, optional): the epoch to start the iterator from
    #             (default: 0).
    #         data_buffer_size (int, optional): number of batches to
    #             preload (default: 0).
    #         disable_iterator_cache (bool, optional): don't cache the
    #             EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
    #             (default: False).
    #         grouped_shuffling (bool, optional): group batches with each groups
    #             containing num_shards batches and shuffle groups. Reduces difference
    #             between sequence lengths among workers for batches sorted by length.
    #         update_epoch_batch_itr (bool optional): if true then donot use the cached
    #             batch iterator for the epoch

    #     Returns:
    #         ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
    #             given dataset split
    #     """
    #     # initialize the dataset with the correct starting epoch
    #     assert isinstance(dataset, FairseqDataset)
    #     if dataset in self.dataset_to_epoch_iter:
    #         return self.dataset_to_epoch_iter[dataset]

    #     # if self.args.sampling_method == "RoundRobin":
    #     #     batch_iter = super().get_batch_iterator(
    #     #         dataset,
    #     #         max_tokens=max_tokens,
    #     #         max_sentences=max_sentences,
    #     #         max_positions=max_positions,
    #     #         ignore_invalid_inputs=ignore_invalid_inputs,
    #     #         required_batch_size_multiple=required_batch_size_multiple,
    #     #         seed=seed,
    #     #         num_shards=num_shards,
    #     #         shard_id=shard_id,
    #     #         num_workers=num_workers,
    #     #         epoch=epoch,
    #     #         data_buffer_size=data_buffer_size,
    #     #         disable_iterator_cache=disable_iterator_cache,
    #     #         skip_remainder_batch=skip_remainder_batch,
    #     #         update_epoch_batch_itr=update_epoch_batch_itr,
    #     #     )
    #     #     self.dataset_to_epoch_iter[dataset] = batch_iter
    #     #     return batch_iter

    #     construct_batch_sampler = self.create_batch_sampler_func(
    #         max_positions,
    #         ignore_invalid_inputs,
    #         max_tokens,
    #         max_sentences,
    #         required_batch_size_multiple=required_batch_size_multiple,
    #         seed=seed,
    #     )

    #     epoch_iter = iterators.EpochBatchIterator(
    #         dataset=dataset,
    #         collate_fn=dataset.collater,
    #         batch_sampler=construct_batch_sampler,
    #         seed=seed,
    #         num_shards=num_shards,
    #         shard_id=shard_id,
    #         num_workers=num_workers,
    #         epoch=epoch,
    #     )
    #     return epoch_iter


    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the language_modeling task is not supported"
                )

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.args.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.source_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        ).next_epoch_itr(shuffle=False)

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary

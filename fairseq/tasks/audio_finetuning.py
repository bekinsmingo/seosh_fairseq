# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import torch
import json

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any

from fairseq.data import AddTargetDataset, Dictionary, encoders
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from . import register_task
from .. import utils
from ..logging import metrics

from fairseq.optim.amp_optimizer import AMPOptimizer

from omegaconf import MISSING, II, OmegaConf
import sys

logger = logging.getLogger(__name__)

from pdb import set_trace as Tra

class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


def label_len_fn(label):
    return len(label.split(" "))


@dataclass
class AudioFinetuningConfig(AudioPretrainingConfig):
    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_detok: Optional[str] = field(
        default=None,
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); "
            "required if using --eval-bleu; use 'space' to disable "
            "detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: str = field(
        default="{}", metadata={"help": "args for building the tokenizer, if needed"}
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None, metadata={"help": "remove BPE before computing BLEU"}
    )
    eval_bleu_args: str = field(
        default="{}",
        metadata={
            "help": "generation args for BLUE scoring, e.g., "
            '\'{"beam": 4, "lenpen": 0.6}\''
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )

    greedy_decoding: bool = field(
        default=False, metadata={"help": "tmp"}
    )

    # s2t_src_joint_ctc: Optional[bool] = II("model.mask_other")
    s2t_src_joint_ctc: bool = field(
        default=False,
        metadata={
            "help": "tmp"
        },
    )
    s2t_src_data: Optional[str] = field(
        default=None, metadata={"help": "tmp"}
    )

    max_source_positions: int = field(
        default=sys.maxsize,
        metadata={
            "help": "tmp"
        },
    )
    max_target_positions: int = field(
        default=sys.maxsize,
        metadata={
            "help": "tmp"
        },
    )

    # ## added for xlsr
    # multiple_train_files: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "tmp",
    #     },
    # )
    # shuffle_by_bucket: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "tmp",
    #     },
    # )
    # shuffle_by_bucket_size: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "tmp",
    #     },
    # )
    # mask_length: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "tmp",
    #     },
    # )
    # mask_prob: float = field(
    #     default=0.5,
    #     metadata={
    #         "help": "tmp",
    #     },
    # )

@register_task("audio_finetuning", dataclass=AudioFinetuningConfig)
class AudioFinetuningTask(AudioPretrainingTask):
    """ """

    cfg: AudioFinetuningConfig

    def __init__(
        self,
        cfg: AudioFinetuningConfig,
    ):
        super().__init__(cfg)
        self.blank_symbol = "<s>"
        self.eval_wer_post_process = cfg.eval_wer_post_process

        self.state.add_factory("target_dictionary", self.load_target_dictionary)
        if self.cfg.s2t_src_joint_ctc:
            self.state.add_factory("source_dictionary", self.load_source_dictionary)
        
        # import pdb; pdb.set_trace()

    def load_target_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.labels}.txt")
            return Dictionary.load(dict_path)
        return None

    def load_source_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(self.cfg.s2t_src_data, f"dict.{self.cfg.labels}.txt")
            return Dictionary.load(dict_path)
        return None

    def load_dataset(
        self, split: str, task_cfg: AudioFinetuningConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        data_path = self.cfg.data
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        text_compressor = TextCompressor(level=text_compression_level)
        with open(label_path, "r") as f:
            labels = [
                text_compressor.compress(l)
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        process_label = LabelEncoder(self.target_dictionary)

        s2t_src_labels = None
        s2t_src_process_label = None
        if self.cfg.s2t_src_joint_ctc:
            data_path = self.cfg.s2t_src_data
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
            text_compressor = TextCompressor(level=text_compression_level)
            with open(label_path, "r") as f:
                s2t_src_labels = [
                    text_compressor.compress(l)
                    for i, l in enumerate(f)
                    if i not in skipped_indices
                ]

            assert len(s2t_src_labels) == len(self.datasets[split]), (
                f"s2t soruce labels length ({len(s2t_src_labels)}) and dataset length "
                f"({len(self.datasets[split])}) do not match"
            )
            s2t_src_process_label = LabelEncoder(self.source_dictionary)

        # Tra()

        self.datasets[split] = AddTargetDataset(
            dataset=self.datasets[split],
            labels=labels,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
            label_len_fn=label_len_fn,
            add_to_input=task_cfg.get("autoregressive", False),
            text_compression_level=text_compression_level,
            s2t_src_labels=s2t_src_labels,
            s2t_src_process_label=s2t_src_process_label,
        )

        '''
        (Pdb) self.datasets[split].__getitem__(0)
        {'id': 0, 'source': tensor([ 0.0013,  0.0013,  0.0020,  ..., -0.0057, -0.0051, -0.0057]), 'label': tensor([11,  5,  4, 15,  7, 16, 21, 11,  5, 14,  4, 24, 16,  6,  4, 10,  6,  4,
                18,  7, 12,  4,  7,  4, 19, 16, 13, 10,  8, 16, 12,  4, 26, 10,  9, 14,
                4,  8, 20,  4, 15,  7, 16, 21, 11,  4, 20, 16, 15, 15,  4,  8, 20,  4,
                25,  5, 28,  7,  6, 10,  8,  9,  4, 10,  9, 29, 16, 13,  5, 14,  4,  7,
                17,  8, 16, 13,  4, 23, 13,  8, 23, 13,  5,  4,  7, 12,  4,  6, 11,  5,
                4, 20, 13,  5,  9, 19, 11,  4, 19,  7, 15, 15,  4,  8, 16, 13,  4, 15,
                8, 25,  5,  4,  8, 20,  4,  8, 16, 13,  4,  8, 18,  9,  4, 14, 10, 21,
                9, 10,  6, 22,  4,  8, 20,  4, 18, 11, 10, 19, 11,  4,  7, 13, 19, 11,
                10, 24,  7, 15, 14,  4, 13,  7, 22, 12,  6,  8, 26,  5,  4, 10,  9,  4,
                6, 11,  5,  4, 20, 16, 15, 15,  4, 20, 15, 16, 12, 11,  4,  8, 20,  4,
                11, 10, 12,  4, 22,  8, 16,  9, 21,  4, 24,  5, 15, 10,  5, 20,  4, 10,
                9,  4, 11, 10, 12,  4, 10, 17, 23,  8, 13,  6,  7,  9, 19,  5,  4,  7,
                12,  4,  7,  4, 24, 13, 10,  6, 10, 12, 11,  4,  8, 20, 20, 10, 19,  5,
                13,  4, 11,  7, 14,  4,  7,  4, 23, 13,  5,  6,  6, 22,  4, 21,  8,  8,
                14,  4, 12,  6,  8, 19, 26,  4], dtype=torch.int32)}
        '''

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.source_dictionary

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
            # with torch.autograd.set_detect_anomaly(True):
            #     optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            if self.cfg.greedy_decoding:
                loss, sample_size, logging_output, net_output = criterion(model, sample, greedy_decoding=self.cfg.greedy_decoding)
                greedy_results = self.greedy_decoding(model, sample, net_output)
            else:
                loss, sample_size, logging_output = criterion(model, sample)
                greedy_results = None

        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model, greedy_results)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        if self.cfg.eval_bleu and self.cfg.autoregressive:
            metrics = self._inference_with_bleu(self.sequence_generator, sample, model, greedy_results)
            logging_output["_bleu_sys_len"] = metrics.sys_len
            logging_output["_bleu_ref_len"] = metrics.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(metrics.counts) == 4
            for i in range(4):
                logging_output[f"_bleu_counts_{i}"] = metrics.counts[i]
                logging_output[f"_bleu_totals_{i}"] = metrics.totals[i]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        # import pdb; pdb.set_trace()
        model = super().build_model(model_cfg, from_checkpoint)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        if self.cfg.eval_bleu and self.cfg.autoregressive:
            assert self.cfg.eval_bleu_detok is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )
            gen_args = json.loads(self.cfg.eval_bleu_args)
            gen_args = Namespace(**gen_args)
            self.sequence_generator = self.build_generator([model], gen_args)

        return model

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def _inference_with_wer(self, generator, sample, model, greedy_results=None):
        import editdistance

        def decode(toks):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_wer_post_process,
                escape_unk=True,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0

        if greedy_results is None:
            gen_out = self.inference_step(generator, [model], sample, None)
        else: 
            gen_out = greedy_results
        
        for i in range(len(gen_out)):
            if greedy_results is None:
                hyp = decode(gen_out[i][0]["tokens"]) # use only best beam 
            else:
                hyp = decode(gen_out[i])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def _inference_with_bleu(self, generator, sample, model, greedy_results=None):
        import sacrebleu

        # # Handle tokenization and BPE
        # tokenizer = task.build_tokenizer(cfg.tokenizer)
        # bpe = task.build_bpe(cfg.bpe)

        # def decode_fn(x):
        #     if bpe is not None:
        #         x = bpe.decode(x)
        #     if tokenizer is not None:
        #         x = tokenizer.decode(x)
        #     return x

        def decode(toks, is_ref):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if is_ref else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            s = s.replace(' ','').replace('|',' ')
            return s

        if greedy_results is None:
            gen_out = self.inference_step(generator, [model], sample, None)
        else: 
            gen_out = greedy_results
        
        hyps, refs = [], []
        for i in range(len(gen_out)):
            if greedy_results is None:
                hyp = gen_out[i][0]["tokens"] # use only best beam 
            else:
                hyp = gen_out[i]
            hyps.append(decode(hyp, is_ref=False))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
                    is_ref=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("H-{} {}".format(sample["id"][0], hyps[0]))
            logger.info("T-{} {}".format(sample["id"][0], refs[0]))
            # logger.info("H-{} {}".format(sample["id"][0], (''.join(hyps[0])).replace('|',' ')))
            # logger.info("T-{} {}".format(sample["id"][0], (''.join(refs[0])).replace('|',' ')))

        eval_tokenization = "none" if self.cfg.eval_tokenized_bleu else "13a"
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize=eval_tokenization)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if self.cfg.eval_wer:
            zero = torch.scalar_tensor(0.0)
            num_char_errors = sum(
                log.get("_num_char_errors", zero) for log in logging_outputs
            )
            num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
            num_word_errors = sum(
                log.get("_num_word_errors", zero) for log in logging_outputs
            )
            num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
            metrics.log_scalar("_num_char_errors", num_char_errors)
            metrics.log_scalar("_num_chars", num_chars)
            metrics.log_scalar("_num_word_errors", num_word_errors)
            metrics.log_scalar("_num_words", num_words)
            if num_chars > 0:
                metrics.log_derived(
                    "uer",
                    lambda meters: meters["_num_char_errors"].sum
                    * 100.0
                    / meters["_num_chars"].sum
                    if meters["_num_chars"].sum > 0
                    else float("nan"),
                )
            if num_words > 0:
                metrics.log_derived(
                    "wer",
                    lambda meters: meters["_num_word_errors"].sum
                    * 100.0
                    / meters["_num_words"].sum
                    if meters["_num_words"].sum > 0
                    else float("nan"),
                )
        if self.cfg.eval_bleu:
            len_keys = ["_bleu_sys_len", "_bleu_ref_len"]
            count_keys = [f"_bleu_counts_{i}" for i in range(4)]
            total_keys = [f"_bleu_totals_{i}" for i in range(4)]
            for k in len_keys + count_keys + total_keys:
                metrics.log_scalar(k, sum(log.get(k, 0) for log in logging_outputs))

            import sacrebleu

            metrics.log_derived(
                "bleu",
                lambda meters: sacrebleu.compute_bleu(
                    correct=[meters[k].sum for k in count_keys],
                    total=[meters[k].sum for k in total_keys],
                    sys_len=meters["_bleu_sys_len"].sum,
                    ref_len=meters["_bleu_ref_len"].sum,
                    smooth_method="exp",
                ).score,
            )

    def greedy_decoding(self, model, sample, net_output):
        with torch.no_grad():
            encoder_out = net_output[-1]
            prev_output_tokens = sample['net_input']['prev_output_tokens'][:,0].unsqueeze(1) # bos token
            max_step = sample['net_input']['prev_output_tokens'].size(1)+5
            bsz = sample['net_input']['prev_output_tokens'].size(0)
            # max_step = model.max_positions()[1]
            # if max_step is None:
            #     max_step = 256
            accum_softmax_prob = torch.Tensor().type_as(encoder_out['encoder_out'])
            for i in range(max_step):
                if (i > 1) and (model.soft_input_training) and (model.soft_input_training_updates <= model.encoder.num_updates) : 
                    softmax_prob = model.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, soft_input=accum_softmax_prob)[0]
                else:
                    softmax_prob = model.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)[0]
                next_output_tokens_prob = softmax_prob[:,-1,:]
                accum_softmax_prob = torch.cat((accum_softmax_prob,next_output_tokens_prob.unsqueeze(1)),1)
                next_output_tokens = torch.argmax(next_output_tokens_prob,-1).unsqueeze(1)
                prev_output_tokens = torch.cat((prev_output_tokens,next_output_tokens),-1)
                if torch.sum(next_output_tokens == self.target_dictionary.eos()).item() == bsz :
                    break;
        return prev_output_tokens
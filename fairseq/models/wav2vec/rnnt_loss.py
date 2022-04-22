# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import math
from dataclasses import dataclass, field

import torchaudio
from torchaudio.models import Hypothesis, RNNTBeamSearch

import editdistance
from fairseq.data.data_utils import post_process
import torch
from fairseq.logging.meters import safe_round


@dataclass
class RNNTCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    clamp: float = field(
        default=1.0,
        metadata={"help": ""},
    )
    post_process: str = field(
        default="wordpiece",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )

@register_criterion("rnnt_loss", dataclass=RNNTCriterionConfig)
class RNNTCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, clamp, post_process):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.clamp = clamp
        self.post_process = post_process
        
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum", clamp=1.0)
        self.blank_idx = 0
        self.decoder = RNNTBeamSearch(None, self.blank_idx)
        self.task = task
        self.tgt_dict = task.target_dictionary

    def forward(self, model, sample, reduce=True):

        targets = sample['target']
        target_lengths = sample['target_lengths'].type('torch.IntTensor').cuda()

        encoder_output, output, src_lengths = model(sample)
        src_lengths = src_lengths.type('torch.IntTensor').cuda()

        loss = self.loss(output, targets, src_lengths, target_lengths)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if not self.training:
            # beam search
            self.decoder.model = model.rnnt_model

            c_err = 0
            c_len = 0
            w_errs = 0
            w_len = 0

            with torch.no_grad():
                for bsz, (t, enc_out) in enumerate(zip(targets, encoder_output)):
                    hyps = self.decoder._search(enc_out, None, 1)
                    hyp_str = post_process_hypos(hyps, self.tgt_dict)[0][0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )

                    targ = t[p]

                    # gold
                    targ_units_arr = targ.tolist()
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_words = post_process(targ_units, self.post_process).split()

                    # pred
                    pred_units_arr = hyps[0][-1] # best beam's indices
                    pred_words = post_process(hyp_str, self.post_process).split() # best beam's word lists

                    # CER
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    # WER
                    dist = editdistance.eval(pred_words, targ_words)
                    w_errs += dist
                    w_len += len(targ_words)

                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)

        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True


def post_process_hypos(hypos, tgt_dict):
    post_process_remove_list = [
        tgt_dict.unk(),
        tgt_dict.eos(),
        tgt_dict.pad(),
    ]

    filtered_hypo_tokens = [
        [token_index for token_index in h.tokens[1:] if token_index not in post_process_remove_list] for h in hypos
    ]

    # hypos_str = [tgt_dict.decode(s) for s in filtered_hypo_tokens]
    hypos_str = [tgt_dict.string(s) for s in filtered_hypo_tokens]
    hypos_ali = [h.alignment[1:] for h in hypos]
    hypos_ids = [h.tokens[1:] for h in hypos]
    hypos_score = [[math.exp(h.score)] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ali, hypos_ids))

    return nbest_batch
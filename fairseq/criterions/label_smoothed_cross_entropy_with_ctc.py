# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.data.data_utils import lengths_to_mask
import numpy as np
import editdistance
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import safe_round

from omegaconf import II

from pdb import set_trace as Tra

@dataclass
class LabelSmoothedCrossEntropyWithCtcCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    ctc_weight: float = field(default=1.0, metadata={"help": "weight for CTC loss"})
    inter_ctc: bool = field(default=False, metadata={"help": "intermediate CTC loss"})
    only_inter_ctc: bool = field(default=False, metadata={"help": "only use intermediate CTC loss"})
    # ctc_weight: float = II("model.ctc_weight")
    s2t_src_joint_ctc: bool = II("task.s2t_src_joint_ctc")

    mwer_training: bool = field(default=False, metadata={"help": "mwer training with nbest hypos"})
    mwer_training_updates: int = field(default=10000, metadata={"help": "weight for CTC loss"})

@register_criterion(
    "label_smoothed_cross_entropy_with_ctc",
    dataclass=LabelSmoothedCrossEntropyWithCtcCriterionConfig,
)
class LabelSmoothedCrossEntropyWithCtcCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        ctc_weight,
        inter_ctc,
        only_inter_ctc,
        mwer_training,
        mwer_training_updates,
        s2t_src_joint_ctc,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.ctc_weight = ctc_weight
        self.inter_ctc = inter_ctc
        self.only_inter_ctc = only_inter_ctc

        self.mwer_training = mwer_training
        self.mwer_training_updates = mwer_training_updates

        self.post_process = self.task.eval_wer_post_process
        self.blank_idx = 0

        self.tgt_dict = self.task.target_dictionary

        self.s2t_src_joint_ctc = s2t_src_joint_ctc
        if self.s2t_src_joint_ctc:
            self.s2t_src_dict = self.task.source_dictionary

    def forward(self, model, sample, reduce=True, greedy_decoding=False):
        net_output = model(**sample["net_input"])
        ce_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ctc_loss = torch.tensor(0.0).type_as(ce_loss)
        inter_ctc_loss = torch.tensor(0.0).type_as(ce_loss)
        if self.ctc_weight > 0.0:
            ctc_lprobs, inter_ctc_lprobs, ctc_lens = model.get_ctc_output(net_output, sample, self.inter_ctc, self.only_inter_ctc)
            ctc_tgt, ctc_tgt_lens = model.get_ctc_target(sample, self.s2t_src_joint_ctc)
            ctc_tgt_mask = lengths_to_mask(ctc_tgt_lens)
            ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
            reduction = "sum" if reduce else "none"

            ctc_loss = (
                F.ctc_loss(
                    ctc_lprobs,
                    ctc_tgt_flat,
                    ctc_lens,
                    ctc_tgt_lens,
                    reduction=reduction,
                    zero_infinity=True,
                )
            )
            if self.inter_ctc and ~self.only_inter_ctc:
                inter_ctc_loss = (
                    F.ctc_loss(
                        inter_ctc_lprobs,
                        ctc_tgt_flat,
                        ctc_lens,
                        ctc_tgt_lens,
                        reduction=reduction,
                        zero_infinity=True,
                    )
                )


        mwer_loss = torch.tensor(0.0).type_as(ce_loss)
        if (self.mwer_training) and (model.encoder.num_updates > self.mwer_training_updates):
            mwer_loss = self.compute_mwer_loss(model, sample)

        # interpolation
        loss = (
            ce_loss * (1-self.ctc_weight) 
            + ctc_loss * self.ctc_weight 
            + inter_ctc_loss * self.ctc_weight 
            + mwer_loss # argmax sampling, inplace operation  
        )

        # Tra()

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "ce_loss": utils.item(ce_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "inter_ctc_loss": utils.item(inter_ctc_loss.data),
            "mwer_loss": utils.item(mwer_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if (not model.training) and (self.ctc_weight > 0.0):
            if "src_lengths" in sample["net_input"]:
                input_lengths = sample["net_input"]["src_lengths"]
            else:
                if net_output[-1]["padding_mask"] is not None:
                    non_padding_mask = ~net_output[-1]["padding_mask"]
                    input_lengths = non_padding_mask.long().sum(-1)
                else:
                    input_lengths = ctc_lprobs.new_full(
                        (ctc_lprobs.size(1),), ctc_lprobs.size(0), dtype=torch.long
                    )

            with torch.no_grad():
                lprobs_t = ctc_lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0

                ctc_target = sample["target_label"] if "target_label" in sample else sample["target"], 
                for lp, t, inp_l in zip(
                    lprobs_t,
                    ctc_target,
                    input_lengths,
                ):
                    ## only support greedy deconding, not ngram decoding
                    lp = lp[:inp_l].unsqueeze(0)

                    p = (t != self.s2t_src_dict.pad()) & (t != self.s2t_src_dict.eos()) if self.s2t_src_joint_ctc else (t != self.tgt_dict.pad()) & (t != self.tgt_dict.eos())
                    targ = t[p]
                    targ_units = self.s2t_src_dict.string(targ) if self.s2t_src_joint_ctc else self.tgt_dict.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.s2t_src_dict.string(pred_units_arr) if self.s2t_src_joint_ctc else self.tgt_dict.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_errs += dist
                    wv_errs += dist

                    w_len += len(targ_words)

                logging_output["ctc_wv_errors"] = wv_errs
                logging_output["ctc_w_errors"] = w_errs
                logging_output["ctc_w_total"] = w_len
                logging_output["ctc_c_errors"] = c_err
                logging_output["ctc_c_total"] = c_len

        if greedy_decoding:
            return loss, sample_size, logging_output, net_output
        else:
            return loss, sample_size, logging_output

    def compute_mwer_loss(self, model, sample):
        # this is currently not supported, because i can't find which part is non-differentiable

        def decode(toks):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.post_process,
                escape_unk=True,
            )
            if self.task.tokenizer:
                s = self.task.tokenizer.decode(s)
            return s

        eps_for_reinforce = np.finfo(np.float32).eps.item()

        # with torch.set_grad_enabled(True): 
        #     batch_nbest_lists = self.task.sequence_generator.generate([model], sample, prefix_tokens=None, constraints=None)

        batch_nbest_lists = self.task.sequence_generator._generate(sample, prefix_tokens=None, mwer_training=True, beam_size_for_mwer_training=1)
            
        mwer_loss = []
        for i, nbest_lists in enumerate(batch_nbest_lists):
            ref_words = decode(utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),).split()
            wers = []
            hypo_probs = []

            for hypos in nbest_lists:
                hypo_tokens = hypos['tokens']
                # hypo_tokens_len = len(hypo_tokens)
                hypo_words = decode(hypo_tokens).split()
                hypo_score = hypos['positional_scores'] # log p(y1|x) log p(y2|y1,x) -> log p(y1,y2|x)
                hypo_prob = torch.exp(hypo_score.sum())

                if torch.isnan(hypo_score).sum().item() > 0 : print('nan is detected in hypo_score (log prob)'); print(hypo_score); Tra()
                if torch.isinf(hypo_score).sum().item() > 0 : print('inf is detected in hypo_score (log prob)'); print(hypo_score); Tra()

                # to avoid -inf loss
                if hypo_prob.item() > 0 :
                    # wers.append(1)
                    wers.append((editdistance.eval(hypo_words, ref_words) * 100.0) / len(ref_words))
                    hypo_probs.append(hypo_prob) #lprobs, minus

            if len(hypo_probs) == 0 :
                continue

            hypo_probs = torch.stack(hypo_probs)

            ## renormalized beam probs
            if hypo_probs.size(0) == 1 :
                mwer_loss += [-torch.log(hypo_probs)]
            else:
                re_normalized_hypo_scores = torch.log(hypo_probs) - torch.log(torch.sum(hypo_probs))

                # if torch.isnan(hypo_scores).sum().item() > 0 : print(hypo_scores); Tra()
                # if torch.isinf(hypo_scores).sum().item() > 0 : print(hypo_scores); Tra()
                if torch.isnan(hypo_probs).sum().item() > 0 : print('nan is detected in hypo_probs'); print(hypo_probs); Tra()
                if torch.isinf(hypo_probs).sum().item() > 0 : print('inf is detected in hypo_probs'); print(hypo_probs); Tra()
                if torch.isnan(re_normalized_hypo_scores).sum().item() > 0 : print('nan is detected in re_normalized_hypo_scores'); print(re_normalized_hypo_scores); Tra()
                if torch.isinf(re_normalized_hypo_scores).sum().item() > 0 : print('inf is detected in re_normalized_hypo_scores'); print(re_normalized_hypo_scores); Tra()

                wers_ = torch.FloatTensor(wers)
                wers = (wers_ - torch.mean(wers_)) / (wers_.std() + eps_for_reinforce)
                # wers_ = F.softmax(wers_, dim=-1) # 0 ~ 1 # to avoid gradient exploding ?
                # wers = (wers_ - torch.mean(wers_)) # -1 ~ 1
                wers = -wers # because WERs are not good Reward, the lower is the better

                mwer_loss += [ -lprob * wer for lprob, wer in zip(re_normalized_hypo_scores, wers)]
                # mwer_loss += [ lprob * wer for lprob, wer in zip(re_normalized_hypo_scores, wers)]

        # with torch.autograd.set_detect_anomaly(True):
        #     torch.stack(mwer_loss).sum().backward()

        # print('torch.stack(mwer_loss)', torch.stack(mwer_loss))

        if torch.isnan(torch.stack(mwer_loss)).sum().item() > 0 : print(mwer_loss); Tra()
        if torch.isinf(torch.stack(mwer_loss)).sum().item() > 0 : print(mwer_loss); Tra()

        return torch.stack(mwer_loss).sum()
        # return torch.stack(mwer_loss).mean()

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        inter_ctc_loss_sum = sum(log.get("inter_ctc_loss", 0) for log in logging_outputs)
        mwer_loss_sum = sum(log.get("mwer_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "inter_ctc_loss", inter_ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mwer_loss", mwer_loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        c_errors = sum(log.get("ctc_c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_c_errors", c_errors)
        c_total = sum(log.get("ctc_c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_c_total", c_total)
        w_errors = sum(log.get("ctc_w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_w_errors", w_errors)
        wv_errors = sum(log.get("ctc_wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_wv_errors", wv_errors)
        w_total = sum(log.get("ctc_w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "ctc_uer",
                lambda meters: safe_round(
                    meters["_ctc_c_errors"].sum * 100.0 / meters["_ctc_c_total"].sum, 3
                )
                if meters["_ctc_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "ctc_wer",
                lambda meters: safe_round(
                    meters["_ctc_w_errors"].sum * 100.0 / meters["_ctc_w_total"].sum, 3
                )
                if meters["_ctc_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "ctc_raw_wer",
                lambda meters: safe_round(
                    meters["_ctc_wv_errors"].sum * 100.0 / meters["_ctc_w_total"].sum, 3
                )
                if meters["_ctc_w_total"].sum > 0
                else float("nan"),
            )


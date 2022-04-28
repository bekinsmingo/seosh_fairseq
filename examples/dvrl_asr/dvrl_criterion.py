# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
from fairseq.dataclass.constants import DDP_BACKEND_CHOICES
import logging
logger = logging.getLogger(__name__)


from fairseq.criterions.ctc import (
    CtcCriterionConfig,
    CtcCriterion,
)
@dataclass
class DVRLCriterionConfig(CtcCriterionConfig):
    iteration: int = field(
        default=50,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    fp16: int = II("common.fp16")

@register_criterion("dvrl_criterion", dataclass=DVRLCriterionConfig)
class DVRLCriterion(CtcCriterion):
    def __init__(self, cfg, task, sentence_avg, iteration, zero_infinity=False):
        super().__init__(cfg, task)
        self.cfg = cfg
        self.task = task
        self.iteration = iteration

    def forward(self, model, sample, optimizer=None, valid_subset_for_dve_training=None, reduce=True):
        '''
        The entire network consists of two model, DVE and predictor.
        The whole process is as follows;
        1. DVE cosnume data inputs and output how good these sample are for learning the model. (selection probabilities)
        2. Sampler decide wheter use these samples according to values (1). 
        3. Predictor finally output log_probs (loss) using data samples.
        4. Update both DVE and Predictor using Gradient Descent (Optimization) 
        '''

        valid_subset = valid_subset_for_dve_training.next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )

        if model.training:
            ## 1. compute selection prob
            selection_prob = model.dve(**sample["net_input"])

            ## 2. Update the predictor model (for N_I iteration) (parametrized with \theta)
            for i in range(self.iteration):
                ## sample from prob distribution
                sampled_selcetion_vector = model.sampler(selection_prob)

                if sampled_selcetion_vector.sum() != 0 :
                    net_output = model(**sample["net_input"])
                    predictor_loss, predictor_lprobs, input_lengths, target_lengths  = self.compute_predictor_loss(model, net_output, sample, sampled_selcetion_vector)

                    if optimizer is not None:
                        with torch.autograd.profiler.record_function("backward"):
                            optimizer.backward(predictor_loss)
                # print('predictor_loss',predictor_loss)

            ## 3. Update the DVE model (1 iteration) (parametrized with \phi)
            valid_w_errs_total = 0
            valid_w_len_total = 0
            with torch.no_grad():
                for valid_sample in valid_subset: # len(valid_subset) = 258
                    if torch.cuda.is_available() :
                        valid_sample = utils.move_to_cuda(valid_sample)
                    if self.cfg.fp16:
                        valid_sample = utils.apply_to_sample(self.apply_half, valid_sample)
                    # import pdb; pdb.set_trace()
                    valid_net_output = model(**valid_sample["net_input"])
                    valid_predictor_loss, valid_predictor_lprobs, valid_input_lengths, valid_target_lengths  = self.compute_predictor_loss(model, valid_net_output, valid_sample)
                    _, _, valid_w_errs, valid_w_len, _ = self.compute_wer(valid_predictor_lprobs, valid_sample, valid_input_lengths)
                    valid_w_errs_total += valid_w_errs
                    valid_w_len_total += valid_w_len

            valid_wer = safe_round(valid_w_errs_total * 100.0 / valid_w_len_total, 3)
            dve_loss = self.compute_dve_loss(selection_prob, sampled_selcetion_vector)
            # print('dve_loss',dve_loss)
            # print('valid_wer',valid_wer)

            ## REINFORCE
            # dve_loss *= (valid_predictor_loss - model.moving_average_previous_loss)
            dve_loss *= (valid_wer - model.moving_average_previous_loss) # use WER as reward instead of loss

            ## update baseline (moving average)
            model.update_moving_average_previous_loss(valid_wer)

            # import pdb; pdb.set_trace()
            
            if optimizer is not None:
                with torch.autograd.profiler.record_function("backward"):
                    optimizer.backward(dve_loss)
        else:
            predictor_loss, predictor_lprobs, input_lengths, target_lengths  = self.compute_predictor_loss(model, net_output, sample)
            dve_loss = torch.tensor(0.0).type_as(predictor_loss)

        ntokens = target_lengths.sum().item()

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(predictor_loss.data),  # * sample['ntokens'],
            "dve_loss": utils.item(dve_loss.data),
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            c_err, c_len, w_errs, w_len, wv_errs = self.compute_wer(predictor_lprobs, sample, input_lengths)

            logging_output["wv_errors"] = wv_errs
            logging_output["w_errors"] = w_errs
            logging_output["w_total"] = w_len
            logging_output["c_errors"] = c_err
            logging_output["c_total"] = c_len

        return predictor_loss, sample_size, logging_output

    def apply_half(self, t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def compute_predictor_loss(self, model, net_output, sample, selection_vector=None, reduce=True):
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder # e.g. torch.Size([634, 8, 32])
        lprobs_ = lprobs.clone()

        target = sample["target"]
        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )

        if net_output["padding_mask"] is not None:
            non_padding_mask = ~net_output["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(1),), lprobs.size(0), dtype=torch.long
            )

        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        if selection_vector is not None:

            new_lprobs = torch.tensor(()).cuda()
            new_target = torch.tensor(()).cuda()
            new_pad_mask = torch.tensor(()).cuda()
            new_input_lengths = []
            new_target_lengths = []

            lprobs = lprobs.transpose(0,1)
            for i, select in enumerate(selection_vector):
                if select :
                    new_lprobs = torch.cat((new_lprobs,lprobs[i].unsqueeze(0)),0)
                    new_target = torch.cat((new_target,target[i].unsqueeze(0)),0)
                    new_pad_mask = torch.cat((new_pad_mask,pad_mask[i].unsqueeze(0)),0)
                    new_input_lengths.append(input_lengths[i])
                    new_target_lengths.append(target_lengths[i])
            lprobs = new_lprobs.transpose(0,1)
            target = new_target
            pad_mask = new_pad_mask.to(torch.bool)
            input_lengths = torch.stack(new_input_lengths)
            target_lengths = torch.stack(new_target_lengths)

        targets_flat = target.masked_select(pad_mask)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        return loss, lprobs_, input_lengths, target_lengths

    def compute_dve_loss(self, selection_prob, sampled_selcetion_vector, reduce=True):
        lprobs = torch.log(torch.sigmoid(selection_prob))
        loss = - (lprobs * sampled_selcetion_vector + (1-lprobs) * (1-sampled_selcetion_vector))
        return loss.sum()

    def compute_wer(self, predictor_lprobs, sample, input_lengths):
        import editdistance
        with torch.no_grad():
            lprobs_t = predictor_lprobs.transpose(0, 1).float().contiguous().cpu()

            c_err = 0
            c_len = 0
            w_errs = 0
            w_len = 0
            wv_errs = 0
            for lp, t, inp_l in zip(
                lprobs_t,
                sample["target_label"]
                if "target_label" in sample
                else sample["target"],
                input_lengths,
            ):
                lp = lp[:inp_l].unsqueeze(0)

                decoded = None
                if self.w2l_decoder is not None:
                    decoded = self.w2l_decoder.decode(lp)
                    if len(decoded) < 1:
                        decoded = None
                    else:
                        decoded = decoded[0]
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]

                p = (t != self.task.target_dictionary.pad()) & (
                    t != self.task.target_dictionary.eos()
                )
                targ = t[p]
                targ_units = self.task.target_dictionary.string(targ)
                targ_units_arr = targ.tolist()

                toks = lp.argmax(dim=-1).unique_consecutive()
                pred_units_arr = toks[toks != self.blank_idx].tolist()

                c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                c_len += len(targ_units_arr)

                targ_words = post_process(targ_units, self.post_process).split()

                pred_units = self.task.target_dictionary.string(pred_units_arr)
                pred_words_raw = post_process(pred_units, self.post_process).split()

                if decoded is not None and "words" in decoded:
                    pred_words = decoded["words"]
                    w_errs += editdistance.eval(pred_words, targ_words)
                    wv_errs += editdistance.eval(pred_words_raw, targ_words)
                else:
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_errs += dist
                    wv_errs += dist

                w_len += len(targ_words)
            
            return c_err, c_len, w_errs, w_len, wv_errs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        dve_loss_sum = utils.item(sum(log.get("dve_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dve_loss_sum", dve_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
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
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


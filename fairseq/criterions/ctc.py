# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

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
import editdistance

from pdb import set_trace as Tra

import logging
logger = logging.getLogger(__name__)


@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )

    # inter CTC
    inter_ctc_weight: float = field(default=0.3, metadata={"help": "weight for CTC loss"})
    inter_ctc: bool = field(default=False, metadata={"help": "intermediate CTC loss"})
    inter_ctc_training_updates: int = field(default=0, metadata={"help": "activate inter ctc training after specific num_updates"})
    
    # eval print
    eval_ctc_print: bool = field(default=True, metadata={"help": "if you want to print hypothesis in every evaluation time"})

    # inter CTC
    kld_loss_weight: float = field(default=0.0, metadata={"help": "weight for CTC loss"})
    kld_loss: bool = field(default=False, metadata={"help": "intermediate CTC loss"})

@register_criterion("ctc", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(
        self, 
        cfg: CtcCriterionConfig, 
        task: FairseqTask,
        ):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        self.tgt_dict = task.target_dictionary

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None and cfg.wer_kenlm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

        self.inter_ctc = cfg.inter_ctc
        self.inter_ctc_weight = cfg.inter_ctc_weight
        self.inter_ctc_training_updates = cfg.inter_ctc_training_updates

        self.eval_ctc_print = cfg.eval_ctc_print

        self.kld_loss = cfg.kld_loss
        self.kld_loss_weight = cfg.kld_loss_weight

    def forward(self, model, sample, reduce=True):
        # Tra()
        net_output = model(**sample["net_input"], inter_ctc = self.inter_ctc)
        if self.inter_ctc:
            inter_net_output = net_output[1]
            net_output = net_output[0]
        else:
            net_output = net_output[0]

        # lprobs = model.get_normalized_probs(
        #     net_output, log_probs=True
        # ).contiguous()  # (T, B, C) from the encoder # e.g. torch.Size([634, 8, 32])

        logits = model.get_logits(net_output)
        probs = utils.softmax(logits.float(), dim=-1).contiguous()
        lprobs = utils.log_softmax(logits.float(), dim=-1).contiguous()

        if self.inter_ctc:
            # inter_ctc_lprobs = model.get_normalized_probs(
            #     inter_net_output, log_probs=True
            # ).contiguous()  # (T, B, C) from the encoder # e.g. torch.Size([634, 8, 32])

            inter_ctc_logits = model.get_logits(inter_net_output)
            inter_ctc_probs = utils.softmax(inter_ctc_logits.float(), dim=-1).contiguous()
            inter_ctc_lprobs = utils.log_softmax(inter_ctc_logits.float(), dim=-1).contiguous()
            
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        '''
        (Pdb) pad_mask.size(); pad_mask[0]
        torch.Size([16, 49])
        tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True, False, False, False],
            device='cuda:0')

        (Pdb) sample.keys(); sample["target"].size(); sample["target"][:2]
        dict_keys(['id', 'net_input', 'target_lengths', 'ntokens', 'target'])
        torch.Size([16, 49])
        tensor([[2115, 5709,    6,   35, 8136,    9,   60, 3343,   19, 1502,    5,  464,
                923,  223,   59,  975,    4, 2608,   30,  223,   40,  689,  138, 7803,
                24,   40,  115,    4,  245,   19,   44,   11,   94,   68,   13,   78,
                741,    6,  515,    7,  163,  770,    6,    7,  959,  269,    1,    1,
                    1],
                [ 276,   12,  166, 1273,   29,   10,  436, 3283,   36,    4,  225,   60,
                632,   17,   22,   59,   63,  251,  223,  194,   16, 2641, 4190,   32,
                147,  151, 3774,  241,  578,  182, 4553,    9,  754,   20, 1923, 1529,
                    9,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                    1]], device='cuda:0', dtype=torch.int32)

        (Pdb) targets_flat.size(); targets_flat[:20];
        torch.Size([602])
        tensor([2115, 5709,    6,   35, 8136,    9,   60, 3343,   19, 1502,    5,  464,
                923,  223,   59,  975,    4, 2608,   30,  223], device='cuda:0',
            dtype=torch.int32)

        (Pdb) sample["target_lengths"]; target_lengths; input_lengths;
        tensor([46, 37, 27, 28, 49, 36, 39, 47, 45, 37, 27, 41, 28, 36, 41, 38],
            device='cuda:0')
        tensor([46, 37, 27, 28, 49, 36, 39, 47, 45, 37, 27, 41, 28, 36, 41, 38],
            device='cuda:0')
        tensor([610, 610, 610, 610, 610, 610, 610, 610, 610, 610, 610, 610, 610, 610,
                610, 610], device='cuda:0')
        '''

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        inter_ctc_loss = torch.tensor(0.0).type_as(ctc_loss)
        if (self.inter_ctc_weight > 0.0) and (self.inter_ctc) and (self.inter_ctc_training_updates <= model.w2v_encoder.num_updates):
            with torch.backends.cudnn.flags(enabled=False):
                inter_ctc_loss = (
                    F.ctc_loss(
                        inter_ctc_lprobs,
                        targets_flat,
                        input_lengths,
                        target_lengths,
                        blank=self.blank_idx,
                        reduction="sum",
                        zero_infinity=self.zero_infinity,
                    )
                )

        kld_loss = torch.tensor(0.0).type_as(ctc_loss)
        if (self.kld_loss) and (self.kld_loss_wegiht > 0.0):
            logits = logits.transpose(0,1)
            probs = probs.transpose(0,1)
            log_probs = lprobs.transpose(0,1)
            bs, _, vocab = logits.size()

            log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))
            loss = torch.mul(probs, log_probs - log_uniform)
            kld_loss_sum = sum([loss[b, :input_lengths[b], :].sum() for b in range(bs)])
            # kld_loss_mean = sum([loss[b, :input_lengths[b], :].sum() for b in range(bs)]) / input_lengths.sum()

            kld_loss = kld_loss_sum
            
        # interpolation
        loss = (
            ctc_loss * (1-self.inter_ctc_weight)
            + inter_ctc_loss * self.inter_ctc_weight
            + kld_loss * self.kld_loss_weight
        )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ctc_loss": utils.item((ctc_loss * (1-self.inter_ctc_weight)).data),
            "inter_ctc_loss": utils.item((inter_ctc_loss * self.inter_ctc_weight).data),
            "kld_loss": utils.item((kld_loss * self.kld_loss_weight).data),
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()
                result, tgt, hyp = self.comput_wer(sample, lprobs_t, input_lengths)

                logging_output["wv_errors"] = result["wv_errors"]
                logging_output["w_errors"] = result["w_errors"]
                logging_output["w_total"] = result["w_total"]
                logging_output["c_errors"] = result["c_errors"]
                logging_output["c_total"] = result["c_total"]

                if self.eval_ctc_print:
                    logger.info("CTC H-{} {}".format(sample["id"][0], ' '.join(hyp)))

                if (self.inter_ctc_weight) > 0.0 and (self.inter_ctc_training_updates <= model.w2v_encoder.num_updates):

                    inter_lprobs_t = inter_ctc_lprobs.transpose(0, 1).float().contiguous().cpu()
                    inter_result, tgt, inter_hyp = self.comput_wer(sample, inter_lprobs_t, input_lengths)

                    logging_output["inter_wv_errors"] = inter_result["wv_errors"]
                    logging_output["inter_w_errors"] = inter_result["w_errors"]
                    logging_output["inter_w_total"] = inter_result["w_total"]
                    logging_output["inter_c_errors"] = inter_result["c_errors"]
                    logging_output["inter_c_total"] = inter_result["c_total"]

                    if self.eval_ctc_print:
                        logger.info("INT H-{} {}".format(sample["id"][0], ' '.join(inter_hyp)))

            if self.eval_ctc_print:
                logger.info("TGT T-{} {}".format(sample["id"][0], ' '.join(tgt)))
                logger.info("============================================================")

        return loss, sample_size, logging_output


    def comput_wer(self, sample, lprobs_t, input_lengths):
        c_err = 0
        c_len = 0
        w_errs = 0
        w_len = 0
        wv_errs = 0

        ctc_target = sample["target_label"] if "target_label" in sample else sample["target"]
            
        for lp, t, inp_l in zip(
            lprobs_t,
            ctc_target,
            input_lengths,
        ):
            ## only support greedy deconding, not ngram decoding
            lp = lp[:inp_l].unsqueeze(0)

            p = (t != self.tgt_dict.pad()) & (t != self.tgt_dict.eos())
            targ = t[p]
            targ_units = self.tgt_dict.string(targ)
            targ_units_arr = targ.tolist()

            toks = lp.argmax(dim=-1).unique_consecutive()
            pred_units_arr = toks[toks != self.blank_idx].tolist()

            c_err += editdistance.eval(pred_units_arr, targ_units_arr)
            c_len += len(targ_units_arr)

            targ_words = post_process(targ_units, self.post_process).split()

            pred_units = self.tgt_dict.string(pred_units_arr)

            pred_words_raw = post_process(pred_units, self.post_process).split()

            dist = editdistance.eval(pred_words_raw, targ_words)
            w_errs += dist
            wv_errs += dist

            w_len += len(targ_words)

            result = {
                "wv_errors" : wv_errs,
                "w_errors" : w_errs,
                "w_total" : w_len,
                "c_errors" : c_err,
                "c_total" : c_len,
            }

            return result, targ_words, pred_words_raw


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        inter_ctc_loss_sum = sum(log.get("inter_ctc_loss", 0) for log in logging_outputs)
        kld_loss_sum = sum(log.get("kld_loss",0) for log in logging_outputs)
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "inter_ctc_loss", inter_ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "kld_loss", kld_loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
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

        ############################## inter ctc wer ##############################

        c_errors = sum(log.get("inter_c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_inter_c_errors", c_errors)
        c_total = sum(log.get("inter_c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_inter_c_total", c_total)
        w_errors = sum(log.get("inter_w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_inter_w_errors", w_errors)
        wv_errors = sum(log.get("inter_wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_inter_wv_errors", wv_errors)
        w_total = sum(log.get("inter_w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_inter_w_total", w_total)


        if c_total > 0:
            metrics.log_derived(
                "inter_uer",
                lambda meters: safe_round(
                    meters["_inter_c_errors"].sum * 100.0 / meters["_inter_c_total"].sum, 3
                )
                if meters["_inter_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "inter_wer",
                lambda meters: safe_round(
                    meters["_inter_w_errors"].sum * 100.0 / meters["_inter_w_total"].sum, 3
                )
                if meters["_inter_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "inter_raw_wer",
                lambda meters: safe_round(
                    meters["_inter_wv_errors"].sum * 100.0 / meters["_inter_w_total"].sum, 3
                )
                if meters["_inter_w_total"].sum > 0
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

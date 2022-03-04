# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.logging.meters import safe_round
from fairseq.dataclass.constants import DDP_BACKEND_CHOICES
import logging
logger = logging.getLogger(__name__)


@dataclass
class CrossEntropyforData2vecCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    ddp_backend: DDP_BACKEND_CHOICES = II("distributed_training.ddp_backend")
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    mwer: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("cross_entropy_for_data2vec", dataclass=CrossEntropyforData2vecCriterionConfig)
class CrossEntropyCriterionforData2Vec(FairseqCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False, mwer=False,):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.mwer = mwer

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # net_output = model(**sample["net_input"])
        net_output, ctc_transcripts, target_padding_mask = model(sample)
        
        if model.decoder.adaptive_softmax is not None:
            loss, lprobs = self.compute_adaptive_loss(model, net_output, sample, reduce=reduce)
            # TODO: adaptive softmax for evaluation mode
        elif self.eps > 0.0:
            loss, lprobs = self.compute_label_smoothed_loss(model, net_output, sample, reduce=reduce)
            # import pdb; pdb.set_trace()
        else:
            loss, lprobs = self.compute_loss(model, net_output, sample, reduce=reduce)

        # for calculate WER (MWER loss)
        # import pdb; pdb.set_trace()

        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0

                ctc_c_err = 0
                ctc_c_len = 0
                ctc_w_errs = 0
                ctc_w_len = 0

                # import pdb; pdb.set_trace()

                rp_ctc_wer = []
                rp_wer = []

                lprobs = lprobs.argmax(dim=-1)
                lprobs = lprobs.masked_fill(target_padding_mask,self.padding_idx)

                for lp, t, ctc_transcript in zip(lprobs, sample["target_label"] if "target_label" in sample else sample["target"], ctc_transcripts):

                    # Processing prediction and target tensors to string
                    p = (t != model.roberta_src_dict.pad()) & (
                        t != model.roberta_src_dict.eos()
                    )
                    # for roberta output
                    targ = t[p]
                    targ_units =  model.roberta_src_dict.string(targ).replace('<pad>','').replace('<mask>','').replace('<s>','').replace('</s>','').replace('<unk>','')
                    targ_units_arr = model.bpe.decode(targ_units)

                    toks = lp
                    # toks = lp.argmax(dim=-1)
                    bpe_sentence = model.roberta_src_dict.string(toks).replace('<pad>','').replace('<mask>','').replace('<s>','').replace('</s>','').replace('<unk>','')
                    pred_units_arr = model.bpe.decode(bpe_sentence)

                    # for ctc output
                    ctc_targ_units_arr = targ_units_arr.upper().replace('.','')
                    ctc_pred_units_arr = ctc_transcript

                    # CER
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    ctc_c_err += editdistance.eval(ctc_pred_units_arr, ctc_targ_units_arr)
                    ctc_c_len += len(ctc_targ_units_arr)

                    # WER
                    targ_words=targ_units_arr.split(' ')
                    pred_words=pred_units_arr.split(' ')
                    w_errs += editdistance.eval(pred_words, targ_words)
                    w_len += len(targ_words)

                    ctc_targ_words=ctc_targ_units_arr.split(' ')
                    ctc_pred_words=ctc_pred_units_arr.split(' ')
                    ctc_w_errs += editdistance.eval(ctc_pred_words, ctc_targ_words)
                    ctc_w_len += len(targ_words)

                    # for report

                    rp_ctc_wer.append(safe_round(editdistance.eval(ctc_pred_words, ctc_targ_words)*100/len(targ_words),3))
                    rp_wer.append(safe_round(editdistance.eval(pred_words, targ_words)*100/len(targ_words),3))

                    # import pdb; pdb.set_trace()

                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

                logging_output["ctc_w_errors"] = ctc_w_errs
                logging_output["ctc_w_total"] = ctc_w_len
                logging_output["ctc_c_errors"] = ctc_c_err
                logging_output["ctc_c_total"] = ctc_c_len

                logger.info("--------------------- ctc ---------------------")
                logger.info("TARG : " + ctc_targ_units_arr)
                logger.info("HYPO : " + ctc_pred_units_arr)
                logger.info("BATCH WER : {}".format(rp_ctc_wer))
                logger.info("BATCH WER : {}".format(safe_round(ctc_w_errs*100/ctc_w_len,3)))
                logger.info("-------------------- Roberta ------------------")
                logger.info("TARG : " + targ_units_arr)
                logger.info("HYPO : " + pred_units_arr)
                logger.info("BATCH WER : {}".format(rp_wer))
                logger.info("BATCH WER : {}".format(safe_round(w_errs*100/w_len,3)))
                logger.info("-----------------------------------------------")


        return loss, sample_size, logging_output

    def compute_adaptive_loss(self, model, net_output, sample, reduce=True):
        adaptive_softmax = model.decoder.adaptive_softmax
        orig_target = model.get_targets(sample, net_output)
        # nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target = adaptive_softmax(net_output, orig_target)
        assert len(target) == len(logits)

        loss = net_output.new(1 if reduce else bsz).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= logits[i].size(1)
                loss += F.cross_entropy(
                    logits[i],
                    target[i],
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )

        # orig = utils.strip_pad(orig_target, self.padding_idx)
        # ntokens = orig.numel()
        # sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        return loss, logits

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # loss = torch.tensor(0)
        if model.training:
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output).view(-1)
            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
            # import pdb; pdb.set_trace()
        else:
            loss = torch.tensor(0)
        return loss, lprobs

    def compute_label_smoothed_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True) # torch.Size([8, 48, 50264])
        # import pdb; pdb.set_trace()
        if model.training:
            target = model.get_targets(sample, net_output)
            lprobs = lprobs.view(-1, lprobs.size(-1)) # torch.Size([384, 50264])
            target = target.view(-1)
            if self.ignore_prefix_size > 0:
                # lprobs: B x T x C
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            # return loss, nll_loss
        else:
            loss = torch.tensor(0)
        # import pdb; pdb.set_trace()
        return loss, lprobs


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        else:
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)


        ctc_c_errors = sum(log.get("ctc_c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_c_errors", ctc_c_errors)
        ctc_c_total = sum(log.get("ctc_c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_c_total", ctc_c_total)
        ctc_w_errors = sum(log.get("ctc_w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_w_errors", ctc_w_errors)
        ctc_w_total = sum(log.get("ctc_w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_w_total", ctc_w_total)

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

        if ctc_c_total > 0:
            metrics.log_derived(
                "ctc_uer",
                lambda meters: safe_round(
                    meters["_ctc_c_errors"].sum * 100.0 / meters["_ctc_c_total"].sum, 3
                )
                if meters["_ctc_c_total"].sum > 0
                else float("nan"),
            )
        if ctc_w_total > 0:
            metrics.log_derived(
                "ctc_wer",
                lambda meters: safe_round(
                    meters["_ctc_w_errors"].sum * 100.0 / meters["_ctc_w_total"].sum, 3
                )
                if meters["_ctc_w_total"].sum > 0
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

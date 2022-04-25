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

@dataclass
class LabelSmoothedCrossEntropyWithCtcCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    ctc_weight: float = field(default=1.0, metadata={"help": "weight for CTC loss"})
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
        mwer_training,
        mwer_training_updates,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.ctc_weight = ctc_weight
        self.mwer_training = mwer_training
        self.mwer_training_updates = mwer_training_updates

    def forward(self, model, sample,reduce=True):
        net_output = model(**sample["net_input"])
        ce_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ctc_loss = torch.tensor(0.0).type_as(ce_loss)
        if self.ctc_weight > 0.0:
            ctc_lprobs, ctc_lens = model.get_ctc_output(net_output, sample)
            ctc_tgt, ctc_tgt_lens = model.get_ctc_target(sample)
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

        mwer_loss = torch.tensor(0.0).type_as(ce_loss)
        if (self.mwer_training) and (model.encoder.num_updates > self.mwer_training_updates):
            mwer_loss = self.compute_mwer_loss(model, sample)

        # interpolation
        loss = (
            ce_loss * (1-self.ctc_weight) 
            + ctc_loss * self.ctc_weight 
            + mwer_loss
        )

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_mwer_loss(self, model, sample):
        import editdistance
        def decode(toks):
            s = self.task.target_dictionary.string(
                toks.int().cpu(),
                self.task.eval_wer_post_process,
                escape_unk=True,
            )
            if self.task.tokenizer:
                s = self.task.tokenizer.decode(s)
            return s

        eps_for_reinforce = np.finfo(np.float32).eps.item()

        # with torch.set_grad_enabled(True): batch_nbest_lists = self.task.sequence_generator.generate([model], sample, prefix_tokens=None, constraints=None)
        with torch.set_grad_enabled(True): batch_nbest_lists = self.task.sequence_generator._generate(sample, prefix_tokens=None)

        mwer_loss = []
        for i, nbest_lists in enumerate(batch_nbest_lists):
            ref_words = decode(utils.strip_pad(sample["target"][i], self.task.target_dictionary.pad()),).split()
            wers = []
            hypo_scores = []

            for hypos in nbest_lists:
                hypo_words = decode(hypos['tokens']).split()
                hypo_score = hypos['positional_scores']

                wers.append((editdistance.eval(hypo_words, ref_words) * 100.0) / len(ref_words))
                hypo_scores.append(hypo_score.sum())

            wers_ = torch.FloatTensor(wers)
            wers = (wers_ - torch.mean(wers_)) / (wers_.std() + eps_for_reinforce)
            wers = -wers # because WERs are not good Reward, the lower is the better

            mwer_loss += [ -lprob * wer for lprob, wer in zip(hypo_scores, wers)]
        
        return torch.stack(mwer_loss).sum()

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "ctc_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

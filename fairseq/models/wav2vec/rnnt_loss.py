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

@dataclass
class RNNTCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    clamp: float = field(
        default=1.0,
        metadata={"help": ""},
    )


@register_criterion("rnnt_loss", dataclass=RNNTCriterionConfig)
class RNNTCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, clamp):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.clamp = clamp
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum", clamp=1.0)
        self.blank_idx = 0

    def forward(self, model, sample, reduce=True):

        targets = sample['target']
        target_lengths = sample['target_lengths'].type('torch.IntTensor').cuda()

        encoder_output, output, src_lengths = model(sample)
        src_lengths = src_lengths.type('torch.IntTensor').cuda()

        loss = self.loss(output, targets, src_lengths, target_lengths)

        # import pdb; pdb.set_trace()

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        if not self.training: 
            # beam search
            self.decoder = RNNTBeamSearch(model.rnnt_model, self.blank_idx)
            hyps = self.decoder._search(encoder_output, None, 20)
            print(hyps[0])

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True


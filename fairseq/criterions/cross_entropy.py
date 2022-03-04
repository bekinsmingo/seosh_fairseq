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


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True) # B, T, vocab
        lprobs = lprobs.view(-1, lprobs.size(-1)) # B*T, vocab <-
        target = model.get_targets(sample, net_output).view(-1)
        '''
        (Pdb) target
        tensor([ 6, 11,  5,  ...,  5, 12,  2], device='cuda:0')
        (Pdb) lprobs
        tensor([[-3.4365, -5.2477, -7.7221,  ..., -3.7599, -7.0720, -3.8067],
                [-5.1323, -4.1260, -5.5255,  ..., -3.6365, -4.0388, -4.0313],
                [-4.7680, -5.4607, -2.6914,  ..., -5.1394, -4.6500, -2.5638],
                ...,
                [-4.8258, -3.7393, -5.3659,  ..., -3.1959, -5.3363, -1.4176],
                [-7.8528, -3.9161, -6.2376,  ..., -1.8106, -6.6340, -1.3399],
                [-5.3327, -5.6267, -7.0789,  ..., -3.9786, -7.4802, -3.0330]],
            device='cuda:0', grad_fn=<ViewBackward>)
        '''
        # import pdb; pdb.set_trace()
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

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
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

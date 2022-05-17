# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
from omegaconf import II

import torch
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class MaskedLmConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("masked_lm", dataclass=MaskedLmConfig)
class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: MaskedLmConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        logits = model(**sample["net_input"], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        #import pdb; pdb.set_trace()
        '''
        (Pdb) sample['target'][-1][-20:]
        tensor([ 1,  1,  1,  1,  1, 10,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                1,  1], device='cuda:0')
        (Pdb) masked_tokens[-1][-20:]
        tensor([False, False, False, False, False,  True, False, False, False, False,
                False, False, False, False, False, False, False, False, False, False],
            device='cuda:0')
        (Pdb) sample["net_input"]['src_tokens'][-1][-20:]
        tensor([  114,    89,    16,    45,  7308, 50264,  9506,    10,  9506,  2649,
                22682,  4321, 31417, 19508, 22465,     8, 30748,  2829,     4,     2],
            device='cuda:0')
        (Pdb) model.get_targets(sample, [logits])[-1][-20:]
        tensor([ 1,  1,  1,  1,  1, 10,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                1,  1], device='cuda:0')
        (Pdb) model.get_targets(sample, [logits]).size()
        torch.Size([16, 499])
        (Pdb) masked_tokens.size()
        torch.Size([16, 499])
        (Pdb) targets.size()
        torch.Size([1198])
        (Pdb) targets.ne(self.padding_idx)
        tensor([True, True, True,  ..., True, True, True], device='cuda:0')
        (Pdb) torch.sum(targets.ne(self.padding_idx))
        tensor(1198, device='cuda:0')
        '''


        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
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

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
from fairseq.tasks.audio_finetuning import AudioFinetuningTask, AudioFinetuningConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from fairseq.tasks import FairseqTask, register_task
from fairseq import utils
from fairseq.logging import metrics

from fairseq.optim.amp_optimizer import AMPOptimizer

from omegaconf import MISSING, II, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class DVRLAudioFinetuningConfig(AudioFinetuningConfig):
    tmp: bool = field(
        default=False,
        metadata={"help": ""},
    )
    dvrl_valid_subset: str = field(
        default='tiny_test_other', metadata={"help": "path to wav2vec ctc model"}
    )
    dvrl_valid_max_tokens: int = II("dataset.max_tokens")
    dvrl_valid_max_sentences: int = II("dataset.batch_size")
    dvrl_valid_max_positions: int = II("dataset.max_tokens")
    dvrl_valid_num_workers: int = II("dataset.num_workers")
    dvrl_valid_data_buffer_size: int = II("dataset.data_buffer_size")

@register_task("dvrl_audio_finetuning", dataclass=DVRLAudioFinetuningConfig)
class DVRLAudioFinetuningTask(AudioFinetuningTask):
    cfg: DVRLAudioFinetuningConfig
    def __init__(
        self,
        cfg: DVRLAudioFinetuningConfig,
    ):
        super().__init__(cfg)

        self.cfg = cfg

        subset = cfg.dvrl_valid_subset
        self.load_dataset(subset, cfg, combine=False, epoch=1)
        self.valid_subset_for_dve_training = self.get_valid_iterator(subset, cfg)


    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, optimizer, self.valid_subset_for_dve_training)
        if ignore_grad:
            loss *= 0
        # with torch.autograd.profiler.record_function("backward"):
        #     optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def get_valid_iterator(
        self,
        subset,
        cfg,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.get_batch_iterator(
            dataset=self.dataset(subset),
            max_tokens=cfg.dvrl_valid_max_tokens,
            max_sentences=cfg.dvrl_valid_max_sentences,
            max_positions=utils.resolve_max_positions(
                self.max_positions(),
                cfg.dvrl_valid_max_tokens,
            ),
            seed=1,
            num_workers=cfg.dvrl_valid_num_workers,
            epoch=1e+8,
            data_buffer_size=cfg.dvrl_valid_data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=False,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def reset_dummy_batch(self, batch):
        self._dummy_batch = batch

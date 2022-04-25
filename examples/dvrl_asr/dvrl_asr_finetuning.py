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
from fairseq.tasks.audio_pretraining import AudioFinetuningTask, AudioFinetuningConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from fairseq.tasks import FairseqTask, register_task
from fairseq import utils
from fairseq.logging import metrics

from fairseq.optim.amp_optimizer import AMPOptimizer


logger = logging.getLogger(__name__)


@dataclass
class DVRLAudioFinetuning(AudioFinetuningConfig):
    tmp: bool = field(
        default=False,
        metadata={"help": ""},
    )

@register_task("dvrl_audio_finetuning", dataclass=DVRLAudioFinetuning)
class DVRLAudioFinetuningTask(AudioFinetuningTask):
    cfg: DVRLAudioFinetuning
    def __init__(
        self,
        cfg: DVRLAudioFinetuning,
    ):
        super().__init__(cfg)
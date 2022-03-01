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
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from fairseq.tasks import FairseqTask, register_task
from fairseq import utils
from fairseq.logging import metrics

from fairseq.optim.amp_optimizer import AMPOptimizer


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


def label_len_fn(label):
    return len(label.split(" "))


@dataclass
class Data2VecAudioTextFinetuning(AudioPretrainingConfig):
    pretraining: bool = field(
        default=False,
        metadata={"help": ""},
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": ""},
    )
    # mask_prob: float = field(
    #     default=0.15,
    #     metadata={"help": ""},
    # )
    refine_interation: int = field(
        default=5,
        metadata={"help": ""},
    )


@register_task("data2vec_audio_text_finetuning", dataclass=Data2VecAudioTextFinetuning)
class Data2VecAudioTextFinetuningTask(AudioPretrainingTask):
    """ """

    cfg: Data2VecAudioTextFinetuning

    def __init__(
        self,
        cfg: Data2VecAudioTextFinetuning,
    ):
        super().__init__(cfg)
        self.blank_symbol = "<s>"
        self.state.add_factory("target_dictionary", self.load_target_dictionary)

        # import pdb; pdb.set_trace()

    def load_target_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.labels}.txt")
            return Dictionary.load(dict_path)
        return None

    def load_dataset(
        self, split: str, task_cfg: Data2VecAudioTextFinetuning = None, **kwargs
    ):
        # Use load dataset of Audio Pretraining Task and add some finetuning 
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        data_path = self.cfg.data
        
        # 이부분이 ltr 인지 wrd, phn 인지를 구분함
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        # label_path = os.path.join(data_path, f"{split}.bin")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        text_compressor = TextCompressor(level=text_compression_level)
        with open(label_path, "r") as f:
            labels = [
                text_compressor.compress(l)
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        process_label = LabelEncoder(self.target_dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
            label_len_fn=label_len_fn,
            add_to_input=False,
            bos=self.target_dictionary.bos(),
            add_bos_and_eos_to_input=True,
            text_compression_level=text_compression_level,
        )

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        # print('update_num',update_num)
        if self.cfg.pretraining:
            # model.set_num_updates(update_num)
            freeze_module_params(model.audio_encoder_ctc)
            freeze_module_params(model.text_encoder)
            # pass
        else:
            freeze_module_params(model.audio_encoder_ctc)
            # model.audio_encoder_ctc.eval()
            freeze_module_params(model.text_encoder)
            # model.text_encoder.eval()

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0

        # import pdb; pdb.set_trace()
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)
        return model

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)


def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False


def check_model_freezed(m):
    if m is not None:
        for n, p in m.named_parameters():
            print(n,p.requires_grad)
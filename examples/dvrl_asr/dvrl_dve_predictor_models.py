
import contextlib
import copy
import logging
import math
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)


from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)

from fairseq.models.wav2vec_asr import (
    Wav2Vec2AsrConfig,
    Wav2VecEncoder,
    Wav2VecCtc,
)

from torch.distributions import Categorical


@dataclass
class Wav2Vec2CtcConfig(Wav2Vec2AsrConfig):
    ## configs for w2v_ctc (refer to Wav2Vec2AsrConfig)
    blank_weight: float = 0
    blank_mode: str = "add"

    ## configs for DVE


@register_model("dvrl_asr", dataclass=Wav2Vec2CtcConfig)
class DVRLforASR(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_ctc, dve, sampler):
        super().__init__()
        self.cfg = cfg
        self.w2v_ctc = w2v_ctc
        self.dve = dve
        self.sampler = sampler


    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_ctc = cls.build_asr_model(cfg, task)
        dve = cls.build_dve_model(cfg, task)
        sampler = cls.build_sampler(cfg)
        return DVRLforASR(cfg, w2v_ctc, dve, sampler)

    ## Task Specific Predictor, for ASR, w2v2-CTC 
    def build_asr_model(cls, cfg, task):
        w2v_encoder = Wav2VecEncoder(cfg, len(task.target_dictionary))
        w2v_ctc = Wav2VecCtc(cfg, w2v_encoder)
        return w2v_ctc

    ## Data Value Evaluator
    def build_dve_model(cls, cfg, task):
        w2v_encoder = Wav2VecEncoder(cfg, len(task.target_dictionary))  
        dve = DVE(cfg, w2v_encoder)
        return dve


    def build_sampler(cls, cfg):
        return Sampler(cfg.encoder_embed_dim)

    def forward(**kwargs):
        '''
        The entire network consists of two model, DVE and predictor.
        The whole process is as follows;
        1. DVE cosnume data inputs and output how good is this sample for learning. (selection probabilities)
        2. Sampler decide wheter use these samples according to values. 
        3. Predictor finally output log_probs (loss) using data samples.
        4. Update both DVE and Predictor using Gradient Descent (Optimization) 
        '''

        ## 1. Get selection probabilities and Sample a selection vector
        dve_out = self.dve(**kwargs)

        ## 2. for N times, repeat sampling and computing loss (for update predictior model)
        for i in range(10):
            samples = self.sampler(dev_out)
            asr_out = self.w2v_ctc(**kwargs)

        ## "The loss of the predictor model is evaluated on a small validation set"
        ## update DVE and baseline (for stability) is on criterion.py
        pass 


## output multinomial distribution, selection prob
class DVE(BaseFairseqModel):
    def __init__(self, cfg, w2v_encoder):
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.proj = Linear(w2v_encoder.encoder.embedding_dim, 10)

    def forward(self, **kwargs):
        x = self.w2v2_encoder.w2v_model.extract_features(**w2v_args)
        out = self.proj(x)
        return out


## output binomial distribution using DVE value output, select or not
class Sampler(nn.Module):
     def __init__(self, input_dim):
        super(Sampler, self).__init__()
        self.proj = Linear(input_dim, 2)
     def forward(self, x):
        prob = F.softmax(self.proj(x))
        prob_dist = Categorical(probs)
        '''
        probs = torch.Tensor([ [0.1, 0.2, 0.7], [0.1, 0.8, 0.1], [0.4, 0.5, 0.1] ])
        prob_dist = torch.distributions.Categorical(probs) # probs should be of size batch x classes
        prob_dist.sample() # tensor([2, 1, 1])
        '''
        return prob_dist.sample()


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
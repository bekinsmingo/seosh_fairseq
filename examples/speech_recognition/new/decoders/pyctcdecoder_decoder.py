#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os.path as osp
import warnings
from collections import deque, namedtuple
from typing import Any, Dict, Tuple

import numpy as np
import torch
from fairseq import tasks
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models.fairseq_model import FairseqModel
from fairseq.utils import apply_to_sample
from omegaconf import open_dict, OmegaConf

from typing import List

from .decoder_config import FlashlightDecoderConfig
from .base_decoder import BaseDecoder


import multiprocessing
import kenlm
from pyctcdecode import build_ctcdecoder
import time


class PyCTCDecoder(BaseDecoder):
    def __init__(self, cfg: FlashlightDecoderConfig, tgt_dict: Dictionary) -> None:
        super().__init__(tgt_dict)

        # import pdb; pdb.set_trace()

        self.labels = [k for k, v in tgt_dict.indices.items()]
        self.pyctcdecoder = build_ctcdecoder(
            labels = self.labels,
            kenlm_model_path = cfg.lmpath,
            alpha=0.5,  # tuned on a val set
            beta=1.0,  # tuned on a val set
        )

        self.beam_size = cfg.beam

        # self.blank is '<s>'
        # self.silence is '|'

    def get_timesteps(self, token_idxs: List[int]) -> List[int]:
        """Returns frame numbers corresponding to every non-blank token.

        Parameters
        ----------
        token_idxs : List[int]
            IDs of decoded tokens.

        Returns
        -------
        List[int]
            Frame numbers corresponding to every non-blank token.
        """
        timesteps = []
        for i, token_idx in enumerate(token_idxs):
            if token_idx == self.blank:
                continue
            if i == 0 or token_idx != token_idxs[i-1]:
                timesteps.append(i)
        return timesteps

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        B, T, N = emissions.size()
        hypos = []


        with multiprocessing.get_context("fork").Pool(15) as pool:
            pred_list = self.pyctcdecoder.decode_batch(pool=pool, logits_list = emissions.numpy(), beam_width=self.beam_size)

        with multiprocessing.get_context("fork").Pool(15) as pool: pred_list = self.pyctcdecoder.decode_batch(pool=pool, logits_list = emissions.numpy(), beam_width=self.beam_size)

        # total time of flashlight -> 6.421210050582886
        # total time of pyctcdecoder -> 4.171117305755615

        '''
        (Pdb) ' '.join(hypos[0][0]['words']);  pred_list[0]
        ## flashlight
        "I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR SINCE THOSE EARLY DAYS IN WHICH 
        I MADE THE MEDALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK 
        I CAN ALSO OUTDO THE COINS I STRUCK FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU 
        LARGE PIECES OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS OF FRANCE THANKS TO THE GREAT CONVENIENCES 
        HE ALLOWED ME WITHOUT EVER LOSING TIME FOR THE EXECUTION OF COLOSSAL STATUES OR OTHER WORKS OF THE SCULPTOR'S CRAFT"

        ## pyctcdecoder
        "I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR SINCE THOSE EARLY DAYS IN WHICH 
        I MADE THE MEDALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK 
        I CAN ALSO OUTDO THE COINS I STRUCK FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU 
        LARGE PIECES OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS OF FRANCE THANKS TO THE GREAT CONVENIENCES 
        HE ALLOWED ME WITHOUT EVER LOSING TIME FOR THE EXECUTION OF COLOSSAL STATUES OR OTHER WORKS OF THE SCULPTOR'S CRAFT"
        '''
            
        return pred_list
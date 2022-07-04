# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import regex as re
from dataclasses import dataclass, field
from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass

R_SPACE = re.compile(r"\s+")
R_PIPE = re.compile(r"\s*\|+\s*")

@dataclass
class koTokenizerConfig(FairseqDataclass):
    to_morph: bool = field(default=False, metadata = {"help": "tmp"})

@register_tokenizer("ko_tokenizer", dataclass=koTokenizerConfig)
class KoTokenizer(object):
    def __init__(self, cfg):
        self.to_morph = cfg.to_morph
        print('to_morph: {}'.format(str(self.to_morph)))
        
        if self.to_morph:
            try:
                from konlpy.tag import Mecab
                self.tokenize = Mecab()
            except ImportError:
                raise ImportError("Please install konlpy with: pip install konlpy")
    
    def encode(self, x: str) -> str:
        return x

    def decode(self, x: str) -> str:
        x = R_SPACE.sub("", x)
        x = R_PIPE.sub(" ", x).strip()
        if self.to_morph:
            return self._morphs(x)
        return x
    
    def _morphs(self, x: str) -> str:
        result = " ".join(self.tokenize.morphs(x.strip()))
        return result

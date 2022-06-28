#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from typing import List, Dict

from .base_decoder import BaseDecoder


class ViterbiDecoder(BaseDecoder):
    def __init__(self, cfg, tgt_dict) -> None:
        super().__init__(tgt_dict)
        self.cfg = cfg

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        '''
        (Pdb) e.size()
        torch.Size([1725, 32])

        (Pdb) e[0]
        tensor([ 15.5281, -13.3430, -13.4655, -13.3544,  -0.2461,  -0.5708,  -1.1782,
                -1.5324,  -0.9952,  -1.6902,  -1.2511,  -0.8639,  -1.6885,  -2.0616,
                -1.9179,  -1.7360,  -2.7441,  -3.3512,  -2.9009,  -3.0602,  -2.7191,
                -2.7791,  -3.6425,  -4.1138,  -2.7297,  -3.9279,  -3.4587,  -4.1157,
                -5.9769,  -5.6747,  -6.0626,  -4.7225])
        '''
        def get_pred(e):
            # # import pdb; pdb.set_trace()
            # from itertools import groupby
            # import torch.nn.functional as F
            # # ctc_probs, ctc_ids = torch.exp(e).max(dim=-1) # if it's log prob
            # ctc_probs, ctc_ids = e.max(dim=-1)
            # mask = (ctc_ids==0)
            # y_hat = torch.stack([x[0] for x in groupby(ctc_ids)])
            # ctc_probs.masked_fill(mask,0).sum()
            # '''
            # (Pdb) ctc_probs.masked_fill(mask,0).sum()
            # tensor(9727.6719)
            # (Pdb) ctc_probs.sum()                                                                                                                           
            # tensor(21543.5996)
            # '''

            # ctc_probs, ctc_ids = (F.log_softmax(e,-1)).max(dim=-1)
            # mask = (ctc_ids==0)
            # y_hat = torch.stack([x[0] for x in groupby(ctc_ids)])
            # ctc_probs.masked_fill(mask,0).sum()
            # '''
            # tensor(-10.7992)
            # '''

            # ctc_probs, ctc_ids = torch.exp(F.log_softmax(e,-1)).max(dim=-1)
            # mask = (ctc_ids==0)
            # y_hat = torch.stack([x[0] for x in groupby(ctc_ids)])
            # ctc_probs.masked_fill(mask,0).sum()

            toks = e.argmax(dim=-1).unique_consecutive()
            return toks[toks != self.blank]

        return [[{"tokens": get_pred(x), "score": 0}] for x in emissions]

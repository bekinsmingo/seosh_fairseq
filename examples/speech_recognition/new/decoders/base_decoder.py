# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools as it
from typing import Any, Dict, List

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.fairseq_model import FairseqModel
import time

from fairseq import checkpoint_utils, distributed_utils, progress_bar, tasks, utils

from pdb import set_trace as Tra


class BaseDecoder:
    def __init__(self, tgt_dict: Dictionary) -> None:
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)

        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        if "<sep>" in tgt_dict.indices:
            self.silence = tgt_dict.index("<sep>")
        elif "|" in tgt_dict.indices:
            self.silence = tgt_dict.index("|")
        else:
            self.silence = tgt_dict.eos()

    def generate(
        self, models: List[FairseqModel], sample: Dict[str, Any], **unused
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        # Tra()
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(
        self,
        models: List[FairseqModel],
        encoder_input: Dict[str, Any],
    ) -> torch.FloatTensor:
        model = models[0]

        encoder_out = model(**encoder_input)

        if 'padding_mask' in encoder_out.keys():
            ## in fairseq latest version, padding mask is renamed
            if encoder_out['padding_mask'] is None:
                output_lengths = [encoder_out['encoder_out'].size(0)]*encoder_out['encoder_out'].size(1)
            else:
                ## if all input sequences have same length, then output has no padding mask 
                output_lengths = (~encoder_out['padding_mask']).sum(-1).tolist()
        else:
            output_lengths = (~encoder_out['encoder_padding_mask']).sum(-1).tolist()

        if hasattr(model, "get_logits"):
            emissions = model.get_logits(encoder_out)
        else:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)
        # emissions = model.get_normalized_probs(encoder_out, log_probs=True) # log probability
        
        '''
        (Pdb) encoder_out['encoder_out'].transpose(0, 1)[0][0]
        tensor([ 15.5281, -13.3430, -13.4655, -13.3544,  -0.2461,  -0.5708,  -1.1782,
                -1.5324,  -0.9952,  -1.6902,  -1.2511,  -0.8639,  -1.6885,  -2.0616,
                -1.9179,  -1.7360,  -2.7441,  -3.3512,  -2.9009,  -3.0602,  -2.7191,
                -2.7791,  -3.6425,  -4.1138,  -2.7297,  -3.9279,  -3.4587,  -4.1157,
                -5.9769,  -5.6747,  -6.0626,  -4.7225], device='cuda:0')
        '''

        '''
        (Pdb) model.get_normalized_probs(encoder_out, log_probs=True).transpose(0, 1)[0][0]
        tensor([-8.3446e-07, -2.8871e+01, -2.8994e+01, -2.8883e+01, -1.5774e+01,
                -1.6099e+01, -1.6706e+01, -1.7060e+01, -1.6523e+01, -1.7218e+01,
                -1.6779e+01, -1.6392e+01, -1.7217e+01, -1.7590e+01, -1.7446e+01,
                -1.7264e+01, -1.8272e+01, -1.8879e+01, -1.8429e+01, -1.8588e+01,
                -1.8247e+01, -1.8307e+01, -1.9171e+01, -1.9642e+01, -1.8258e+01,
                -1.9456e+01, -1.8987e+01, -1.9644e+01, -2.1505e+01, -2.1203e+01,
                -2.1591e+01, -2.0251e+01], device='cuda:0')
        '''
        # return emissions.transpose(0, 1).float().cpu().contiguous()

        if self.cfg.blankcollapsethreshold < 1.0:
            # import pdb; pdb.set_trace()
            emissions = self.collapse_blanks(emissions.transpose(0, 1).float().cpu().contiguous(), output_lengths, self.cfg.blankcollapsethreshold)
            '''
            (Pdb) emissions.size()
            torch.Size([1725, 7, 32])
            
            (Pdb) self.collapse_blanks(emissions.transpose(0, 1).float().cpu().contiguous(), output_lengths, 0.99).transpose(0,1).size()
            torch.Size([1175, 7, 32])
            (Pdb) self.collapse_blanks(emissions.transpose(0, 1).float().cpu().contiguous(), output_lengths, 0.95).transpose(0,1).size()          
            torch.Size([798, 7, 32])
            (Pdb) self.collapse_blanks(emissions.transpose(0, 1).float().cpu().contiguous(), output_lengths, 0.9).transpose(0,1).size()
            torch.Size([557, 7, 32])
            '''
            return emissions
        else:
            return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs: torch.IntTensor) -> torch.LongTensor:
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        raise NotImplementedError



    def collapse_blanks(self, emissions, output_lengths, blank_collapse_threshold, use_logit=True):

        if use_logit:
            blanks = (utils.softmax(emissions.transpose(0, 1), dim=-1)).transpose(0, 1)[:, :, self.blank] > blank_collapse_threshold
            collapsed_emissions = torch.ones_like(emissions).type_as(emissions) * float("-inf")
            collapsed_emissions[:,:,self.blank] = 0 # because we use logit, it should be 0
        else:
            blanks = emissions[:, :, self.blank] > blank_collapse_threshold
            collapsed_emissions = torch.zeros_like(emissions).type_as(emissions)
            collapsed_emissions[:,:,self.blank] = 1 # because we use exact prob, dummy timestep should get blank prob as 1 


        for i in range(emissions.size(0)):
            u, c = torch.unique_consecutive(blanks[i], dim=0, return_counts=True)
            u = u.tolist()
            c = c.tolist()
            cc = []
            first_blanks = 0
            k = 0
            for j in range(len(c)):
                c[j] = min(c[j], output_lengths[i] - k)
                if u[j]:    # if blank
                    if j == 0:
                        first_blanks = c[j]
                    else:
                        if j < len(c) - 1:
                            cc.append(c[j])
                else:
                    cc += [1] * c[j]
                k += c[j]
                if k >= output_lengths[i]:
                    break
            if len(cc) == 0:    # case: every frame is a blank frame
                cc = [0]
                first_blanks = 0
            org_index = torch.cumsum(torch.tensor(cc), dim=0) + first_blanks - 1
            emission = emissions[i, org_index]
            collapsed_emissions[i, :len(emission)] = emission
            output_lengths[i] = len(emission)

        collapsed_emissions = collapsed_emissions[:, :max(output_lengths)]

        return collapsed_emissions
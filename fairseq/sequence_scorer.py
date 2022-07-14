# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
from fairseq import utils


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(
        self,
        tgt_dict,
        softmax_batch=None,
        compute_alignment=False,
        eos=None,
        symbols_to_strip_from_output=None,
    ):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            import pdb
            pdb.set_trace()

            '''
            *bos 실험*

            결론은 bos 있이 학습한 모델이 bos 없는 경우, 있는 경우 전부 좋다? 
            -> 아 맞다, 이건 좀 그런게 best모델끼리 비교하는데 40k vs 70k 라서 비교 좀 그런듯?


            ================ 1. training without bos (40k) ================
            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 0,  6, 19,  8,  4,  6, 11, 13,  5,  5,  4]], device='cuda:0')
            tensor([[ 6, 19,  8,  4,  6, 11, 13,  5,  5,  4,  2]], device='cuda:0')
            tensor([[ 8,  4,  8,  4,  2, 11,  5,  5,  5,  4,  2]], device='cuda:0')
            tensor([[0.2881, 0.3472, 0.3598, 0.9312, 0.1781, 0.4105, 0.3445, 0.6256, 0.9386,
                    0.9716, 0.1113]], device='cuda:0')

            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 2,  6, 19,  8,  4,  6, 11, 13,  5,  5,  4]], device='cuda:0')
            tensor([[ 6, 19,  8,  4,  6, 11, 13,  5,  5,  4,  2]], device='cuda:0')
            tensor([[12, 11,  8,  4, 11, 11, 10,  5,  5,  4,  2]], device='cuda:0')
            tensor([[0.1087, 0.4353, 0.6454, 0.9877, 0.1048, 0.6926, 0.6281, 0.8508, 0.9356,
                    0.9907, 0.1697]], device='cuda:0')

            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 2, 12,  8,  9, 22,  7,  4, 12, 17, 10, 15,  5, 14,  4,  9,  8,  4]],
            tensor([[12,  8,  9, 22,  7,  4, 12, 17, 10, 15,  5, 14,  4,  9,  8,  4,  2]],
            tensor([[10,  7,  4,  4,  7,  4,  2,  7, 10, 15,  5, 14,  4,  2,  8,  6, 17]],
            tensor([[0.2204, 0.2754, 0.5156, 0.4444, 0.8640, 0.9625, 0.3399, 0.7339, 0.9295,
                    0.9045, 0.9660, 0.9240, 0.9998, 0.4702, 0.8612, 0.4052, 0.2309]],

            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6, 19,  8,  4,  6, 11, 13,
                    5,  5,  4]], device='cuda:0')
            tensor([[ 6, 19,  8,  4,  6, 11, 13,  5,  5,  4,  2]], device='cuda:0')
            tensor([[ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  4,  4,  6, 11,  5,  5,
                    5,  4, 17]], device='cuda:0')
            tensor([[0.2379, 0.1723, 0.1696, 0.3470, 0.3489, 0.3933, 0.3904, 0.3614, 0.3336,
                    0.3125, 0.2991, 0.8980, 0.3719, 0.7614, 0.1134, 0.4760, 0.4913, 0.5067,
                    0.8972, 0.9454, 0.1164]], device='cuda:0')


            ================ 2. training with bos (70k) ================
            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 0,  6, 19,  8,  4,  6, 11, 13,  5,  5,  4]], device='cuda:0')
            tensor([[ 6, 19,  8,  4,  6, 11, 13,  5,  5,  4,  2]], device='cuda:0')
            tensor([[ 2, 11,  8,  4, 11, 11, 10,  5,  5,  4,  2]], device='cuda:0')
            tensor([[0.1794, 0.5462, 0.4569, 0.9413, 0.0804, 0.7287, 0.4673, 0.7823, 0.9777,
                    0.9782, 0.1678]], device='cuda:0')

            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 2,  6, 19,  8,  4,  6, 11, 13,  5,  5,  4]], device='cuda:0')
            tensor([[ 6, 19,  8,  4,  6, 11, 13,  5,  5,  4,  2]], device='cuda:0')
            tensor([[ 8,  8,  5,  4, 11, 11, 10,  5,  5,  4,  2]], device='cuda:0')
            tensor([[0.1971, 0.3839, 0.4294, 0.9121, 0.0906, 0.6030, 0.4651, 0.7309, 0.9642,
                    0.9753, 0.1166]], device='cuda:0')

            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 2, 12,  8,  9, 22,  7,  4, 12, 17, 10, 15,  5, 14,  4,  9,  8,  4]],
            tensor([[12,  8,  9, 22,  7,  4, 12, 17, 10, 15,  5, 14,  4,  9,  8,  4,  2]],
            tensor([[ 7,  4,  9,  4,  4,  4,  2,  7, 10, 15,  5, 14,  4,  2,  8,  6, 17]],
            tensor([[0.3436, 0.5625, 0.2175, 0.6814, 0.8106, 0.8345, 0.4753, 0.6018, 0.8881,
                    0.6927, 0.9684, 0.9522, 0.9999, 0.5030, 0.7621, 0.4375, 0.1978]],

            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6, 19,  8,  4,  6, 11, 13,
                    5,  5,  4]], device='cuda:0')
            tensor([[ 6, 19,  8,  4,  6, 11, 13,  5,  5,  4,  2]], device='cuda:0')
            tensor([[12,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 11,  8,  4, 17, 11, 10,  5,
                    5,  4,  6]], device='cuda:0')
            tensor([[0.0931, 0.1141, 0.1566, 0.1851, 0.1923, 0.1944, 0.1956, 0.1959, 0.1964,
                    0.1970, 0.1974, 0.8803, 0.4686, 0.9615, 0.1167, 0.7512, 0.4217, 0.7087,
                    0.9659, 0.9821, 0.1084]], device='cuda:0')


            ================ 3. 번외, tfxl cocnat loader로 Libri 학습 (500k) ================
            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 0,  6, 19,  8,  4,  6, 11, 13,  5,  5,  4]], device='cuda:0')
            tensor([[ 6, 19,  8,  4,  6, 11, 13,  5,  5,  4,  2]], device='cuda:0')
            tensor([[ 4, 11,  7, 13, 12, 11,  5,  5,  5,  4, 11]], device='cuda:0')
            tensor([[0.7392, 0.4517, 0.4070, 0.5706, 0.0750, 0.2926, 0.4273, 0.5897, 0.9637,
                    0.9732, 0.1001]], device='cuda:0')

            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 2,  6, 19,  8,  4,  6, 11, 13,  5,  5,  4]], device='cuda:0')
            tensor([[ 6, 19,  8,  4,  6, 11, 13,  5,  5,  4,  2]], device='cuda:0')
            tensor([[ 6, 11,  8,  4,  8, 11, 10,  5,  5,  4, 20]], device='cuda:0')
            tensor([[0.1969, 0.9204, 0.7070, 0.9765, 0.1723, 0.6457, 0.3805, 0.7746, 0.9741,
                    0.9863, 0.2267]], device='cuda:0')

            (Pdb) net_input['src_tokens']; sample['target']; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[1]; torch.nn.functional.softmax(decoder_out[0],dim=-1).max(dim=-1)[0]
            tensor([[ 2, 12,  8,  9, 22,  7,  4, 12, 17, 10, 15,  5, 14,  4,  9,  8,  4]],
            tensor([[12,  8,  9, 22,  7,  4, 12, 17, 10, 15,  5, 14,  4,  9,  8,  4,  2]],
            tensor([[ 6, 11,  4,  9,  7,  4, 12,  7, 10, 15,  5, 14,  4,  7,  8,  6, 17]],
            tensor([[0.1909, 0.4689, 0.5532, 0.3374, 0.9762, 0.9674, 0.1154, 0.3844, 0.9476,
                    0.9221, 0.9494, 0.9842, 1.0000, 0.4470, 0.8257, 0.4856, 0.2830]],
            '''

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=len(models) == 1, sample=sample
                ).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                if torch.is_tensor(attn):
                    attn = attn.data
                else:
                    attn = attn[0]
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append(
                [
                    {
                        "tokens": ref,
                        "score": score_i,
                        "attention": avg_attn_i,
                        "alignment": alignment,
                        "positional_scores": avg_probs_i,
                    }
                ]
            )

        return hypos

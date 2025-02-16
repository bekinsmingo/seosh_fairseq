# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset, data_utils
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from pdb import set_trace as Tra

class AddTargetDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        pad,
        eos,
        batch_targets,
        process_label=None,
        label_len_fn=None,
        add_to_input=False,
        bos=None,
        add_bos_and_eos_to_input=False,
        text_compression_level=TextCompressionLevel.none,
        s2t_src_labels=None,
        s2t_src_process_label=None,
    ):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.label_len_fn = label_len_fn
        self.add_to_input = add_to_input
        self.bos = bos
        self.add_bos_and_eos_to_input = add_bos_and_eos_to_input
        self.text_compressor = TextCompressor(level=text_compression_level)

        self.s2t_src_labels=s2t_src_labels
        self.s2t_src_process_label=s2t_src_process_label

    def get_label(self, index, process_fn=None):
        lbl = self.labels[index]
        # print('lbl', lbl)
        lbl = self.text_compressor.decompress(lbl)
        # print('lbl after', lbl)
        return lbl if process_fn is None else process_fn(lbl)

    def get_s2t_src_label(self, index, process_fn=None):
        lbl = self.s2t_src_labels[index]
        # print('src lbl', lbl)
        lbl = self.text_compressor.decompress(lbl)
        # print('src lbl after', lbl)
        return lbl if process_fn is None else process_fn(lbl)

    def __getitem__(self, index):
        item = self.dataset[index]
        # print('item',item)
        item["label"] = self.get_label(index, process_fn=self.process_label)
        # print('target', self.text_compressor.decompress(self.labels[index]))
        # print('lbl after all', item["label"])
        if self.s2t_src_labels:
            item["s2t_src_label"] = self.get_s2t_src_label(index, process_fn=self.s2t_src_process_label)
            # print('src target', self.text_compressor.decompress(self.s2t_src_labels[index]))
            # print('src lbl after all', item["s2t_src_label"])
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = self.label_len_fn(self.get_label(index))
        if self.s2t_src_labels:
            s2t_src_own_sz = len(self.get_s2t_src_label(index))
        return sz, own_sz


    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())

        if self.add_bos_and_eos_to_input:
            target = [torch.cat((torch.LongTensor([self.bos]), s['label'],torch.LongTensor([self.eos]))) for s in samples if s["id"] in indices]
        else:
            target = [s["label"] for s in samples if s["id"] in indices]

        # Tra()

        if self.s2t_src_labels:
            s2t_src_target = [s["s2t_src_label"] for s in samples if s["id"] in indices]

        # print('s2t_src_target', s2t_src_target)
        # print('target', target)

        if self.add_to_input:
            eos = torch.LongTensor([self.eos])
            prev_output_tokens = [torch.cat([eos, t], axis=-1) for t in target]
            target = [torch.cat([t, eos], axis=-1) for t in target]
            collated["net_input"]["prev_output_tokens"] = prev_output_tokens

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()

            if self.s2t_src_labels:
                collated["s2t_src_target_lengths"] = torch.LongTensor([len(t) for t in s2t_src_target])
                s2t_src_target = data_utils.collate_tokens(s2t_src_target, pad_idx=self.pad, left_pad=False)
                collated["s2t_src_ntokens"] = collated["s2t_src_target_lengths"].sum().item()

            # if getattr(collated["net_input"], "prev_output_tokens", None):
            if 'prev_output_tokens' in collated["net_input"].keys():
                collated["net_input"]["prev_output_tokens"] = data_utils.collate_tokens(
                    collated["net_input"]["prev_output_tokens"],
                    pad_idx=self.pad,
                    left_pad=False,
                )
        else:
            collated["ntokens"] = sum([len(t) for t in target])
            if self.s2t_src_labels:
                collated["s2t_src_ntokens"] = sum([len(t) for t in s2t_src_target])

        collated["target"] = target
        if self.s2t_src_labels:
            collated["s2t_src_target"] = s2t_src_target.long()

        # Tra()

        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored
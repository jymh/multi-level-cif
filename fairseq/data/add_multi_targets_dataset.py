## implemented by RYM for multiscale cif model data loading

import torch

from . import BaseWrapperDataset, data_utils
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

class AddMultiTargetsDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        phone_labels,
        char_labels,
        pad,
        eos,
        batch_targets,
        phone_process_label=None,
        char_process_label=None,
        label_len_fn=None,
        add_to_input=False,
        text_compression_level=TextCompressionLevel.none
    ):
        super().__init__(dataset)
        self.phone_labels = phone_labels
        self.char_labels = char_labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.phone_process_label = phone_process_label
        self.char_process_label = char_process_label
        self.label_len_fn = label_len_fn
        self.add_to_input = add_to_input
        self.text_compressor = TextCompressor(level=text_compression_level)

    def get_label(self, index, phone_process_fn=None, char_process_fn=None):
        p_lbl = self.phone_labels[index]
        c_lbl = self.char_labels[index]
        p_lbl = self.text_compressor.decompress(p_lbl)
        c_lbl = self.text_compressor.decompress(c_lbl)

        if phone_process_fn is not None:
            p_lbl = phone_process_fn(p_lbl)
        if char_process_fn is not None:
            c_lbl = char_process_fn(c_lbl)

        return p_lbl, c_lbl

    def __getitem__(self, index):
        item = self.dataset[index]
        item["p_label"], item["c_label"] = self.get_label(index, self.phone_process_label, self.char_process_label)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        p_lbl, c_lbl = self.get_label(index)
        p_sz = self.label_len_fn(p_lbl)
        c_sz = self.label_len_fn(c_lbl)
        own_sz = max(p_sz, c_sz)
        return sz, own_sz

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        p_target = [s["p_label"] for s in samples if s["id"] in indices]
        c_target = [s["c_label"] for s in samples if s["id"] in indices]

        p_sample = {"target": p_target}
        c_sample = {"target": c_target}
        if self.batch_targets:
            p_sample["target_lengths"] = torch.LongTensor([len(t) for t in p_target])
            p_target = data_utils.collate_tokens(p_target, pad_idx=self.pad, left_pad=False)
            p_sample["ntokens"] = p_sample["target_lengths"].sum().item()

            c_sample["target_lengths"] = torch.LongTensor([len(t) for t in c_target])
            c_target = data_utils.collate_tokens(c_target, pad_idx=self.pad, left_pad=False)
            c_sample["ntokens"] = c_sample["target_lengths"].sum().item()
        else:
            p_sample["ntokens"] = sum([len(t) for t in p_target])
            c_sample["ntokens"] = sum([len(t) for t in c_target])

        p_sample["target"] = p_target
        c_sample["target"] = c_target
        collated["p_sample"] = p_sample
        collated["c_sample"] = c_sample

        if self.add_to_input:
            p_eos = p_target.new_full((p_target.size(0), 1), self.eos)
            collated["p_sample"]["target"] = torch.cat([p_target, p_eos], dim=-1).long()
            collated["net_input"]["phone_prev_output_tokens"] = torch.cat(
                [p_eos, p_target], dim=-1
            ).long()
            collated["p_sample"]["ntokens"] += p_target.size(0)

            c_eos = c_target.new_full((c_target.size(0), 1), self.eos)
            collated["c_sample"]["target"] = torch.cat([c_target, c_eos], dim=-1).long()
            collated["net_input"]["char_prev_output_tokens"] = torch.cat(
                [c_eos, c_target], dim=-1
            ).long()
            collated["c_sample"]["ntokens"] += c_target.size(0)

        '''
        collated : { 
            "net_input": ...
            "c_sample": {
                "target"
                "target_lengths"
                "n_tokens"
            }
            "p_sample": ...
        }
        '''
        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored
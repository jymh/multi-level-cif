## implemented by RYM for multi-level lexicon constraint cif

import torch

from . import BaseWrapperDataset, data_utils
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

class AddLexiconConstraintMlDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        phone_segs,
        integrate_weights
    ):
        super().__init__(dataset)
        self.phone_segs = phone_segs
        self.integrate_weights = integrate_weights

    def __getitem__(self, index):
        item = self.dataset[index]
        item["phone_segs"], item["integrate_weights"] = torch.IntTensor(self.phone_segs[index]), torch.IntTensor(self.integrate_weights[index])
        return item

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        phone_segs = [s["phone_segs"] for s in samples if s["id"] in indices]
        integrate_weights = [s["integrate_weights"] for s in samples if s["id"] in indices]

        collated["phone_segs"] = phone_segs
        collated["integrate_weights"] = integrate_weights

        return collated


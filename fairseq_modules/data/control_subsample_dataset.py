from fairseq.data import SubsampleDataset, FairseqDataset
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ControlSubsampleDataset(SubsampleDataset):

    def __init__(self, dataset, size_ratio, shuffle=False, same_samples=False):
        super(ControlSubsampleDataset, self).__init__(dataset, size_ratio, shuffle)
        # print("same_samples: ", same_samples)
        if same_samples:
            # breakpoint()
            print(f"........ use the same samples to at every iteration for {self.dataset.datasets}, with {size_ratio} .....")
            orig_indices = np.random.choice(list(range(len(self.dataset))), len(dataset), replace=False)
            self.indices = orig_indices[:self.actual_size]
            print(" first few orig_indices before getting sampled dataset: ", orig_indices)
            print("first few sliced indices: ", self.indices[:10])
            # breakpoint()





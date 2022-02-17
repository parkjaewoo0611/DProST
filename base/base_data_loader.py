from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler = DistributedSampler(dataset, shuffle=shuffle)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory':True
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
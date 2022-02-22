from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, RandomSampler

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, rank, is_dist=False, mode='train'):
        if is_dist:
            self.sampler = DistributedSampler(dataset, rank=rank, shuffle=shuffle)
        elif 'train' in mode and shuffle:
            self.sampler =  RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)
        self.batch_idx = 0
        self.n_samples = len(self.sampler)
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': False,
            'persistent_workers': True
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
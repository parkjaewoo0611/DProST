import random
import torch.utils.data.dataset

class MultipleDatasets(torch.utils.data.dataset.Dataset):
    def __init__(self, datasets):
        self.frame_step = 1
        self.datasets = datasets
        # The begin and end indexes of datasets
        self.indexes = [0]
        for dataset, repeat_times in datasets:
            self.indexes.append(self.indexes[-1] + int(len(dataset) * repeat_times))

    def __len__(self):
        return self.indexes[-1]

    def __getitem__(self, idx):
        # Determine which dataset to use in self.datasets
        dataset_idx = 0
        for i, dataset_end_idx in enumerate(self.indexes):
            if idx < dataset_end_idx:
                dataset_idx = i - 1
                break

        dataset, repeat_times = self.datasets[dataset_idx]
        if repeat_times >= 1:
            return dataset[(idx - self.indexes[dataset_idx]) % len(dataset)]
        else:
            return dataset[random.randint(0, len(dataset) - 1)]
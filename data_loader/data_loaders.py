import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pickle
from base import BaseDataLoader
from utils.util import replace_bg, get_param
from torch.utils.data.dataset import Dataset
import glob

class PoseDataset(Dataset):
    """ load data instance of 6D pose"""
    def __init__(self, data_dir, obj_list, mode, img_ratio):
        """
            data_dir : path to root dataset dir
            obj_list : list of obj id
            mode : 'train', 'test', 'train_pbr', 'train_syn'
            img_ratio : resize scale, default = 0.5
        """
        H = int(480 * img_ratio)
        W = int(640 * img_ratio)
        self.img_ratio = img_ratio
        self.transform = transforms.Compose([transforms.ToTensor(), 
                                             transforms.ConvertImageDtype(torch.float),
                                             transforms.Resize(size=(H, W))])
        self.idx2radius = get_param(data_dir, 'idx2radius')
        self._bg_img_paths = glob.glob(f'{data_dir}/background/*')
        self.K_scaler = torch.tensor(np.diag([self.img_ratio, self.img_ratio, 1])).to(torch.float)
        self.default = Image.fromarray(np.zeros([H, W]))
        self.mode = mode
        self.data_dir = data_dir

        with open(os.path.join(data_dir, f'{mode}.pickle'), 'rb') as f:
            dataset = pickle.load(f)
        dataset = [sample for sample in dataset if sample['obj_id'] in obj_list]
        dataset = [sample for sample in dataset if sample['visib_fract'] > 0.0]
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = Image.open(self.dataset[idx]['image'])
        depth = Image.open(self.dataset[idx]['depth']) * self.dataset[idx]['depth_scale'] if self.dataset[idx]['depth'] is not None else self.default.copy()
        mask = Image.open(self.dataset[idx]['mask']) if self.dataset[idx]['mask'] is not None else self.default.copy().astype(bool)
        image = replace_bg(image, mask, self._bg_img_paths) if self.mode == 'train_syn' else image
        bbox = np.array(self.dataset[idx]['bbox_obj'].copy()) if 'train' in self.mode else np.array(self.dataset[idx]['bbox_test'].copy())
        obj_id = self.dataset[idx]['obj_id']
        RT = self.dataset[idx]['RT'].copy()
        RT[:3, 3] = RT[:3, 3] / self.idx2radius[obj_id]

        sample = {
            'images': self.transform(image),
            'depths': self.transform(depth),
            'masks' : self.transform(mask),
            'RTs' : torch.tensor(RT).to(torch.float),
            'bboxes' :  torch.tensor(bbox).to(torch.float) * self.img_ratio,
            'obj_ids' : torch.tensor(obj_id),
            'K_origins' : torch.tensor(self.dataset[idx]['K']).to(torch.float),
            'Ks' : self.K_scaler @ torch.tensor(self.dataset[idx]['K']).to(torch.float),
            'visib_fracts' : torch.tensor(self.dataset[idx]['visib_fract']),
            'px_count_visibs' : torch.tensor(self.dataset[idx]['px_count_visib'])
        }
        return sample

class DataLoader(BaseDataLoader):
    """
    DataLoader to construct batch from multiple Dataset class
    """
    def __init__(self, data_dir, batch_size, obj_list, mode='train', img_ratio=1.0, 
                 shuffle=True, num_workers=4, **kwargs):
        self.dataset = PoseDataset(data_dir, obj_list, mode, img_ratio)

        #### self.dataset --> batch
        super().__init__(self.dataset, batch_size, shuffle, num_workers)


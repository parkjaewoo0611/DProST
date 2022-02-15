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
    def __init__(self, data_dir, obj_list, name, img_ratio):
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
        self.name = name
        self.data_dir = data_dir

        dataset = pickle.load(open(os.path.join(data_dir, f'{name}.pickle'), 'rb'))
        dataset = [sample for sample in dataset if sample['obj_id'] in obj_list]
        dataset = [sample for sample in dataset if sample['visib_fract'] > 0.0]
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = Image.open(self.dataset[idx]['image'])
        depth = Image.open(self.dataset[idx]['depth']) * self.dataset[idx]['depth_scale'] if self.dataset[idx]['depth'] is not None else self.default.copy()
        mask = Image.open(self.dataset[idx]['mask']) if self.dataset[idx]['mask'] is not None else self.default.copy().astype(bool)
        image = replace_bg(image, mask, self._bg_img_paths) if self.name == 'train_syn' else image
        bbox = np.array(self.dataset[idx]['bbox_obj'].copy()) if 'train' in self.name else np.array(self.dataset[idx]['bbox_test'].copy())
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
    def __init__(self, data_dir, batch_size, obj_list, reference_N=8, is_pbr=True, is_syn=False, img_ratio=1.0, 
                 shuffle=True, validation_split=0.0, num_workers=4, training=True, FPS=True, **kwargs):
        self.reference_N = reference_N
        self.FPS = FPS
        self.is_pbr = is_pbr
        self.is_syn = is_syn
        self.training = training
        if self.training:
            self.dataset = PoseDataset(data_dir, obj_list, 'train', img_ratio)
        else:
            self.dataset = PoseDataset(data_dir, obj_list, 'test', img_ratio)

        if self.is_pbr:
            self.syn_dataset = PoseDataset(data_dir, obj_list, 'train_pbr', img_ratio)
        elif self.is_syn:
            self.syn_dataset = PoseDataset(data_dir, obj_list, 'train_syn', img_ratio)
        else:
            self.syn_dataset = PoseDataset(data_dir, obj_list, 'train', img_ratio)

        #### self.dataset --> batch
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, data):
        batch_N = len(data)
        if self.training:
            data = data + [self.syn_dataset[int(np.random.random()*len(self.syn_dataset))] for _ in range(batch_N)]
        batch = {key : torch.stack([d[key] for d in data]) for key in list(data[0].keys())}
        return batch




import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pickle
from base import BaseDataLoader
from utils.util import LM_idx2radius, farthest_rotation_sampling
import random
import torch.utils.data.dataset

class DataLoader(BaseDataLoader):
    """
    DataLoader for pickle data which has location of images to load 
    and etc labels
    """
    def __init__(self, data_dir, batch_size, obj_list, reference_N=8, is_pbr=True, img_ratio=1.0, shuffle=True, validation_split=0.0, num_workers=1, training=True, FPS=True):
        H = int(480 * img_ratio)
        W = int(640 * img_ratio)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(H, W))])
        self.data_dir = data_dir
        self.img_ratio = img_ratio
        self.drop_last = True
        self.training = training
        self.reference_N = reference_N
        self.is_pbr = is_pbr
        self.obj_list = obj_list
        self.FPS = FPS

        with open(os.path.join(self.data_dir, 'train.pickle'), 'rb') as f:
            self.dataset_train = pickle.load(f)

        if training:
            self.dataset = self.dataset_train
            if self.is_pbr:
                with open(os.path.join(data_dir, 'train_pbr.pickle'), 'rb') as f:
                    self.dataset_pbr = pickle.load(f)
                self.dataset_pbr = [(batch, target) for i, (batch, target) in enumerate(self.dataset_pbr) if batch['obj_id'] in self.obj_list]
        else:
            with open(os.path.join(data_dir, 'test.pickle'), 'rb') as f:
                self.dataset = pickle.load(f)

        self.dataset = [(batch, target) for i, (batch, target) in enumerate(self.dataset) if batch['obj_id'] in self.obj_list]
        # self.dataset = self.dataset[0:4]


        #### self.dataset --> (batch, target) tuple
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)


    def select_reference(self):
        ##### for reference
        references = {}
        for obj_id in self.obj_list:
            obj_dataset = [(batch, target) for i, (batch, target) in enumerate(self.dataset_train) if batch['obj_id'] is obj_id]
            if self.FPS:
                ref = farthest_rotation_sampling(obj_dataset, self.reference_N)
            else:
                ref = random.sample(obj_dataset, self.reference_N)
            images, masks, _, bboxes, RTs = self.collate_fn(ref, True)
        references[obj_id] = {
            'images': images,
            'masks': masks,
            'bboxes': bboxes,
            'RTs': RTs
        }
        return references


    def collate_fn(self, data, reference=False):
        """
            data : a list of tuples with (batch, target)
            batch['image'] :  path to image
            batch['obj_id'] : integer of obj_id
            target['mask'] : path to mask image
            target['bbox_obj'] : [xmin, ymin, xmax, ymax]
            target['RT'] : 4x4 matrix of [R | t]
            to
            torch tensor batch ready to go
        """  
        images = []
        masks = []
        obj_ids = []
        bboxes = []
        RTs = []
        for idx, (batch_sample, target_sample) in enumerate(data):
            images.append(self.transform(np.array(Image.open(batch_sample['image']))))
            masks.append(self.transform(np.array(Image.open(target_sample['mask']))))
            obj_ids.append(torch.tensor(batch_sample['obj_id']))
            if self.training or reference:
                bboxes.append(torch.tensor(target_sample['bbox_obj']) * self.img_ratio)
            else:
                bboxes.append(torch.tensor(target_sample['bbox_faster']) * self.img_ratio)      # 'bbox_obj', 'bbox_yolo', 'bbox_faster'
            RT = torch.tensor(target_sample['RT'], dtype=torch.float)
            RT[:3, 3] = RT[:3, 3] / LM_idx2radius[batch_sample['obj_id']]
            RTs.append(RT)

        if self.is_pbr and self.training:
            data_pbr = random.sample(self.dataset_pbr, len(data))
            for idx, (batch_sample, target_sample) in enumerate(data_pbr):
                images.append(self.transform(np.array(Image.open(batch_sample['image']))))
                masks.append(self.transform(np.array(Image.open(target_sample['mask']))))
                obj_ids.append(torch.tensor(batch_sample['obj_id']))
                bboxes.append(torch.tensor(target_sample['bbox_obj']) * self.img_ratio)
                RT = torch.tensor(target_sample['RT'], dtype=torch.float)
                RT[:3, 3] = RT[:3, 3] / LM_idx2radius[batch_sample['obj_id']]
                RTs.append(RT)

        images = torch.stack(images)
        masks = torch.stack(masks)
        obj_ids = torch.stack(obj_ids)
        bboxes = torch.stack(bboxes)
        RTs = torch.stack(RTs)
        return images, masks, obj_ids, bboxes, RTs
        
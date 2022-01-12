import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pickle
from base import BaseDataLoader
from utils.util import farthest_rotation_sampling
from utils.LM_parameter import LM_idx2radius, LM_idx2synradius
import random
import torch.utils.data.dataset

class DataLoader(BaseDataLoader):
    """
    DataLoader for pickle data which has location of images to load 
    and etc labels
    """
    def __init__(self, data_dir, batch_size, obj_list, reference_N=8, is_pbr=False, is_syn=True, img_ratio=1.0, 
                 shuffle=True, validation_split=0.0, num_workers=1, training=True, FPS=True, **kwargs):
        H = int(480 * img_ratio)
        W = int(640 * img_ratio)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(H, W))])
        self.data_dir = data_dir
        self.img_ratio = img_ratio
        self.drop_last = True
        self.training = training
        self.reference_N = reference_N
        self.is_pbr = is_pbr
        self.is_syn = is_syn
        self.obj_list = obj_list
        self.FPS = FPS

        with open(os.path.join(self.data_dir, 'train.pickle'), 'rb') as f:
            self.dataset_train = pickle.load(f)

        if training:
            self.dataset = self.dataset_train
            if self.is_pbr:
                with open(os.path.join(data_dir, 'train_pbr.pickle'), 'rb') as f:
                    self.dataset_syn = pickle.load(f)
                self.dataset_syn = [batch for i, batch in enumerate(self.dataset_syn) if batch['obj_id'] in self.obj_list]
                self.dataset_syn = [batch for i, batch in enumerate(self.dataset_syn) if batch['visib_fract'] > 0.2]
            elif self.is_syn:
                with open(os.path.join(data_dir, 'train_syn.pickle'), 'rb') as f:
                    self.dataset_syn = pickle.load(f)
                self.dataset_syn = [batch for i, batch in enumerate(self.dataset_syn) if batch['obj_id'] in self.obj_list]               
        else:
            with open(os.path.join(data_dir, 'test.pickle'), 'rb') as f:
                self.dataset = pickle.load(f)

        self.dataset = [batch for i, batch in enumerate(self.dataset) if batch['obj_id'] in self.obj_list]
        
        self.obj_dataset = {}
        for obj in self.obj_list:
            self.obj_dataset[obj] = [batch for i, batch in enumerate(self.dataset_train) if batch['obj_id'] is obj]

        #### self.dataset --> (batch, target) tuple
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)


    def select_reference(self):
        ##### for reference
        references = {}
        for obj_id in self.obj_list:
            obj_dataset = self.obj_dataset[obj_id]
            if self.FPS:
                ref = farthest_rotation_sampling(obj_dataset, self.reference_N)
            else:
                ref = random.sample(obj_dataset, self.reference_N)
            images, masks, _, _, bboxes, RTs = self.collate_fn(ref, True)
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
        depths = []
        obj_ids = []
        bboxes = []
        RTs = []
        for idx, batch_sample in enumerate(data):
            images.append(self.transform(np.array(Image.open(batch_sample['image']))))
            depths.append(torch.tensor(np.array(Image.open(batch_sample['depth'])) * batch_sample['depth_scale']))
            if batch_sample['mask'] is None:
                mask = np.array(Image.open(batch_sample['depth'])).astype(bool)
            else:
                mask = np.array(Image.open(batch_sample['mask']))
            masks.append(self.transform(mask))
            obj_ids.append(torch.tensor(batch_sample['obj_id']))
            if self.training or reference:
                bboxes.append(torch.tensor(batch_sample['bbox_obj']) * self.img_ratio)
            else:
                bboxes.append(torch.tensor(batch_sample['bbox_faster']) * self.img_ratio)      # 'bbox_obj', 'bbox_yolo', 'bbox_faster'
            RT = torch.tensor(batch_sample['RT'], dtype=torch.float)
            RT[:3, 3] = RT[:3, 3] / LM_idx2radius[batch_sample['obj_id']]
            RTs.append(RT)

        if (self.is_pbr or self.is_syn) and self.training and not reference:
            dataset_syn = random.sample(self.dataset_syn, len(data))
            for idx, batch_sample in enumerate(dataset_syn):
                images.append(self.transform(np.array(Image.open(batch_sample['image']))))
                if self.is_pbr:
                    depths.append(torch.tensor(np.array(Image.open(batch_sample['depth'])) * batch_sample['depth_scale']))
                else:
                    depths.append(torch.zeros_like(depths[0]))
                
                if batch_sample['mask'] is None:
                    mask = np.array(Image.open(batch_sample['depth'])).astype(bool)
                else:
                    mask = np.array(Image.open(batch_sample['mask']))
                masks.append(self.transform(mask))
                obj_ids.append(torch.tensor(batch_sample['obj_id']))
                bboxes.append(torch.tensor(batch_sample['bbox_obj']) * self.img_ratio)
                RT = torch.tensor(batch_sample['RT'], dtype=torch.float)
                if self.is_pbr:
                    RT[:3, 3] = RT[:3, 3] / LM_idx2radius[batch_sample['obj_id']]
                else:
                    RT[:3, 3] = RT[:3, 3] / LM_idx2synradius[batch_sample['obj_id']]
                RTs.append(RT)

        images = torch.stack(images)
        depths = torch.stack(depths)
        masks = torch.stack(masks)
        obj_ids = torch.stack(obj_ids)
        bboxes = torch.stack(bboxes)
        RTs = torch.stack(RTs)
        return images, masks, depths, obj_ids, bboxes, RTs
        
import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pickle
from base import BaseDataLoader
from utils.util import LM_idx2radius
import random
import torch.utils.data.dataset
from pytorch3d.transforms import so3_relative_angle

class DataLoader(BaseDataLoader):
    """
    DataLoader for pickle data which has location of images to load 
    and etc labels
    """
    def __init__(self, data_dir, batch_size, obj_list, is_pbr=False, img_ratio=1.0, shuffle=True, validation_split=0.0, num_workers=1, training=True, FPS=True):
        H = int(480 * img_ratio)
        W = int(640 * img_ratio)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(H, W))])
        self.data_dir = data_dir
        self.img_ratio = img_ratio
        self.drop_last = True
        self.training = training
        self.reference_N = 8
        
        if training:
            with open(os.path.join(data_dir, 'train_pbr.pickle'), 'rb') as f:
                dataset_pbr = pickle.load(f)
            with open(os.path.join(data_dir, 'train.pickle'), 'rb') as f:
                dataset = pickle.load(f)
            dataset_pbr.extend(dataset * (len(dataset_pbr)//len(dataset)))
            self.dataset = dataset
        else:
            with open(os.path.join(data_dir, 'test.pickle'), 'rb') as f:
                self.dataset = pickle.load(f)

        self.dataset = [(batch, target) for i, (batch, target) in enumerate(self.dataset) if batch['obj_id'] in obj_list]
        # self.dataset = self.dataset[0:4]

        ##### for reference

        with open(os.path.join(data_dir, 'train.pickle'), 'rb') as f:
            dataset = pickle.load(f)
        references = {}
        self.references = {}
        for obj_id in obj_list:
            obj_dataset = [(batch, target) for i, (batch, target) in enumerate(dataset) if batch['obj_id'] is obj_id]
            if FPS:
                references[obj_id] = self.farthest_rotation_sampling(obj_dataset, self.reference_N)
            else:
                references[obj_id] = random.sample(obj_dataset, self.reference_N)
            images, masks, _, bboxes, RTs = self.collate_fn(references[obj_id], True)
            self.references[obj_id] = {
                'images': images,
                'masks': masks,
                'bboxes': bboxes,
                'RTs': RTs
            }

        #### self.dataset --> (batch, target) tuple
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)

    def farthest_rotation_sampling(self, dataset, N):
        references = []
        farthest_idx = np.zeros(N)
        farthest_Rs = np.zeros([N, 3, 3])
        Rs = torch.tensor(np.stack([target['RT'][:3, :3] for i, (batch, target) in enumerate(dataset)]))
        farthest_idx[0] = np.random.randint(Rs.shape[0])
        farthest_Rs[0] = Rs[int(farthest_idx[0])]
        distances = so3_relative_angle(torch.tensor(farthest_Rs[0]).unsqueeze(0).repeat(Rs.shape[0], 1, 1), Rs)
        for i in range(1, N):
            farthest_idx[i] = torch.argmax(distances)
            farthest_Rs[i] = Rs[int(farthest_idx[i])]
            distances = torch.minimum(distances, so3_relative_angle(torch.tensor(farthest_Rs[i]).unsqueeze(0).repeat(Rs.shape[0], 1, 1), Rs))

        for idx in list(farthest_idx.astype(int)):
            references.append(dataset[idx])
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

        images = torch.stack(images)
        masks = torch.stack(masks)
        obj_ids = torch.stack(obj_ids)
        bboxes = torch.stack(bboxes)
        RTs = torch.stack(RTs)
        return images, masks, obj_ids, bboxes, RTs
        

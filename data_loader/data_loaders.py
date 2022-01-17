import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pickle
from base import BaseDataLoader
from utils.util import farthest_rotation_sampling, resize_short_edge
from utils.LM_parameter import LM_idx2radius, LM_idx2synradius
import random
import torch.utils.data.dataset
import glob

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
        self.training = training
        self.reference_N = reference_N
        self.is_pbr = is_pbr
        self.is_syn = is_syn
        self.obj_list = obj_list
        self.FPS = FPS
        if self.training:                
            self._bg_img_paths = glob.glob(f'{data_dir}/background/*')

        with open(os.path.join(self.data_dir, 'train.pickle'), 'rb') as f:
            self.dataset_train = pickle.load(f)

        #### load real dataset indexes
        if training:
            file = 'train'
        else:
            file = 'test'
        with open(os.path.join(self.data_dir, f'{file}.pickle'), 'rb') as f:
            dataset = pickle.load(f)
        self.dataset = [batch for i, batch in enumerate(dataset) if batch['obj_id'] in self.obj_list]

        #### load synthetic dataset indexes
        if training and (self.is_pbr or self.is_syn):
            if self.is_pbr:
                file_syn = 'train_pbr'
            elif self.is_syn:
                file_syn = 'train_syn'      
            with open(os.path.join(self.data_dir, f'{file_syn}.pickle'), 'rb') as f:
                dataset_syn = pickle.load(f)
            dataset_syn = [batch for i, batch in enumerate(dataset_syn) if batch['obj_id'] in self.obj_list]
            self.dataset_syn = [batch for i, batch in enumerate(dataset_syn) if batch['visib_fract'] > 0.2]

        self.obj_dataset = {}
        for obj in self.obj_list:
            self.obj_dataset[obj] = [batch for i, batch in enumerate(self.dataset_train) if batch['obj_id'] is obj]

        #### self.dataset --> batch
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, data, reference=False):
        """
            data : batch dictionary
            batch['image'] :  path to image
            batch['obj_id'] : integer of obj_id
            batch['mask'] : path to mask image
            batch['bbox_obj'] : [xmin, ymin, xmax, ymax]
            batch['RT'] : 4x4 matrix of [R | t]
            to
            torch tensor batch ready to go
        """  
        images = []
        depths = []
        obj_ids = []
        RTs = []
        masks = []
        bboxes = []
        for idx, batch_sample in enumerate(data):
            image = self.load_img(batch_sample['image'])
            obj_id = batch_sample['obj_id']
            RT = batch_sample['RT'].copy()
            RT[:3, 3] = RT[:3, 3] / LM_idx2radius[batch_sample['obj_id']]
            if batch_sample['depth'] is not None:
                depth = self.load_img(batch_sample['depth'])* batch_sample['depth_scale']
            else:
                depth = np.zeros_like(image[...,0])

            if batch_sample['mask'] is not None:
                mask = self.load_img(batch_sample['mask'])
            else:
                mask = depth.copy().astype(bool)

            if self.training or reference:
                bbox = np.array(batch_sample['bbox_obj'].copy()) * self.img_ratio
            else:
                bbox = np.array(batch_sample['bbox_faster'].copy()) * self.img_ratio    # 'bbox_obj', 'bbox_yolo', 'bbox_faster'
            images.append(self.transform(image))
            obj_ids.append(torch.tensor(obj_id))
            RTs.append(torch.tensor(RT, dtype=torch.float32))
            depths.append(torch.tensor(depth, dtype=torch.float32))
            masks.append(self.transform(mask))
            bboxes.append(torch.tensor(bbox, dtype=torch.float32))

        if self.training and (self.is_pbr or self.is_syn) and not reference:
            dataset_syn = random.sample(self.dataset_syn, len(data))
            for idx, batch_sample in enumerate(dataset_syn):
                image = self.load_img(batch_sample['image'])
                obj_id = batch_sample['obj_id']
                RT = batch_sample['RT'].copy()
                if self.is_pbr:
                    RT[:3, 3] = RT[:3, 3] / LM_idx2radius[batch_sample['obj_id']]
                else:
                    RT[:3, 3] = RT[:3, 3] / LM_idx2synradius[batch_sample['obj_id']]

                if batch_sample['depth'] is not None:
                    depth = self.load_img(batch_sample['depth'])* batch_sample['depth_scale']
                else:
                    depth = np.zeros_like(image[...,0])

                if batch_sample['mask'] is not None:
                    mask = self.load_img(batch_sample['mask'])
                else:
                    mask = depth.copy().astype(bool)
                
                if self.is_syn:
                    image = self.replace_bg(image, mask)
                else:
                    pass

                bbox = np.array(batch_sample['bbox_obj'].copy()) * self.img_ratio

                images.append(self.transform(image))
                obj_ids.append(torch.tensor(obj_id))
                RTs.append(torch.tensor(RT, dtype=torch.float32))
                depths.append(torch.tensor(depth, dtype=torch.float32))
                masks.append(self.transform(mask))
                bboxes.append(torch.tensor(bbox, dtype=torch.float32))

        images = torch.stack(images)
        depths = torch.stack(depths)
        masks = torch.stack(masks)
        obj_ids = torch.stack(obj_ids)
        bboxes = torch.stack(bboxes)
        RTs = torch.stack(RTs)
        return images, masks, depths, obj_ids, bboxes, RTs

    def load_img(self, path):
        return np.array(Image.open(path))

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

    def replace_bg(self, im, im_mask, return_mask=False):
        ## editted from GDR-Net git https://github.com/THU-DA-6D-Pose-Group/GDR-Net/core/base_data_loader.py
        # add background to the image
        H, W = im.shape[:2]
        ind = random.randint(0, len(self._bg_img_paths) - 1)
        filename = self._bg_img_paths[ind]
        bg_img = self.get_bg_image(filename, H, W)

        if len(bg_img.shape) != 3:
            bg_img = np.zeros((H, W, 3), dtype=np.uint8)

        mask = im_mask.copy().astype(np.bool)
        nonzeros = np.nonzero(mask.astype(np.uint8))
        x1, y1 = np.min(nonzeros, axis=1)
        x2, y2 = np.max(nonzeros, axis=1)
        c_h = 0.5 * (x1 + x2)
        c_w = 0.5 * (y1 + y2)
        rnd = random.random()
        # print(x1, x2, y1, y2, c_h, c_w, rnd, mask.shape)
        if rnd < 0.2:  # block upper
            c_h_ = int(random.uniform(x1, c_h))
            mask[:c_h_, :] = False
        elif rnd < 0.4:  # block bottom
            c_h_ = int(random.uniform(c_h, x2))
            mask[c_h_:, :] = False
        elif rnd < 0.6:  # block left
            c_w_ = int(random.uniform(y1, c_w))
            mask[:, :c_w_] = False
        elif rnd < 0.8:  # block right
            c_w_ = int(random.uniform(c_w, y2))
            mask[:, c_w_:] = False
        else:
            pass
        mask_bg = ~mask
        im[mask_bg] = bg_img[mask_bg]
        im = im.astype(np.uint8)
        if return_mask:
            return im, mask  # bool fg mask
        else:
            return im


    def get_bg_image(self, filename, imH, imW, channel=3):
        ## editted from GDR-Net git https://github.com/THU-DA-6D-Pose-Group/GDR-Net/core/base_data_loader.py
        """keep aspect ratio of bg during resize target image size:

        imHximWxchannel.
        """
        target_size = min(imH, imW)
        max_size = max(imH, imW)
        real_hw_ratio = float(imH) / float(imW)
        bg_image = np.array(Image.open(filename))

        bg_h, bg_w, bg_c = bg_image.shape
        bg_image_resize = np.zeros((imH, imW, channel), dtype="uint8")
        if (float(imH) / float(imW) < 1 and float(bg_h) / float(bg_w) < 1) or (
            float(imH) / float(imW) >= 1 and float(bg_h) / float(bg_w) >= 1
        ):
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                if bg_h_new < bg_h:
                    bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                else:
                    bg_image_crop = bg_image
            else:
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                if bg_w_new < bg_w:
                    bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                else:
                    bg_image_crop = bg_image
        else:
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:  # bg_h < bg_w
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                # logger.info(bg_w_new)
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
        bg_image_resize_0 = resize_short_edge(bg_image_crop, target_size, max_size)
        h, w, c = bg_image_resize_0.shape
        bg_image_resize[0:h, 0:w, :] = bg_image_resize_0
        return bg_image_resize
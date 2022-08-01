import torch
import random 
import numpy as np
from pytorch3d.transforms import so3_relative_angle
from utils.util import image_cropping, squaring_boxes, get_K_crop_resize
import torch.nn.functional as F

class ReferenceLoader():
    def __init__(self, ref_dataset, reference_N, FPS, obj_list, ref_size, img_ratio, N_z, use_mesh, device='cpu', **kwargs):
        self.obj_list = obj_list
        self.ref_N = reference_N
        self.ref_size = ref_size
        self.is_FPS = FPS
        self.device = device
        self.use_mesh = use_mesh
        self.H = int(480 * img_ratio)
        self.W = int(640 * img_ratio)
        self.N_z = N_z
        self.REF_DICT, self.REF_MASK_DICT = self.load_refs(ref_dataset)
        

    def load_refs(self, ref_dataset):
        if not self.use_mesh:
            materials = ['Ks', 'bboxes', 'images', 'masks', 'RTs']
            ref_dict = {}
            ref_mask_dict = {}
            for obj_id in self.obj_list:
                print(f'Generating Reference Feature of obj {obj_id}')
                ref_idx = self.select_ref(ref_dataset, obj_id)
                ref_samples = {k : torch.stack([ref_dataset[idx][k].to(self.device) for idx in ref_idx]) for k in materials}
                ref_dict[obj_id], ref_mask_dict[obj_id] =  self.build_ref(ref_samples)
        else:
            ref_dict = None
            ref_mask_dict = None
        return ref_dict, ref_mask_dict

    def select_ref(self, dataset, obj_id):
        if self.is_FPS:
            ref_idx = self.farthest_rotation_sampling(dataset.dataset, obj_id)
        else:
            ref_idx = random.sample(dataset, self.ref_N)
        return ref_idx

    def farthest_rotation_sampling(self, dataset, obj_id):
        """
        return idx of reference samples
        """
        farthest_idx = np.zeros(self.ref_N).astype(int)

        obj_dataset = [(i, sample) for i, sample in enumerate(dataset) if (sample['visib_fract'] > 0.95) and (sample['obj_id'] == obj_id)]
        Rs = torch.tensor(np.stack([data[1]['RT'][:3, :3] for data in obj_dataset]))
        mask_pixel_N = [data[1]['px_count_visib'] for data in obj_dataset]
        obj_index = np.array(mask_pixel_N).argmax()
        farthest_idx[0] = obj_dataset[obj_index][0]
        farthest_R = torch.tensor(Rs[obj_index][None])
        distances = so3_relative_angle(torch.tensor(farthest_R).repeat(Rs.shape[0], 1, 1), Rs)
        for i in range(1, self.ref_N):
            obj_index = torch.argmax(distances).item()
            farthest_idx[i] = obj_dataset[obj_index][0]
            farthest_R = torch.tensor(Rs[obj_index][None])
            distances = torch.minimum(distances, so3_relative_angle(torch.tensor(farthest_R).repeat(Rs.shape[0], 1, 1), Rs))
        return farthest_idx


    def build_ref(self, ref_samples):
        bboxes_crop = squaring_boxes(ref_samples['bboxes'])
        K_crop = get_K_crop_resize(ref_samples['Ks'], bboxes_crop, (self.ref_size, self.ref_size))
        roi_feature = image_cropping(bboxes_crop, ref_samples['images'], (self.ref_size, self.ref_size))
        roi_mask = image_cropping(bboxes_crop, ref_samples['masks'], (self.ref_size, self.ref_size))
        ref, ref_mask = self.carving_feature(roi_mask, roi_feature, ref_samples['RTs'], K_crop)
        return ref, ref_mask

    def carving_feature(self, masks, features, RT, K_crop):
        N_ref = features.shape[0]
        index_3d = torch.zeros([self.ref_size, self.ref_size, self.ref_size, 3])
        idx = torch.arange(0, self.ref_size)
        index_3d[..., 0], index_3d[..., 1], index_3d[..., 2] = torch.meshgrid(idx, idx, idx)
        normalized_idx = (index_3d - self.ref_size/2)/(self.ref_size/2)
        X = normalized_idx.reshape(1, -1, 3).repeat(N_ref, 1, 1)

        homogeneous_X = torch.cat((X, torch.ones(X.shape[0], X.shape[1], 1)), 2).transpose(1, 2).to(RT.device)
        xyz_KRT = torch.bmm(K_crop, torch.bmm(RT[:, :3, :], homogeneous_X))
        xyz = (xyz_KRT/xyz_KRT[:, [2], :]).transpose(1, 2).reshape(N_ref, self.ref_size, self.ref_size, self.ref_size, 3)
        xyz[..., :2] = (xyz[..., :2] - self.ref_size/2)/(self.ref_size/2)
        xyz[... ,2] = 0
        
        features_3d = features.unsqueeze(2)
        masks_3d = masks.unsqueeze(2)

        ref_mask_3d = F.grid_sample(masks_3d, xyz)
        ref_3d = F.grid_sample(features_3d, xyz)
        ref_mask_3d = torch.prod(ref_mask_3d, 0, keepdim=True)
        ref_3d = ref_3d.sum(0, keepdim=True)

        ref_3d = ref_3d * ref_mask_3d
        ref_3d = ref_3d.transpose(2, 4)                    # XYZ to ZYX (DHW)
        ref_mask_3d = ref_mask_3d.transpose(2, 4)          # XYZ to ZYX (DHW)
        return ref_3d, ref_mask_3d

    def batch_refs(self, id_batch):
        if not self.use_mesh:
            ref_batch = torch.cat([self.REF_DICT[id] for id in id_batch], 0)
            ref_mask_batch = torch.cat([self.REF_MASK_DICT[id] for id in id_batch], 0)
        else:
            ref_batch = [None] * len(id_batch)
            ref_mask_batch = [None] * len(id_batch)
        return ref_batch, ref_mask_batch


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from pytorch3d.transforms.transform3d import Transform3d
from pytorch3d.transforms import (
    random_rotations, rotation_6d_to_matrix, euler_angles_to_matrix,
    so3_relative_angle, quaternion_to_matrix, quaternion_multiply, matrix_to_euler_angles
)

import numpy as np
from base import BaseModel
from utils.util import (
    apply_imagespace_predictions, deepim_crops, crop_inputs, RT_from_boxes, bbox_add_noise, invert_T, orthographic_pool, proj_visualize, projective_pool, dynamic_projective_stn,
    FX, FY, PX, PY, UNIT_CUBE_VERTEX, z_buffer_min, z_buffer_max, grid_sampler, grid_transformer, get_roi_feature, ProST_grid
)

class LocalizationNetwork(nn.Module):
    def __init__(self, model_name, occlusion):
        super().__init__()
        if occlusion:
            input_channel = 9
        else:
            input_channel = 6

        if model_name == 'res18':
            backbone = models.resnet18(pretrained=True)
            backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone[0] = nn.Conv2d(input_channel, 64, 3, 2, 2)
            setattr(backbone, "n_features", 512)
        
        if model_name == 'res34':
            backbone = models.resnet34(pretrained=True)
            backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone[0] = nn.Conv2d(input_channel, 64, 3, 2, 2)
            setattr(backbone, "n_features", 512)

        if model_name == 'res50':
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone[0] = nn.Conv2d(input_channel, 64, 3, 2, 2)
            setattr(backbone, "n_features", 2048)
        self.model = backbone

        self.trans_fc = nn.Linear(self.model.n_features, 3, bias=True)
        self.trans_fc.weight.data = nn.Parameter(torch.zeros_like(self.trans_fc.weight.data))
        self.trans_fc.bias.data = nn.Parameter(torch.Tensor([0,0,1]))

        self.rotat_fc = nn.Linear(self.model.n_features, 6, bias=True)
        self.rotat_fc.weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc.weight.data))
        self.rotat_fc.bias.data = nn.Parameter(torch.Tensor([1,0,0,0,1,0]))
    
    def forward(self, feature):
        encoded = self.model(feature)
        encoded = F.relu(encoded)
        encoded = encoded.flatten(2, 3).mean(dim=-1)
        rotation = self.rotat_fc(encoded)
        translation = self.trans_fc(encoded)
        result = torch.cat([rotation, translation], -1)
        return result
    
        

class ProjectivePose(BaseModel):
    def __init__(self, img_ratio, input_size, ftr_size, start_level, end_level, model_name='res18', occlusion=True, pose_dim=9, N_z = 100, training=True):
        super(ProjectivePose, self).__init__()
        self.pose_dim = pose_dim

        # Projective STN default grid with camera parameter
        self.H = int(480 * img_ratio)
        self.W = int(640 * img_ratio)
        fx = FX * img_ratio
        fy = FY * img_ratio
        px = PX * img_ratio
        py = PY * img_ratio
        self.K = torch.tensor([[fx,  0, px],
                              [ 0, fy, py],
                              [ 0,  0,  1]]).unsqueeze(0)
        self.projstn_grid, self.coefficient = ProST_grid(self.H, self.W, (fx+fy)/2, px, py, N_z)

        self.input_size = input_size
        self.ftr_size = ftr_size

        self.start_level = start_level
        self.end_level  = end_level
        self.training = training
        self.occlusion = occlusion

        # self.proj_backbone = resnet_fpn_backbone('resnet18', pretrained=True, trainable_layers=0)
        # self.image_backbone = resnet_fpn_backbone('resnet18', pretrained=True, trainable_layers=0)
        self.local_network = nn.ModuleDict()
        for level in range(self.start_level, self.end_level-1, -1):
            self.local_network[str(level)] = LocalizationNetwork(model_name, self.occlusion)
        # self.local_network = LocalizationNetwork()
        ### for Orthographic Pooling ###
        t0 = torch.tensor([0, 0, 0]).unsqueeze(0).unsqueeze(2)
        # R_f = euler_angles_to_matrix(torch.tensor([np.pi/2, 0, 0]), 'XYZ').unsqueeze(0)
        # R_t = torch.eye(3, 3).unsqueeze(0)
        # R_r = euler_angles_to_matrix(torch.tensor([0, -np.pi/2, 0]), 'XYZ').unsqueeze(0)
        # RT_f = torch.cat((R_f, t0), 2)
        # RT_t = torch.cat((R_t, t0), 2)
        # RT_r = torch.cat((R_r, t0), 2)

        R_p = euler_angles_to_matrix(torch.tensor([0, -np.pi/2, np.pi]), 'XYZ').unsqueeze(0)
        RT_p = torch.cat((R_p, t0), 2)


        # # orthographic pool front grid
        # self.grid_f = F.affine_grid(RT_f[:, :3, :], [1, 1, self.ftr_size, self.ftr_size, self.ftr_size])      ## -1 ~ 1 is valid area
        # # orthographic pool top grid
        # self.grid_t = F.affine_grid(RT_t[:, :3, :], [1, 1, self.ftr_size, self.ftr_size, self.ftr_size])      ## -1 ~ 1 is valid area
        # # orthographic pool right grid
        # self.grid_r = F.affine_grid(RT_r[:, :3, :], [1, 1, self.ftr_size, self.ftr_size, self.ftr_size])      ## -1 ~ 1 is valid area

        # projective pool right grid
        self.grid_p = F.affine_grid(RT_p[:, :3, :], [1, 1, self.ftr_size, self.ftr_size, self.ftr_size])      ## -1 ~ 1 is valid area

        self.vxvyvz_W_scaler = torch.tensor([self.W, 1, 1]).unsqueeze(0)
        self.vxvyvz_H_scaler = torch.tensor([1, self.H, 1]).unsqueeze(0)


    def build_ref(self, ref):
        bsz = ref['images'].shape[0]
        K_batch = self.K.repeat(bsz, 1, 1)
        projstn_grid = self.projstn_grid.repeat(bsz, 1, 1, 1, 1)
        coefficient = self.coefficient.repeat(bsz, 1, 1, 1, 1)

        _, _, ref['K_crop'], ref['bboxes_crop'] = crop_inputs(projstn_grid, coefficient, K_batch, ref['bboxes'], (self.ftr_size, self.ftr_size))
        ref['roi_feature'] = get_roi_feature(ref['bboxes_crop'], ref['images'], (self.H, self.W), (self.ftr_size, self.ftr_size))
        ref['roi_mask'] = get_roi_feature(ref['bboxes_crop'], ref['masks'], (self.H, self.W), (self.ftr_size, self.ftr_size))
        ftr, ftr_mask = projective_pool(self.grid_p, ref['roi_mask'], ref['roi_feature'], ref['RTs'], ref['K_crop'], self.ftr_size)
        return ftr, ftr_mask


    def forward(self, images, ftr, ftr_mask, bboxes, obj_ids, gt_RT):
        bsz = images.shape[0]
        K_batch = self.K.repeat(bsz, 1, 1).to(bboxes.device)
        # unit_cube_vertex = UNIT_CUBE_VERTEX.unsqueeze(0).repeat(bsz, 1, 1).to(bboxes.device)
        projstn_grid = self.projstn_grid.repeat(bsz, 1, 1, 1, 1).to(bboxes.device)
        coefficient = self.coefficient.repeat(bsz, 1, 1, 1, 1).to(bboxes.device)
        
        ####################### 3D feature module ########################################
        P = {
            'ftr': ftr, 
            'ftr_mask': ftr_mask
        }
        pr_RT = {}
        """
        P = {
                'ftr': ftr, 
                'ftr_mask': ftr_mask, 
                'grid_crop': grid_crop, 
                'coeffi_crop': coeffi_crop,
                'K_crop': K_crop,
                'bboxes_crop': bboxes_crop,
                'roi_feature': roi_feature
        }
        """
        ####################### image feature module #####################################
        # img_feature = self.image_backbone(images)
        """
        # returns
            [('0', torch.Size([1, 256, 120, 160])),
            ('1', torch.Size([1, 256, 60, 80])),
            ('2', torch.Size([1, 256, 30, 40])),
            ('3', torch.Size([1, 256, 15, 20])),
            ('pool', torch.Size([1, 256, 1, 1]))]
        """
        # img_feature = {
        #     '0': images,
        #     '1': images,
        #     '2': images,
        #     '3': images,
        # }


        ######################## initial pose estimate ##############################
        if self.training:
            bboxes = bbox_add_noise(bboxes, std_rate=0.1)
        pr_RT[self.start_level+1] = RT_from_boxes(bboxes, K_batch).detach()  
        
        # RT[4] = gt_RT
        
        # RT[4] = add_noise(gt_RT, euler_deg_std=[0, 0, 0], trans_std=[0.3, 0, 0])

        # const = torch.zeros_like(gt_RT)
        # const[:,:3,3] = -0.5
        # RT[4] = gt_RT + const 

        ####################### Projective STN grid #####################################
        ###### grid zoom-in 
        P['grid_crop'], P['coeffi_crop'], P['K_crop'], P['bboxes_crop'] = crop_inputs(projstn_grid, coefficient, K_batch, bboxes, (self.ftr_size, self.ftr_size))
        
        ####################### crop from image ##################################
        ####### get RoI feature from image    
        P['roi_feature'] = get_roi_feature(P['bboxes_crop'], images, (self.H, self.W), (self.input_size, self.input_size))
        # P['roi_mask'] = get_roi_feature(P['bboxes_crop'], masks, (self.H, self.W), (self.input_size, self.input_size))
        

        ####################### Orthographic pool ##################################
        ##### get ftr feature
        # front = front.permute(0, 3, 1, 2)
        # top = top.permute(0, 3, 1, 2)
        # right = right.permute(0, 3, 1, 2)
        # f_mask = front[:, 3:4, ...]
        # t_mask = top[:, 3:4, ...]
        # r_mask = right[:, 3:4, ...]
        # f_img = ((front[:, :3, ...] * f_mask) / 255.0)
        # t_img = ((top[:, :3, ...] * t_mask) / 255.0)
        # r_img = ((right[:, :3, ...] * r_mask) / 255.0)
        # ftr_img = torch.cat((f_img, t_img, r_img))
        # ftr_feature = self.proj_backbone(ftr_img)


        # P['ftr'], P['ftr_mask'] = orthographic_pool(
        #     (self.grid_f, self.grid_t, self.grid_r), 
        #     (f_mask, t_mask, r_mask), 
        #     (f_img, t_img, r_img), 
        #     self.ftr_size)


        for level in range(self.start_level, self.end_level-1, -1):
            pr_RT[level] = self.projective_pose(
                self.local_network[str(level)], 
                pr_RT[level+1].detach(), 
                P['ftr'], 
                P['ftr_mask'], 
                P['roi_feature'], 
                P['grid_crop'], 
                P['coeffi_crop'], 
                P['K_crop'])

        return pr_RT, P


    def projective_pose(self, local_network, previous_RT, ftr, ftr_mask, roi_feature, grid_crop, coeffi_crop, K_crop):
        ###### Dynamic Projective STN
        pr_grid_proj, obj_dist = dynamic_projective_stn(previous_RT, grid_crop, coeffi_crop)
        ####### sample ftr to 2D
        pr_ftr, pr_ftr_mask = grid_sampler(ftr, ftr_mask, pr_grid_proj)

        ###### concatenate
        if self.occlusion:
            ###### z-buffering
            pr_proj_min, _ = z_buffer_min(pr_ftr, pr_ftr_mask)
            pr_proj_max, _ = z_buffer_max(pr_ftr, pr_ftr_mask)
            pr_proj_min = F.interpolate(pr_proj_min, (self.input_size, self.input_size))
            pr_proj_max = F.interpolate(pr_proj_max, (self.input_size, self.input_size))
            loc_input = torch.cat((pr_proj_min, pr_proj_max, roi_feature), 1)
        else:
            ###### z-buffering
            pr_proj, _ = z_buffer_min(pr_ftr, pr_ftr_mask)
            pr_proj = F.interpolate(pr_proj, (self.input_size, self.input_size))
            loc_input = torch.cat((pr_proj, roi_feature), 1)
        ###### Localization Network 
        prediction = local_network(loc_input)
        ###### update pose
        next_RT = self.update_pose(previous_RT, K_crop, prediction)
        return next_RT


    def update_pose(self, TCO, K_crop, pose_outputs):
        if self.pose_dim == 9:
            dR = rotation_6d_to_matrix(pose_outputs[:, 0:6]).transpose(1, 2)
            vxvyvz = pose_outputs[:, 6:9]
        elif self.pose_dim == 7:
            dR = quaternion_to_matrix(pose_outputs[:, 0:4]).transpose(1, 2)
            vxvyvz = pose_outputs[:, 4:7]
        else:
            raise ValueError(f'pose_dim={self.pose_dim} not supported')
        vxvyvz = vxvyvz * self.vxvyvz_W_scaler.repeat(vxvyvz.shape[0], 1).to(vxvyvz.device) * self.vxvyvz_H_scaler.repeat(vxvyvz.shape[0], 1).to(vxvyvz.device)
        TCO_updated = apply_imagespace_predictions(TCO, K_crop, vxvyvz, dR)
        return TCO_updated

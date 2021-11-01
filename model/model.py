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
    apply_imagespace_predictions, deepim_crops, crop_inputs, RT_from_boxes, add_noise, invert_T,
    FX, FY, PX, PY, UNIT_CUBE_VERTEX, z_buffer_min, grid_sampler, grid_transformer, get_roi_feature, ProST_grid
)

class LocalizationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        backbone[0] = nn.Conv2d(3+3, 64, 3, 2, 2)
        setattr(backbone, "n_features", 512)
        self.model = backbone

        # self.model = nn.Sequential(
        #     nn.Conv2d(256+3, 256, 3, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(256, 256, bias=True),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256, bias=True),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU()
        # )
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
    def __init__(self, img_ratio, render_size, start_level, end_level, pose_dim=9, N_z = 100):
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
                              [ 0,  0,  1]])
        self.projstn_grid, self.coefficient = ProST_grid(self.H, self.W, (fx+fy)/2, px, py, N_z)
        self.render_size = render_size//4

        # feature size of each level
        # self.size = {
        #     0: self.render_size//4, 
        #     1: self.render_size//8, 
        #     2: self.render_size//16, 
        #     3: self.render_size//32
        #     }

        self.proj_backbone = resnet_fpn_backbone('resnet18', pretrained=True, trainable_layers=0)
        self.image_backbone = resnet_fpn_backbone('resnet18', pretrained=True, trainable_layers=0)
        self.local_network = LocalizationNetwork()

        ### for Orthographic Pooling ###
        t0 = torch.tensor([0, 0, 0]).unsqueeze(0).unsqueeze(2)
        R_f = euler_angles_to_matrix(torch.tensor([np.pi/2, 0, 0]), 'XYZ').unsqueeze(0)
        R_t = torch.eye(3, 3).unsqueeze(0)
        R_r = euler_angles_to_matrix(torch.tensor([0, -np.pi/2, 0]), 'XYZ').unsqueeze(0)
        RT_f = torch.cat((R_f, t0), 2).to(torch.device("cuda:0")).float()
        RT_t = torch.cat((R_t, t0), 2).to(torch.device("cuda:0")).float()
        RT_r = torch.cat((R_r, t0), 2).to(torch.device("cuda:0")).float()

        R_p = euler_angles_to_matrix(torch.tensor([0, -np.pi/2, np.pi]), 'XYZ').unsqueeze(0)
        RT_p = torch.cat((R_p, t0), 2).to(torch.device("cuda:0")).float()


        # orthographic pool front grid
        self.grid_f = F.affine_grid(RT_f[:, :3, :], [1, 1, self.render_size, self.render_size, self.render_size])      ## -1 ~ 1 is valid area
        # orthographic pool top grid
        self.grid_t = F.affine_grid(RT_t[:, :3, :], [1, 1, self.render_size, self.render_size, self.render_size])      ## -1 ~ 1 is valid area
        # orthographic pool right grid
        self.grid_r = F.affine_grid(RT_r[:, :3, :], [1, 1, self.render_size, self.render_size, self.render_size])      ## -1 ~ 1 is valid area

        # projective pool right grid
        self.grid_p = F.affine_grid(RT_p[:, :3, :], [1, 1, self.render_size, self.render_size, self.render_size])      ## -1 ~ 1 is valid area

        self.vxvyvz_W_scaler = torch.tensor([self.W, 1, 1]).unsqueeze(0)
        self.vxvyvz_H_scaler = torch.tensor([1, self.H, 1]).unsqueeze(0)

        self.start_level = start_level
        self.end_level  = end_level

    def forward(self, images, masks, front, top, right, bboxes, obj_ids, gt_RT):
        bsz = images.shape[0]
        K_batch = self.K.unsqueeze(0).repeat(bsz, 1, 1).to(bboxes.device)
        # unit_cube_vertex = UNIT_CUBE_VERTEX.unsqueeze(0).repeat(bsz, 1, 1).to(bboxes.device)
        projstn_grid = self.projstn_grid.repeat(bsz, 1, 1, 1, 1).to(bboxes.device)
        coefficient = self.coefficient.repeat(bsz, 1, 1, 1, 1).to(bboxes.device)
        
        ####################### 3D feature module ########################################
        M = [{}, {}, {}, {}]
        P = {}
        pr_RT = {}
        """
        # M = [
            {
                # 'ftr': ftr, 
                # 'ftr_mask': ftr_mask, 
                'pr_proj': pr_proj,
                'roi_feature': roi_feature,
                'obj_dist': obj_dist,
            }, ...
        # ]
        P = {
                'ftr': ftr, 
                'ftr_mask': ftr_mask, 
                'grid_crop': grid_crop, 
                'coeffi_crop': coeffi_crop,
                'K_crop': K_crop,
                'bboxes_crop': bboxes_crop
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
        pr_RT[self.start_level+1] = RT_from_boxes(bboxes, K_batch).detach()  
        
        # RT[4] = gt_RT
        
        # RT[4] = add_noise(gt_RT, euler_deg_std=[0, 0, 0], trans_std=[0.3, 0, 0])

        # const = torch.zeros_like(gt_RT)
        # const[:,:3,3] = -0.5
        # RT[4] = gt_RT + const 

        ####################### Projective STN grid #####################################
        ###### grid zoom-in 
        P['grid_crop'], P['coeffi_crop'], P['K_crop'], P['bboxes_crop'] = crop_inputs(projstn_grid, coefficient, K_batch, bboxes, (self.render_size, self.render_size))

        ####################### crop from image ##################################
        ####### get RoI feature from image    
        P['roi_feature'] = get_roi_feature(P['bboxes_crop'], images, (self.H, self.W), (self.render_size, self.render_size))
        P['roi_mask'] = get_roi_feature(P['bboxes_crop'], masks, (self.H, self.W), (self.render_size, self.render_size))

        ####################### Orthographic pool ##################################
        ##### get ftr feature
        front = front.permute(0, 3, 1, 2)
        top = top.permute(0, 3, 1, 2)
        right = right.permute(0, 3, 1, 2)
        f_mask = front[:, 3:4, ...]
        t_mask = top[:, 3:4, ...]
        r_mask = right[:, 3:4, ...]
        f_img = ((front[:, :3, ...] * f_mask) / 255.0)
        t_img = ((top[:, :3, ...] * t_mask) / 255.0)
        r_img = ((right[:, :3, ...] * r_mask) / 255.0)
        # ftr_img = torch.cat((f_img, t_img, r_img))
        # ftr_feature = self.proj_backbone(ftr_img)

        # for level in range(self.start_level, self.end_level-1, -1):
            # M[level]['ftr'], M[level]['ftr_mask'] = self.orthographic_pool((f_mask, t_mask, r_mask), torch.split(ftr_feature[str(level)].clone(), bsz, 0))
        P['ftr'], P['ftr_mask'] = self.orthographic_pool((f_mask, t_mask, r_mask), (f_img, t_img, r_img))
        P['ftr'], P['ftr_mask'] = self.projective_pool(P['roi_mask'], P['roi_feature'], gt_RT, P['K_crop'])



        for level in range(self.start_level, self.end_level-1, -1):
            # pr_RT[level], M[level]['pr_proj'], M[level]['obj_dist'] = self.projective_pose(pr_RT[level+1].detach(), M[level]['ftr'], M[level]['ftr_mask'], P['roi_feature'], P['grid_crop'], P['coeffi_crop'], P['K_crop'])
            pr_RT[level], M[level]['pr_proj'], M[level]['obj_dist'] = self.projective_pose(pr_RT[level+1].detach(), P['ftr'], P['ftr_mask'], P['roi_feature'], P['grid_crop'], P['coeffi_crop'], P['K_crop'])

        return M, pr_RT, P


    def projective_pose(self, previous_RT, ftr, ftr_mask, roi_feature, grid_crop, coeffi_crop, K_crop):
        ###### grid distance change
        obj_dist = torch.norm(previous_RT[:, :3, 3], 2, -1)
        grid_proj_origin = grid_crop + coeffi_crop * obj_dist.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)       
        ###### Projective Grid Generator
        pr_grid_proj = grid_transformer(grid_proj_origin, previous_RT)
        ####### sample ftr to 2D
        pr_ftr, pr_ftr_mask = grid_sampler(ftr, ftr_mask, pr_grid_proj)
        ###### z-buffering
        pr_proj, pr_proj_indx = z_buffer_min(pr_ftr, pr_ftr_mask)
        ###### concatenate
        loc_input = torch.cat((pr_proj, roi_feature), 1)
        ###### Localization Network 
        prediction = self.local_network(loc_input)
        ###### update pose
        next_RT = self.update_pose(previous_RT, K_crop, prediction)
        return next_RT, pr_proj, obj_dist


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

    def orthographic_pool(self, mask, feature):
        f_mask, t_mask, r_mask = mask
        f_feature, t_feature, r_feature = feature
        bsz = f_feature.shape[0]

        grid_f = self.grid_f.clone().repeat(bsz, 1, 1, 1, 1)
        grid_t = self.grid_t.clone().repeat(bsz, 1, 1, 1, 1)
        grid_r = self.grid_r.clone().repeat(bsz, 1, 1, 1, 1)

        ## 3d mask
        f_mask_3d = F.interpolate(f_mask, (self.render_size, self.render_size))  
        t_mask_3d = F.interpolate(t_mask, (self.render_size, self.render_size))  
        r_mask_3d = F.interpolate(r_mask, (self.render_size, self.render_size))  

        f_mask_3d = f_mask_3d.unsqueeze(2).repeat(1, 1, self.render_size, 1, 1)                
        t_mask_3d = t_mask_3d.unsqueeze(2).repeat(1, 1, self.render_size, 1, 1)
        r_mask_3d = r_mask_3d.unsqueeze(2).repeat(1, 1, self.render_size, 1, 1)

        f_mask_3d = F.grid_sample(f_mask_3d, grid_f, mode='nearest')
        t_mask_3d = F.grid_sample(t_mask_3d, grid_t, mode='nearest')
        r_mask_3d = F.grid_sample(r_mask_3d, grid_r, mode='nearest')
        ftr_mask_3d = f_mask_3d * t_mask_3d * r_mask_3d

        ## 3d image
        f_3d = F.interpolate(f_feature, (self.render_size, self.render_size))  
        t_3d = F.interpolate(t_feature, (self.render_size, self.render_size))  
        r_3d = F.interpolate(r_feature, (self.render_size, self.render_size))  

        f_3d = f_3d.unsqueeze(2).repeat(1, 1, self.render_size, 1, 1)                
        t_3d = t_3d.unsqueeze(2).repeat(1, 1, self.render_size, 1, 1)
        r_3d = r_3d.unsqueeze(2).repeat(1, 1, self.render_size, 1, 1)

        f_3d = F.grid_sample(f_3d, grid_f)
        t_3d = F.grid_sample(t_3d, grid_t)
        r_3d = F.grid_sample(r_3d, grid_r)

        ftr_3d = (f_3d + t_3d + r_3d) / 3   
        ftr_3d = ftr_3d * ftr_mask_3d
        return ftr_3d, ftr_mask_3d

    def projective_pool(self, masks, features, RT, K_crop):
        bsz = features.shape[0]

        index_3d = torch.zeros([self.render_size, self.render_size, self.render_size, 3])
        idx = torch.arange(0, self.render_size)
        index_3d[..., 0], index_3d[..., 1], index_3d[..., 2] = torch.meshgrid(idx, idx, idx)
        normalized_idx = (index_3d - self.render_size/2)/(self.render_size/2)
        X = normalized_idx.reshape(1, -1, 3).repeat(bsz, 1, 1)

        homogeneous_X = torch.cat((X, torch.ones(X.shape[0], X.shape[1], 1)), 2).transpose(1, 2).to(RT.device)
        xyz_KRT = torch.bmm(K_crop, torch.bmm(RT[:, :3, :], homogeneous_X))
        xyz = (xyz_KRT/xyz_KRT[:, [2], :]).transpose(1, 2).reshape(bsz, self.render_size, self.render_size, self.render_size, 3)
        xyz[..., :2] = (xyz[..., :2] - self.render_size/2)/(self.render_size/2)
        xyz[... ,2] = 0
        
        features_3d = features.unsqueeze(2)
        masks_3d = masks.unsqueeze(2)

        ftr_mask_3d = F.grid_sample(masks_3d, xyz, mode='nearest')
        ftr_3d = F.grid_sample(features_3d, xyz, mode='nearest')

        ftr_mask_3d = torch.prod(ftr_mask_3d, 0, keepdim=True)
        ftr_3d = ftr_3d.sum(0, keepdim=True)

        ftr_3d = ftr_3d * ftr_mask_3d

        grid_p = self.grid_p.clone()
        ftr_mask_3d = F.grid_sample(ftr_mask_3d, grid_p, mode='nearest')
        ftr_3d = F.grid_sample(ftr_3d, grid_p, mode='nearest')
                
        return ftr_3d.repeat(bsz, 1, 1, 1, 1), ftr_mask_3d.repeat(bsz, 1, 1, 1, 1)
from pytorch3d.transforms.transform3d import Transform3d
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor, MaskRCNNHeads
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, roi_align
from torchvision.transforms.functional import resize
import torchvision.utils

from collections import OrderedDict

import torch
from torch.jit.annotations import Tuple, List, Dict, Optional

from pytorch3d.transforms import (
    random_rotations, rotation_6d_to_matrix, euler_angles_to_matrix,
    so3_relative_angle, quaternion_to_matrix, quaternion_multiply, matrix_to_euler_angles
)
from pytorch3d import transforms
from pytorch3d.renderer import (
    OrthographicCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, HardPhongShader, TexturesVertex,
    get_world_to_view_transform, PerspectiveCameras
)
import numpy as np
from base import BaseModel
from utils.util import (
    apply_imagespace_predictions, deepim_crops, crop_inputs, RT_from_boxes, TCO_to_vxvyvz,
    FX, FY, PX, PY, UNIT_CUBE_VERTEX
)
class LocalizationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(256+3, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.trans_fc = nn.Linear(256, 3, bias=True)
        self.trans_fc.weight.data = nn.Parameter(torch.zeros_like(self.trans_fc.weight.data))
        self.trans_fc.bias.data = nn.Parameter(torch.Tensor([0,0,1]))

        self.rotat_fc = nn.Linear(256, 6, bias=True)
        self.rotat_fc.weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc.weight.data))
        self.rotat_fc.bias.data = nn.Parameter(torch.Tensor([1,0,0,0,1,0]))
    
    def forward(self, feature):
        encoded = self.model(feature)
        rotation = self.rotat_fc(encoded)
        translation = self.trans_fc(encoded)
        result = torch.cat([rotation, translation], -1)
        return result
    
        

class ProjectivePose(BaseModel):
    def __init__(self, img_ratio, render_size, pose_dim=9):
        super(ProjectivePose, self).__init__()
        self.pose_dim = pose_dim

        # Projective STN default grid with camera parameter
        self.H = 480 * img_ratio
        self.W = 640 * img_ratio
        fx = FX * img_ratio
        fy = FY * img_ratio
        px = PX * img_ratio
        py = PY * img_ratio
        N_z = 100
        self.projstn_grid, self.coefficient = ProST_grid(self.H, self.W, (fx+fy)/2, px, py, N_z)
        self.render_size = render_size
        self.K = torch.tensor([[fx,  0, px],
                               [ 0, fy, py],
                               [ 0,  0,  1]])

        # feature size of each level
        self.scale = {
            0: 4, 
            1: 8, 
            2: 16, 
            3: 32
            }

        self.proj_backbone = resnet_fpn_backbone('resnet18', pretrained=True, trainable_layers=3)
        self.image_backbone = resnet_fpn_backbone('resnet18', pretrained=True, trainable_layers=3)
        self.local_network = LocalizationNetwork()


        ### for Orthographic Pooling ###
        t0 = torch.tensor([0, 0, 0]).unsqueeze(0).unsqueeze(2)
        R_f = euler_angles_to_matrix(torch.tensor([np.pi/2, 0, 0]), 'XYZ').unsqueeze(0)
        R_r = euler_angles_to_matrix(torch.tensor([0, -np.pi/2, 0]), 'XYZ').unsqueeze(0)
        RT_f = torch.cat((R_f, t0), 2).to(torch.device("cuda:0")).float()
        RT_r = torch.cat((R_r, t0), 2).to(torch.device("cuda:0")).float()

        # orthographic pool top grid
        self.grid_f = {
            0: F.affine_grid(RT_f[:, :3, :], [1, 1, self.render_size//self.scale[0], self.render_size//self.scale[0], self.render_size//self.scale[0]]),         ## -1 ~ 1 is valid area
            1: F.affine_grid(RT_f[:, :3, :], [1, 1, self.render_size//self.scale[1], self.render_size//self.scale[1], self.render_size//self.scale[1]]),         ## -1 ~ 1 is valid area
            2: F.affine_grid(RT_f[:, :3, :], [1, 1, self.render_size//self.scale[2], self.render_size//self.scale[2], self.render_size//self.scale[2]]),         ## -1 ~ 1 is valid area
            3: F.affine_grid(RT_f[:, :3, :], [1, 1, self.render_size//self.scale[3], self.render_size//self.scale[3], self.render_size//self.scale[3]]),         ## -1 ~ 1 is valid area
        }

        # orthographic pool right grid
        self.grid_r = {
            0: F.affine_grid(RT_r[:, :3, :], [1, 1, self.render_size//self.scale[0], self.render_size//self.scale[0], self.render_size//self.scale[0]]),         ## -1 ~ 1 is valid area
            1: F.affine_grid(RT_r[:, :3, :], [1, 1, self.render_size//self.scale[1], self.render_size//self.scale[1], self.render_size//self.scale[1]]),         ## -1 ~ 1 is valid area
            2: F.affine_grid(RT_r[:, :3, :], [1, 1, self.render_size//self.scale[2], self.render_size//self.scale[2], self.render_size//self.scale[2]]),         ## -1 ~ 1 is valid area
            3: F.affine_grid(RT_r[:, :3, :], [1, 1, self.render_size//self.scale[3], self.render_size//self.scale[3], self.render_size//self.scale[3]]),         ## -1 ~ 1 is valid area
        }

        # for change RT to pytorch3d style
        self.R_comp = torch.eye(3, 3)[None]
        self.R_comp[0, 0, 0] = -1
        self.R_comp[0, 1, 1] = -1
        self.R_comp.requires_grad = True






    def forward(self, images, front, top, right, masks, bboxes, obj_ids, gt_RT=None):
        bsz = images.shape[0]
        K_batch = self.K.unsqueeze(0).repeat(bsz, 1, 1).to(bboxes.device)
        unit_cube_vertex = UNIT_CUBE_VERTEX.unsqueeze(0).repeat(bsz, 1, 1).to(bboxes.device)
        projstn_grid = self.projstn_grid.repeat(bsz, 1, 1, 1, 1).to(bboxes.device)
        coefficient = self.coefficient.repeat(bsz, 1, 1, 1, 1).to(bboxes.device)
        
        ####################### 3D feature module ########################################

        ftr = {}
        ftr_mask = {}

        front = front.permute(0, 3, 1, 2)
        top = top.permute(0, 3, 1, 2)
        right = right.permute(0, 3, 1, 2)
        f_mask = front[:, 3:4, ...]
        t_mask = top[:, 3:4, ...]
        r_mask = right[:, 3:4, ...]
        f_img = ((front[:, :3, ...] * f_mask) / 255.0)
        t_img = ((top[:, :3, ...] * t_mask) / 255.0)
        r_img = ((right[:, :3, ...] * r_mask) / 255.0)
        ftr_img = torch.cat((f_img, t_img, r_img))
        ftr_feature = self.proj_backbone(ftr_img)

        ftr[3], ftr_mask[3] = self.orthographic_pool((f_mask, t_mask, r_mask), torch.split(ftr_feature['3'].clone(), bsz, 0), 3)       
        ftr[2], ftr_mask[2] = self.orthographic_pool((f_mask, t_mask, r_mask), torch.split(ftr_feature['2'].clone(), bsz, 0), 2)   
        ftr[1], ftr_mask[1] = self.orthographic_pool((f_mask, t_mask, r_mask), torch.split(ftr_feature['1'].clone(), bsz, 0), 1)
        ftr[0], ftr_mask[0] = self.orthographic_pool((f_mask, t_mask, r_mask), torch.split(ftr_feature['0'].clone(), bsz, 0), 0)
 

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

        import matplotlib.pyplot as plt
        aaa = images[0].mean(0).detach().cpu().numpy()
        bb = (aaa - aaa.min())
        bb = bb/(bb.max() + 1e-6)
        plt.imsave('image.png', bb)

        ######################## initial pose estimate ##############################
        pr_RT = {}
        grid_crop = {}
        coeffi_crop = {}
        pr_grid_proj = {}
        loss_dict = {}
        obj_dist = {}
        K_crop = {}
        pr_RT[4] = RT_from_boxes(bboxes, K_batch).detach()  
        # pr_RT[4] = gt_RT

        ####################### Projective STN #####################################
        # pr_RT[3], grid_crop[3], coeffi_crop[3], pr_grid_proj[3], pr_proj, roi_feature, obj_dist[3], K_crop[3] = self.projective_pose(pr_RT[4], self.scale[3], ftr[3], ftr_mask[3], img_feature['3'], projstn_grid, coefficient, K_batch, bboxes, unit_cube_vertex)
        # pr_RT[2], grid_crop[2], coeffi_crop[2], pr_grid_proj[2], pr_proj, roi_feature, obj_dist[2], K_crop[2] = self.projective_pose(pr_RT[3], self.scale[2], ftr[2], ftr_mask[2], img_feature['2'], projstn_grid, coefficient, K_batch, bboxes, unit_cube_vertex)
        # pr_RT[1], grid_crop[1], coeffi_crop[1], pr_grid_proj[1], pr_proj, roi_feature, obj_dist[1], K_crop[1] = self.projective_pose(pr_RT[2], self.scale[1], ftr[1], ftr_mask[1], img_feature['1'], projstn_grid, coefficient, K_batch, bboxes, unit_cube_vertex)
        # pr_RT[0], grid_crop[0], coeffi_crop[0], pr_grid_proj[0], pr_proj, roi_feature, obj_dist[0], K_crop[0] = self.projective_pose(pr_RT[1], self.scale[0], ftr[0], ftr_mask[0], img_feature['0'], projstn_grid, coefficient, K_batch, bboxes, unit_cube_vertex)


        pr_RT[3], grid_crop[3], coeffi_crop[3], pr_grid_proj[3], pr_proj, roi_feature, obj_dist[3], K_crop[3] = self.projective_pose(pr_RT[4], self.scale[3], ftr[3], ftr_mask[3], images, projstn_grid, coefficient, K_batch, bboxes, unit_cube_vertex)
        pr_RT[2], grid_crop[2], coeffi_crop[2], pr_grid_proj[2], pr_proj, roi_feature, obj_dist[2], K_crop[2] = self.projective_pose(pr_RT[3], self.scale[2], ftr[2], ftr_mask[2], images, projstn_grid, coefficient, K_batch, bboxes, unit_cube_vertex)
        pr_RT[1], grid_crop[1], coeffi_crop[1], pr_grid_proj[1], pr_proj, roi_feature, obj_dist[1], K_crop[1] = self.projective_pose(pr_RT[2], self.scale[1], ftr[1], ftr_mask[1], images, projstn_grid, coefficient, K_batch, bboxes, unit_cube_vertex)
        pr_RT[0], grid_crop[0], coeffi_crop[0], pr_grid_proj[0], pr_proj, roi_feature, obj_dist[0], K_crop[0] = self.projective_pose(pr_RT[1], self.scale[0], ftr[0], ftr_mask[0], images, projstn_grid, coefficient, K_batch, bboxes, unit_cube_vertex)


        import matplotlib.pyplot as plt
        aaa = pr_proj[0].mean(0).detach().cpu().numpy()
        bb = (aaa - aaa.min())
        bb = bb/(bb.max() + 1e-6)
        plt.imsave('prediction.png', bb)

        import matplotlib.pyplot as plt
        aaa = roi_feature[0].detach().cpu().permute(1, 2, 0).numpy()
        bb = (aaa - aaa.min())
        bb = bb/(bb.max() + 1e-6)
        plt.imsave('roi_feature.png', bb)

        # import matplotlib.pyplot as plt
        # aaa = img_feature['0'][0].mean(0).detach().cpu().numpy()
        # bb = (aaa - aaa.min())
        # bb = bb/(bb.max() + 1e-6)
        # plt.imsave('img_feature.png', bb)

        if gt_RT is not None:
            loss_dict[3], gt_proj = self.loss(gt_RT.clone(), pr_RT[3], pr_RT[4], K_crop[3], grid_crop[3], coeffi_crop[3], pr_grid_proj[3], ftr[3], ftr_mask[3], obj_dist[3])
            loss_dict[2], gt_proj = self.loss(gt_RT.clone(), pr_RT[2], pr_RT[3], K_crop[2], grid_crop[2], coeffi_crop[2], pr_grid_proj[2], ftr[2], ftr_mask[2], obj_dist[2])
            loss_dict[1], gt_proj = self.loss(gt_RT.clone(), pr_RT[1], pr_RT[2], K_crop[1], grid_crop[1], coeffi_crop[1], pr_grid_proj[1], ftr[1], ftr_mask[1], obj_dist[1])
            loss_dict[0], gt_proj = self.loss(gt_RT.clone(), pr_RT[0], pr_RT[1], K_crop[0], grid_crop[0], coeffi_crop[0], pr_grid_proj[0], ftr[0], ftr_mask[0], obj_dist[0])
            
            import matplotlib.pyplot as plt
            aaa = gt_proj[0].mean(0).detach().cpu().numpy()
            bb = (aaa - aaa.min())
            bb = bb/(bb.max() + 1e-6)
            plt.imsave('gt.png', bb)

        return loss_dict, pr_RT

    def loss(self, gt_RT, pr_RT, ppr_RT, K_crop, grid_crop, coeffi_crop, pr_grid_proj, ftr, ftr_mask, obj_dist):
        ### grid to gt distance
        obj_dist_gt = torch.norm(gt_RT[:, :3, 3], 2, -1)
        # print(obj_dist_gt)
        grid_proj_origin_gt = grid_crop + coeffi_crop * obj_dist_gt.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        ### transform grid to gt grid
        grid_proj_gt_shape = grid_proj_origin_gt.shape
        grid_proj_origin_gt = grid_proj_origin_gt.flatten(1, 3)

        pytorch3d_pr_RT = change_RT(pr_RT, self.R_comp.to(pr_RT.device))
        pytorch3d_gt_RT = change_RT(gt_RT, self.R_comp.to(gt_RT.device))
        pytorch3d_gt_RT_inv = pytorch3d_gt_RT.inverse()
        gt_tf = Transform3d(matrix=pytorch3d_gt_RT_inv)
        gt_grid_proj = gt_tf.transform_points(grid_proj_origin_gt).reshape(grid_proj_gt_shape)

        ###### Grid Sampler
        ftr = torch.flip(ftr, [2, 3, 4])
        ftr_mask = torch.flip(ftr_mask, [2, 3, 4])    
        gt_ftr = F.grid_sample(ftr, gt_grid_proj, mode='bilinear')
        gt_ftr_mask = F.grid_sample(ftr_mask, gt_grid_proj, mode='bilinear')

        ###### z-buffering
        gt_proj, gt_proj_indx = z_buffer_min(gt_ftr, gt_ftr_mask)
        
        # loss = torch.norm((pr_grid_proj - gt_grid_proj.detach()), p=2, dim=-1).mean()
        loss = transforms.so3_relative_angle(pytorch3d_pr_RT[:, :3, :3], pytorch3d_gt_RT[:, :3, :3]).mean()
        pred_vxvyvz = TCO_to_vxvyvz(ppr_RT, pr_RT, K_crop)
        labe_vxvyvz = TCO_to_vxvyvz(ppr_RT, gt_RT, K_crop)
        vx_loss = F.smooth_l1_loss(pred_vxvyvz[:, 0]/self.W, labe_vxvyvz[:, 0]/self.W)
        vy_loss = F.smooth_l1_loss(pred_vxvyvz[:, 1]/self.H, labe_vxvyvz[:, 1]/self.H)
        vz_loss = F.smooth_l1_loss(pred_vxvyvz[:, 2], labe_vxvyvz[:, 2])
        loss += vx_loss + vy_loss + vz_loss

        loss += F.mse_loss(obj_dist, obj_dist_gt.detach()) / 10
        # loss = F.mse_loss(pr_grid_proj, gt_grid_proj)          # --> loss with 3d grid
        return loss, gt_proj

    def projective_pose(self, previous_RT, scale, ftr, ftr_mask, img_feature, projstn_grid, coefficient, K_batch, bboxes, unit_cube_vertex):
        ###### grid zoom-in & grid distance change
        grid_crop, coeffi_crop, K_crop, bboxes_crop = crop_inputs(projstn_grid, coefficient, K_batch, previous_RT, bboxes, (self.render_size//scale, self.render_size//scale), unit_cube_vertex)
        obj_dist = torch.norm(previous_RT[:, :3, 3], 2, -1)
        # print(obj_dist)
        grid_proj_origin = grid_crop + coeffi_crop * obj_dist.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        ###### Projective Grid Generator
        grid_proj_shape = grid_proj_origin.shape
        grid_proj_origin = grid_proj_origin.flatten(1, 3)        

        pytorch3d_previous_RT = change_RT(previous_RT, self.R_comp.to(previous_RT.device))
        pytorch3d_previous_RT_inv = pytorch3d_previous_RT.inverse()
        pr_tf = Transform3d(matrix=pytorch3d_previous_RT_inv)
        pr_grid_proj = pr_tf.transform_points(grid_proj_origin).reshape(grid_proj_shape)

        ###### Grid Sampler
        ftr = torch.flip(ftr, [2, 3, 4])
        ftr_mask = torch.flip(ftr_mask, [2, 3, 4])    
        pr_ftr = F.grid_sample(ftr, pr_grid_proj, mode='bilinear')
        pr_ftr_mask = F.grid_sample(ftr_mask, pr_grid_proj, mode='bilinear')

        ###### z-buffering
        pr_proj, pr_proj_indx = z_buffer_min(pr_ftr, pr_ftr_mask)

        ####### get RoI feature from image    
        bboxes_img_feature = bboxes_crop * (torch.tensor(img_feature.shape[-2:])/torch.tensor([self.H, self.W])).unsqueeze(0).repeat(1, 2).to(bboxes_crop.device)
        idx = torch.arange(bboxes_img_feature.shape[0]).to(torch.float).unsqueeze(1).to(bboxes_img_feature.device)
        bboxes_img_feature = torch.cat([idx, bboxes_img_feature], 1).to(torch.float) # bboxes_img_feature.shape --> [N, 5] and first column is index of batch for region
        roi_feature = roi_align(img_feature, bboxes_img_feature, output_size=(pr_proj.shape[2], pr_proj.shape[3]), sampling_ratio=4)
        
        ###### concatenate
        loc_input = torch.cat((pr_proj, roi_feature), 1)
        
        ###### Localization Network 
        prediction = self.local_network(loc_input)

        ###### update pose
        next_RT = self.update_pose(previous_RT, K_crop, prediction)
        return next_RT, grid_crop, coeffi_crop, pr_grid_proj, pr_proj, roi_feature, obj_dist, K_crop



    def update_pose(self, TCO, K_crop, pose_outputs):
        if self.pose_dim == 9:
            dR = rotation_6d_to_matrix(pose_outputs[:, 0:6]).transpose(1, 2)
            vxvyvz = pose_outputs[:, 6:9]
        elif self.pose_dim == 7:
            dR = quaternion_to_matrix(pose_outputs[:, 0:4]).transpose(1, 2)
            vxvyvz = pose_outputs[:, 4:7]
        else:
            raise ValueError(f'pose_dim={self.pose_dim} not supported')
        TCO_updated = apply_imagespace_predictions(TCO, K_crop, vxvyvz, dR)
        return TCO_updated

    def orthographic_pool(self, mask, feature, level=0):
        f_mask, t_mask, r_mask = mask
        f_feature, t_feature, r_feature = feature

        ## 3d mask
        f_mask_3d = F.interpolate(f_mask, (self.render_size//self.scale[level], self.render_size//self.scale[level])).unsqueeze(2).repeat(1, 1, self.render_size//self.scale[level], 1, 1)  
        t_mask_3d = F.interpolate(t_mask, (self.render_size//self.scale[level], self.render_size//self.scale[level])).unsqueeze(2).repeat(1, 1, self.render_size//self.scale[level], 1, 1)  
        r_mask_3d = F.interpolate(r_mask, (self.render_size//self.scale[level], self.render_size//self.scale[level])).unsqueeze(2).repeat(1, 1, self.render_size//self.scale[level], 1, 1)  

        grid_f = self.grid_f[level].clone().repeat(t_mask_3d.shape[0], 1, 1, 1, 1)
        grid_r = self.grid_r[level].clone().repeat(r_mask_3d.shape[0], 1, 1, 1, 1)
        
        f_mask_3d = F.grid_sample(f_mask_3d, grid_f, mode='nearest')
        t_mask_3d = t_mask_3d
        r_mask_3d = F.grid_sample(r_mask_3d, grid_r, mode='nearest')

        ftr_mask_3d = f_mask_3d * t_mask_3d * r_mask_3d

        ## 3d image
        f_3d = f_feature.unsqueeze(2).repeat(1, 1, self.render_size//self.scale[level], 1, 1)                
        t_3d = t_feature.unsqueeze(2).repeat(1, 1, self.render_size//self.scale[level], 1, 1)
        r_3d = r_feature.unsqueeze(2).repeat(1, 1, self.render_size//self.scale[level], 1, 1)
 

        f_3d = F.grid_sample(f_3d, grid_f, mode='nearest')
        t_3d = t_3d
        r_3d = F.grid_sample(r_3d, grid_r, mode='nearest')

        ftr_3d = (f_3d + t_3d + r_3d) / 3   
        ftr_3d = ftr_3d * ftr_mask_3d
        return ftr_3d, ftr_mask_3d

def ProST_grid(H, W, f, cx, cy, rc_pts):
    meshgrid = torch.meshgrid(torch.range(0, H-1), torch.range(0, W-1))
    X = (cx - meshgrid[1])
    Y = (cy - meshgrid[0])
    Z = torch.ones_like(X) * f
    XYZ = torch.cat((X.unsqueeze(2), Y.unsqueeze(2), Z.unsqueeze(2)), -1)
    L = torch.norm(XYZ, dim=-1)

    dist_min = -1
    dist_max = 1
    step_size = (dist_max - dist_min) / rc_pts
    steps = torch.arange(dist_min, dist_max - step_size/2, step_size)
    normalized_XYZ = XYZ.unsqueeze(0).repeat(rc_pts, 1, 1, 1) / L.unsqueeze(0).unsqueeze(3).repeat(rc_pts, 1, 1, 1)
    ProST_grid = steps.unsqueeze(1).unsqueeze(2).unsqueeze(3) * normalized_XYZ

    return ProST_grid.unsqueeze(0), normalized_XYZ.unsqueeze(0)


# def change_RT(RT, R_comp):    
#     R_comp =  R_comp.repeat(RT.shape[0], 1, 1).to(RT.device)
#     # new_RT = torch.eye(4).unsqueeze(0).repeat(RT.shape[0], 1, 1).to(RT.device)
#     new_R = torch.bmm(R_comp, RT[:, :3, :3]).transpose(1, 2)
#     ################### T = -RC#########new_T = Rt @ T @ new_R = ###########3
#     new_T = torch.bmm(torch.bmm(RT[:, :3, :3].transpose(1, 2), RT[:, :3, 3].view(-1, 3, 1)).transpose(1, 2), new_R)
#     new_RT = torch.cat([torch.cat([new_R, new_T], 1), torch.tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(2).repeat(R_comp.shape[0], 1, 1).to(new_R.device)], 2)
#     return new_RT

def change_RT(RT, R_comp):    
    R_comp =  R_comp.repeat(RT.shape[0], 1, 1)
    new_R = torch.bmm(R_comp, RT[:, :3, :3]).transpose(1, 2)
    ################### T = -RC#########new_T = Rt @ T @ new_R = ###########3
    new_T = torch.bmm(torch.bmm(RT[:, :3, :3].transpose(1, 2), RT[:, :3, 3].view(-1, 3, 1)).transpose(1, 2), new_R).transpose(1, 2).squeeze(2)
    new_RT = torch.cat([torch.cat([new_R, new_T.unsqueeze(1)], 1), torch.tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(2).repeat(R_comp.shape[0], 1, 1).to(new_R.device)], 2)
    return new_RT


def z_buffer_max(ftr, mask):
    z_grid = torch.arange(ftr.shape[2]).unsqueeze(1).unsqueeze(2).repeat(1, ftr.shape[3], ftr.shape[4]).to(ftr.device)
    index = mask.squeeze(1) * z_grid.unsqueeze(0)
    index = torch.max(index, 1).indices
    img = torch.gather(ftr, 2, index.unsqueeze(1).unsqueeze(2).repeat(1, ftr.shape[1], 1, 1, 1)).squeeze(2)
    return img, index


def z_buffer_min(ftr, mask):
    z_grid = (ftr.shape[2] - 1) - torch.arange(ftr.shape[2]).unsqueeze(1).unsqueeze(2).repeat(1, ftr.shape[3], ftr.shape[4]).to(ftr.device)
    index = mask.squeeze(1) * z_grid.unsqueeze(0)
    index = torch.max(index, 1).indices
    img = torch.gather(ftr, 2, index.unsqueeze(1).unsqueeze(2).repeat(1, ftr.shape[1], 1, 1, 1)).squeeze(2)
    return img, index

def z_buffer_mean(ftr, mask):
    # z_grid = (ftr.shape[2] - 1) - torch.arange(ftr.shape[2]).unsqueeze(1).unsqueeze(2).repeat(1, ftr.shape[3], ftr.shape[4]).to(ftr.device)
    # index = mask.squeeze(1) * z_grid.unsqueeze(0)
    # index = torch.max(index, 1).indices
    # img = torch.gather(ftr, 2, index.unsqueeze(1).unsqueeze(2).repeat(1, ftr.shape[1], 1, 1, 1)).squeeze(2)
    index = None
    img = ftr.mean(2)
    return img, index
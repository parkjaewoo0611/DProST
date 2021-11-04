import torch.nn.functional as F
import torch
from pytorch3d import transforms
from utils.util import TCO_to_vxvyvz, grid_transformer, dynamic_projective_stn

def geodesic_vxvyvz_loss(in_RT, out_RT, gt_RT, K_crop, **kwargs):
    loss = transforms.so3_relative_angle(out_RT[:, :3, :3], gt_RT[:, :3, :3]).mean()
    out_vxvyvz = TCO_to_vxvyvz(in_RT, out_RT, K_crop)
    gt_vxvyvz = TCO_to_vxvyvz(in_RT, gt_RT, K_crop)
    vx_loss = F.l1_loss(out_vxvyvz[:, 0], gt_vxvyvz[:, 0])
    vy_loss = F.l1_loss(out_vxvyvz[:, 1], gt_vxvyvz[:, 1])
    vz_loss = F.l1_loss(out_vxvyvz[:, 2], gt_vxvyvz[:, 2])
    loss += vx_loss + vy_loss + vz_loss
    return loss

def grid_distance_loss(in_RT, out_RT, gt_RT, grid_crop, coeffi_crop, **kwargs):
    pr_grid_proj, obj_dist = dynamic_projective_stn(out_RT, grid_crop, coeffi_crop)
    # ###### grid distance change
    # obj_dist = torch.sqrt(out_RT[:, :3, 3].pow(2).sum(-1) + 1e-9)           # --> to avoid nan in norm
    # grid_proj_origin = grid_crop + coeffi_crop * obj_dist.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    # ### transform grid to gt grid
    # pr_grid_proj = grid_transformer(grid_proj_origin, out_RT)

    gt_grid_proj, obj_dist_gt = dynamic_projective_stn(gt_RT, grid_crop, coeffi_crop)
    # ###### grid distance change
    # obj_dist_gt = torch.sqrt(gt_RT[:, :3, 3].pow(2).sum(-1) + 1e-9)         # --> to avoid nan in norm
    # grid_proj_origin_gt = grid_crop + coeffi_crop * obj_dist_gt.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    # ### transform grid to gt grid
    # gt_grid_proj = grid_transformer(grid_proj_origin_gt, gt_RT)
    
    # accurately, sqrt using and sum over xyz first is rightis right, 
    # but since grad at 0 of sqrt = inf, which need 1e-9 term hinder to loss to go 0, approaximate to mse
    #TODO: check above is right
    loss = F.mse_loss(pr_grid_proj, gt_grid_proj.detach())
    loss += F.mse_loss(obj_dist, obj_dist_gt.detach())

    return loss

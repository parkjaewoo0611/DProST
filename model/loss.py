import torch.nn.functional as F
import torch
from pytorch3d import transforms
from utils.util import TCO_to_vxvyvz, grid_transformer

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
    ###### grid distance change
    obj_dist = torch.sqrt(out_RT[:, :3, 3].pow(2).sum(-1) + 1e-6)           # --> to avoid nan in norm
    grid_proj_origin = grid_crop + coeffi_crop * obj_dist.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    ### transform grid to gt grid
    pr_grid_proj = grid_transformer(grid_proj_origin, out_RT)

    ###### grid distance change
    obj_dist_gt = torch.sqrt(gt_RT[:, :3, 3].pow(2).sum(-1) + 1e-6)         # --> to avoid nan in norm
    grid_proj_origin_gt = grid_crop + coeffi_crop * obj_dist_gt.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    ### transform grid to gt grid
    gt_grid_proj = grid_transformer(grid_proj_origin_gt, gt_RT)
    
    grid_subtract = pr_grid_proj - gt_grid_proj.detach()
    loss = torch.sqrt(grid_subtract.pow(2).sum(-1) + 1e-6).mean()     # --> loss with 3d grid, sqrt(pow) to avoid nan in norm
    loss += F.mse_loss(obj_dist, obj_dist_gt.detach())
    return loss

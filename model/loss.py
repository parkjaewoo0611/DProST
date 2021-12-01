import torch.nn.functional as F
import torch
from pytorch3d import transforms
from utils.util import TCO_to_vxvyvz, grid_transformer, dynamic_projective_stn, transform_pts

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
    gt_grid_proj, obj_dist_gt = dynamic_projective_stn(gt_RT, grid_crop, coeffi_crop)
    loss = torch.sqrt(F.mse_loss(pr_grid_proj, gt_grid_proj.detach(), reduce=False).sum(-1) + 1e-9).mean()
    loss += F.l1_loss(obj_dist, obj_dist_gt.detach())
    return loss

def point_matching_loss(in_RT, out_RT, gt_RT, vertexes, **kwargs):
    pr_pts = transform_pts(out_RT, vertexes)
    gt_pts = transform_pts(gt_RT, vertexes)
    loss = torch.sqrt(F.mse_loss(pr_pts, gt_pts.detach(), reduce=False).sum(-1) + 1e-9).mean()
    return loss

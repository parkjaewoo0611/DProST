import torch
import torch.nn.functional as F
from pytorch3d.transforms import so3_relative_angle
from utils.util import RT_to_vxvyvz, transform_pts

# TODO: fix geodesic_R + translation loss
def geodesic_vxvyvz_loss(gt_RT, output, K_crop, **kwargs):
    out_RT = output['RT']
    loss = so3_relative_angle(out_RT[:, :3, :3], gt_RT[:, :3, :3]).mean()
    out_vxvyvz = RT_to_vxvyvz(in_RT, out_RT, K_crop)
    gt_vxvyvz = RT_to_vxvyvz(in_RT, gt_RT, K_crop)
    vx_loss = F.l1_loss(out_vxvyvz[:, 0], gt_vxvyvz[:, 0])
    vy_loss = F.l1_loss(out_vxvyvz[:, 1], gt_vxvyvz[:, 1])
    vz_loss = F.l1_loss(out_vxvyvz[:, 2], gt_vxvyvz[:, 2])
    loss += vx_loss + vy_loss + vz_loss
    return loss

def grid_matching_loss(gt_RT, output, gt_grid, gt_dist, **kwargs):
    pr_grid, pr_dist = output['grid'], output['dist']
    loss = torch.sqrt(F.mse_loss(pr_grid, gt_grid.detach(), reduce=False).sum(-1) + 1e-9).mean()
    loss += F.l1_loss(pr_dist, gt_dist.detach())
    return loss

def grid_matching_wo_dist_loss(gt_RT, output, gt_grid, **kwargs):
    pr_grid = output['grid']
    loss = torch.sqrt(F.mse_loss(pr_grid, gt_grid.detach(), reduce=False).sum(-1) + 1e-9).mean()
    return loss

def image_matching_loss(gt_RT, output, gt_proj, **kwargs):
    pr_proj = output['proj']
    loss = F.l1_loss(pr_proj, gt_proj.detach())
    return loss

def point_matching_loss(gt_RT, output, full_vertexes, **kwargs):
    out_RT = output['RT']
    pr_pts = transform_pts(out_RT, full_vertexes)
    gt_pts = transform_pts(gt_RT, full_vertexes)
    loss = torch.sqrt(F.mse_loss(pr_pts, gt_pts.detach(), reduce=False).sum(-1) + 1e-9).mean()
    return loss

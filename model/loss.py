import torch
import torch.nn.functional as F

def grid_matching_loss(gt_RT, output, gt_grid, gt_dist, **kwargs):
    pr_grid, pr_dist = output['grid'], output['dist']
    loss = torch.sqrt(F.mse_loss(pr_grid, gt_grid.detach(), reduce=False).sum(-1) + 1e-9).mean()
    loss += F.l1_loss(pr_dist, gt_dist.detach())
    return loss

def grid_matching_wo_dist_loss(gt_RT, output, gt_grid, **kwargs):
    pr_grid = output['grid']
    loss = torch.sqrt(F.mse_loss(pr_grid, gt_grid.detach(), reduce=False).sum(-1) + 1e-9).mean()
    return loss


import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch.nn.functional as F

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(gpu_id):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """

    gpu_list = [int(gpu) for gpu in gpu_id.split(",")]
    
    n_gpu_use = len(gpu_list)
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu

    device = torch.device(f'cuda:0' if n_gpu_use > 0 else 'cpu')
    
    return device, gpu_list

class MetricTracker:
    def __init__(self, error_ftns=[], metric_ftns=[], writer=None):
        self.writer = writer
        self._error_ftns = error_ftns
        self._stack = {
            err.__name__: [] for err in error_ftns       # each error for each samples
        }
        self._stack['diameter'] = []
        self._stack['id'] = []
        self._metric_ftns = metric_ftns

    def reset(self):
        for key in self._stack.keys():
            self._stack[key] = []
    
    def loss_update(self, value, write, n=1):
        if write and self.writer is not None:
            self.writer.add_scalar('loss', value)

    def update(self, key, value):
        self._stack[key] += value       # stack errors and infos for each sample

    def result(self, obj_id=None):
        if obj_id == None:
            _result = {
                met.__name__ : met(**self._stack) for met in self._metric_ftns
            }
        else:
            _obj_stack = {}
            ind = [i for i, id in enumerate(self._stack['id']) if id == obj_id]
            for k, v in self._stack.items():
                _obj_stack[k] = [v[i] for i in ind]
            _result = {
                met.__name__ : met(**_obj_stack) for met in self._metric_ftns
            }
        return _result

######################################################
import torch
import torchvision
import numpy as np
from bop_toolkit.bop_toolkit_lib.misc import get_symmetry_transformations
from pytorch3d.transforms import euler_angles_to_matrix, so3_relative_angle, Transform3d
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PerspectiveCameras
)
from torchvision.ops import roi_align
from torchvision.utils import make_grid
import cv2
from flatten_dict import flatten
import random
from PIL import Image

def R_T_to_RT(R, T):
    if R.dtype == np.float64:
        RT = np.concatenate((R, T), axis=-1)
        const = np.zeros_like(RT)[..., [0], :]
        const[..., 0, -1] = 1
        RT = np.concatenate((RT, const), axis=-2)
    else:
        RT = torch.cat((R, T), dim=-1)
        const = torch.zeros_like(RT)[..., [0], :]
        const[..., 0, -1] = 1
        RT = torch.cat((RT, const), dim=-2)
    return RT

def RT_symmetry(RT_label, mesh_info_batch, continuous_symmetry_N=8):
    bsz = RT_label.shape[0]
    max_symmetry_N = 1    
    symmetry_label_list = []
    for i in range(bsz):
        symmetry_Rt_list = get_symmetry_transformations(mesh_info_batch[i], max_sym_disc_step = 1/continuous_symmetry_N)
        symmetry_RT = torch.tensor([R_T_to_RT(Rt['R'], Rt['t']) for Rt in symmetry_Rt_list], dtype=RT_label.dtype, device=RT_label.device)
        symmetry_N = symmetry_RT.shape[0]
        if max_symmetry_N < symmetry_N: 
            max_symmetry_N = symmetry_N
        symmetry_label_list.append(RT_label[i] @ symmetry_RT)
    possible_label_batch = torch.eye(4, 4)[None][None].repeat(bsz, max_symmetry_N, 1, 1).to(RT_label.device)
    for i, symmetry_label in enumerate(symmetry_label_list):
        symmetry_N = symmetry_label.shape[0]
        possible_label_batch[i, :symmetry_N] = symmetry_label
        possible_label_batch[i, symmetry_N:] = symmetry_label[-1][None].repeat(max_symmetry_N - symmetry_N, 1, 1)
    return possible_label_batch

################################################# DeepIM, Cosypose functions ########################################################

def RT_to_vxvyvz(RT_in, RT_gt, K):
    assert RT_in.shape[-2:] == (4, 4)
    assert RT_gt.shape[-2:] == (4, 4)
    assert K.shape[-2:] == (3, 3)
    # Translation in image space
    tz_in = RT_in[:, 2, [3]]
    tz_gt = RT_gt[:, 2, [3]]
    vz = tz_gt / tz_in
    fxfy = K[:, [0, 1], [0, 1]]
    txty_in = RT_in[:, :2, 3]
    txty_gt = RT_gt[:, :2, 3]
    vxvy = fxfy * ((txty_gt / tz_gt.repeat(1, 2)) - (txty_in / tz_in.repeat(1, 2)))
    vxvyvz = torch.cat([vxvy, vz], 1)
    return vxvyvz

def bbox_add_noise(bbox, std_rate=0.2):
    ### from https://github.com/ylabbe/cosypose/cosypose/lib3d/transform.py
    """
    bbox : batch_size x 4
    noisy_bbox : batch_size x 4
    """
    device = bbox.device
    bbox_size = bbox[:, 2:4] - bbox[:, 0:2]
    bbox_std = torch.cat((bbox_size * std_rate, bbox_size * std_rate), 1)
    noisy_bbox = torch.normal(bbox, bbox_std).to(device)
    return noisy_bbox

def add_noise(RT, euler_deg_std=[15, 15, 15], trans_std=[0.01, 0.01, 0.05]):
    ### from https://github.com/ylabbe/cosypose/cosypose/lib3d/transform.py
    RT_out = RT.clone()
    device = RT_out.device
    bsz = RT.shape[0]
    euler_noise_deg = np.concatenate(
        [np.random.normal(loc=0, scale=euler_deg_std_i, size=bsz)[:, None]
         for euler_deg_std_i in euler_deg_std], axis=1)
    #euler_noise_deg = np.ones_like(euler_noise_deg) * 10
    euler_noise_rad = torch.tensor(euler_noise_deg) * np.pi / 180
    R_noise = euler_angles_to_matrix(euler_noise_rad, 'XYZ').float().to(device)
    
    trans_noise = np.concatenate(
        [np.random.normal(loc=0, scale=trans_std_i, size=bsz)[:, None]
         for trans_std_i in trans_std], axis=1)
    trans_noise = torch.tensor(trans_noise).float().to(device)
    RT_out[:, :3, :3] = R_noise @ RT_out[:, :3, :3]
    RT_out[:, :3, 3] += trans_noise
    return RT_out

def RT_from_boxes(boxes_2d, K):
    # User in BOP20 challenge
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]

    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    RT = torch.eye(4, 4).to(torch.float).to(boxes_2d.device)[None].repeat(bsz, 1, 1)

    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2

    deltax_3d = 2   # radius of sphere
    deltay_3d = 2   # radius of sphere

    bb_deltax = (boxes_2d[:, 2] - boxes_2d[:, 0]) + 1
    bb_deltay = (boxes_2d[:, 3] - boxes_2d[:, 1]) + 1

    z_from_dx = fxfy[:, 0] * deltax_3d / bb_deltax
    z_from_dy = fxfy[:, 1] * deltay_3d / bb_deltay

    # z = z_from_dx.unsqueeze(1)
    # z = z_from_dy.unsqueeze(1)
    z = (z_from_dy.unsqueeze(1) + z_from_dx.unsqueeze(1)) / 2

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    RT[:, :2, 3] = xy_init
    RT[:, 2, 3] = z.flatten()
    return RT

def get_K_crop_resize(K, boxes, crop_resize):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4, )
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    crop_resize = torch.tensor(crop_resize, dtype=torch.float)
    final_width, final_height = crop_resize[1], crop_resize[0]
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2
    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci
    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff
    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K

def crop_inputs(grids, coeffi, K, obs_boxes, output_size, lamb=1.1):
    ### from https://github.com/ylabbe/cosypose/cosypose/lib3d/transform.py
    boxes_crop, grid_cropped, coeffi_cropped = deepim_crops(grids=grids,
                                                            coefficients=coeffi,
                                                            obs_boxes=obs_boxes,
                                                            output_size=output_size,
                                                            lamb=lamb)
    K_crop = get_K_crop_resize(K=K.clone(),
                               boxes=boxes_crop,
                               crop_resize=output_size)
    return grid_cropped, coeffi_cropped, K_crop.detach(), boxes_crop


def deepim_crops(grids, coefficients, obs_boxes, output_size=None, lamb=1.4):
    ### from https://github.com/ylabbe/cosypose/cosypose/lib3d/cropping.py
    batch_size, _, h, w, _ = grids.shape
    device = grids.device
    if output_size is None:
        output_size = (h, w)
    # uv = project_points(O_vertices, K, TCO_pred)
    # rend_boxes = boxes_from_uv(uv)
    # rend_center_uv = project_points(torch.zeros(batch_size, 1, 3).to(device), K, TCO_pred)
    # boxes = deepim_boxes(rend_center_uv, obs_boxes, rend_boxes, im_size=(h, w), lamb=lamb)
    center_boxes = ((obs_boxes[:, [0, 1]] + obs_boxes[:, [2, 3]])/2).unsqueeze(1)
    boxes = deepim_boxes(center_boxes, obs_boxes, obs_boxes, im_size=(h, w), lamb=lamb)

    boxes = torch.cat((torch.arange(batch_size).unsqueeze(1).to(device).float(), boxes), dim=1)
    crops_X = torchvision.ops.roi_align(grids.permute(4, 0, 1, 2, 3)[0], boxes, output_size=output_size, sampling_ratio=4)
    crops_Y = torchvision.ops.roi_align(grids.permute(4, 0, 1, 2, 3)[1], boxes, output_size=output_size, sampling_ratio=4)
    crops_Z = torchvision.ops.roi_align(grids.permute(4, 0, 1, 2, 3)[2], boxes, output_size=output_size, sampling_ratio=4)
    crops = torch.stack((crops_X, crops_Y, crops_Z)).permute(1, 2, 3, 4, 0)

    coefficient_crops = torchvision.ops.roi_align(coefficients.permute(0, 3, 1, 2), boxes, output_size=output_size, sampling_ratio=4).permute(0, 2, 3, 1)
    return boxes[:, 1:], crops, coefficient_crops


def project_points(points_3d, K, RT):
    assert K.shape[-2:] == (3, 3)
    assert RT.shape[-2:] == (4, 4)
    batch_size = points_3d.shape[0]
    n_points = points_3d.shape[1]
    device = points_3d.device
    if points_3d.shape[-1] == 3:
        points_3d = torch.cat((points_3d, torch.ones(batch_size, n_points, 1).to(device)), dim=-1)
    P = K @ RT[:, :3]
    suv = (P.unsqueeze(1) @ points_3d.unsqueeze(-1)).squeeze(-1)
    suv = suv / suv[..., [-1]]
    return suv[..., :2]

def boxes_from_uv(uv):
    assert uv.shape[-1] == 2
    x1 = uv[..., [0]].min(dim=1)[0]
    y1 = uv[..., [1]].min(dim=1)[0]

    x2 = uv[..., [0]].max(dim=1)[0]
    y2 = uv[..., [1]].max(dim=1)[0]

    return torch.cat((x1, y1, x2, y2), dim=1)

def deepim_boxes(rend_center_uv, obs_boxes, rend_boxes, lamb=1.4, im_size=(240, 320), clamp=False):
    """
    gt_boxes: N x 4
    crop_boxes: N x 4
    """
    lobs, robs, uobs, dobs = obs_boxes[:, [0, 2, 1, 3]].t()
    lrend, rrend, urend, drend = rend_boxes[:, [0, 2, 1, 3]].t()
    xc = rend_center_uv[..., 0, 0]
    yc = rend_center_uv[..., 0, 1]
    lobs, robs = lobs.unsqueeze(-1), robs.unsqueeze(-1)
    uobs, dobs = uobs.unsqueeze(-1), dobs.unsqueeze(-1)
    lrend, rrend = lrend.unsqueeze(-1), rrend.unsqueeze(-1)
    urend, drend = urend.unsqueeze(-1), drend.unsqueeze(-1)

    xc, yc = xc.unsqueeze(-1), yc.unsqueeze(-1)
    # w = im_size[1]
    # h = im_size[0]
    # r = w / h
    r = 1

    xdists = torch.cat(
        ((lobs - xc).abs(), (lrend - xc).abs(),
         (robs - xc).abs(), (rrend - xc).abs()),
        dim=1)
    ydists = torch.cat(
        ((uobs - yc).abs(), (urend - yc).abs(),
         (dobs - yc).abs(), (drend - yc).abs()),
        dim=1)
    xdist = xdists.max(dim=1)[0]
    ydist = ydists.max(dim=1)[0]
    width = torch.max(xdist, ydist * r) * 2 * lamb
    height = torch.max(xdist / r, ydist) * 2 * lamb

    xc, yc = xc.squeeze(-1), yc.squeeze(-1)
    x1, y1, x2, y2 = xc - width/2, yc - height / 2, xc + width / 2, yc + height / 2
    boxes = torch.cat(
        (x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1)
    assert not clamp
    # if clamp:
    #     boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, w - 1)
    #     boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, h - 1)
    return boxes

def apply_imagespace_predictions(RT, K, vxvyvz, dRCO):
    assert RT.shape[-2:] == (4, 4)
    assert K.shape[-2:] == (3, 3)
    assert dRCO.shape[-2:] == (3, 3)
    assert vxvyvz.shape[-1] == 3
    RT_out = RT.clone()

    # Translation in image space
    zsrc = RT[:, 2, [3]]
    vz = vxvyvz[:, [2]]
    ztgt = vz * zsrc

    vxvy = vxvyvz[:, :2]
    fxfy = K[:, [0, 1], [0, 1]]
    xsrcysrc = RT[:, :2, 3]
    RT_out[:, 2, 3] = ztgt.flatten()
    RT_out[:, :2, 3] = ((vxvy / fxfy) + (xsrcysrc / zsrc.repeat(1, 2))) * ztgt.repeat(1, 2) ### 논문에서는 fxfy가1이므로 scale issue

    # Rotation in camera frame
    # TC1' = TC2' @  T2'1' where TC2' = T22' = dCRO is predicted and T2'1'=T21=TC1
    RT_out[:, :3, :3] = dRCO @ RT[:, :3, :3]
    return RT_out

############################################# DProST module functions ######################################
def ProST_grid(H, W, f, cx, cy):
    meshgrid = torch.meshgrid(torch.range(0, H-1), torch.range(0, W-1))
    X = (cx - meshgrid[1])
    Y = (cy - meshgrid[0])
    Z = torch.ones_like(X) * f
    XYZ = torch.cat((X.unsqueeze(2), Y.unsqueeze(2), Z.unsqueeze(2)), -1)
    return XYZ

def reshape_grid(K_batch, K_d, XYZ, N_z):
    bsz = K_batch.shape[0]
    f, px, py = (K_batch[..., 0, 0] + K_batch[..., 1, 1])/2, K_batch[..., 0, 2], K_batch[..., 1, 2]
    f_d, px_d, py_d = (K_d[..., 0, 0] + K_d[..., 1, 1])/2,  K_d[..., 0, 2],  K_d[..., 1, 2]
    XYZ = XYZ.unsqueeze(0).repeat(bsz, 1, 1, 1)
    XYZ[...,2] = XYZ[...,2] * (f / f_d).unsqueeze(1).unsqueeze(2)
    sim_ratio = XYZ[..., 2] / f.unsqueeze(1).unsqueeze(2)
    delta_px, delta_py = px - px_d, py - py_d
    XYZ[..., 0] = XYZ[...,0] + delta_px.unsqueeze(1).unsqueeze(2) * sim_ratio
    XYZ[..., 1] = XYZ[...,1] + delta_py.unsqueeze(1).unsqueeze(2) * sim_ratio
    L = torch.norm(XYZ, dim=-1)
    normalized_XYZ = XYZ / L.unsqueeze(-1)
    dist_min = -1
    dist_max = 1
    step_size = (dist_max - dist_min) / N_z
    steps = torch.arange(dist_min, dist_max - step_size/2, step_size)
    projstn_grid = steps.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(bsz, 1, 1, 1, 1).to(normalized_XYZ.device) * normalized_XYZ.unsqueeze(1)
    return projstn_grid, normalized_XYZ

def grid_transformer(grid, RT):
    ###### Projective Grid Generator
    grid_shape = grid.shape
    grid = grid.flatten(1, 3)            
    transformed_grid = transform_pts(RT, grid, inverse=True).reshape(grid_shape)
    return transformed_grid

R_comp = torch.eye(4, 4)[None]
R_comp[0, 0, 0] = -1
R_comp[0, 1, 1] = -1
R_comp.requires_grad = True
def RT_to_pytorch3d_RT(RT):
    pytorch3d_RT = torch.bmm(R_comp.repeat(RT.shape[0], 1, 1).to(RT.device), RT).permute(0, 2, 1) # pytorch style RT
    return pytorch3d_RT

def transform_pts(RT, pts, inverse=False):
    RT = RT_to_pytorch3d_RT(RT)
    t = Transform3d(matrix=RT)
    if inverse:
        t = t.inverse()
    pts_transformed = t.transform_points(pts)
    return pts_transformed

def get_roi_feature(bboxes_crop, img_feature, original_size, output_size):
    bboxes_img_feature = bboxes_crop * (torch.tensor(img_feature.shape[-2:])/torch.tensor(original_size)).unsqueeze(0).repeat(1, 2).to(bboxes_crop.device)
    idx = torch.arange(bboxes_img_feature.shape[0]).to(torch.float).unsqueeze(1).to(bboxes_img_feature.device)
    bboxes_img_feature = torch.cat([idx, bboxes_img_feature], 1).to(torch.float) # bboxes_img_feature.shape --> [N, 5] and first column is index of batch for region
    roi_feature = roi_align(img_feature, bboxes_img_feature, output_size=output_size, sampling_ratio=4)
    return roi_feature

def obj_visualize(RT, grid_crop, coeffi_crop, ftr, ftr_mask, mesh, K_crop, img_size, **kwargs):
    ####### Dynamic Projective STN
    obj_grid, obj_dist = dynamic_projective_stn(RT, grid_crop, coeffi_crop)
    if mesh == None:
        ####### sample ftr to 2D
        NDC_ftr, NDC_ftr_mask = grid_sampler(ftr, ftr_mask, obj_grid)
        ###### z-buffering
        proj_img, proj_index = z_buffer_min(NDC_ftr, NDC_ftr_mask)
    else:
        proj_img = meshes_visualize(K_crop, RT, img_size, mesh)
    return proj_img, obj_dist, obj_grid

def dynamic_projective_stn(RT, grid_crop, coeffi_crop):
    ###### grid pushing
    obj_dist = torch.norm(RT[:, :3, 3], 2, -1)
    cam_grid = grid_crop + coeffi_crop.unsqueeze(1) * obj_dist.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    ###### Projective Grid transform
    obj_grid = grid_transformer(cam_grid, RT)
    return obj_grid, obj_dist

def grid_sampler(ftr, ftr_mask, grid):
    ###### Grid Sampler  
    pr_ftr = F.grid_sample(ftr, grid, mode='bilinear', align_corners=True)
    pr_ftr_mask = F.grid_sample(ftr_mask, grid, mode='bilinear', align_corners=True)
    return pr_ftr, pr_ftr_mask

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

def contour(render, img, is_label):
    if is_label: 
        color = (0, 255, 0) 
    else: 
        color = (0, 0, 255)
    img = ((img - np.min(img))/(np.max(img) - np.min(img)) * 255).astype(np.uint8).copy() # convert to contiguous array
    render = ((render - np.min(render))/(np.max(render) - np.min(render)) * 255).astype(np.uint8)
    render = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(render, 2, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.drawContours(img, contours, -1, color, 2)
    return result

def visualize(RTs, output, P):
    vis_size = 1
    img = P['roi_feature'][:vis_size]
    lab, _, _ = obj_visualize(RTs, **P)
    lab = lab[:vis_size]
    lev = []
    for idx in list(output.keys()):
        lev_, _, _ = obj_visualize(output[idx]['RT'], **P)
        lev.append(lev_[:vis_size])
    lev = torch.cat(lev, 0)
    img_g = make_grid(img, nrow=vis_size, normalize=True, padding=0).permute(1, 2, 0).detach().cpu().numpy()
    lab_g = make_grid(lab, nrow=vis_size, normalize=True, padding=0).permute(1, 2, 0).detach().cpu().numpy()
    lev_g = make_grid(lev, nrow=vis_size, normalize=True, padding=0).permute(1, 2, 0).detach().cpu().numpy()
    g = np.concatenate((img_g, lab_g, lev_g), 0)

    lab_c = contour(lab_g, img_g, is_label=True)
    lev_c = contour(lev_g, np.tile(lab_c, (len(output.keys()), 1, 1)), is_label=False)
    c = np.concatenate((img_g, lab_c/255.0, lev_c/255.0), 0)
    return c, g

########################################### reference function ######################################################

def farthest_rotation_sampling(dataset, obj_id, N):
    """
    return idx of reference samples
    """
    farthest_idx = np.zeros(N).astype(int)

    obj_dataset = [(i, sample) for i, sample in enumerate(dataset) if (sample['visib_fract'] > 0.95) and (sample['obj_id'] == obj_id)]
    Rs = torch.tensor(np.stack([data[1]['RT'][:3, :3] for data in obj_dataset]))
    mask_pixel_N = [data[1]['px_count_visib'] for data in obj_dataset]
    obj_index = np.array(mask_pixel_N).argmax()
    farthest_idx[0] = obj_dataset[obj_index][0]
    farthest_R = torch.tensor(Rs[obj_index][None])
    distances = so3_relative_angle(torch.tensor(farthest_R).repeat(Rs.shape[0], 1, 1), Rs)
    for i in range(1, N):
        obj_index = torch.argmax(distances).item()
        farthest_idx[i] = obj_dataset[obj_index][0]
        farthest_R = torch.tensor(Rs[obj_index][None])
        distances = torch.minimum(distances, so3_relative_angle(torch.tensor(farthest_R).repeat(Rs.shape[0], 1, 1), Rs))
    return farthest_idx

def carving_feature(masks, features, RT, K_crop, ftr_size):
    N_ref = features.shape[0]
    index_3d = torch.zeros([ftr_size, ftr_size, ftr_size, 3])
    idx = torch.arange(0, ftr_size)
    index_3d[..., 0], index_3d[..., 1], index_3d[..., 2] = torch.meshgrid(idx, idx, idx)
    normalized_idx = (index_3d - ftr_size/2)/(ftr_size/2)
    X = normalized_idx.reshape(1, -1, 3).repeat(N_ref, 1, 1)

    homogeneous_X = torch.cat((X, torch.ones(X.shape[0], X.shape[1], 1)), 2).transpose(1, 2).to(RT.device)
    xyz_KRT = torch.bmm(K_crop, torch.bmm(RT[:, :3, :], homogeneous_X))
    xyz = (xyz_KRT/xyz_KRT[:, [2], :]).transpose(1, 2).reshape(N_ref, ftr_size, ftr_size, ftr_size, 3)
    xyz[..., :2] = (xyz[..., :2] - ftr_size/2)/(ftr_size/2)
    xyz[... ,2] = 0
    
    features_3d = features.unsqueeze(2)
    masks_3d = masks.unsqueeze(2)

    ftr_mask_3d = F.grid_sample(masks_3d, xyz)
    ftr_3d = F.grid_sample(features_3d, xyz)
    ftr_mask_3d = torch.prod(ftr_mask_3d, 0, keepdim=True)
    ftr_3d = ftr_3d.sum(0, keepdim=True)

    ftr_3d = ftr_3d * ftr_mask_3d

    ftr_mask_3d = ftr_mask_3d.transpose(2, 4)          # XYZ to ZYX (DHW)
    ftr_3d = ftr_3d.transpose(2, 4)                    # XYZ to ZYX (DHW)
    return ftr_3d, ftr_mask_3d

def build_ref(ref_dataset, ref_idx, K_d, XYZ, N_z, ftr_size, H, W):
    keys = ['Ks', 'bboxes', 'images', 'masks', 'RTs']
    ref = {k : torch.stack([ref_dataset[idx][k].to(XYZ.device) for idx in ref_idx]) for k in keys}
    projstn_grid, coefficient = reshape_grid(ref['Ks'], K_d, XYZ, N_z)
    _, _, K_crop, bboxes_crop = crop_inputs(projstn_grid, coefficient, ref['Ks'], ref['bboxes'], (ftr_size, ftr_size))
    roi_feature = get_roi_feature(bboxes_crop, ref['images'], (H, W), (ftr_size, ftr_size))
    roi_mask = get_roi_feature(bboxes_crop, ref['masks'], (H, W), (ftr_size, ftr_size))
    ftr, ftr_mask = carving_feature(roi_mask, roi_feature, ref['RTs'], K_crop, ftr_size)
    return ftr, ftr_mask

############################################### mesh rendering function for ablation ###########################################
def camera_update(K, R, T, size):
    f = torch.stack([K[:, 0, 0], K[:, 1, 1]]).permute(1, 0)
    p = torch.stack([K[:, 0, 2], K[:, 1, 2]]).permute(1, 0)
    raster_settings = RasterizationSettings(image_size=size,
                                            blur_radius=0.0,
                                            faces_per_pixel=1)
    cameras = PerspectiveCameras(focal_length=f,
                                 principal_point=p,
                                 image_size = torch.tensor(size)[None].repeat(K.shape[0], 1),
                                 R=R,
                                 T=T,
                                 in_ndc=False)
    phong_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                                  shader=HardPhongShader(cameras=cameras))
    return phong_renderer

def meshes_visualize(K, RT, size, mesh):
    RT = RT_to_pytorch3d_RT(RT)
    phong_renderer = camera_update(K, RT[:, :3, :3], RT[:, 3, :3], size.tolist()).to(RT.device)
    phong = phong_renderer(meshes_world=mesh)[..., :3].permute(0, 3, 1, 2) / 255.0
    return phong

############################################### background substiture function ####################################

def replace_bg(im, im_mask, bg_img_paths, return_mask=False):
    ## editted from GDR-Net git https://github.com/THU-DA-6D-Pose-Group/GDR-Net/core/base_data_loader.py
    # add background to the image
    H, W = im.shape[:2]
    ind = random.randint(0, len(bg_img_paths) - 1)
    filename = bg_img_paths[ind]
    bg_img = get_bg_image(filename, H, W)

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


def get_bg_image(filename, imH, imW, channel=3):
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

def resize_short_edge(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR, return_scale=False):
    ## editted from GDR-Net git https://github.com/THU-DA-6D-Pose-Group/GDR-Net/core/base_data_loader.py
    """Scale the shorter edge to the given size, with a limit of `max_size` on
    the longer edge. If `max_size` is reached, then downscale so that the
    longer edge does not exceed max_size. only resize input image to target
    size and return scale.

    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        if return_scale:
            return im, im_scale
        else:
            return im
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        if return_scale:
            return padded_im, im_scale
        else:
            return padded_im

########################### hparams_function #########################
def hparams_key(config):
    required_hparams = [
        'ftr_size', 'iteration', 'model_name', 'N_z', 'data_dir',
        'batch_size', 'obj_list', 'reference_N', 'is_pbr', 'is_syn', 
        'FPS', 'loss', 'step_size', 'epochs'
    ]
    hparams = {}
    config = flatten(config, reducer='path')
    for k, v in config.items():
        name = k.split('/')[-1]
        if name in required_hparams: 
            hparams[name]=f"{v}"
    return hparams

################################# Dataset parameter loader ######################
def get_param(data_dir, param_name=None):
    if 'YCBV' in data_dir:
        import utils.YCBV_parameter as p
    else:
        import utils.LM_parameter as p
    if param_name:
        return p.DATA_PARAM[param_name]
    else:
        return p.DATA_PARAM
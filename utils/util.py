import json
import torch
from pathlib import Path
import shutil
from itertools import repeat
from collections import OrderedDict, Counter

def reset_dir(dirname):
    dirname = Path(dirname)
    if dirname.is_dir():
        shutil.rmtree(dirname)
    dirname.mkdir(parents=True, exist_ok=False)

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

def prepare_device(gpu_list):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """   
    n_gpu = len(gpu_list)
    n_possible_gpu = torch.cuda.device_count()
    if n_gpu > 0 and n_possible_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu = 0
    if n_gpu > n_possible_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu}, but only {n_possible_gpu} are "
              "available on this machine.")
        n_gpu = n_possible_gpu   
    return gpu_list, n_gpu

class MetricTracker:
    def __init__(self, error_ftns=[], metric_ftns=[]):
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

    def update(self, key, value):
        self._stack[key] += value       # stack errors and infos for each sample

    def result(self, obj_id=None):
        if obj_id == None:
            obj_ids = sorted(Counter(self._stack['id']).keys())
            _result = {met.__name__: 0 for met in self._metric_ftns}
            for obj_id in obj_ids:
                obj_result = self.result(obj_id=obj_id)
                _result = {k: _result[k] + v/len(obj_ids) for k, v in obj_result.items()}
        else:
            _obj_stack = self.error(obj_id)
            _result = {
                met.__name__ : met(**_obj_stack) for met in self._metric_ftns
            }
        return _result
    
    def error(self, obj_id=None):
        _error = {}
        ind = [i for i, id in enumerate(self._stack['id']) if id == obj_id]
        for k, v in self._stack.items():
            _error[k] = list(map(float, [v[i] for i in ind]))       # change to float type to dump on json
        return _error

######################################################
import torch
import torch.nn.functional as F
import numpy as np
from bop_toolkit.bop_toolkit_lib.misc import get_symmetry_transformations, project_pts
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import euler_angles_to_matrix, Transform3d
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PerspectiveCameras
)
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from torchvision.ops import roi_align
from torchvision.utils import make_grid
import cv2
from flatten_dict import flatten
import random
from PIL import Image
from itertools import product

def R_T_to_RT(R, T):
    if R.dtype == np.float64 or R.dtype == np.float32:
        RT = np.concatenate((R, T), axis=-1)
        const = np.zeros_like(RT)[..., [0], :]
        const[..., 0, -1] = 1
        RT = np.concatenate((RT, const), axis=-2)
    else:
        if R.shape != T.shape:
            R, T = R.view(-1, 3, 3), T.view(-1, 3, 1)
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

def squaring_boxes(obs_boxes, lamb=1.1):
    centers = ((obs_boxes[:, [0, 1]] + obs_boxes[:, [2, 3]])/2)
    xc, yc = centers[..., 0], centers[..., 1]
    lobs, robs, uobs, dobs = obs_boxes[:, [0, 2, 1, 3]].t()
    xdist = (lobs - xc).abs()
    ydist = (uobs - yc).abs()
    size = torch.max(xdist, ydist) * 2 * lamb
    x1, y1, x2, y2 = xc - size/2, yc - size/2, xc + size/2, yc + size/2
    boxes = torch.cat((x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1)
    return boxes

def rescale_boxes(obs_boxes, view_boxes, img_size):
    view_box_w = view_boxes[:, 2] - view_boxes[:, 0]
    view_box_h = view_boxes[:, 3] - view_boxes[:, 1]
    obs_boxes[:, 0] = (obs_boxes[:, 0] - view_boxes[:, 0]) * img_size[0] / view_box_w   
    obs_boxes[:, 1] = (obs_boxes[:, 1] - view_boxes[:, 1]) * img_size[1] / view_box_h
    obs_boxes[:, 2] = (obs_boxes[:, 2] - view_boxes[:, 0]) * img_size[0] / view_box_w 
    obs_boxes[:, 3] = (obs_boxes[:, 3] - view_boxes[:, 1]) * img_size[1] / view_box_h 
    return obs_boxes


def image_cropping(bboxes_crop, img_feature, output_size):
    idx = torch.arange(bboxes_crop.shape[0]).to(torch.float).unsqueeze(1).to(bboxes_crop.device)
    bboxes_crop = torch.cat([idx, bboxes_crop], 1).to(torch.float) # bboxes_img_feature.shape --> [N, 5] and first column is index of batch for region
    roi_feature = roi_align(img_feature, bboxes_crop, output_size=output_size, sampling_ratio=4)
    return roi_feature

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
def grid_forming(K_batch, H, W, N_z):
    f, px, py = (K_batch[..., 0, 0] + K_batch[..., 1, 1])/2, K_batch[..., 0, 2], K_batch[..., 1, 2]
    device = K_batch.device
    # generate 3D image plane
    meshgrid = torch.meshgrid(torch.range(0, H-1, device=device), torch.range(0, W-1, device=device))
    X = px.view(px.shape[0], 1, 1, 1) - meshgrid[1].view(1, H, W, 1)
    Y = py.view(py.shape[0], 1, 1, 1) - meshgrid[0].view(1, H, W, 1)
    Z = torch.ones_like(X, device=device) * f.view(f.shape[0], 1, 1, 1)
    XYZ = torch.cat((X, Y, Z), -1)
    # generate 3D grid on the ray from camera to image plane
    L = torch.norm(XYZ, dim=-1).unsqueeze(-1)
    normalized_XYZ = XYZ / L
    steps = torch.arange(-1, 1, 2/N_z, device=device).view(1, N_z, 1, 1, 1).to(normalized_XYZ.device)
    projstn_grid = steps * normalized_XYZ.unsqueeze(1)
    return projstn_grid


def grid_cropping(grids, boxes, output_size=None):
    batch_size, _, h, w, _ = grids.shape
    device = grids.device
    if output_size is None:
        output_size = (h, w)
    boxes = torch.cat((torch.arange(batch_size).unsqueeze(1).to(device).float(), boxes), dim=1)
    crops_X = roi_align(grids.permute(4, 0, 1, 2, 3)[0], boxes, output_size=output_size, sampling_ratio=4)
    crops_Y = roi_align(grids.permute(4, 0, 1, 2, 3)[1], boxes, output_size=output_size, sampling_ratio=4)
    crops_Z = roi_align(grids.permute(4, 0, 1, 2, 3)[2], boxes, output_size=output_size, sampling_ratio=4)
    crops = torch.stack((crops_X, crops_Y, crops_Z)).permute(1, 2, 3, 4, 0)
    return crops


def grid_pushing(RT, grid_crop):
    dist = torch.norm(RT[:, :3, 3], 2, -1).view(RT.shape[0], 1, 1, 1, 1)
    pushing_direction = F.normalize(grid_crop[:,[-1],...], dim=4)
    cam_grid = grid_crop + pushing_direction * dist 
    return cam_grid, dist


def grid_transformation(grid, RT, inverse=True):
    grid_shape = grid.shape
    grid = grid.flatten(1, 3)            
    transformed_grid = transform_pts(RT, grid, inverse=inverse).reshape(grid_shape)
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

def obj_visualize(RT, grid_crop, ref, ref_mask, mesh, K_crop, bbox_size, **kwargs):
    ####### DProST grid push & transform
    cam_grid, dist = grid_pushing(RT, grid_crop)
    obj_grid = grid_transformation(cam_grid, RT)

    ####### Projector
    if all(m == None for m in mesh):
        NDC_ref = F.grid_sample(ref, obj_grid, mode='bilinear', align_corners=True)
        NDC_mask = F.grid_sample(ref_mask, obj_grid, mode='bilinear', align_corners=True)
        proj_img, proj_index = z_buffer_min(NDC_ref, NDC_mask)
    else:
        proj_img = meshes_visualize(K_crop, RT, bbox_size, mesh)
    return proj_img, dist, obj_grid

def z_buffer_max(NDC_ref, NDC_mask):
    bsz, c, N_z, h, w = NDC_ref.shape
    z_grid = torch.arange(0, N_z, 1).view(1, 1, N_z, 1, 1).to(NDC_ref.device)
    index = NDC_mask * z_grid
    index = torch.max(index, 2, keepdim=True).indices.repeat(1, c, 1, 1, 1)
    img = torch.gather(NDC_ref, 2, index).squeeze(2)
    return img, index

def z_buffer_min(NDC_ref, NDC_mask):
    bsz, c, N_z, h, w = NDC_ref.shape
    z_grid = torch.arange(N_z-1, -1, -1).view(1, 1, N_z, 1, 1).to(NDC_ref.device)   # bsz, d, h, w
    index = NDC_mask * z_grid
    index = torch.max(index, 2, keepdim=True).indices.repeat(1, c, 1, 1, 1)
    img = torch.gather(NDC_ref, 2, index).squeeze(2)
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

def visualize(RT, output, P, g_vis=False, vis_size=1):
    img = P['roi_feature']
    img = make_grid(img[:vis_size], nrow=vis_size, normalize=True, padding=0).permute(1, 2, 0).detach().cpu().numpy()
    lab_f, _, _ = obj_visualize(RT, **P)
    lab_f = make_grid(lab_f[:vis_size], nrow=vis_size, normalize=False, padding=0).permute(1, 2, 0).detach().cpu().numpy()
    lev_f = []
    for idx in list(output.keys()):
        lev_, _, _ = obj_visualize(output[idx]['RT'], **P)
        lev_f.append(lev_[:vis_size])
    lev_f = torch.cat(lev_f, 0)
    lev_f = make_grid(lev_f, nrow=vis_size, normalize=False, padding=0).permute(1, 2, 0).detach().cpu().numpy()
    f = np.concatenate((lab_f, lev_f), 0)
    f = f / f.max()
    f = np.concatenate((img, f), 0)

    lab_c = contour(lab_f, img, is_label=True)
    lev_c = contour(lev_f, np.tile(lab_c, (len(output.keys()), 1, 1)), is_label=False)
    c = np.concatenate((img, lab_c/255.0, lev_c/255.0), 0)
    
    if g_vis:
        lab_g = grid_visualize(P['gt_grid'][:vis_size], P['ref'][:vis_size], P['ref_mask'][:vis_size], P['mesh'][:vis_size])
        lab_g = make_grid(lab_g, nrow=vis_size, normalize=False, padding=0).detach().cpu().numpy()
        lev_g = []
        for idx in list(output.keys()):
            lev_g_ = grid_visualize(output[idx]['grid'][:vis_size], P['ref'][:vis_size], P['ref_mask'][:vis_size], P['mesh'][:vis_size])
            lev_g.append(lev_g_[:vis_size])
        lev_g = torch.cat(lev_g, 0).permute(0, 3, 1, 2)
        lev_g = make_grid(lev_g, nrow=vis_size, normalize=False, padding=0).permute(1, 2, 0).detach().cpu().numpy()
        g = np.concatenate((lab_g, lev_g), 0)
    else:
        g = None

    return c, f, g

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

############################################## grid visualization function #######################################
def grid_visualize(grid, ref, ref_mask, mesh, img_size=512, f=512, N_z=64):
    bsz = grid.shape[0]
    
    # grid color x-->R, y-->G, z-->B
    color = torch.zeros_like(grid)
    idx_z = torch.arange(0, grid.shape[1]) / grid.shape[1]
    idx_y = torch.arange(0, grid.shape[2]) / grid.shape[2]
    idx_x = torch.arange(0, grid.shape[3]) / grid.shape[3]
    color[..., 2], color[..., 1], color[..., 0] = torch.meshgrid(idx_z, idx_y, idx_x)

    # grid edge coloring to white
    items = [[-1,0], [-1,0]]
    combi = list(product(*items))
    for edge in combi:
        color[..., edge[0], edge[1], :, :] = 1
        color[..., edge[0], :, edge[1], :] = 1
        color[..., :, edge[0], edge[1], :] = 1
    
    color = torch.flatten(color, 1, 3)
    grid = torch.flatten(grid, 1, 3)
    point_cloud = Pointclouds(points=grid, features=color)

    # unit sphere
    sphere = ico_sphere(4, grid.device)
    sphere_points = sample_points_from_meshes(sphere, 5000)
    sphere_colors = torch.ones_like(sphere_points)
    sphere_point_cloud = Pointclouds(points=sphere_points, features=sphere_colors)

    # camera setting
    K = torch.tensor([[f, 0, img_size/2],
                      [0, f, img_size/2],
                      [0, 0, 1]])[None].repeat(bsz, 1, 1).to(grid.device)
    bbox = torch.tensor([img_size*2.5/10, img_size*2.5/10, img_size*7.5/10, img_size*7.5/10])[None].repeat(bsz, 1).to(grid.device)
    RT = RT_from_boxes(bbox, K)
    view_R, _ = look_at_view_transform(1, 90, 90, device=grid.device)
    RT[:, :3, :3] = RT[:, :3, :3] @ view_R
    pytorch3d_RT = RT_to_pytorch3d_RT(RT)
    f = torch.stack([K[:, 0, 0], K[:, 1, 1]]).permute(1, 0)
    p = torch.stack([K[:, 0, 2], K[:, 1, 2]]).permute(1, 0)
    cameras = PerspectiveCameras(focal_length=f,
                                 principal_point=p,
                                 image_size=torch.tensor([img_size, img_size])[None].repeat(bsz, 1).to(grid.device),
                                 R=pytorch3d_RT[:, :3, :3],
                                 T=pytorch3d_RT[:, 3, :3],
                                 in_ndc=False,
                                 device=grid.device)
    raster_settings = PointsRasterizationSettings(
        image_size=[img_size, img_size], 
        radius = 0.006,
        points_per_pixel = 10
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    ).to(grid.device)

    images = renderer(point_cloud)              # --> visualize grid
    sphere_image = renderer(sphere_point_cloud) # --> visualize unit sphere

    base_grid = grid_forming(K, img_size, img_size, N_z).to(grid.device)
    obj_image, _, _ = obj_visualize(RT, base_grid, ref, ref_mask, mesh, K, torch.tensor([img_size, img_size])) # --> visualize object
    obj_image = obj_image.contiguous().view(obj_image.size(0), -1)
    obj_image -= obj_image.min()
    obj_image /= obj_image.max()
    obj_image = obj_image.view(obj_image.size(0), -1, img_size, img_size)
    result = images[..., :3] * 7 / 10 + sphere_image[..., :3] / 10 + obj_image.permute(0, 2, 3, 1)[..., :3] * 2 / 10
    return result

def get_3d_bbox_from_pts(pts):
  """Calculates 3D bounding box of the given set of 3D points.
  :param pts: N x 3 ndarray with xyz-coordinates of 3D points.
  :return: 3D bounding box (8,2) of bbox 8 points
      7 -------- 6
     /|         /|
    4 -------- 5 .
    | |        | |
    . 3 -------- 2
    |/         |/
    0 -------- 1
  """
  xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
  x_min, x_max, y_min, y_max, z_min, z_max = xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max()
  bbox_3d = [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
             [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]]
  bbox_3d = torch.tensor(bbox_3d)
  return bbox_3d

def draw_projected_box3d(image, qs, is_gt=False, thickness=2):
    """Draw 3d bounding box in image
    qs: (8,2), projected 3d points array of vertices for the 3d box in following order:
      7 -------- 6
     /|         /|
    4 -------- 5 .
    | |        | |
    . 3 -------- 2
    |/         |/
    0 -------- 1
    """
    # Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py
    image = np.ascontiguousarray(image * 255, dtype=np.uint8)
    qs = qs.astype(np.int32)
    if is_gt: 
        color=(0, 255, 0)
    else:
        color=(0, 0, 255)
    _bottom_color = _middle_color = _top_color = color
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _bottom_color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _middle_color, thickness, cv2.LINE_AA)

        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _top_color, thickness, cv2.LINE_AA)
    return image / 255.0

def bbox_3d_visualize(gt_RT, pr_RT, K, corners_3d, img, bbox, img_size=256, vis_size=1):
    box_view = squaring_boxes(bbox, lamb=1.7)
    K_view = get_K_crop_resize(K, box_view, [img_size, img_size])
    img_view = image_cropping(box_view, img, [img_size, img_size])

    # to numpy
    img_view = img_view.permute(0, 2, 3, 1).detach().cpu().numpy()
    K_view = K_view.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    pr_RT = pr_RT.detach().cpu().numpy()
    corners_3d = corners_3d.detach().cpu().numpy()

    for i in range(vis_size):
        gt_corners_2d = project_pts(corners_3d[i], K_view[i], gt_RT[i][:3, :3], gt_RT[i][:3, [3]])
        pr_corners_2d = project_pts(corners_3d[i], K_view[i], pr_RT[i][:3, :3], pr_RT[i][:3, [3]])
        img_view[[i]] = draw_projected_box3d(img_view[i], gt_corners_2d, is_gt=True)
        img_view[[i]] = draw_projected_box3d(img_view[i], pr_corners_2d, is_gt=False)
    img_view = make_grid(torch.tensor(img_view[:vis_size]).permute(0, 3, 1, 2), nrow=vis_size, normalize=False, padding=0).permute(1, 2, 0).detach().cpu().numpy()
    return img_view

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
        'gpu_id', 'ref_size', 'iteration', 'model_name', 'N_z', 'data_dir',
        'batch_size', 'obj_list', 'use_mesh', 'reference_N', 'mode', 
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
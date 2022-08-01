from utils.bop_toolkit.bop_toolkit_lib.pose_error import vsd, mssd, mspd, re, te, proj, add, adi
import numpy as np

def VSD(out_RT, gt_RT, ids, depth_maps, K, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    K = K.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        diameter = DATA_PARAM['idx2diameter'][id]
        depth_map = depth_maps[i].numpy()
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        K_ = K[i]
        e = vsd(R_e, t_e, R_g, t_g, depth_map, K_, DATA_PARAM['vsd_delta'], DATA_PARAM['taus'], DATA_PARAM['vsd_normalized_by_diameter'], diameter, DATA_PARAM['vsd_ren'], id, 'step')
        error.append(e)
    return error

def MSSD(out_RT, gt_RT, ids, points, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        pts = points[i].detach().cpu().numpy() * DATA_PARAM['idx2radius'][id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        syms = DATA_PARAM['idx2syms'][id]
        e = mssd(R_e, t_e, R_g, t_g, pts, syms)
        error.append(e)
    return error

def MSPD(out_RT, gt_RT, ids, points, K, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    K = K.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        pts = points[i].detach().cpu().numpy() * DATA_PARAM['idx2radius'][id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        syms = DATA_PARAM['idx2syms'][id]
        K_ = K[i]
        e = mspd(R_e, t_e, R_g, t_g, K_, pts, syms)
        error.append(e)
    return error

def RE(out_RT, gt_RT, ids, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        e = re(R_e, R_g)
        error.append(e)
    return error

def TE(out_RT, gt_RT, ids, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        e = te(t_e, t_g)
        error.append(e)
    return error

def TXE(out_RT, gt_RT, ids, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        e = np.abs(t_e[0] - t_g[0])
        error.append(e)
    return error

def TYE(out_RT, gt_RT, ids, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        e = np.abs(t_e[1] - t_g[1])
        error.append(e)
    return error

def TZE(out_RT, gt_RT, ids, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        e = np.abs(t_e[2] - t_g[2])
        error.append(e)
    return error

def PROJ(out_RT, gt_RT, points, ids, K, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    K = K.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        pts = points[i].detach().cpu().numpy() * DATA_PARAM['idx2radius'][id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        K_ = K[i]
        e = proj(R_e, t_e, R_g, t_g, K_, pts)
        error.append(e)
    return error

def ADD(out_RT, gt_RT, points, ids, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        pts = points[i].detach().cpu().numpy() * DATA_PARAM['idx2radius'][id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        e = add(R_e, t_e, R_g, t_g, pts)
        error.append(e)
    return error

def ADD_S(out_RT, gt_RT, points, ids, DATA_PARAM, **kwargss):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        pts = points[i].detach().cpu().numpy() * DATA_PARAM['idx2radius'][id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        if DATA_PARAM['idx2symmetry'][id] == 'none':
            e = add(R_e, t_e, R_g, t_g, pts)
        else:
            e = adi(R_e, t_e, R_g, t_g, pts)
        error.append(e)
    return error

def ADD_SS(out_RT, gt_RT, points, ids, DATA_PARAM, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    error = []
    for i, id in enumerate(ids):
        pts = points[i].detach().cpu().numpy() * DATA_PARAM['idx2radius'][id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * DATA_PARAM['idx2radius'][id]
        e = adi(R_e, t_e, R_g, t_g, pts)
        error.append(e)
    return error
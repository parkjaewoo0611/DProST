import torch
from utils.util import TCO_to_RT
from utils.bop_toolkit.bop_toolkit_lib.pose_error import vsd, mssd, mspd, re, te, proj, add, adi
from utils.LM_parameter import (
    VSD_DELTA, TAUS, VSD_NORMALIZED_BY_DIAMETER, VSD_REN, VSD_THRESHOLD, 
    MSSD_THRESHOLD,
    MSPD_THRESHOLD,
    LM_idx2symmetry, LM_idx2diameter, LM_idx2radius, LM_idx2syms, K)
import numpy as np

def VSD_score(out_RT, gt_RT, ids, depth_maps, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    AR_VSD = []
    for i, id in enumerate(ids):
        diameter = LM_idx2diameter[id]
        depth_map = depth_maps[i].squeeze(0).numpy()
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        e = vsd(R_e, t_e, R_g, t_g, depth_map, K, VSD_DELTA, TAUS, VSD_NORMALIZED_BY_DIAMETER, diameter, VSD_REN, id, 'step')
        result = np.mean(np.array(e)[None] < VSD_THRESHOLD)
        AR_VSD.append(result)
    mAR_VSD_score = np.mean(AR_VSD)
    return mAR_VSD_score

def MSSD_score(out_RT, gt_RT, ids, points, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    points = points.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    AR_MSSD = []
    for i, id in enumerate(ids):
        pts = points[i] * LM_idx2radius[id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        syms = LM_idx2syms[id]
        e = mssd(R_e, t_e, R_g, t_g, pts, syms)
        result = np.mean(np.array(e)[None] < MSSD_THRESHOLD * LM_idx2diameter[id])
        AR_MSSD.append(result)            
    mAR_MSSD = np.mean(AR_MSSD)
    return mAR_MSSD

def MSPD_score(out_RT, gt_RT, ids, points, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    points = points.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    AR_MSPD = []
    for i, id in enumerate(ids):
        pts = points[i] * LM_idx2radius[id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        syms = LM_idx2syms[id]
        e = mspd(R_e, t_e, R_g, t_g, K, pts, syms)
        result = np.mean(np.array(e)[None] < MSPD_THRESHOLD)
        AR_MSPD.append(result)            
    mAR_MSPD = np.mean(AR_MSPD)
    return mAR_MSPD

def RE_score(out_RT, gt_RT, ids, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    RE = []
    for i, id in enumerate(ids):
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        e = re(R_e, R_g)
        result = np.mean(np.array(e))
        RE.append(result)    
    mRE = np.mean(RE)
    return mRE

def TE_score(out_RT, gt_RT, ids, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    TE = []
    for i, id in enumerate(ids):
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        e = te(t_e, t_g)
        result = np.mean(np.array(e))
        TE.append(result)    
    mTE = np.mean(TE)
    return mTE

def PROJ_score_02(out_RT, gt_RT, points, ids, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    points = points.detach().cpu().numpy()
    PROJ = []
    for i, id in enumerate(ids):
        pts = points[i] * LM_idx2radius[id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        e = proj(R_e, t_e, R_g, t_g, K, pts)
        PROJ.append((e - 2)<0)
    mPROJ = np.mean(PROJ)
    return mPROJ

def PROJ_score_05(out_RT, gt_RT, points, ids, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    points = points.detach().cpu().numpy()
    PROJ = []
    for i, id in enumerate(ids):
        pts = points[i] * LM_idx2radius[id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        e = proj(R_e, t_e, R_g, t_g, K, pts)
        PROJ.append((e - 5)<0)
    mPROJ = np.mean(PROJ)
    return mPROJ

def PROJ_score_10(out_RT, gt_RT, points, ids, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    points = points.detach().cpu().numpy()
    PROJ = []
    for i, id in enumerate(ids):
        pts = points[i] * LM_idx2radius[id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        e = proj(R_e, t_e, R_g, t_g, K, pts)
        PROJ.append((e - 10)<0)
    mPROJ = np.mean(PROJ)
    return mPROJ

def ADD_score_02(out_RT, gt_RT, points, ids, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    points = points.detach().cpu().numpy()
    ADD02 = []
    for i, id in enumerate(ids):
        pts = points[i] * LM_idx2radius[id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        if LM_idx2symmetry[id] == 'none':
            e = add(R_e, t_e, R_g, t_g, pts)
        else:
            e = adi(R_e, t_e, R_g, t_g, pts)
        ADD02.append((e - LM_idx2diameter[id]*0.02)<0)
    mADD02 = np.mean(ADD02)
    return mADD02

def ADD_score_05(out_RT, gt_RT, points, ids, **kwargss):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    points = points.detach().cpu().numpy()
    ADD05 = []
    for i, id in enumerate(ids):
        pts = points[i] * LM_idx2radius[id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        if LM_idx2symmetry[id] == 'none':
            e = add(R_e, t_e, R_g, t_g, pts)
        else:
            e = adi(R_e, t_e, R_g, t_g, pts)
        ADD05.append((e - LM_idx2diameter[id]*0.05)<0)
    mADD05 = np.mean(ADD05)
    return mADD05

def ADD_score_10(out_RT, gt_RT, points, ids, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    points = points.detach().cpu().numpy()
    ADD10 = []
    for i, id in enumerate(ids):
        pts = points[i] * LM_idx2radius[id]
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        t_g = gt_RT[i, :3, 3][:, np.newaxis] * LM_idx2radius[id]
        if LM_idx2symmetry[id] == 'none':
            e = add(R_e, t_e, R_g, t_g, pts)
        else:
            e = adi(R_e, t_e, R_g, t_g, pts)
        ADD10.append((e - LM_idx2diameter[id]*0.10)<0)
    mADD10 = np.mean(ADD10)
    return mADD10
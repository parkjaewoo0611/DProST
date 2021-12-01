import torch
from utils.util import TCO_to_RT, LM_idx2symmetry, LM_idx2diameter, LM_idx2radius, LM_idx2syms, K
from utils.bop_toolkit.bop_toolkit_lib.pose_error import mssd, mspd, re, te, proj
import numpy as np

def MSSD_score(prediction, RTs, meshes, ids):
    TCO_output = prediction.clone()
    TCO_label = RTs.clone()
    points = meshes.clone().verts_list()
    ids = ids.cpu().numpy()

    MSSD = {'0.05':[], 
            '0.10':[],
            '0.15':[],
            '0.20':[],
            '0.25':[],
            '0.30':[],
            '0.35':[],
            '0.40':[],
            '0.45':[],
            '0.50':[],
    }

    for i, id in enumerate(ids):
        pts = points[i].unsqueeze(0) * LM_idx2radius[id]
        pred_RT = TCO_output[i:i+1]
        labe_RT = TCO_label[i:i+1]
        pred_RT[:, :3, 3] = pred_RT[:, :3, 3] * LM_idx2radius[id]
        labe_RT[:, :3, 3] = labe_RT[:, :3, 3] * LM_idx2radius[id]

        pred_RT = pred_RT.detach().cpu().numpy()
        labe_RT = labe_RT.detach().cpu().numpy()
        pts = pts.detach().cpu().numpy()[0]

        syms = LM_idx2syms[id]
        out_d = mssd(pred_RT[0, :3, :3], pred_RT[0, :3, 3:], labe_RT[0, :3, :3], labe_RT[0, :3, 3:], pts, syms)
        for key in MSSD.keys():
            thr = LM_idx2diameter[id] * float(key)
            MSSD[key].append((out_d - thr)<0)
        
    for key in MSSD.keys():
        MSSD[key] = sum(MSSD[key])/len(MSSD[key])
    
    AR_MSSD = np.array(list(MSSD.values())).mean()
    return AR_MSSD


def R_score(prediction, RTs, meshes, ids):
    TCO_output = prediction.clone()
    TCO_label = RTs.clone()
    points = meshes.clone().verts_list()
    ids = ids.cpu().numpy()

    R_dist = []

    for i, id in enumerate(ids):
        pred_RT = TCO_output[i:i+1]
        labe_RT = TCO_label[i:i+1]
        pred_RT[:, :3, 3] = pred_RT[:, :3, 3] * LM_idx2radius[id]
        labe_RT[:, :3, 3] = labe_RT[:, :3, 3] * LM_idx2radius[id]

        pred_RT = pred_RT.detach().cpu().numpy()
        labe_RT = labe_RT.detach().cpu().numpy()

        out_d = re(pred_RT[0, :3, :3], labe_RT[0, :3, :3])

        R_dist.append(out_d)
        
    
    R_dist = np.array(list(R_dist)).mean()
    return R_dist

def t_score(prediction, RTs, meshes, ids):
    TCO_output = prediction.clone()
    TCO_label = RTs.clone()
    points = meshes.clone().verts_list()
    ids = ids.cpu().numpy()

    t_dist = []

    for i, id in enumerate(ids):
        pred_RT = TCO_output[i:i+1]
        labe_RT = TCO_label[i:i+1]
        pred_RT[:, :3, 3] = pred_RT[:, :3, 3] * LM_idx2radius[id]
        labe_RT[:, :3, 3] = labe_RT[:, :3, 3] * LM_idx2radius[id]

        pred_RT = pred_RT.detach().cpu().numpy()
        labe_RT = labe_RT.detach().cpu().numpy()

        out_d = te(pred_RT[0, :3, 3:], labe_RT[0, :3, 3:])

        t_dist.append(out_d)
        
    
    t_dist = np.array(list(t_dist)).mean()
    return t_dist

def proj_score_2(prediction, RTs, meshes, ids):
    TCO_output = prediction.clone()
    TCO_label = RTs.clone()
    points = meshes.clone().verts_list()
    ids = ids.cpu().numpy()

    proj_dist = []

    for i, id in enumerate(ids):
        pts = points[i].unsqueeze(0) * LM_idx2radius[id]
        pred_RT = TCO_output[i:i+1]
        labe_RT = TCO_label[i:i+1]
        pred_RT[:, :3, 3] = pred_RT[:, :3, 3] * LM_idx2radius[id]
        labe_RT[:, :3, 3] = labe_RT[:, :3, 3] * LM_idx2radius[id]

        pred_RT = pred_RT.detach().cpu().numpy()
        labe_RT = labe_RT.detach().cpu().numpy()
        pts = pts.detach().cpu().numpy()[0]

        out_d = proj(pred_RT[0, :3, :3], pred_RT[0, :3, 3:], labe_RT[0, :3, :3], labe_RT[0, :3, 3:], K, pts)

        proj_dist.append((out_d - 2)<0)
        
    return sum(proj_dist)/len(proj_dist)


def proj_score_5(prediction, RTs, meshes, ids):
    TCO_output = prediction.clone()
    TCO_label = RTs.clone()
    points = meshes.clone().verts_list()
    ids = ids.cpu().numpy()

    proj_dist = []

    for i, id in enumerate(ids):
        pts = points[i].unsqueeze(0) * LM_idx2radius[id]
        pred_RT = TCO_output[i:i+1]
        labe_RT = TCO_label[i:i+1]
        pred_RT[:, :3, 3] = pred_RT[:, :3, 3] * LM_idx2radius[id]
        labe_RT[:, :3, 3] = labe_RT[:, :3, 3] * LM_idx2radius[id]

        pred_RT = pred_RT.detach().cpu().numpy()
        labe_RT = labe_RT.detach().cpu().numpy()
        pts = pts.detach().cpu().numpy()[0]

        out_d = proj(pred_RT[0, :3, :3], pred_RT[0, :3, 3:], labe_RT[0, :3, :3], labe_RT[0, :3, 3:], K, pts)

        proj_dist.append((out_d - 5)<0)
        
    return sum(proj_dist)/len(proj_dist)


def proj_score_10(prediction, RTs, meshes, ids):
    TCO_output = prediction.clone()
    TCO_label = RTs.clone()
    points = meshes.clone().verts_list()
    ids = ids.cpu().numpy()

    proj_dist = []

    for i, id in enumerate(ids):
        pts = points[i].unsqueeze(0) * LM_idx2radius[id]
        pred_RT = TCO_output[i:i+1]
        labe_RT = TCO_label[i:i+1]
        pred_RT[:, :3, 3] = pred_RT[:, :3, 3] * LM_idx2radius[id]
        labe_RT[:, :3, 3] = labe_RT[:, :3, 3] * LM_idx2radius[id]

        pred_RT = pred_RT.detach().cpu().numpy()
        labe_RT = labe_RT.detach().cpu().numpy()
        pts = pts.detach().cpu().numpy()[0]

        out_d = proj(pred_RT[0, :3, :3], pred_RT[0, :3, 3:], labe_RT[0, :3, :3], labe_RT[0, :3, 3:], K, pts)

        proj_dist.append((out_d - 10)<0)
        
    return sum(proj_dist)/len(proj_dist)

def ADD_score_02(prediction, RTs, meshes, ids):
    TCO_output = prediction.clone()
    TCO_label = RTs.clone()
    points = meshes.clone().verts_list()
    ids = ids.cpu().numpy()

    ADD10 = []

    for i, id in enumerate(ids):
        pts = points[i].unsqueeze(0) * LM_idx2radius[id]
        pred_RT = TCO_output[i:i+1]
        labe_RT = TCO_label[i:i+1]
        pred_RT[:, :3, 3] = pred_RT[:, :3, 3] * LM_idx2radius[id]
        labe_RT[:, :3, 3] = labe_RT[:, :3, 3] * LM_idx2radius[id]
        if LM_idx2symmetry[id] == 'none':
            out_d = ADD(pred_RT, labe_RT, pts)
        else:
            out_d = ADD_S(pred_RT, labe_RT, pts)
        ADD10.append((out_d.detach().cpu().numpy() - LM_idx2diameter[id]*0.02)<0)

    return sum(ADD10)/len(ADD10)

def ADD_score_05(prediction, RTs, meshes, ids):
    TCO_output = prediction.clone()
    TCO_label = RTs.clone()
    points = meshes.clone().verts_list()
    ids = ids.cpu().numpy()

    ADD10 = []

    for i, id in enumerate(ids):
        pts = points[i].unsqueeze(0) * LM_idx2radius[id]
        pred_RT = TCO_output[i:i+1]
        labe_RT = TCO_label[i:i+1]
        pred_RT[:, :3, 3] = pred_RT[:, :3, 3] * LM_idx2radius[id]
        labe_RT[:, :3, 3] = labe_RT[:, :3, 3] * LM_idx2radius[id]
        if LM_idx2symmetry[id] == 'none':
            out_d = ADD(pred_RT, labe_RT, pts)
        else:
            out_d = ADD_S(pred_RT, labe_RT, pts)
        ADD10.append((out_d.detach().cpu().numpy() - LM_idx2diameter[id]*0.05)<0)

    return sum(ADD10)/len(ADD10)

def ADD_score_10(prediction, RTs, meshes, ids):
    TCO_output = prediction.clone()
    TCO_label = RTs.clone()
    points = meshes.clone().verts_list()
    ids = ids.cpu().numpy()

    ADD10 = []

    for i, id in enumerate(ids):
        pts = points[i].unsqueeze(0) * LM_idx2radius[id]
        pred_RT = TCO_output[i:i+1]
        labe_RT = TCO_label[i:i+1]
        pred_RT[:, :3, 3] = pred_RT[:, :3, 3] * LM_idx2radius[id]
        labe_RT[:, :3, 3] = labe_RT[:, :3, 3] * LM_idx2radius[id]
        if LM_idx2symmetry[id] == 'none':
            out_d = ADD(pred_RT, labe_RT, pts)
        else:
            out_d = ADD_S(pred_RT, labe_RT, pts)
        ADD10.append((out_d.detach().cpu().numpy() - LM_idx2diameter[id]*0.1)<0)

    return sum(ADD10)/len(ADD10)


def ADD(output, target, points):
    pred_out_R, pred_out_T = TCO_to_RT(output)
    labe_R, labe_T = TCO_to_RT(target)

    pred_out_pts = torch.bmm(pred_out_R, points.permute(0, 2, 1)) + pred_out_T.repeat(1, 1, points.shape[1])
    labe_pts = torch.bmm(labe_R, points.permute(0, 2, 1)) + labe_T.repeat(1, 1, points.shape[1])

    labe_pts = labe_pts.permute(0, 2, 1)
    pred_out_pts = pred_out_pts.permute(0, 2, 1)

    out_lossvalue = torch.norm(pred_out_pts - labe_pts, p=2, dim=2).mean(1)

    return out_lossvalue

def ADD_S(output, target, points):
    pred_out_R, pred_out_T = TCO_to_RT(output)
    labe_R, labe_T = TCO_to_RT(target)

    pred_out_pts = torch.bmm(pred_out_R, points.permute(0, 2, 1)) + pred_out_T.repeat(1, 1, points.shape[1])
    labe_pts = torch.bmm(labe_R, points.permute(0, 2, 1)) + labe_T.repeat(1, 1, points.shape[1])

    labe_pts = labe_pts.permute(0, 2, 1)
    pred_out_pts = pred_out_pts.permute(0, 2, 1)

    out_lossvalue = torch.cdist(pred_out_pts, labe_pts).min(2)[0].mean(1)

    return out_lossvalue
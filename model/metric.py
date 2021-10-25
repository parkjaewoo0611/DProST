import torch
from utils.util import TCO_to_RT, LM_idx2symmetry, LM_idx2diameter, LM_idx2radius

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def ADD_score(prediction, RTs, meshes, ids):
    TCO_output = prediction[0].detach()
    TCO_label = RTs.detach()
    points = meshes.verts_list()
    ids = ids.cpu().numpy()

    ADD10 = []

    for i, id in enumerate(ids):
        if LM_idx2symmetry[id] == 'none':
            out_d = ADD(
                TCO_output[i:i+1], 
                TCO_label[i:i+1],
                points[i].unsqueeze(0) * LM_idx2radius[id])
        else:
            out_d = ADD_S(
                TCO_output[i:i+1], 
                TCO_label[i:i+1],
                points[i].unsqueeze(0) * LM_idx2radius[id])
        ADD10.append((out_d.detach().cpu().numpy() - LM_idx2diameter[id]/10)<0)

    return sum(ADD10)/len(ADD10)


def ADD(output, target, points):
    pred_out_R, pred_out_T = TCO_to_RT(output)
    labe_R, labe_T = TCO_to_RT(target)

    pred_out_pts = torch.bmm(pred_out_R, points.permute(0, 2, 1)) + pred_out_T.repeat(1, 1, points.shape[1])
    labe_pts = torch.bmm(labe_R, points.permute(0, 2, 1)) + labe_T.repeat(1, 1, points.shape[1])

    labe_pts = labe_pts.permute(0, 2, 1)
    pred_out_pts = pred_out_pts.permute(0, 2, 1)

    out_lossvalue = torch.norm(pred_out_pts - labe_pts, p=1, dim=2).mean(1)

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
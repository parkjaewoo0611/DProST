import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix
from base import BaseModel
from utils.util import (
    apply_imagespace_predictions, crop_inputs, RT_from_boxes, bbox_add_noise, 
    obj_visualize, reshape_grid, get_roi_feature, ProST_grid
)
class LocalizationNetwork(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        input_channel = 6

        if model_name == 'res18':
            backbone = models.resnet18(pretrained=True)
            backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone[0] = nn.Conv2d(input_channel, 64, 3, 2, 2)
            setattr(backbone, "n_features", 512)
        
        if model_name == 'res34':
            backbone = models.resnet34(pretrained=True)
            backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone[0] = nn.Conv2d(input_channel, 64, 3, 2, 2)
            setattr(backbone, "n_features", 512)

        if model_name == 'res50':
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone[0] = nn.Conv2d(input_channel, 64, 3, 2, 2)
            setattr(backbone, "n_features", 2048)
        self.model = backbone

        self.trans_fc = nn.Linear(self.model.n_features, 3, bias=True)
        self.trans_fc.weight.data = nn.Parameter(torch.zeros_like(self.trans_fc.weight.data))
        self.trans_fc.bias.data = nn.Parameter(torch.Tensor([0,0,1]))

        self.rotat_fc = nn.Linear(self.model.n_features, 6, bias=True)
        self.rotat_fc.weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc.weight.data))
        self.rotat_fc.bias.data = nn.Parameter(torch.Tensor([1,0,0,0,1,0]))
    
    def forward(self, feature):
        encoded = self.model(feature)
        encoded = F.relu(encoded)
        encoded = encoded.flatten(2, 3).mean(dim=-1)
        rotation = self.rotat_fc(encoded)
        translation = self.trans_fc(encoded)
        result = torch.cat([rotation, translation], -1)
        return result

class DProST(BaseModel):
    def __init__(self, img_ratio, ftr_size, img_size, iteration, model_name='res34', pose_dim=9, N_z = 64, mode='train', device='cpu'):
        super(DProST, self).__init__()
        self.pose_dim = pose_dim
        self.device = device

        # Projective STN default grid with camera parameter
        self.img_ratio = img_ratio
        self.H = int(480 * self.img_ratio)
        self.W = int(640 * self.img_ratio)
        self.N_z = N_z
        self.ftr_size = ftr_size
        self.img_size = torch.tensor([img_size, img_size])
        self.iteration = iteration
        self.mode = mode

        f_d = 1000
        self.K_d = torch.tensor([[f_d,  0,  self.W//2],
                                 [ 0,  f_d, self.H//2],
                                 [ 0,  0,  1]]).unsqueeze(0).to(self.device)
        XYZ = ProST_grid(self.H, self.W, f_d, self.W//2, self.H//2)
        self.XYZ = XYZ.to(self.device)
        dist_min = -1
        dist_max = 1
        step_size = (dist_max - dist_min) / N_z
        self.steps = torch.arange(dist_min, dist_max - step_size/2, step_size).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        self.L = torch.norm(self.XYZ, dim=-1)

        self.local_network = nn.ModuleDict()
        for i in range(1, self.iteration+1):
            self.local_network[str(i)] = LocalizationNetwork(model_name)

        self.vxvyvz_W_scaler = torch.tensor([self.W, 1, 1]).unsqueeze(0).to(self.device)
        self.vxvyvz_H_scaler = torch.tensor([1, self.H, 1]).unsqueeze(0).to(self.device)
        

    def forward(self, images, ftr, ftr_mask, bboxes, K_batch, gt_RT=None, mesh=None):
        projstn_grid, coefficient = reshape_grid(K_batch, self.K_d, self.XYZ, self.steps)
        ####################### 3D feature module ###################################
        P = {
            'ftr': ftr, 
            'ftr_mask': ftr_mask,
            'mesh': mesh,
            'img_size': self.img_size,
        }
        pred = {}

        ######################## initial pose estimate ##############################
        if self.mode == 'train':
            bboxes = bbox_add_noise(bboxes, std_rate=0.1)
        pred[0] = {'RT': RT_from_boxes(bboxes, K_batch).detach()}  

        ####################### DProST grid cropping #################################
        P['grid_crop'], P['coeffi_crop'], P['K_crop'], P['bboxes_crop'] = crop_inputs(projstn_grid, coefficient, K_batch, bboxes, P['img_size'])
        ####################### crop from image ######################################s
        P['roi_feature'] = get_roi_feature(P['bboxes_crop'], images, (self.H, self.W), P['img_size'])
        ####################### DProST grid push & transform #########################
        for i in range(1, self.iteration+1):
            ###### Dynamic Projective STN & sampling & z-buffering
            pred[i-1]['proj'], pred[i-1]['dist'], pred[i-1]['grid'] = obj_visualize(pred[i-1]['RT'], **P)
            ###### Localization Network 
            loc_input = torch.cat((pred[i-1]['proj'], P['roi_feature']), 1)
            prediction = self.local_network[str(i)](loc_input.detach())
            ###### update pose
            next_RT = self.update_pose(pred[i-1]['RT'], P['K_crop'], prediction)
            pred[i] = {'RT': next_RT}

        if self.mode == 'train' or self.mode == 'valid':
            pred[self.iteration]['proj'], pred[self.iteration]['dist'], pred[self.iteration]['grid'] = obj_visualize(pred[self.iteration]['RT'], **P)
            P['gt_proj'], P['gt_dist'], P['gt_grid'] = obj_visualize(gt_RT, **P)
        return pred, P

    def update_pose(self, TCO, K_crop, pose_outputs):
        ##### from https://github.com/ylabbe/cosypose
        if self.pose_dim == 9:
            dR = rotation_6d_to_matrix(pose_outputs[:, 0:6]).transpose(1, 2)
            vxvyvz = pose_outputs[:, 6:9]
        elif self.pose_dim == 7:
            dR = quaternion_to_matrix(pose_outputs[:, 0:4]).transpose(1, 2)
            vxvyvz = pose_outputs[:, 4:7]
        else:
            raise ValueError(f'pose_dim={self.pose_dim} not supported')
        vxvyvz = vxvyvz * self.vxvyvz_W_scaler.repeat(vxvyvz.shape[0], 1) * self.vxvyvz_H_scaler.repeat(vxvyvz.shape[0], 1)
        TCO_updated = apply_imagespace_predictions(TCO, K_crop, vxvyvz, dR)
        return TCO_updated

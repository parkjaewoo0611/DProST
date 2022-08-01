import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix
from base import BaseModel
from utils.util import (
    apply_imagespace_predictions, RT_from_boxes, bbox_add_noise, 
    obj_visualize, grid_forming, grid_cropping, image_cropping, squaring_boxes, get_K_crop_resize
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
    def __init__(self, img_ratio, bbox_size, iteration, model_name='res34', pose_dim=9, N_z = 64, mode='train', device='cpu'):
        super(DProST, self).__init__()
        self.pose_dim = pose_dim
        self.device = device

        self.H = int(480 * img_ratio)
        self.W = int(640 * img_ratio)
        self.N_z = N_z
        self.bbox_size = torch.tensor([bbox_size, bbox_size])
        self.iteration = iteration
        self.mode = mode

        self.local_network = nn.ModuleDict()
        for i in range(1, self.iteration+1):
            self.local_network[str(i)] = LocalizationNetwork(model_name).to(self.device)

        self.vxvyvz_W_scaler = torch.tensor([self.W, 1, 1], device=self.device).unsqueeze(0)
        self.vxvyvz_H_scaler = torch.tensor([1, self.H, 1], device=self.device).unsqueeze(0)
        self.visualize = False
        

    def forward(self, images, ref, ref_mask, bbox, K_batch, gt_RT=None, mesh=None):
        ####################### 3D feature module ###################################
        P = {
            'ref': ref, 
            'ref_mask': ref_mask,
            'mesh': mesh,
            'bbox_size': self.bbox_size,
        }
        pred = {}

        ##################### initial pose estimate from bbox #########################
        if self.mode == 'train':
            bbox = bbox_add_noise(bbox, std_rate=0.1)
        pred[0] = {'RT': RT_from_boxes(bbox, K_batch).detach()}  
        P['bboxes_crop'] = squaring_boxes(bbox)
        P['K_crop'] = get_K_crop_resize(K_batch, P['bboxes_crop'], P['bbox_size'])
        P['roi_feature'] = image_cropping(P['bboxes_crop'], images, P['bbox_size'])

        ####################### DProST grid forming & cropping (once) #################################
        projstn_grid = grid_forming(K_batch, self.H, self.W, self.N_z)
        P['grid_crop'] = grid_cropping(projstn_grid, P['bboxes_crop'], P['bbox_size'])


        for i in range(1, self.iteration+1):
            ####################### DProST grid push & transform + Projcetor #########################
            pred[i-1]['proj'], pred[i-1]['dist'], pred[i-1]['grid'] = obj_visualize(pred[i-1]['RT'], **P)
            ###### Localization Network 
            loc_input = torch.cat((pred[i-1]['proj'], P['roi_feature']), 1)
            prediction = self.local_network[str(i)](loc_input.detach())
            ###### update pose
            next_RT = self.update_pose(pred[i-1]['RT'], P['K_crop'], prediction)
            pred[i] = {'RT': next_RT}

        if self.mode == 'train' or self.mode == 'valid' or self.visualize:
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

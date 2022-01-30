import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix
from base import BaseModel
from utils.util import (
    apply_imagespace_predictions, crop_inputs, RT_from_boxes, bbox_add_noise, 
    carving_feature, dynamic_projective_stn,
    z_buffer_min, z_buffer_max, grid_sampler, get_roi_feature, ProST_grid
)
from utils.LM_parameter import FX, FY, PX, PY
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
    def __init__(self, img_ratio, ftr_size, iteration, model_name='res34', pose_dim=9, N_z = 64, mode='train', device='cpu'):
        super(DProST, self).__init__()
        self.pose_dim = pose_dim
        self.device = device

        # Projective STN default grid with camera parameter
        self.H = int(480 * img_ratio)
        self.W = int(640 * img_ratio)
        fx = FX * img_ratio
        fy = FY * img_ratio
        px = PX * img_ratio
        py = PY * img_ratio
        self.K = torch.tensor([[fx,  0, px],
                              [ 0, fy, py],
                              [ 0,  0,  1]]).unsqueeze(0).to(self.device)
        projstn_grid, coefficient = ProST_grid(self.H, self.W, (fx+fy)/2, px, py, N_z)
        self.projstn_grid, self.coefficient = projstn_grid.to(self.device), coefficient.to(self.device)

        self.ftr_size = ftr_size

        self.iteration = iteration
        self.mode = mode

        self.local_network = nn.ModuleDict()
        for i in range(1, self.iteration+1):
            self.local_network[str(i)] = LocalizationNetwork(model_name)

        self.vxvyvz_W_scaler = torch.tensor([self.W, 1, 1]).unsqueeze(0).to(self.device)
        self.vxvyvz_H_scaler = torch.tensor([1, self.H, 1]).unsqueeze(0).to(self.device)


    def build_ref(self, ref):
        N_ref = ref['images'].shape[0]
        K_batch = self.K.repeat(N_ref, 1, 1).detach().cpu()
        projstn_grid = self.projstn_grid.repeat(N_ref, 1, 1, 1, 1).detach().cpu()
        coefficient = self.coefficient.repeat(N_ref, 1, 1, 1, 1).detach().cpu()

        _, _, ref['K_crop'], ref['bboxes_crop'] = crop_inputs(projstn_grid, coefficient, K_batch, ref['bboxes'], (self.ftr_size, self.ftr_size))
        ref['roi_feature'] = get_roi_feature(ref['bboxes_crop'], ref['images'], (self.H, self.W), (self.ftr_size, self.ftr_size))
        ref['roi_mask'] = get_roi_feature(ref['bboxes_crop'], ref['masks'], (self.H, self.W), (self.ftr_size, self.ftr_size))
        ftr, ftr_mask = carving_feature(ref['roi_mask'], ref['roi_feature'], ref['RTs'], ref['K_crop'], self.ftr_size)
        return ftr, ftr_mask


    def forward(self, images, ftr, ftr_mask, bboxes, obj_ids, gt_RT=None):
        bsz = images.shape[0]
        K_batch = self.K.repeat(bsz, 1, 1).to(images.device)
        projstn_grid = self.projstn_grid.repeat(bsz, 1, 1, 1, 1).to(images.device)
        coefficient = self.coefficient.repeat(bsz, 1, 1, 1, 1).to(images.device)
        ####################### 3D feature module ###################################
        P = {
            'ftr': ftr, 
            'ftr_mask': ftr_mask
        }
        pred = {}

        ######################## initial pose estimate ##############################
        if self.mode == 'train':
            bboxes = bbox_add_noise(bboxes, std_rate=0.1)
        pred[0] = {'RT': RT_from_boxes(bboxes, K_batch).detach()}  

        ####################### DProST grid cropping #################################
        ###### grid zoom-in 
        P['grid_crop'], P['coeffi_crop'], P['K_crop'], P['bboxes_crop'] = crop_inputs(projstn_grid, coefficient, K_batch, bboxes, (self.ftr_size, self.ftr_size))
        ####################### crop from image ######################################
        ####### get RoI feature from image    
        P['roi_feature'] = get_roi_feature(P['bboxes_crop'], images, (self.H, self.W), (self.ftr_size, self.ftr_size))
        ####################### DProST grid push & transform #########################
        ##### get ftr feature
        for i in range(1, self.iteration+1):
            pred[i], pred[i-1]['grid'], pred[i-1]['dist'] = self.projective_pose(
                self.local_network[str(i)], 
                pred[i-1]['RT'], 
                P['ftr'], 
                P['ftr_mask'], 
                P['roi_feature'], 
                P['grid_crop'], 
                P['coeffi_crop'], 
                P['K_crop'])

        if self.mode == 'train' or self.mode == 'valid':
            pred[self.iteration]['grid'], pred[self.iteration]['dist'] = dynamic_projective_stn(pred[self.iteration]['RT'], P['grid_crop'], P['coeffi_crop'])
            P['gt_grid'], P['gt_dist'] = dynamic_projective_stn(gt_RT, P['grid_crop'], P['coeffi_crop'])
        return pred, P


    def projective_pose(self, local_network, previous_RT, ftr, ftr_mask, roi_feature, grid_crop, coeffi_crop, K_crop):
        ###### Dynamic Projective STN
        obj_grid, obj_dist = dynamic_projective_stn(previous_RT, grid_crop, coeffi_crop)
        ####### sample ftr to 3D feature
        NDC_ftr, NDC_ftr_mask = grid_sampler(ftr, ftr_mask, obj_grid)

        ###### z-buffering
        proj_img, _ = z_buffer_min(NDC_ftr, NDC_ftr_mask)
        loc_input = torch.cat((proj_img, roi_feature), 1)

        ###### Localization Network 
        prediction = local_network(loc_input.detach())
        ###### update pose
        next_RT = self.update_pose(previous_RT, K_crop, prediction)
        return {'RT': next_RT}, obj_grid, obj_dist


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
        vxvyvz = vxvyvz * self.vxvyvz_W_scaler.repeat(vxvyvz.shape[0], 1).to(vxvyvz.device) * self.vxvyvz_H_scaler.repeat(vxvyvz.shape[0], 1).to(vxvyvz.device)
        TCO_updated = apply_imagespace_predictions(TCO, K_crop, vxvyvz, dR)
        return TCO_updated

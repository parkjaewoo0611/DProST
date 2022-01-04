import sys
import os
import shutil
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
import argparse
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
import data_loader.mesh_loader as module_mesh
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import proj_visualize, contour_visualize_2
import matplotlib.pyplot as plt
import numpy as np
import warnings
from utils.util import crop_inputs, get_roi_feature
warnings.filterwarnings("ignore") 

def main(config, start_level, end_level):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= config["gpu_id"]
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        obj_list=config['data_loader']['args']['obj_list'],
        img_ratio=config['data_loader']['args']['img_ratio'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
        FPS=True#config['data_loader']['args']['FPS']
    )
    mesh_loader = config.init_obj('mesh_loader', module_mesh)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    # metrics = ['ADD_score', 'MSSD_score', 'R_score', 't_score', 'proj_score']
    metrics=[]
    metric_fns = [getattr(module_metric, met) for met in metrics]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)

    # test result visualized folder
    result_path = config["result_path"]
    if os.path.isdir(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    model = model.to(device)
    model.eval()
    model.training = False

    # set iteration setting
    model.start_level = start_level
    model.end_level = end_level

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    ftr = {}
    ftr_mask = {}

    obj_references = data_loader.select_reference()
    for obj_id, references in obj_references.items():
        ftr[obj_id], ftr_mask[obj_id] = model.build_ref(references)
        ftr[obj_id], ftr_mask[obj_id] = ftr[obj_id].to(device), ftr_mask[obj_id].to(device)

    with torch.no_grad():
        for batch_idx, (images, masks, obj_ids, bboxes, RTs) in enumerate(tqdm(data_loader)):
            images, masks, bboxes, RTs = images.to(device), masks.to(device), bboxes.to(device), RTs.to(device)
            meshes = mesh_loader.batch_meshes(obj_ids)
            ftrs = torch.cat([ftr[obj_id] for obj_id in obj_ids.tolist()], 0)
            ftr_masks = torch.cat([ftr_mask[obj_id] for obj_id in obj_ids.tolist()], 0)

            prediction, P = model(images, ftrs, ftr_masks, bboxes, obj_ids, RTs)
            P['vertexes'] = torch.stack([mesh_loader.PTS_DICT[obj_id.tolist()] for obj_id in obj_ids])

            # computing loss, metrics on test set          
            loss = 0
            for idx in list(prediction.keys())[1:]:
                loss += loss_fn(prediction[idx+1], prediction[idx], RTs, **P)

            batch_size = images.shape[0]
            total_loss += loss.detach().item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(prediction[list(prediction.keys())[-1]], RTs, meshes, obj_ids) * batch_size

            #### visualize images
            size = 256
            K_batch = model.K.repeat(1, 1, 1).to(bboxes.device)
            grid_crop, coeffi_crop, K_crop, bbox_crop = crop_inputs(model.projstn_grid.to(RTs.device), model.coefficient.to(RTs.device), K_batch, bboxes, (size, size), lamb=2.5)

            img = get_roi_feature(bbox_crop, images, (model.H, model.W), (size, size)).detach().cpu()
            img = make_grid(img, nrow=batch_size, normalize=True).permute(1,2,0).numpy()
            img_vis = ((img - np.min(img))/(np.max(img) - np.min(img)) * 255).astype(np.uint8)
            
            pr_proj_labe = proj_visualize(RTs, grid_crop, coeffi_crop, P['ftr'], P['ftr_mask'])
            pr_proj_labe = F.interpolate(pr_proj_labe, (size, size), mode='bilinear', align_corners=True)
            labe = make_grid(pr_proj_labe.detach().cpu(), nrow=batch_size, normalize=True).permute(1,2,0).numpy()
            labe_vis = ((labe - np.min(labe))/(np.max(labe) - np.min(labe)) * 255).astype(np.uint8)
            
            labe_c = contour_visualize_2(labe, labe, img, only_label=True)

            pr_proj_input = proj_visualize(prediction[start_level+1], grid_crop, coeffi_crop, P['ftr'], P['ftr_mask'])
            pr_proj_input = F.interpolate(pr_proj_input, (size, size), mode='bilinear', align_corners=True)
            input = make_grid(pr_proj_input.detach().cpu(), nrow=batch_size, normalize=True).permute(1,2,0).numpy()
            input_vis = ((input - np.min(input))/(np.max(input) - np.min(input)) * 255).astype(np.uint8)

            compare_c  = contour_visualize_2(input, labe, img)
            
            result = np.concatenate((labe_c, compare_c), 0)
            result_vis = np.concatenate((labe_vis, input_vis), 0)

            for idx in list(prediction.keys())[1:]:
                pr_proj_pred = proj_visualize(prediction[idx], grid_crop, coeffi_crop, P['ftr'], P['ftr_mask'])    
                pr_proj_pred = F.interpolate(pr_proj_pred, (size, size), mode='bilinear', align_corners=True)
                pred = make_grid(pr_proj_pred.detach().cpu(), nrow=batch_size, normalize=True).permute(1,2,0).numpy()
                pred_vis = ((pred - np.min(pred))/(np.max(pred) - np.min(pred)) * 255).astype(np.uint8)
    
                compare_c  = contour_visualize_2(pred, labe, img)
    
                result = np.concatenate((result, compare_c), 0)
                result_vis = np.concatenate((result_vis, pred_vis), 0)

            plt.imsave(f'{result_path}/result_{batch_idx}.png', result)
            plt.imsave(f'{result_path}/resultvis_{batch_idx}.png', result_vis)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='saved/models/OCCLUSION_1000/5/config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/models/OCCLUSION_1000/5/checkpoint-epoch1000.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--result_path', default='saved/results/OCCLUSION_1000/5', type=str,
                      help='result saved path')
    args.add_argument('-s', '--start_level', default=2, type=int,
                      help='start level')
    args.add_argument('-e', '--end_level', default=0, type=int,
                      help='end level')
    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args.start_level, args.end_level)

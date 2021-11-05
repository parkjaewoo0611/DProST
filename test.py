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
from utils import proj_visualize, contour_visualize
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

def main(config):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= config["gpu_id"]
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=4,
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
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

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

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    start_level = config["arch"]["args"]["start_level"]
    end_level = config["arch"]["args"]["end_level"]

    ftr = {}
    ftr_mask = {}

    obj_references = data_loader.select_reference()
    for obj_id, references in obj_references.items():
        ftr[obj_id], ftr_mask[obj_id] = model.build_ref(references)
        ftr[obj_id], ftr_mask[obj_id] = ftr[obj_id].to(device), ftr_mask[obj_id].to(device)

    with torch.no_grad():
        for batch_idx, (images, masks, obj_ids, bboxes, RTs) in enumerate(tqdm(data_loader)):
            images, masks, bboxes, RTs = images.to(device), masks.to(device), bboxes.to(device), RTs.to(device)
            front, top, right = mesh_loader.batch_render(obj_ids)
            meshes = mesh_loader.batch_meshes(obj_ids)
            ftrs = torch.cat([ftr[obj_id] for obj_id in obj_ids.tolist()], 0)
            ftr_masks = torch.cat([ftr_mask[obj_id] for obj_id in obj_ids.tolist()], 0)

            prediction, P = model(images, ftrs, ftr_masks, front, top, right, bboxes, obj_ids, RTs)

            # computing loss, metrics on test set          
            loss = 0
            for idx in list(prediction.keys())[1:]:
                loss += loss_fn(prediction[idx+1], prediction[idx], RTs, **P)

            batch_size = images.shape[0]
            total_loss += loss.detach().item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(prediction[list(prediction.keys())[-1]], RTs, meshes, obj_ids) * batch_size

            ##### visualize images
            img = make_grid(P['roi_feature'].detach().cpu(), nrow=batch_size, normalize=True).permute(1,2,0).numpy()
            img_vis = ((img - np.min(img))/(np.max(img) - np.min(img)) * 255).astype(np.uint8)
            
            pr_proj_labe = proj_visualize(RTs, P['grid_crop'], P['coeffi_crop'], P['ftr'], P['ftr_mask'])
            pr_proj_labe = F.interpolate(pr_proj_labe, (model.input_size, model.input_size), mode='bilinear', align_corners=True)
            labe = make_grid(pr_proj_labe.detach().cpu(), nrow=batch_size, normalize=True).permute(1,2,0).numpy()
            labe_vis = ((labe - np.min(labe))/(np.max(labe) - np.min(labe)) * 255).astype(np.uint8)
            labe_c  = contour_visualize(labe, img)
            
            pr_proj_input = proj_visualize(prediction[start_level+1], P['grid_crop'], P['coeffi_crop'], P['ftr'], P['ftr_mask'])
            pr_proj_input = F.interpolate(pr_proj_input, (model.input_size, model.input_size), mode='bilinear', align_corners=True)
            input = make_grid(pr_proj_input.detach().cpu(), nrow=batch_size, normalize=True).permute(1,2,0).numpy()
            input_vis = ((input - np.min(input))/(np.max(input) - np.min(input)) * 255).astype(np.uint8)
            input_c = contour_visualize(input, img, (0, 0, 255))

            result = np.concatenate((img_vis, labe_vis, labe_c, input_vis, input_c), 0)

            for idx in list(prediction.keys())[1:]:
                pr_proj_pred = proj_visualize(prediction[idx], P['grid_crop'], P['coeffi_crop'], P['ftr'], P['ftr_mask'])    
                pr_proj_pred = F.interpolate(pr_proj_pred, (model.input_size, model.input_size), mode='bilinear', align_corners=True)
                pred = make_grid(pr_proj_pred.detach().cpu(), nrow=batch_size, normalize=True).permute(1,2,0).numpy()
                pred_vis = ((pred - np.min(pred))/(np.max(pred) - np.min(pred)) * 255).astype(np.uint8)
                pred_c = contour_visualize(pred, img, (0, 0, 255))

                result = np.concatenate((result, pred_vis, (pred_c//2 + labe_c//2)), 0)

            plt.imsave(f'{result_path}/result_{batch_idx}.png', result)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='saved/models/ProjectivePose/1105_193132/config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/models/ProjectivePose/1105_193132/checkpoint-epoch1.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--result_path', default='saved/results/1105_193132', type=str,
                      help='result saved path')
    args.add_argument('-s', '--start_level', default=None, type=int,
                      help='start level')
    args.add_argument('-e', '--end_level', default=None, type=int,
                      help='end level')
    config = ConfigParser.from_args(args)
    main(config)

import sys
import os
import shutil
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
import argparse
from parse_config import str2bool
import collections
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import data_loader.mesh_loader as module_mesh
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.util import get_roi_feature, contour_visualize, crop_inputs, get_proj_grid
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

def main(config, is_test=True, data_loader=None, mesh_loader=None, model=None, criterion=None, best_path=None, **kwargs):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= config["gpu_id"]

    # test result visualized folder
    if is_test and config["visualize"]:
        result_path = config["result_path"]
        if os.path.isdir(result_path):
            shutil.rmtree(result_path)
        os.makedirs(result_path, exist_ok=True)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = config.get_logger('test')

    if is_test:
        # setup data_loader instances
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=1,
            reference_N=config['data_loader']['args']['reference_N'],
            obj_list=config['data_loader']['args']['obj_list'],
            img_ratio=config['data_loader']['args']['img_ratio'],
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2,
            FPS=config['data_loader']['args']['FPS']
        )
        mesh_loader = config.init_obj('mesh_loader', module_mesh)

        # build model architecture
        model = config.init_obj('arch', module_arch)
        # logger.info(model)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        best_path = config.resume

    logger.info('Loading checkpoint: {} ...'.format(best_path))
    checkpoint = torch.load(best_path)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.training = False

    # set iteration setting
    model.start_level = config["arch"]["args"]["start_level"]
    model.end_level = config["arch"]["args"]["end_level"]

    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    metrics = ["VSD_score", "MSSD_score", "MSPD_score", 
               "ADD_score_02", "ADD_score_05", "ADD_score_10", 
               "PROJ_score_02", "PROJ_score_05", "PROJ_score_10"]
    metric_fns = [getattr(module_metric, met) for met in metrics]

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    ftr = {}
    ftr_mask = {}

    obj_references = data_loader.select_reference()
    for obj_id, references in obj_references.items():
        ftr[obj_id], ftr_mask[obj_id] = model.build_ref(references)
        ftr[obj_id], ftr_mask[obj_id] = ftr[obj_id].to(device), ftr_mask[obj_id].to(device)

    with torch.no_grad():
        for batch_idx, (images, masks, depths, obj_ids, bboxes, RTs) in enumerate(tqdm(data_loader)):
            images, masks, bboxes, RTs = images.to(device), masks.to(device), bboxes.to(device), RTs.to(device)
            ftrs = torch.cat([ftr[obj_id] for obj_id in obj_ids.tolist()], 0)
            ftr_masks = torch.cat([ftr_mask[obj_id] for obj_id in obj_ids.tolist()], 0)
            output, P = model(images, ftrs, ftr_masks, bboxes, obj_ids, RTs)
            P['vertexes'] = torch.stack([mesh_loader.FULL_PTS_DICT[obj_id.tolist()] for obj_id in obj_ids])

            # computing loss, metrics on test set          
            loss = 0
            for idx in list(output.keys())[1:]:
                loss += criterion(RTs, output[idx], **P)

            batch_size = images.shape[0]
            total_loss += loss.detach().item() * batch_size
            M = {
                'out_RT' : output[list(output.keys())[-1]]['RT'],
                'gt_RT' : RTs,
                'ids' : obj_ids,
                'points' : P['vertexes'],
                'depth_maps' : depths
            }
            for i, met in enumerate(metric_fns):
                total_metrics[i] += met(**M) * batch_size

            #### visualize images
            if is_test and config["visualize"]:
                size = 256
                K_batch = model.K.repeat(1, 1, 1).to(bboxes.device)
                grid_crop, coeffi_crop, K_crop, bbox_crop = crop_inputs(model.projstn_grid.to(RTs.device), model.coefficient.to(RTs.device), 
                                                                        K_batch, bboxes, (size, size), lamb=2.5)
                img = get_roi_feature(bbox_crop, images, (model.H, model.W), (size, size)).detach().cpu()
                img = make_grid(img, nrow=batch_size, normalize=True).permute(1,2,0).numpy()
                label_proj = get_proj_grid(RTs, grid_crop, coeffi_crop, P['ftr'], P['ftr_mask'], size)
                label_contour = contour_visualize(label_proj, label_proj, img, only_label=True)
                input_proj = get_proj_grid(output[model.start_level+1]['RT'], grid_crop, coeffi_crop, P['ftr'], P['ftr_mask'], size)
                input_contour  = contour_visualize(input_proj, label_proj, img)
                result_proj = np.concatenate((label_proj, input_proj), 0)                
                result_contour = np.concatenate((label_contour, input_contour), 0)
                for idx in list(output.keys())[1:]:
                    level_proj = get_proj_grid(output[idx]['RT'], grid_crop, coeffi_crop, P['ftr'], P['ftr_mask'], size)
                    level_contour  = contour_visualize(level_proj, label_proj, img)
                    result_proj = np.concatenate((result_proj, level_proj), 0)
                    result_contour = np.concatenate((result_contour, level_contour), 0)
                plt.imsave(f'{result_path}/result_{batch_idx}.png', result_contour)
                plt.imsave(f'{result_path}/resultvis_{batch_idx}.png', result_proj)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: round(total_metrics[i].item() / n_samples * 100, 1) for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-v', '--visualize', default=False, type=str2bool,
                      help='visualize results in result folder')
    args.add_argument('--result_path', default=None, type=str,
                      help='result saved path')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--start_level'], type=int, target='arch;args;start_level'),
        CustomArgs(['--end_level'], type=int, target='arch;args;end_level'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

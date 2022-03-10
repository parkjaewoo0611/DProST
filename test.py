import sys
import os
import shutil
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
import argparse
from parse_config import str2bool
import collections
import torch
from tqdm import tqdm
import data_loader.mesh_loader as module_mesh
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.error as module_error
import model.model as module_arch
from parse_config import ConfigParser
from utils.util import visualize
import matplotlib.pyplot as plt
import csv
import warnings
from utils.util import hparams_key, build_ref, get_param, MetricTracker, farthest_rotation_sampling, ensure_dir
import random

warnings.filterwarnings("ignore") 

def main(config, is_test=True, data_loader=None, mesh_loader=None, model=None, best_path=None, writer=None, 
         error_ftns=None, metric_ftns=None, ftr=None, ftr_mask=None, use_mesh=False, **kwargs):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= config["gpu_id"]

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = config.get_logger('test')

    if is_test:
        # test result visualized folder
        if config["visualize"]:
            result_path = config["result_path"]
            if os.path.isdir(result_path):
                shutil.rmtree(result_path)
            os.makedirs(result_path, exist_ok=True)
        # set repeated args required
        config['data_loader']['img_ratio'] = config['arch']['args']['img_ratio']
        config['data_loader']['is_dist'] = False

        config['mesh_loader']['args']['data_dir'] = config['data_loader']['data_dir']
        config['mesh_loader']['args']['obj_list'] = config['data_loader']['obj_list'] 
        config['arch']['args']['device'] = device

        # setup data_loader instances
        data_loader = getattr(module_data, 'DataLoader')(
            config['data_loader']['data_dir'],
            batch_size=3,
            obj_list=config['data_loader']['obj_list'],
            mode='test',
            img_ratio=config['arch']['args']['img_ratio'],
            shuffle=False,
            num_workers=1,
            rank=0,
            is_dist = False
        )
        ref_dataset = getattr(module_data, 'PoseDataset')(
            config['data_loader']['data_dir'], 
            config['data_loader']['obj_list'], 
            'train', 
            config['arch']['args']['img_ratio'])
        mesh_loader = config.init_obj('mesh_loader', module_mesh)

        # build model architecture
        model = config.init_obj('arch', module_arch)
        model.visualize = config['visualize']

        # get function handles of loss and metrics
        best_path = config.resume

        metric_ftns = [getattr(module_metric, met) for met in config['test_metrics']]
        error_ftns = [getattr(module_error, met) for met in config['test_errors']]

        # reference build
        ref_param = {
            'K_d': model.K_d,
            'XYZ' : model.XYZ, 
            'steps' : model.steps, 
            'ftr_size' : model.ftr_size, 
            'H' : model.H, 
            'W' : model.W
        }
        use_mesh = config['mesh_loader']['args']['use_mesh']
        if not use_mesh:
            ftr = {}
            ftr_mask = {}
            for obj_id in config['mesh_loader']['args']['obj_list']:
                print(f'Generating Reference Feature of obj {obj_id}')
                if config['reference']['FPS']:
                    ref_idx = farthest_rotation_sampling(ref_dataset.dataset, obj_id, config['reference']['reference_N'])
                else:
                    ref_idx = random.sample(ref_dataset.dataset, config['reference']['reference_N'])

                ftr[obj_id], ftr_mask[obj_id] = build_ref(ref_dataset, ref_idx, **ref_param)
        

    test_metrics = MetricTracker(error_ftns=error_ftns, metric_ftns=metric_ftns)

    logger.info('Loading checkpoint: {} ...'.format(best_path))
    checkpoint = torch.load(best_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.mode = 'test'
    
    DATA_PARAM = get_param(data_loader.dataset.data_dir)

    # set iteration setting
    model.iteration = config["arch"]["args"]["iteration"]

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, disable=config['gpu_scheduler'])):
            images, masks, bboxes, RTs, Ks = batch['images'].to(device), batch['masks'].to(device), batch['bboxes'].to(device), batch['RTs'].to(device), batch['Ks'].to(device)
            obj_ids, depths, K_origins = batch['obj_ids'], batch['depths'], batch['K_origins']
            if use_mesh:
                meshes = mesh_loader.batch_meshes(obj_ids.tolist())
                ftrs = None
                ftr_masks = None
            else:
                meshes = None
                ftrs = torch.cat([ftr[obj_id] for obj_id in obj_ids.tolist()], 0)
                ftr_masks = torch.cat([ftr_mask[obj_id] for obj_id in obj_ids.tolist()], 0)

            output, P = model(images, ftrs, ftr_masks, bboxes, Ks, RTs, meshes)

            P['vertexes'] = [mesh_loader.FULL_PTS_DICT[obj_id.tolist()] for obj_id in obj_ids]

            # computing loss, metrics on test set          
            M = {
                'out_RT' : output[list(output.keys())[-1]]['RT'],
                'gt_RT' : RTs,
                'K' : K_origins,
                'ids' : obj_ids,
                'points' : P['vertexes'],
                'depth_maps' : depths,
                'DATA_PARAM' : DATA_PARAM
            }

            for err in test_metrics._error_ftns:
                test_metrics.update(err.__name__, err(**M))
            test_metrics.update('diameter', [DATA_PARAM['idx2diameter'][id] for id in obj_ids.tolist()])
            test_metrics.update('id', obj_ids.tolist())

            #### visualize images
            if is_test and config["visualize"] and (batch_idx % 10 == 0):
                c, f, g = visualize(RTs, output, P, g_vis=True)
                plt.imsave(f'{result_path}/result_{batch_idx}.png', c)
                plt.imsave(f'{result_path}/resultvis_{batch_idx}.png', f)
                plt.imsave(f'{result_path}/grid_{batch_idx}.png', g)
                
            
            if config['trainer']['is_toy'] and batch_idx==5:
                break

    for obj_test in config['data_loader']['obj_list']:
        log = {}
        log.update({
            k : v for k, v in test_metrics.result(obj_test).items()
        })
        logger.info(log)
        
        result_csv_path = list(best_path.parts)
        if not is_test:
            result_csv_path[1], result_csv_path[-1] = 'log', f'test_result_{obj_test}.csv'
        else:
            result_csv_path[-2], result_csv_path[-1] = 'log', f'test_result_{obj_test}.csv'
        ensure_dir(os.path.join(*result_csv_path[:-1]))
        result_csv_path = os.path.join(*result_csv_path)
        with open(result_csv_path, "w") as csv_file:
            metric_csv = csv.writer(csv_file)
            for k, v in log.items():
                metric_csv.writerow([k, v])

    if not is_test:
        hparams = hparams_key(config.config)
        writer.add_hparams(hparams, log)


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
    args.add_argument('-p', '--result_path', default=None, type=str,
                      help='result saved path')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--iteration'], type=int, target='arch;args;iteration'),
        CustomArgs(['--obj_list'], type=list, target='data_loader;obj_list'),
        CustomArgs(['--batch_size'], type=int, target='data_loader;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

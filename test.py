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
from utils.util import hparams_key, build_ref, get_param, MetricTracker

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
        config['data_loader']['args']['img_ratio'] = config['arch']['args']['img_ratio']
        config['mesh_loader']['args']['data_dir'] = config['data_loader']['args']['data_dir']
        config['mesh_loader']['args']['obj_list'] = config['data_loader']['args']['obj_list'] 
        config['arch']['args']['device'] = device

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
            num_workers=4,
            FPS=config['data_loader']['args']['FPS']
        )
        mesh_loader = config.init_obj('mesh_loader', module_mesh)

        # build model architecture
        model = config.init_obj('arch', module_arch)
        # logger.info(model)

        # get function handles of loss and metrics
        best_path = config.resume

        metric_ftns = [getattr(module_metric, met) for met in config['test_metrics']]
        error_ftns = [getattr(module_error, met) for met in config['test_errors']]

        ftr = {}
        ftr_mask = {}
        for obj_id in config['mesh_loader']['args']['obj_list']:
            print(f'Generating Reference Feature of obj {obj_id}')
            ref = data_loader.select_reference(obj_id)
            ftr[obj_id], ftr_mask[obj_id] = build_ref(ref, model.K_d, model.XYZ, model.N_z, model.ftr_size, model.H, model.W)

    test_metrics = MetricTracker(error_ftns=error_ftns, metric_ftns=metric_ftns)

    logger.info('Loading checkpoint: {} ...'.format(best_path))
    checkpoint = torch.load(best_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.mode = 'test'
    
    DATA_PARAM = get_param(data_loader.data_dir)

    # set iteration setting
    model.iteration = config["arch"]["args"]["iteration"]


    with torch.no_grad():
        for batch_idx, (images, masks, depths, obj_ids, bboxes, RTs, Ks, K_origins) in enumerate(tqdm(data_loader, disable=config['gpu_scheduler'])):
            images, masks, bboxes, RTs, Ks = images.to(device), masks.to(device), bboxes.to(device), RTs.to(device), Ks.to(device)
            if use_mesh:
                meshes = mesh_loader.batch_meshes(obj_ids)
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

            #### visualize images
            if is_test and config["visualize"]:
                c, g = visualize(RTs, output, P)
                plt.imsave(f'{result_path}/result_{batch_idx}.png', c)
                plt.imsave(f'{result_path}/resultvis_{batch_idx}.png', g)

    log = {}
    log.update({
        k : v for k, v in test_metrics.result().items()
    })
    logger.info(log)
    
    result_csv_path = list(best_path.parts)
    result_csv_path[1], result_csv_path[-1] = 'log', 'test_result.csv'
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
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

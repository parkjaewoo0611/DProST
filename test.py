import sys
import os
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
import argparse
from parse_config import str2bool
import collections
import torch
from tqdm import tqdm
import data_loader.mesh_loader as module_mesh
import data_loader.data_loaders as module_data
import data_loader.reference_loader as module_ref
import model.metric as module_metric
import model.error as module_error
import model.model as module_arch
from parse_config import ConfigParser
from utils.util import visualize
import matplotlib.pyplot as plt
import csv
import warnings
from utils.util import hparams_key, get_param, MetricTracker, bbox_3d_visualize, reset_dir
from logger import TensorboardWriter

warnings.filterwarnings("ignore")

def main(config, is_test=True, best_path=None, **kwargs):
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = config.get_logger('test')

    if is_test:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]= ",".join([str(id) for id in config["gpu_id"]])
        # set repeated args required
        config['arch']['args']['device'] = device
        best_path = config.resume

    # data loader
    data_loader = config.init_obj('data_loader', module_data,
        batch_size=9,
        img_ratio=config['arch']['args']['img_ratio'],
        mode='test',
        shuffle=False,
        num_worker=8)

    #  mesh loader
    mesh_loader = config.init_obj('mesh_loader', module_mesh,
        data_dir=config['data_loader']['args']['data_dir'],
        obj_list=config['data_loader']['args']['obj_list'],
        device=device)

    # reference loader
    ref_dataset = getattr(module_data, 'PoseDataset')(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['obj_list'],
        'train',
        config['arch']['args']['img_ratio'])
    ref_loader = config.init_obj('reference_loader', module_ref,
        ref_dataset=ref_dataset,
        obj_list=config['data_loader']['args']['obj_list'],
        use_mesh=config['mesh_loader']['args']['use_mesh'],
        img_ratio=config['arch']['args']['img_ratio'],
        N_z=config['arch']['args']['N_z'],
        device=device)

    metric_ftns = [getattr(module_metric, met) for met in config['test_metrics']]
    error_ftns = [getattr(module_error, met) for met in config['test_errors']]
    test_metrics = MetricTracker(error_ftns=error_ftns, metric_ftns=metric_ftns)

    # test result visualized folder
    if config.visualize:
        result_path = config.result_path
        reset_dir(config.result_path)

    # build model architecture
    model = config.init_obj('arch', module_arch, device=device)

    logger.info('Loading checkpoint: {} ...'.format(best_path))
    checkpoint = torch.load(best_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.mode = 'test'
    model.iteration = config["arch"]["args"]["iteration"]
    model.visualize = config.visualize

    DATA_PARAM = get_param(data_loader.dataset.data_dir)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, disable=config['gpu_scheduler'])):
            images, bboxes, RTs, Ks = batch['images'].to(device), batch['bboxes'].to(device), batch['RTs'].to(device), batch['Ks'].to(device)
            obj_ids, K_origins = batch['obj_ids'], batch['K_origins']

            meshes = mesh_loader.batch_meshes(obj_ids.tolist())            # load mesh or not by use_mesh
            refs, ref_masks = ref_loader.batch_refs(obj_ids.tolist())      # load ref or not by use_mesh

            output, P = model(images, refs, ref_masks, bboxes, Ks, RTs, meshes)

            # for quantitative & qualitative comparison
            P['vertexes'] = mesh_loader.batch_full_pts(obj_ids.tolist())
            P['bbox_3d'] = mesh_loader.batch_bbox_3d(obj_ids.tolist())

            # computing loss, metrics on test set
            M = {
                'out_RT' : output[list(output.keys())[-1]]['RT'],
                'gt_RT' : RTs,
                'K' : K_origins,
                'ids' : obj_ids,
                'points' : P['vertexes'],
                'DATA_PARAM' : DATA_PARAM
            }

            for err in test_metrics._error_ftns:
                test_metrics.update(err.__name__, err(**M))
            test_metrics.update('diameter', [DATA_PARAM['idx2diameter'][id] for id in obj_ids.tolist()])
            test_metrics.update('id', obj_ids.tolist())

            #### visualize images
            if is_test and config.visualize and (batch_idx % 10 == 0):
                c, f, g = visualize(RTs, output, P, g_vis=True)
                b = bbox_3d_visualize(RTs, output[list(output.keys())[-1]]['RT'], Ks, P['bbox_3d'], images, bboxes)
                plt.imsave(f'{result_path}/result_{batch_idx}.png', c)
                plt.imsave(f'{result_path}/resultvis_{batch_idx}.png', f)
                plt.imsave(f'{result_path}/grid_{batch_idx}.png', g)
                plt.imsave(f'{result_path}/bbox_{batch_idx}.png', b)

            if config['trainer']['is_toy'] and batch_idx==5:
                break

    out_path = best_path.parent / 'out'         # result inside model folder (to math the pattern in train and test)
    reset_dir(out_path)
    for obj_test in config['data_loader']['args']['obj_list']:
        error_path = out_path / f'test_error_{obj_test}.csv'
        error = test_metrics.error(obj_test)
        with open(error_path, "w") as csv_file:
            error_csv = csv.writer(csv_file)
            for k, v in error.items():
                error_csv.writerow([k, v])

        metric_path = out_path / f'test_result_{obj_test}.csv'
        result = test_metrics.result(obj_test)
        with open(metric_path, "w") as csv_file:
            metric_csv = csv.writer(csv_file)
            for k, v in result.items():
                metric_csv.writerow([k, v])

        logger.info(f'result of obj {obj_test}: {result}')

    total_result = test_metrics.result()
    logger.info(f'total result: {total_result}')

    if not is_test:
        writer = TensorboardWriter(config.log_dir, logger, config['trainer']['tensorboard'])
        hparams = hparams_key(config.config)
        writer.add_hparams(hparams, total_result)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-v', '--visualize', default=False, type=str2bool,
                      help='visualize results in result folder')
    args.add_argument('-p', '--result_path', default=None, type=str,
                      help='result saved path')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['--gpu_id'], type=list, target='gpu_id'),
        CustomArgs(['--iteration'], type=int, target='arch;args;iteration'),
        CustomArgs(['--obj_list'], type=list, target='data_loader;args;obj_list'),
        CustomArgs(['--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--gpu_scheduler'], type=bool, target='gpu_scheduler'),
        CustomArgs(['--is_toy'], type=bool, target='trainer;is_toy'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

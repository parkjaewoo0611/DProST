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
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.util import visualize
import matplotlib.pyplot as plt
import csv
import warnings
from flatten_dict import flatten
warnings.filterwarnings("ignore") 

def main(config, is_test=True, data_loader=None, mesh_loader=None, model=None, criterion=None, best_path=None, writer=None, metric_ftns=None, ftr=None, ftr_mask=None, **kwargs):
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
        criterion = getattr(module_loss, config['loss'])
        best_path = config.resume

        metric_ftns = [getattr(module_metric, met) for met in config['test_metrics']]

        ftr = {}
        ftr_mask = {}
        obj_references = data_loader.select_reference()
        for obj_id, references in obj_references.items():
            ftr[obj_id], ftr_mask[obj_id] = model.build_ref(references)
            ftr[obj_id], ftr_mask[obj_id] = ftr[obj_id].to(device), ftr_mask[obj_id].to(device)

    logger.info('Loading checkpoint: {} ...'.format(best_path))
    checkpoint = torch.load(best_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.training = False

    # set iteration setting
    model.iteration = config["arch"]["args"]["iteration"]

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_ftns))

    with torch.no_grad():
        for batch_idx, (images, masks, depths, obj_ids, bboxes, RTs) in enumerate(tqdm(data_loader, disable=config['gpu_scheduler'])):
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
            for i, met in enumerate(metric_ftns):
                total_metrics[i] += met(**M) * batch_size

            #### visualize images
            if is_test and config["visualize"]:
                c, g = visualize(RTs, output, P)
                plt.imsave(f'{result_path}/result_{batch_idx}.png', c)
                plt.imsave(f'{result_path}/resultvis_{batch_idx}.png', g)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: round(total_metrics[i].item() / n_samples * 100, 1) for i, met in enumerate(metric_ftns)
    })
    logger.info(log)
    
    result_csv_path = best_path.split('/')
    result_csv_path[1], result_csv_path[-1] = 'log', 'test_result.csv'
    result_csv_path = os.path.join(*result_csv_path)
    with open(result_csv_path, "w") as csv_file:
        metric_csv = csv.writer(csv_file)
        for k, v in log.items():
            metric_csv.writerow([k, v])

    if not is_test:
        hparams = flatten(config.config, reducer='path')
        for k, v in hparams.items(): hparams[k]=f"{v}"
        log['saved_epoch'] = 'test'
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

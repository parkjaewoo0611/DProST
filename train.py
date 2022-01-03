import sys
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
import argparse
import collections
import torch
import numpy as np
import data_loader.mesh_loader as module_mesh
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import os

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= config['gpu_id']
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['gpu_id'])
    
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    if config['data_loader']['args']['test_as_valid']:
        test_args = config['data_loader']['args'].copy()
        test_args['shuffle'], test_args['training'] = False, False
        valid_data_loader = getattr(module_data, config['data_loader']['type'])(**test_args)
    else:
        valid_data_loader = data_loader.split_validation()

    # mesh loader
    mesh_loader = config.init_obj('mesh_loader', module_mesh)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch )
    logger.info(model)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      mesh_loader=mesh_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--result_path', default=None, type=str,
                      help='result saved path')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--reference_N'], type=int, target='data_loader;args;reference_N'),
        CustomArgs(['--N_z'], type=int, target='arch;args;N_z'),
        CustomArgs(['--occlusion'], type=bool, target='arch;args;occlusion'),
        CustomArgs(['--is_pbr'], type=bool, target='data_loader;args;is_pbr'),
        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['--mesh_dir'], type=str, target='mesh_loader;args;mesh_dir'),
        CustomArgs(['--data_obj_list'], type=list, target='data_loader;args;obj_list'),
        CustomArgs(['--mesh_obj_list'], type=list, target='mesh_loader;args;obj_list'),
        CustomArgs(['--loss'], type=str, target='loss'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

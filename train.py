import sys
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
import argparse
from parse_config import str2bool
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
import test

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    if not config['gpu_scheduler']:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]= config['gpu_id']
    if config['gpu_scheduler']:
        config['trainer']['verbosity'] = 0

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['gpu_id'])
    
    logger = config.get_logger('train')

    # set repeated args required
    config['data_loader']['args']['img_ratio'] = config['arch']['args']['img_ratio']
    config['mesh_loader']['args']['data_dir'] = config['data_loader']['args']['data_dir']
    config['mesh_loader']['args']['obj_list'] = config['data_loader']['args']['obj_list'] 
    config['arch']['args']['device'] = device
    config['data_loader']['args']['batch_size'] = len(device_ids) * config['data_loader']['args']['batch_size']

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch )

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    if config['data_loader']['args']['test_as_valid']:
        test_args = config['data_loader']['args'].copy()
        test_args['shuffle'], test_args['training'], test_args['validation_split'] = False, False, 0.0
        valid_data_loader = getattr(module_data, config['data_loader']['type'])(**test_args)
    else:
        valid_data_loader = data_loader.split_validation()

    # mesh loader
    mesh_loader = config.init_obj('mesh_loader', module_mesh)

    ftr = {}
    ftr_mask = {}
    obj_references = data_loader.select_reference()
    for obj_id, references in obj_references.items():
        print(f'Generating Reference Feature of obj {obj_id}')
        ftr[obj_id], ftr_mask[obj_id] = model.build_ref(references)
        ftr[obj_id], ftr_mask[obj_id] = ftr[obj_id].to(device), ftr_mask[obj_id].to(device)
    
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    valid_metrics = [getattr(module_metric, met) for met in config['valid_metrics']]
    test_metrics = [getattr(module_metric, met) for met in config['test_metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, valid_metrics, test_metrics, optimizer,
                      ftr, ftr_mask,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      mesh_loader=mesh_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    test_dict = {
        "data_loader": valid_data_loader,
        "mesh_loader": mesh_loader,
        "model": model,
        "criterion": criterion,
        "best_path": f"{trainer.best_dir}/model_best.pth",
        "writer": trainer.writer.set_mode('test'),
        "metric_ftns": test_metrics,
        "ftr": ftr,
        "ftr_mask": ftr_mask,
    }
    test.main(config, is_test=False, **test_dict)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
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
        CustomArgs(['--gpu_scheduler'], type=bool, target='gpu_scheduler'),
        CustomArgs(['--ftr_size'], type=int, target='arch;args;ftr_size'),
        CustomArgs(['--iteration'], type=int, target='arch;args;iteration'),
        CustomArgs(['--model_name'], type=str, target='arch;args;model_name'),
        CustomArgs(['--N_z'], type=int, target='arch;args;N_z'),

        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--obj_list'], type=list, target='data_loader;args;obj_list'),
        CustomArgs(['--reference_N'], type=int, target='data_loader;args;reference_N'),
        CustomArgs(['--is_pbr'], type=bool, target='data_loader;args;is_pbr'),
        CustomArgs(['--is_syn'], type=bool, target='data_loader;args;is_syn'),
        CustomArgs(['--FPS'], type=bool, target='data_loader;args;FPS'),

        CustomArgs(['--lr'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--loss'], type=str, target='loss'),
        CustomArgs(['--lr_step_size'], type=int, target='lr_scheduler;args;step_size'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

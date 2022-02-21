import sys
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
import argparse
from parse_config import str2bool
import collections
import torch
from torch.multiprocessing import spawn
import torch.distributed as dist
import numpy as np
import data_loader.mesh_loader as module_mesh
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.error as module_error
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import prepare_device, build_ref, farthest_rotation_sampling
import os
import test
from pathlib import Path
import random

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(gpu, config, n_gpu):
    port = random.randint(1111, 9999)
    config['arch']['args']['device'] = gpu    
    dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{port}',
            world_size=n_gpu,
            rank=gpu)
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch )
    torch.cuda.set_device(gpu)
    model = model.to(gpu)
    ref_param = {
        'K_d': model.K_d,
        'XYZ' : model.XYZ, 
        'steps' : model.steps, 
        'ftr_size' : model.ftr_size, 
        'H' : model.H, 
        'W' : model.W
    }
    if n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    # setup data_loader instances
    print('Data Loader setting...')
    train_loader_args = config['data_loader'].copy()
    synth_loader_args = config['data_loader'].copy()
    valid_loader_args = config['data_loader'].copy()    

    train_loader_args['mode'] = 'train'
    valid_loader_args['mode'] = 'test'
    train_loader_args['is_dist'] = True
    synth_loader_args['is_dist'] = True
    valid_loader_args['is_dist'] = False
    valid_loader_args['shuffle'] = False
    train_data_loader = getattr(module_data, 'DataLoader')(rank=gpu, **train_loader_args)
    synth_data_loader = getattr(module_data, 'DataLoader')(rank=gpu, **synth_loader_args)    
    valid_data_loader = getattr(module_data, 'DataLoader')(rank=gpu, **valid_loader_args)

    # mesh loader
    print('Mesh Loader setting...')
    mesh_loader = config.init_obj('mesh_loader', module_mesh)

    ftr = {}
    ftr_mask = {}
    for obj_id in config['mesh_loader']['args']['obj_list']:
        print(f'Generating Reference Feature of obj {obj_id}')
        if synth_data_loader.dataset.mode == 'train_pbr' and ('YCBV' in synth_data_loader.dataset.data_dir):
            ref_dataset = synth_data_loader.dataset
        else:
            ref_dataset = train_data_loader.dataset

        if config['reference']['FPS']:
            ref_idx = farthest_rotation_sampling(ref_dataset.dataset, obj_id, config['reference']['reference_N'])
        else:
            ref_idx = random.sample(ref_dataset.dataset, config['reference']['reference_N'])

        ftr[obj_id], ftr_mask[obj_id] = build_ref(ref_dataset, ref_idx, **ref_param)
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    valid_errors = [getattr(module_error, met) for met in config['valid_errors']]
    valid_metrics = [getattr(module_metric, met) for met in config['valid_metrics']]
    test_errors = [getattr(module_error, met) for met in config['test_errors']]
    test_metrics = [getattr(module_metric, met) for met in config['test_metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    material = {
        "model" : model,
        "criterion" : criterion,
        "valid_error_ftns" : valid_errors,
        "valid_metric_ftns" : valid_metrics,
        "test_metric_ftns" : test_metrics,
        "optimizer" : optimizer,
        "ftr" : ftr,
        "ftr_mask" : ftr_mask,
        "device" : gpu,
        "train_data_loader" : train_data_loader,
        "synth_data_loader" : synth_data_loader,
        "valid_data_loader" : valid_data_loader,
        "mesh_loader" : mesh_loader,
        "lr_scheduler" : lr_scheduler,
        "use_mesh" : config['mesh_loader']['args']['use_mesh'],
        "save_period": config['trainer']['save_period'],
        "is_toy": config['trainer']['is_toy'],
        "gpu_scheduler": config['gpu_scheduler']
    }
    trainer = Trainer(config, **material)
    trainer.train()

    dist.barrier()
    if gpu == 0 :
        material['data_loader'] = valid_data_loader
        material["best_path"] = Path(f"{trainer.best_dir}/model_best.pth")
        material['writer'] = trainer.writer.set_mode('test')
        material['error_ftns'] = test_errors
        material['metric_ftns'] = test_metrics
        test.main(config, is_test=False, **material)

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
    args.add_argument('-p', '--result_path', default=None, type=str,
                      help='result saved path')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--gpu_scheduler'], type=bool, target='gpu_scheduler'),
        CustomArgs(['--ftr_size'], type=int, target='arch;args;ftr_size'),
        CustomArgs(['--iteration'], type=int, target='arch;args;iteration'),
        CustomArgs(['--model_name'], type=str, target='arch;args;model_name'),
        CustomArgs(['--N_z'], type=int, target='arch;args;N_z'),

        CustomArgs(['--data_dir'], type=str, target='data_loader;data_dir'),
        CustomArgs(['--batch_size'], type=int, target='data_loader;batch_size'),
        CustomArgs(['--obj_list'], type=list, target='data_loader;obj_list'),
        CustomArgs(['--reference_N'], type=int, target='data_loader;reference_N'),
        CustomArgs(['--mode'], type=bool, target='data_loader;mode'),
        CustomArgs(['--FPS'], type=bool, target='data_loader;FPS'),

        CustomArgs(['--lr'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--loss'], type=str, target='loss'),
        CustomArgs(['--lr_step_size'], type=int, target='lr_scheduler;args;step_size'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--save_period'], type=int, target='trainer;save_period'),
        CustomArgs(['--is_toy'], type=bool, target='trainer;is_toy'),
        CustomArgs(['--use_mesh'], type=bool, target='mesh_loader;args;use_mesh'),
    ]
    config = ConfigParser.from_args(args, options)

    if not config['gpu_scheduler']:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]= config['gpu_id']
    if config['gpu_scheduler']:
        config['trainer']['verbosity'] = 0


    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # prepare for (multi-device) GPU training
    device, device_ids, n_gpu = prepare_device(config['gpu_id'])

    # set repeated args required
    config['data_loader']['img_ratio'] = config['arch']['args']['img_ratio']
    config['mesh_loader']['args']['data_dir'] = config['data_loader']['data_dir']
    config['mesh_loader']['args']['obj_list'] = config['data_loader']['obj_list']

    logger = config.get_logger('train')

    spawn(main, nprocs=n_gpu, args=(config, n_gpu, ))

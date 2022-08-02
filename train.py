import sys
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
import argparse
from parse_config import str2bool
import collections
import torch
from torch.multiprocessing import spawn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.modules.batchnorm import SyncBatchNorm
import numpy as np
import data_loader.mesh_loader as module_mesh
import data_loader.data_loaders as module_data
import data_loader.reference_loader as module_ref
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.error as module_error
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import prepare_device
import os
import test
from pathlib import Path
import random
import builtins

# fix random seeds for reproducibility
SEED = 123

def main(rank, world_size, config):
    torch.manual_seed(SEED + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED + rank)

    if config['trainer']['is_toy']:
        config['trainer']['epochs'] = 3
        config['trainer']['save_period'] = 1

    # DDP setting
    if world_size > 1 and rank !=0:
        def print_pass(*args):
            pass
        builtins.print=print_pass
        
    print(f"Use GPU: {world_size} for training")
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(rank)

    # model loader
    model = config.init_obj('arch', module_arch, device=device)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    # data loader
    print('Data Loader setting...')
    train_data_loader = config.init_obj('data_loader', module_data, 
        img_ratio=config['arch']['args']['img_ratio'],
        mode='train')
    synth_data_loader = config.init_obj('data_loader', module_data,
        img_ratio=config['arch']['args']['img_ratio'])
    valid_data_loader = config.init_obj('data_loader', module_data, 
        img_ratio=config['arch']['args']['img_ratio'],
        mode='test', 
        shuffle=False)

    # mesh loader
    print('Mesh Loader setting...')
    mesh_loader = config.init_obj('mesh_loader', module_mesh, 
        data_dir=config['data_loader']['args']['data_dir'],
        obj_list=config['data_loader']['args']['obj_list'],
        device=device)

    # reference loader
    print('Reference Loader setting...')
    ref_loader = config.init_obj('reference_loader', module_ref,
        ref_dataset=train_data_loader.dataset,
        obj_list=config['data_loader']['args']['obj_list'],
        use_mesh=config['mesh_loader']['args']['use_mesh'],
        img_ratio=config['arch']['args']['img_ratio'],
        N_z=config['arch']['args']['N_z'],
        device=device)
    
    # get function handles of loss and metrics
    print('Metric setting...')
    criterion = getattr(module_loss, config['loss'])
    valid_errors = [getattr(module_error, met) for met in config['valid_errors']]
    valid_metrics = [getattr(module_metric, met) for met in config['valid_metrics']]
    test_metrics = [getattr(module_metric, met) for met in config['test_metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    material = {
        "model" : model,
        "optimizer" : optimizer,
        "criterion" : criterion,
        "lr_scheduler" : lr_scheduler,
        "valid_error_ftns" : valid_errors,
        "valid_metric_ftns" : valid_metrics,
        "test_metric_ftns" : test_metrics,
        "train_data_loader" : train_data_loader,
        "synth_data_loader" : synth_data_loader,
        "valid_data_loader" : valid_data_loader,
        "mesh_loader" : mesh_loader,
        "ref_loader" : ref_loader,
        "save_period": config['trainer']['save_period'],
        "is_toy": config['trainer']['is_toy'],
        "gpu_scheduler": config['gpu_scheduler'],
        "device" : device,
        "rank" : rank
    }
    print('Trainer setting...')
    trainer = Trainer(config, **material)
    trainer.train()
    print('Training Done')


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
        CustomArgs(['--gpu_id'], type=list, target='gpu_id'),
        CustomArgs(['--gpu_scheduler'], type=bool, target='gpu_scheduler'),
        CustomArgs(['--iteration'], type=int, target='arch;args;iteration'),
        CustomArgs(['--model_name'], type=str, target='arch;args;model_name'),
        CustomArgs(['--N_z'], type=int, target='arch;args;N_z'),

        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--obj_list'], type=list, target='data_loader;args;obj_list'),
        CustomArgs(['--mode'], type=str, target='data_loader;args;mode'),

        CustomArgs(['--reference_N'], type=int, target='reference_loader;args;reference_N'),
        CustomArgs(['--FPS'], type=bool, target='reference_loader;args;FPS'),
        CustomArgs(['--ref_size'], type=int, target='reference_loader;args;ref_size'),
        CustomArgs(['--use_mesh'], type=bool, target='mesh_loader;args;use_mesh'),

        CustomArgs(['--lr'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--loss'], type=str, target='loss'),
        CustomArgs(['--valid_metrics'], type=list, target='valid_metrics'),
        CustomArgs(['--lr_step_size'], type=int, target='lr_scheduler;args;step_size'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--save_period'], type=int, target='trainer;save_period'),
        CustomArgs(['--early_stop'], type=int, target='trainer;early_stop'),
        CustomArgs(['--is_toy'], type=bool, target='trainer;is_toy'),
    ]
    config = ConfigParser.from_args(args, options)
    if config['gpu_scheduler']:
        config['trainer']['verbosity'] = 0

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{random.randint(6000, 6999)}'    # --> for multiple single gpu training
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join([str(id) for id in config["gpu_id"]])
    
    # prepare for (multi-device) GPU training
    gpu_list, world_size = prepare_device(config['gpu_id'])

    # main(rank=0, world_size=world_size, config=config)       # uncomment for debug
    spawn(fn=main, args=(world_size, config), nprocs=world_size)
    print('Test the best model')
    test.main(config, is_test=False, best_path=Path(f"{config.save_dir}/model_best.pth"))

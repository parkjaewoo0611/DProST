import sys
import os
import shutil
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
import argparse
import torch
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

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        obj_list=config['data_loader']['args']['obj_list'],
        img_ratio=config['data_loader']['args']['img_ratio'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    mesh_loader = config.init_obj('mesh_loader', module_mesh)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # test result visualized folder
    result_path = os.fspath(config.resume).split('/')
    result_path[-4] = 'results'
    result_path = os.path.join(*result_path[:-1])
    shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for batch_idx, (images, masks, obj_ids, bboxes, RTs) in enumerate(tqdm(data_loader)):
            images, masks, bboxes, RTs = images.to(device), masks.to(device), bboxes.to(device), RTs.to(device)
            front, top, right = mesh_loader.batch_render(obj_ids)
            meshes = mesh_loader.batch_meshes(obj_ids)

            M, prediction = model(images, front, top, right, bboxes, obj_ids, RTs)

            # computing loss, metrics on test set          
            loss = 0
            # for i, dict in enumerate(M):
            #     idx = len(M) - (i + 1)
            #     loss += self.criterion(prediction[idx+1], prediction[idx], RTs, **M[idx])
            #     loss += self.criterion(RTs, **M[idx])
            loss += loss_fn(prediction[4], prediction[0], RTs, **M[0])

            batch_size = images.shape[0]
            total_loss += loss.detach().item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(prediction, RTs, meshes, obj_ids) * batch_size

            pr_proj_pred = proj_visualize(prediction[0], M[0]['grid_crop'], M[0]['coeffi_crop'], M[0]['ftr'], M[0]['ftr_mask'])
            pr_proj_labe = proj_visualize(RTs, M[0]['grid_crop'], M[0]['coeffi_crop'], M[0]['ftr'], M[0]['ftr_mask'])
            
            input_vis = make_grid(M[0]['roi_feature'].detach().cpu(), nrow=2, normalize=True).permute(1,2,0).numpy()
            labe_vis = make_grid(pr_proj_labe.detach().cpu().mean(1, keepdim=True), nrow=2, normalize=True).permute(1,2,0).numpy()
            pred_vis = make_grid(pr_proj_pred.detach().cpu().mean(1, keepdim=True), nrow=2, normalize=True).permute(1,2,0).numpy()

            pred = contour_visualize(pred_vis, input_vis)
            labe = contour_visualize(labe_vis, input_vis)

            result = np.concatenate((labe, pred), 1)

            plt.imsave(f'{result_path}/result_{batch_idx}.png', result)



    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='saved/models/ProjectivePose/1026_220541/config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/models/ProjectivePose/1026_220541/checkpoint-epoch300.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

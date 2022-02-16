import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker, visualize, get_param
from tqdm import tqdm

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config, device, model, criterion, test_metric_ftns, valid_error_ftns, valid_metric_ftns, optimizer, ftr, ftr_mask, 
                 mesh_loader, train_data_loader, synth_data_loader, valid_data_loader=None, use_mesh=False, 
                 lr_scheduler=None, len_epoch=None, save_period=100, 
                 gpu_scheduler=False, is_toy=False, **kwargs):
        super().__init__(model, test_metric_ftns, optimizer, config)
        self.device = device
        self.criterion = criterion
        self.train_metrics = MetricTracker(writer=self.writer)
        self.valid_metrics = MetricTracker(error_ftns=valid_error_ftns, metric_ftns=valid_metric_ftns, writer=self.writer)

        self.ftr = ftr
        self.ftr_mask = ftr_mask

        self.train_data_loader = train_data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(self.train_data_loader)
            self.len_epoch = len_epoch
        self.synth_data_loader = inf_loop(synth_data_loader)
        self.valid_data_loader = valid_data_loader
        self.mesh_loader = mesh_loader

        self.do_validation = self.valid_data_loader is not None
        
        self.save_period = save_period
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

        self.gpu_scheduler = gpu_scheduler
        self.DATA_PARAM = get_param(self.train_data_loader.dataset.data_dir)
        self.use_mesh = use_mesh
        self.is_toy = is_toy

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.model.mode = 'train'
        self.train_metrics.reset()
        
        for batch_idx, (train_batch, synth_batch) in enumerate(zip(self.train_data_loader, self.synth_data_loader)):
            batch = {key : torch.cat([train_batch[key], synth_batch[key]], 0) for key in list(train_batch.keys())}
            images, bboxes, RTs, Ks = batch['images'].to(self.device), batch['bboxes'].to(self.device), batch['RTs'].to(self.device), batch['Ks'].to(self.device)
            obj_ids = batch['obj_ids']
            if self.use_mesh:
                meshes = self.mesh_loader.batch_meshes(obj_ids.tolist())
                ftrs = None
                ftr_masks = None
            else:
                meshes = None
                ftrs = torch.cat([self.ftr[obj_id] for obj_id in obj_ids.tolist()], 0)
                ftr_masks = torch.cat([self.ftr_mask[obj_id] for obj_id in obj_ids.tolist()], 0)
            
            self.optimizer.zero_grad()
            output, P = self.model(images, ftrs, ftr_masks, bboxes, Ks, RTs, meshes)

            if self.criterion.__name__ == 'point_matching_loss':
                P['full_vertexes'] = torch.stack([self.mesh_loader.FULL_PTS_DICT[obj_id.tolist()] for obj_id in obj_ids])
            loss = 0
            for idx in list(output.keys())[1:]:
                loss += self.criterion(RTs, output[idx], **P)
            loss.backward()

            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.loss_update(loss.detach().item(), write=True)

            if batch_idx % self.log_step == 0:
                self.logger.debug('({}) Train Epoch: {} {} Loss: {:.6f}  Best {}: {:.6f}'.format(
                    self.checkpoint_dir.name,
                    epoch,
                    self._progress(batch_idx),
                    loss.detach().item(),
                    self.mnt_metric,
                    self.mnt_best))

            if batch_idx == self.len_epoch:
                break
            if self.is_toy and batch_idx==5:
                break
        log = self.train_metrics.result()

        if self.do_validation and epoch % self.save_period == 0 :
            c, g = visualize(RTs, output, P)
            self.writer.add_image(f'contour', torch.tensor(c).permute(2, 0, 1))
            self.writer.add_image(f'rendering', torch.tensor(g).permute(2, 0, 1))
            
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.model.mode = 'valid'
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.valid_data_loader, disable=self.gpu_scheduler)):
                images, bboxes, RTs, Ks = batch['images'].to(self.device), batch['bboxes'].to(self.device), batch['RTs'].to(self.device), batch['Ks'].to(self.device)
                obj_ids, depths, K_origins = batch['obj_ids'], batch['depths'], batch['K_origins']
                
                if self.use_mesh:
                    meshes = self.mesh_loader.batch_meshes(obj_ids)
                    ftrs = None
                    ftr_masks = None
                else:
                    meshes = None
                    ftrs = torch.cat([self.ftr[obj_id] for obj_id in obj_ids.tolist()], 0)
                    ftr_masks = torch.cat([self.ftr_mask[obj_id] for obj_id in obj_ids.tolist()], 0)

                output, P = self.model(images, ftrs, ftr_masks, bboxes, Ks, RTs, meshes)

                P['vertexes'] = [self.mesh_loader.PTS_DICT[obj_id.tolist()] for obj_id in obj_ids]
                if self.criterion.__name__ == 'point_matching_loss':
                    P['full_vertexes'] = torch.stack([self.mesh_loader.FULL_PTS_DICT[obj_id.tolist()] for obj_id in obj_ids])

                loss = 0
                for idx in list(output.keys())[1:]:
                    loss += self.criterion(RTs, output[idx], **P)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.loss_update(loss.detach().item(), write=False)
                M = {
                    'out_RT' : output[list(output.keys())[-1]]['RT'],
                    'gt_RT' : RTs,
                    'K' : K_origins,
                    'ids' : obj_ids,
                    'points' : P['vertexes'],
                    'depth_maps' : depths,
                    'DATA_PARAM' : self.DATA_PARAM
                }

                for err in self.valid_metrics._error_ftns:
                    self.valid_metrics.update(err.__name__, err(**M))
                self.valid_metrics.update('diameter', [self.DATA_PARAM['idx2diameter'][id] for id in obj_ids.tolist()])
                self.valid_metrics.update('id', obj_ids.tolist())
                if self.is_toy and batch_idx==5:
                    break

            c, g = visualize(RTs, output, P)
            self.writer.add_image(f'contour', torch.tensor(c).permute(2, 0, 1))
            self.writer.add_image(f'rendering', torch.tensor(g).permute(2, 0, 1))

        result = self.valid_metrics.result()
        for k, v in result.items():
            self.writer.add_scalar(k, v)

        return result

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    


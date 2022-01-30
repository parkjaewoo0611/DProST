import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker, visualize
from tqdm import tqdm

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, test_metric_ftns, optimizer, ftr, ftr_mask, config, device,
                 data_loader, mesh_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, test_metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.mesh_loader = mesh_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.save_period = self.config['trainer']['save_period']
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.ftr = ftr
        self.ftr_mask = ftr_mask

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.model.mode = 'train'

        for batch_idx, (images, masks, depths, obj_ids, bboxes, RTs) in enumerate(self.data_loader):
            images, masks, bboxes, RTs = images.to(self.device), masks.to(self.device), bboxes.to(self.device), RTs.to(self.device)
            ftrs = torch.cat([self.ftr[obj_id] for obj_id in obj_ids.tolist()], 0)
            ftr_masks = torch.cat([self.ftr_mask[obj_id] for obj_id in obj_ids.tolist()], 0)

            self.optimizer.zero_grad()
            output, P = self.model(images, ftrs, ftr_masks, bboxes, obj_ids, RTs)
            P['vertexes'] = torch.stack([self.mesh_loader.PTS_DICT[obj_id.tolist()] for obj_id in obj_ids])
            if self.criterion.__name__ == 'point_matching_loss':
                P['full_vertexes'] = torch.stack([self.mesh_loader.FULL_PTS_DICT[obj_id.tolist()] for obj_id in obj_ids])
            loss = 0
            for idx in list(output.keys())[1:]:
                loss += self.criterion(RTs, output[idx], **P)
            loss.backward()

            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.detach().item(), write=True)

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
            for batch_idx, (images, masks, depths, obj_ids, bboxes, RTs) in enumerate(tqdm(self.valid_data_loader, disable=self.config['gpu_scheduler'])):
                images, masks, bboxes, RTs = images.to(self.device), masks.to(self.device), bboxes.to(self.device), RTs.to(self.device)
                ftrs = torch.cat([self.ftr[obj_id] for obj_id in obj_ids.tolist()], 0)
                ftr_masks = torch.cat([self.ftr_mask[obj_id] for obj_id in obj_ids.tolist()], 0)
                
                output, P = self.model(images, ftrs, ftr_masks, bboxes, obj_ids, RTs)
                P['vertexes'] = torch.stack([self.mesh_loader.PTS_DICT[obj_id.tolist()] for obj_id in obj_ids])
                if self.criterion.__name__ == 'point_matching_loss':
                    P['full_vertexes'] = torch.stack([self.mesh_loader.FULL_PTS_DICT[obj_id.tolist()] for obj_id in obj_ids])

                loss = 0
                for idx in list(output.keys())[1:]:
                    loss += self.criterion(RTs, output[idx], **P)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.detach().item(), write=False)
                M = {
                    'out_RT' : output[list(output.keys())[-1]]['RT'],
                    'gt_RT' : RTs,
                    'ids' : obj_ids,
                    'points' : P['vertexes'],
                    'depth_maps' : depths
                }
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(**M), write=False)

            c, g = visualize(RTs, output, P)
            self.writer.add_image(f'contour', torch.tensor(c).permute(2, 0, 1))
            self.writer.add_image(f'rendering', torch.tensor(g).permute(2, 0, 1))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        for met in self.metric_ftns:
            v = self.valid_metrics.avg(met.__name__)
            self.writer.add_scalar(met.__name__, v)

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    


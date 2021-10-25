import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, image_mean_std_check, proj_visualize


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, mesh_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
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
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.mean, self.std = image_mean_std_check(self.data_loader)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (images, masks, obj_ids, bboxes, RTs) in enumerate(self.data_loader):
            images, masks, bboxes, RTs = images.to(self.device), masks.to(self.device), bboxes.to(self.device), RTs.to(self.device)
            front, top, right = self.mesh_loader.batch_render(obj_ids)
            meshes = self.mesh_loader.batch_meshes(obj_ids)

            self.optimizer.zero_grad()
            M, prediction = self.model(images, front, top, right, bboxes, obj_ids, RTs)
            loss = 0
            # for i, dict in enumerate(M):
            #     idx = len(M) - (i + 1)
            #     loss += self.criterion(prediction[idx+1], prediction[idx], RTs, **M[idx])
            #     loss += self.criterion(RTs, **M[idx])
            loss += self.criterion(prediction[4], prediction[0], RTs, **M[0])
            # loss += self.criterion(RTs, **M[0])

            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.detach().item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(prediction, RTs, meshes, obj_ids))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.detach().item()))   

                pr_proj_pred = proj_visualize(prediction[0], M[0]['grid_crop'], M[0]['coeffi_crop'], M[0]['ftr'], M[0]['ftr_mask'])
                pr_proj_labe = proj_visualize(RTs, M[0]['grid_crop'], M[0]['coeffi_crop'], M[0]['ftr'], M[0]['ftr_mask'])

                self.writer.add_image('image', make_grid(images.detach().cpu(), nrow=2, normalize=True))
                self.writer.add_image('input', make_grid(M[0]['pr_proj'].detach().mean(1, True).cpu(), nrow=2, normalize=True))
                self.writer.add_image('roi_feature', make_grid(M[0]['roi_feature'].detach().mean(1, True).cpu(), nrow=2, normalize=True))
                self.writer.add_image('prediction', make_grid(pr_proj_pred.detach().mean(1, True).cpu(), nrow=2, normalize=True))
                self.writer.add_image('gt', make_grid(pr_proj_labe.detach().mean(1, True).cpu(), nrow=2, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
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
        self.valid_metrics.reset()
        with torch.no_grad():

            for batch_idx, (images, masks, obj_ids, bboxes, RTs) in enumerate(self.valid_data_loader):
                images, masks, bboxes, RTs = images.to(self.device), masks.to(self.device), bboxes.to(self.device), RTs.to(self.device)
                front, top, right = self.mesh_loader.batch_render(obj_ids)
                meshes = self.mesh_loader.batch_meshes(obj_ids)

                M, prediction = self.model(images, front, top, right, bboxes, obj_ids, RTs)
                loss = 0
                # for i, dict in enumerate(M):
                #     idx = len(M) - (i + 1)
                #     loss += self.criterion(prediction[idx+1], prediction[idx], RTs, **M[idx])
                #     loss += self.criterion(RTs, **M[idx])
                loss += self.criterion(prediction[4], prediction[0], RTs, **M[0])
                # loss += self.criterion(RTs, **M[0])

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.detach().item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(prediction, RTs, meshes, obj_ids))

                pr_proj_pred = proj_visualize(prediction[0], M[0]['grid_crop'], M[0]['coeffi_crop'], M[0]['ftr'], M[0]['ftr_mask'])
                pr_proj_labe = proj_visualize(RTs, M[0]['grid_crop'], M[0]['coeffi_crop'], M[0]['ftr'], M[0]['ftr_mask'])

                self.writer.add_image('image', make_grid(images.detach().cpu(), nrow=2, normalize=True))
                self.writer.add_image('input', make_grid(M[0]['pr_proj'].detach().mean(1, True).cpu(), nrow=2, normalize=True))
                self.writer.add_image('roi_feature', make_grid(M[0]['roi_feature'].detach().mean(1, True).cpu(), nrow=2, normalize=True))
                self.writer.add_image('prediction', make_grid(pr_proj_pred.detach().mean(1, True).cpu(), nrow=2, normalize=True))
                self.writer.add_image('gt', make_grid(pr_proj_labe.detach().mean(1, True).cpu(), nrow=2, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
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

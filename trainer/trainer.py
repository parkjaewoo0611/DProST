import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker, visualize, get_param
from tqdm import tqdm
import torch.distributed as dist

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config, model, optimizer, criterion, lr_scheduler, 
                 valid_error_ftns, valid_metric_ftns, test_metric_ftns, 
                 train_data_loader, synth_data_loader, valid_data_loader,
                 mesh_loader, ref_loader, 
                 save_period=100, is_toy=False,  
                 gpu_scheduler=False, device='cpu', rank=0, **kwargs):
        super().__init__(model, test_metric_ftns, optimizer, config, device, rank)
        
        self.criterion = criterion        
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

        self.valid_metrics = MetricTracker(error_ftns=valid_error_ftns, metric_ftns=valid_metric_ftns)

        self.mesh_loader = mesh_loader
        self.ref_loader = ref_loader

        self.train_data_loader = train_data_loader

        self.len_epoch = len(self.train_data_loader)

        self.synth_data_loader = inf_loop(synth_data_loader)
        self.valid_data_loader = valid_data_loader
        self.DATA_PARAM = get_param(self.train_data_loader.dataset.data_dir)

        self.save_period = save_period
        self.is_toy = is_toy
        self.gpu_scheduler = gpu_scheduler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.module.train()
        self.model.module.mode = 'train'
        self.writer.set_mode('train')

        self.train_data_loader.sampler.set_epoch(epoch)
        for batch_idx, (train_batch, synth_batch) in enumerate(zip(self.train_data_loader, self.synth_data_loader)):
            batch = {key : torch.cat([train_batch[key], synth_batch[key]], 0) for key in list(train_batch.keys())}
            images, bboxes, RTs, Ks = batch['images'].to(self.device), batch['bboxes'].to(self.device), batch['RTs'].to(self.device), batch['Ks'].to(self.device)
            obj_ids = batch['obj_ids']
            meshes = self.mesh_loader.batch_meshes(obj_ids.tolist())            # load mesh or not according to use_mesh
            refs, ref_masks = self.ref_loader.batch_refs(obj_ids.tolist())      # load ref or not according to use_mesh

            self.optimizer.zero_grad()
            output, P = self.model(images, refs, ref_masks, bboxes, Ks, RTs, meshes)

            loss = 0
            for idx in list(output.keys())[1:]:
                loss += self.criterion(RTs, output[idx], **P)
            loss.backward()
            self.optimizer.step()

            loss = loss.detach().item()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss)

            if batch_idx % self.log_step == 0 and self.is_rank_0:
                self.logger.info('({}) Train Epoch: {} {} Loss: {:.6f}  Best {}: {:.6f}'.format(
                    self.checkpoint_dir.name,
                    epoch,
                    self._progress(batch_idx),
                    loss,
                    self.mnt_metric,
                    self.mnt_best))

            if self.is_toy and batch_idx==5:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        if epoch % self.save_period == 0 and self.is_rank_0:
            c, f, _ = visualize(RTs, output, P)
            self.writer.add_image(f'contour', torch.tensor(c).permute(2, 0, 1))
            self.writer.add_image(f'rendering', torch.tensor(f).permute(2, 0, 1))


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.module.eval()
        self.model.module.mode = 'valid'
        self.writer.set_mode('valid')
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.valid_data_loader, disable=self.gpu_scheduler)):
                images, bboxes, RTs, Ks = batch['images'].to(self.device), batch['bboxes'].to(self.device), batch['RTs'].to(self.device), batch['Ks'].to(self.device)
                obj_ids, K_origins = batch['obj_ids'], batch['K_origins']
                
                meshes = self.mesh_loader.batch_meshes(obj_ids.tolist())               # load mesh or not according to use_mesh
                refs, ref_masks = self.ref_loader.batch_refs(obj_ids.tolist())         # load ref or not according to use_mesh

                output, P = self.model(images, refs, ref_masks, bboxes, Ks, RTs, meshes)

                P['vertexes'] = [self.mesh_loader.PTS_DICT[obj_id.tolist()] for obj_id in obj_ids]
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)     # steps per second & to match steps in training
 
                M = {
                    'out_RT' : output[list(output.keys())[-1]]['RT'],
                    'gt_RT' : RTs,
                    'K' : K_origins,
                    'ids' : obj_ids,
                    'points' : P['vertexes'],
                    'DATA_PARAM' : self.DATA_PARAM
                }

                for err in self.valid_metrics._error_ftns:
                    self.valid_metrics.update(err.__name__, err(**M))
                self.valid_metrics.update('diameter', [self.DATA_PARAM['idx2diameter'][id] for id in obj_ids.tolist()])
                self.valid_metrics.update('id', obj_ids.tolist())

                if self.is_toy and batch_idx==5:
                    break

            c, f, _ = visualize(RTs, output, P)
            self.writer.add_image(f'contour', torch.tensor(c).permute(2, 0, 1))
            self.writer.add_image(f'rendering', torch.tensor(f).permute(2, 0, 1))

        result = self.valid_metrics.result()
        return result

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    


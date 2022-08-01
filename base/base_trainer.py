import torch
import torch.distributed as dist
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import os
from utils.util import hparams_key
class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, test_metric_ftns, optimizer, config, device, rank):
        self.device = device
        self.is_rank_0 = (rank == 0)
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = torch.tensor(0)
        else:
            self.mnt_mode = self.monitor
            self.mnt_metric = config['valid_metrics'][0]
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = torch.tensor(inf) if self.mnt_mode == 'min' else torch.tensor(-inf)
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.best_dir = config.save_dir
        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        self.hparams = hparams_key(config.config)
        self.hparams_result = {met.__name__: 0 for met in test_metric_ftns}
        self.hparams_result['current_epoch'] = self.start_epoch
        self.hparams_result['best_epoch'] = self.start_epoch
        self.hparams_result[self.mnt_metric] = self.mnt_best.item()
        self.writer.add_hparams(self.hparams, self.hparams_result)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)

            if self.is_rank_0:
                log = {'epoch': epoch}
                if epoch % self.save_period == 0 and self.mnt_mode != 'off':
                    # evaluate model performance according to configured metric, save best checkpoint as model_best
                    best = False
                    val_log = self._valid_epoch(epoch)
                    log.update(val_log)

                    self.hparams_result['current_epoch'] = epoch
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and val_log[self.mnt_metric] <= self.mnt_best.item()) or \
                                (self.mnt_mode == 'max' and val_log[self.mnt_metric] >= self.mnt_best.item())
                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.hparams_result[self.mnt_metric] = val_log[self.mnt_metric]
                        self.hparams_result['best_epoch'] = epoch
                        self.mnt_best = torch.tensor(val_log[self.mnt_metric])
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                        "Training stops.".format(self.early_stop))
                        break

                    for k, v in val_log.items():
                        self.writer.add_scalar(k, v)
                    self.writer.add_hparams(self.hparams, self.hparams_result)
                    self._save_checkpoint(epoch, save_best=best)

                # print logged informations to the screen
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))


    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model.module).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best.item(),
            'config': self.config
        }
        # filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        # torch.save(state, filename)
        # self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            self.best_dir = self.checkpoint_dir
            best_path = str(self.best_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.best_dir = os.path.dirname(resume_path)
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = torch.tensor(checkpoint['monitor_best'])

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.module.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


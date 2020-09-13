import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import show_message, load_latest_checkpoint, plot_tensor_to_numpy


class Logger(SummaryWriter):
    def __init__(self, config):
        self.logdir = config.training_config.logdir
        self.continue_training = config.training_config.continue_training
        self.sample_rate = config.data_config.sample_rate
        
        if not self.continue_training and os.path.exists(self.logdir):
            raise RuntimeError(
                f"You're trying to run training from scratch, "
                f"but logdir `{self.logdir} already exists. Remove it or specify new one.`"
            )
        if not self.continue_training:
            os.makedirs(self.logdir)
        super(Logger, self).__init__(self.logdir)
        with open(f'{self.logdir}/config.json', 'w') as f:
            json.dump(config.to_dict_type(), f)
        
    def _log_losses(self, iteration, loss_stats: dict):
        for key, value in loss_stats.items():
            self.add_scalar(key, value, iteration)

    def log_training(self, iteration, stats, verbose=True):
        stats = {f'training/{key}': value for key, value in stats.items()}
        self._log_losses(iteration, loss_stats=stats)
        show_message(
            f'Iteration: {iteration} | Losses: {[value for value in stats.values()]}',
            verbose=verbose
        )

    def log_test(self, epoch, stats, verbose=True):
        stats = {f'test/{key}': value for key, value in stats.items()}
        self._log_losses(epoch, loss_stats=stats)
        show_message(
            f'Epoch: {epoch} | Losses: {[value for value in stats.values()]}',
            verbose=verbose
        )
    
    def log_audios(self, iteration, audios: dict):
        for key, audio in audios.items():
            self.add_audio(key, audio, iteration, sample_rate=self.sample_rate)

    def log_specs(self, iteration, specs: dict):
        for key, image in specs.items():
            self.add_image(key, plot_tensor_to_numpy(image), iteration, dataformats='HWC')

    def save_model_config(self, config):
        with open(f'{self.log_dir}/config.json', 'w') as f:
            json.dump(config, f)
    
    def save_checkpoint(self, iteration, model, optimizer=None):
        d = {}
        d['iteration'] = iteration
        d['model'] = model.state_dict()
        if not isinstance(optimizer, type(None)):
            d['optimizer'] = optimizer.state_dict()
        filename = f'{self.log_dir}/checkpoint_{iteration}.pt'
        torch.save(d, filename)

    def load_latest_checkpoint(self, model, optimizer=None):
        if not self.continue_training:
            raise RuntimeError(
                f"Trying to load the latest checkpoint from logdir {self.logdir}, "
                "but did not set `continue_training=true` in configuration."
            )
        model, optimizer, iteration = load_latest_checkpoint(self.logdir, model, optimizer)
        return model, optimizer, iteration

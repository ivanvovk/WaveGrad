import argparse
import json

import torch
from torch.utils.data import DataLoader

from datetime import datetime

from logger import Logger
from model import WaveGrad
from data import AudioDataset, MelSpectrogramFixed
from benchmark import compute_rtf
from utils import ConfigWrapper, show_message


def run(config, args):
    show_message('Initializing logger...', verbose=args.verbose)
    logger = Logger(config)
    
    show_message('Initializing model...', verbose=args.verbose)
    model = WaveGrad(config).cuda()
    show_message(f'Number of parameters: {model.nparams}', verbose=args.verbose)
    mel_fn = MelSpectrogramFixed(
        sample_rate=config.data_config.sample_rate,
        n_fft=config.data_config.n_fft,
        win_length=config.data_config.win_length,
        hop_length=config.data_config.hop_length,
        f_min=config.data_config.f_min,
        f_max=config.data_config.f_max,
        n_mels=config.data_config.n_mels,
        window_fn=torch.hann_window
    ).cuda()

    show_message('Initializing optimizer, scheduler and losses...', verbose=args.verbose)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training_config.scheduler_step_size,
        gamma=config.training_config.scheduler_gamma
    )

    show_message('Initializing data loaders...', verbose=args.verbose)
    train_dataset = AudioDataset(config, training=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.training_config.batch_size, drop_last=True
    )
    test_dataset = AudioDataset(config, training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    test_batch = test_dataset.sample_test_batch(
        config.training_config.n_samples_to_test
    )

    if config.training_config.continue_training:
        show_message('Loading latest checkpoint to continue training...', verbose=args.verbose)
        model, optimizer, iteration = logger.load_latest_checkpoint(model, optimizer)
        epoch_size = len(train_dataset) // config.training_config.batch_size
        epoch_start = iteration // epoch_size
    else:
        iteration = 0
        epoch_start = 0

    # Log ground truth test batch
    audios = {
        f'audio_{index}/gt': audio
        for index, audio in enumerate(test_batch)
    }
    logger.log_audios(0, audios)
    specs = {
        f'mel_{index}/gt': mel_fn(audio.cuda()).cpu().squeeze()
        for index, audio in enumerate(test_batch)
    }
    logger.log_specs(0, specs)

    show_message('Start training...', verbose=args.verbose)
    try:
        for epoch in range(epoch_start, config.training_config.n_epoch):
            # Training step
            model.set_new_noise_schedule(
                n_iter=config.training_config.training_noise_schedule.n_iter,
                betas_range=config.training_config.training_noise_schedule.betas_range
            )
            for batch in train_dataloader:
                batch = batch.cuda()
                mels = mel_fn(batch)
                
                # Training step
                model.zero_grad()
                loss = model.compute_loss(mels, batch)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=config.training_config.grad_clip_threshold
                )
                optimizer.step()
                loss_stats = {
                    'total_loss': loss.item(),
                    'grad_norm': grad_norm.item()
                }
                logger.log_training(iteration, loss_stats, verbose=args.verbose)
                
                iteration += 1

            # Test step
            if epoch % config.training_config.test_interval == 0:
                model.set_new_noise_schedule(
                    n_iter=config.training_config.test_noise_schedule.n_iter,
                    betas_range=config.training_config.test_noise_schedule.betas_range
                )
                with torch.no_grad():
                    # Calculate test set loss
                    test_loss = 0
                    for i, batch in enumerate(test_dataloader):
                        batch = batch.cuda()
                        mels = mel_fn(batch)
                        test_loss_ = model.compute_loss(mels, batch)
                        test_loss += test_loss_
                    test_loss /= (i + 1)
                    loss_stats = {'total_loss': test_loss.item()}

                    # Restore random batch from test dataset
                    audios = {}
                    specs = {}
                    test_l1_loss = 0
                    test_l1_spec_loss = 0
                    average_rtf = 0

                    for index, test_sample in enumerate(test_batch):
                        test_sample = test_sample[None].cuda()
                        test_mel = mel_fn(test_sample.cuda())

                        start = datetime.now()
                        y_0_hat = model.forward(
                            test_mel, store_intermediate_states=False
                        )
                        y_0_hat_mel = mel_fn(y_0_hat)
                        end = datetime.now()
                        generation_time = (end - start).total_seconds()
                        average_rtf += compute_rtf(
                            y_0_hat, generation_time, config.data_config.sample_rate
                        )

                        test_l1_loss += torch.nn.L1Loss()(y_0_hat, test_sample).item()
                        test_l1_spec_loss += torch.nn.L1Loss()(y_0_hat_mel, test_mel).item()

                        audios[f'audio_{index}/predicted'] = y_0_hat.cpu().squeeze()
                        specs[f'mel_{index}/predicted'] = y_0_hat_mel.cpu().squeeze()

                    average_rtf /= len(test_batch)
                    show_message(f'Device: GPU. average_rtf={average_rtf}', verbose=args.verbose)

                    test_l1_loss /= len(test_batch)
                    loss_stats['l1_test_batch_loss'] = test_l1_loss
                    test_l1_spec_loss /= len(test_batch)
                    loss_stats['l1_spec_test_batch_loss'] = test_l1_spec_loss

                    logger.log_test(epoch, loss_stats, verbose=args.verbose)
                    logger.log_audios(epoch, audios)
                    logger.log_specs(epoch, specs)

                logger.save_checkpoint(iteration, model, optimizer)
            if epoch % (epoch//10 + 1) == 0:
                scheduler.step()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: training has been stopped.')
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str)
    parser.add_argument('-v', '--verbose', required=False, default=True, type=bool)
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))
    
    run(config, args)

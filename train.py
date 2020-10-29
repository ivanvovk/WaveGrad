import os
import argparse
import json
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from datetime import datetime
from tqdm import tqdm

from logger import Logger
from model import WaveGrad
from data import AudioDataset, MelSpectrogramFixed
from benchmark import compute_rtf
from utils import ConfigWrapper, show_message, str2bool


def run_training(rank, config, args):
    if args.n_gpus > 1:
        init_distributed(rank, args.n_gpus, config.dist_config)
        torch.cuda.set_device(f'cuda:{rank}')

    show_message('Initializing logger...', verbose=args.verbose, rank=rank)
    logger = Logger(config, rank=rank)
    
    show_message('Initializing model...', verbose=args.verbose, rank=rank)
    model = WaveGrad(config).cuda()
    show_message(f'Number of WaveGrad parameters: {model.nparams}', verbose=args.verbose, rank=rank)
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

    show_message('Initializing optimizer, scheduler and losses...', verbose=args.verbose, rank=rank)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training_config.scheduler_step_size,
        gamma=config.training_config.scheduler_gamma
    )
    if config.training_config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    show_message('Initializing data loaders...', verbose=args.verbose, rank=rank)
    train_dataset = AudioDataset(config, training=True)
    train_sampler = DistributedSampler(train_dataset) if args.n_gpus > 1 else None
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.training_config.batch_size,
        sampler=train_sampler, drop_last=True
    )
    test_dataset = AudioDataset(config, training=False)
    test_sampler = DistributedSampler(test_dataset) if args.n_gpus > 1 else None
    test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler)
    test_batch = test_dataset.sample_test_batch(
        config.training_config.n_samples_to_test
    )

    if config.training_config.continue_training:
        show_message('Loading latest checkpoint to continue training...', verbose=args.verbose, rank=rank)
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

    if args.n_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        show_message(f'INITIALIZATION IS DONE ON RANK {rank}.')

    show_message('Start training...', verbose=args.verbose, rank=rank)
    try:
        for epoch in range(epoch_start, config.training_config.n_epoch):
            # Training step
            model.train()
            (model if args.n_gpus == 1 else model.module).set_new_noise_schedule(
                init=torch.linspace,
                init_kwargs={
                    'steps': config.training_config.training_noise_schedule.n_iter,
                    'start': config.training_config.training_noise_schedule.betas_range[0],
                    'end': config.training_config.training_noise_schedule.betas_range[1]
                }
            )
            for batch in (
                tqdm(train_dataloader, leave=False) \
                if args.verbose and rank == 0 else train_dataloader
            ):
                model.zero_grad()

                batch = batch.cuda()
                mels = mel_fn(batch)
                
                if config.training_config.use_fp16:
                    with torch.cuda.amp.autocast():
                        loss = (model if args.n_gpus == 1 else model.module).compute_loss(mels, batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss = (model if args.n_gpus == 1 else model.module).compute_loss(mels, batch)
                    loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=config.training_config.grad_clip_threshold
                )

                if config.training_config.use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                loss_stats = {
                    'total_loss': loss.item(),
                    'grad_norm': grad_norm.item()
                }
                logger.log_training(iteration, loss_stats, verbose=False)
                
                iteration += 1

            # Test step after epoch on rank==0 GPU
            if epoch % config.training_config.test_interval == 0 and rank == 0:
                model.eval()
                (model if args.n_gpus == 1 else model.module).set_new_noise_schedule(
                    init=torch.linspace,
                    init_kwargs={
                        'steps': config.training_config.test_noise_schedule.n_iter,
                        'start': config.training_config.test_noise_schedule.betas_range[0],
                        'end': config.training_config.test_noise_schedule.betas_range[1]
                    }
                )
                with torch.no_grad():
                    # Calculate test set loss
                    test_loss = 0
                    for i, batch in enumerate(
                        tqdm(test_dataloader) \
                        if args.verbose and rank == 0 else test_dataloader
                    ):
                        batch = batch.cuda()
                        mels = mel_fn(batch)
                        test_loss_ = (model if args.n_gpus == 1 else model.module).compute_loss(mels, batch)
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
                        y_0_hat = (model if args.n_gpus == 1 else model.module).forward(
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

                logger.save_checkpoint(
                    iteration,
                    model if args.n_gpus == 1 else model.module,
                    optimizer
                )
            if epoch % (epoch//10 + 1) == 0:
                scheduler.step()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: training has been stopped.')
        cleanup()
        return


def run_distributed(fn, config, args):
    try:
        mp.spawn(fn, args=(config, args), nprocs=args.n_gpus, join=True)
    except:
        cleanup()


def init_distributed(rank, n_gpus, dist_config):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    torch.cuda.set_device(rank % n_gpus)

    os.environ['MASTER_ADDR'] = dist_config.MASTER_ADDR
    os.environ['MASTER_PORT'] = dist_config.MASTER_PORT
    
    torch.distributed.init_process_group(
        backend='nccl', world_size=n_gpus, rank=rank
    )


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='configuration file')
    parser.add_argument(
        '-v', '--verbose', required=False, type=str2bool,
        nargs='?', const=True, default=True, help='verbosity level'
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    n_gpus = torch.cuda.device_count()
    args.__setattr__('n_gpus', n_gpus)

    if args.n_gpus > 1:
        run_distributed(run_training, config, args)
    else:
        run_training(0, config, args)

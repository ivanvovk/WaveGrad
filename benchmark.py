import os
import itertools
import numpy as np

import torch

from datetime import datetime
from tqdm import tqdm
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

from data import AudioDataset, MelSpectrogramFixed
from utils import show_message


def compute_rtf(sample, generation_time, sample_rate=22050):
    """
    Computes RTF for a given sample.
    """
    total_length = sample.shape[-1]
    return float(generation_time * sample_rate / total_length)


def estimate_average_rtf_on_filelist(filelist_path, config, model, verbose=True):
    """
    Runs RTF estimation of filelist of audios and computes statistics.
    :param filelist_path (str): path to a filelist with needed audios
    :param config (utils.ConfigWrapper): configuration dict
    :param model (torch.nn.Module): WaveGrad model
    :param verbose (bool, optional): verbosity level
    :return stats: statistics dict
    """
    device = next(model.parameters()).device
    config.training_config.test_filelist_path = filelist_path
    dataset = AudioDataset(config, training=False)
    mel_fn = MelSpectrogramFixed(
        sample_rate=config.data_config.sample_rate,
        n_fft=config.data_config.n_fft,
        win_length=config.data_config.win_length,
        hop_length=config.data_config.hop_length,
        f_min=config.data_config.f_min,
        f_max=config.data_config.f_max,
        n_mels=config.data_config.n_mels,
        window_fn=torch.hann_window
    ).to(device)
    rtfs = []
    for i in (tqdm(range(len(dataset))) if verbose else range(len(dataset))):
        datapoint = dataset[i].to(device)
        mel = mel_fn(datapoint)[None]
        start = datetime.now()
        sample = model.forward(mel, store_intermediate_states=False)
        end = datetime.now()
        generation_time = (end - start).total_seconds()
        rtf = compute_rtf(
            sample, generation_time, sample_rate=config.data_config.sample_rate
        )
        rtfs.append(rtf)
    average_rtf = np.mean(rtfs)
    std_rtf = np.std(rtfs)

    show_message(f'DEVICE: {device}. average_rtf={average_rtf}, std={std_rtf}', verbose=verbose)
    
    rtf_stats = {
        'rtfs': rtfs,
        'average': average_rtf,
        'std': std_rtf
    }
    return rtf_stats


def _betas_estimate(betas, model, mels, mel_fn):
    n_iter = len(betas)
    init_fn = lambda **kwargs: torch.FloatTensor(betas)
    model.set_new_noise_schedule(init=init_fn, init_kwargs={'steps': n_iter})

    outputs = model.forward(mels, store_intermediate_states=False)
    test_pred_mels = mel_fn(outputs)

    loss = torch.nn.L1Loss()(test_pred_mels, mels).item()
    return loss


def generate_betas_grid(n_iter, betas_range, verbose):
    betas_range = torch.FloatTensor(betas_range).log10()
    exp_step = (betas_range[1] - betas_range[0]) / (n_iter - 1)
    exponents = 10**torch.arange(betas_range[0], betas_range[1] + exp_step, step=exp_step)

    grid = []
    state = int(''.join(['1'] * n_iter))  # initial state
    final_state = 9**n_iter
    max_grid_size = 9**5
    step = int(np.ceil(final_state / (max_grid_size)))
    for _ in range(max_grid_size):
        multipliers = list(map(int, str(state)))
        if 0 in multipliers:
            state += step
            continue
        betas = [mult * exp for mult, exp in zip(multipliers, exponents)]
        grid.append(betas)
        state += step
    return grid


def iters_schedule_grid_search(model, config,
                               n_iter=6,
                               betas_range=(1e-6, 1e-2),
                               test_batch_size=2,
                               step=1,
                               path_to_store_schedule=None,
                               save_stats_for_grid=True,
                               verbose=True,
                               n_jobs=1):
    """
    Performs grid search for 6 iterations schedule. Run it only on GPU and only for a small number of iterations!
    :param model (torch.nn.Module): WaveGrad model
    :param config (ConfigWrapper): model configuration
    :param n_iter (int, optional): number of iterations to search for
    :param test_batch_size (int, optional): number of one second samples to be tested grid sets on
    :param path_to_store_schedule (str, optional): path to store stats. If not specified, then it will no be saved and would be just returned.
    :param save_stats_for_grid (str, optional): flag to save stats for whole grid or not
    :param verbose (bool, optional): output all the process
    :param n_jobs(int, optional): number of parallel threads to use
    :return betas (list): list of betas, which gives the lowest log10-mel-spectrogram absolute error
    :return stats (dict): dict of type {betas: loss} for the whole grid
    """
    device = next(model.parameters()).device
    if 'cpu' in str(device):
        show_message('WARNING: running grid search on CPU will be slow.')
    
    show_message('Initializing betas grid...', verbose=verbose)
    grid = generate_betas_grid(n_iter, betas_range, verbose=verbose)[::step]
    
    show_message('Initializing utils...', verbose=verbose)
    mel_fn = MelSpectrogramFixed(
        sample_rate=config.data_config.sample_rate,
        n_fft=config.data_config.n_fft,
        win_length=config.data_config.win_length,
        hop_length=config.data_config.hop_length,
        f_min=config.data_config.f_min,
        f_max=config.data_config.f_max,
        n_mels=config.data_config.n_mels,
        window_fn=torch.hann_window
    ).to(device)
    dataset = AudioDataset(config, training=True)
    idx = np.random.choice(range(len(dataset)), size=test_batch_size, replace=False)
    test_batch = torch.stack([dataset[i] for i in idx]).to(device)
    test_mels = mel_fn(test_batch)

    show_message('Starting search...', verbose=verbose)
    with ThreadPool(processes=n_jobs) as pool:
        process_fn = partial(_betas_estimate, model=model, mels=test_mels, mel_fn=mel_fn)
        stats = list(tqdm(pool.imap(process_fn, grid), total=len(grid)))
    stats = {i : (grid[i], stats[i]) for i in range(len(stats))}

    if save_stats_for_grid:
        tmp_stats_path = f'{os.path.dirname(path_to_store_schedule)}/{n_iter}stats.pt'
        show_message(f'Saving tmp stats for whole grid to `{tmp_stats_path}`...', verbose=verbose)
        torch.save(stats, tmp_stats_path)
    
    best_idx = np.argmin(list([value for _, value in stats.values()]))
    best_betas = grid[best_idx]
    
    if not isinstance(path_to_store_schedule, type(None)):
        show_message(f'Saving best schedule to `{path_to_store_schedule}`...', verbose=verbose)
        torch.save(best_betas, path_to_store_schedule)
    
    return best_betas, stats


def fibonacci(b1=1e-6, b2=9e-6, n_iter=25):
    betas = [b1, b2]
    for _ in range(n_iter - 2):
        betas.append(sum(betas[-2:]))
    return betas[:n_iter]

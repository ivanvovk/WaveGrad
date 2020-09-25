import itertools
import numpy as np

import torch

from datetime import datetime
from tqdm import tqdm

from data import AudioDataset, MelSpectrogramFixed
from utils import show_message


def compute_rtf(sample, generation_time, sample_rate=22050):
    total_length = sample.shape[-1]
    return float(generation_time * sample_rate / total_length)


def estimate_average_rtf_on_filelist(filelist_path, config, model, verbose=True):
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


def iters_schedule_grid_search(model, config, n_iter=6, step=1, test_batch_size=2, path_to_store_stats=None, verbose=True):
    """
    Performs grid search for 6 iterations schedule. Run it only on GPU!
    :param model (torch.nn.Module): WaveGrad model
    :param config (ConfigWrapper): model configuration
    :param step (int, optional): the step to make when going through grid.
        `step=1` means that search will compute losses for the whole grid, else grid will be reduced by [::step].
        It is recommended to use `step=1`.
    :param test_batch_size (int, optional): number of one second samples to be tested grid sets on
    :path_to_store_stats (str, optional): path to store stats. If not specified, then it will no be saved.
    :param verbose (bool, optional): output all the process
    :return best_betas (list): list of betas, which gives the lowest log10-mel-spectrogram absolute error
    :return stats (dict): dict of type {betas: loss} for the whole grid
    """
    device = next(model.parameters()).device
    if device == 'cpu':
        show_message('WARNING: running grid search on CPU will be slow.')

    show_message('Initializing betas grid...', verbose=verbose)
    nums = np.linspace(1, 9, num=9).astype(float)
    scale = n_iter // 5
    residual = n_iter % 5
    exps_types = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    exps = []
    for exps_type in exps_types:
        exps += [exps_type] * scale
    exps += [exps_types[-1]] * residual
    exps = [exps,]
    assert len(exps[0]) == n_iter
    grid = []
    for exp in exps:
        grid_ = [list(zip(x,exp)) for x in itertools.permutations(nums,len(exp))]
        grid.append([[item[0] * item[1] for item in betas] for betas in grid_])
    grid = np.concatenate(grid)[::step]
    show_message(f'Grid size: {grid.shape[0]}', verbose=verbose)

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
    idx = np.random.choice(range(len(dataset)), size=2, replace=False)
    test_batch = torch.stack([dataset[i] for i in idx]).to(device)
    test_mels = mel_fn(test_batch)

    show_message('Starting search...', verbose=verbose)
    stats = {}
    for i, betas in enumerate(tqdm(grid, leave=False) if verbose else grid):
        init_fn = lambda **kwargs: torch.FloatTensor(betas)
        model.set_new_noise_schedule(init=init_fn, init_kwargs={'steps': n_iter})
        outputs = model.forward(test_mels, store_intermediate_states=False)
        test_pred_mels = mel_fn(outputs)
        stats[i] = torch.nn.L1Loss()(test_pred_mels, test_mels)

    if path_to_store_stats:
        show_message(f'Saving stats to {path_to_store_stats}...')
        torch.save(stats, path_to_store_stats)

    best_idx = np.argmin(list(stats.values()))
    best_betas = grid[best_idx]
    show_message(f'Best betas on {n_iter} iterations: {best_betas}')
    
    return best_betas, stats


def init_from_iters6_schedule(iters6_betas, n_iter):
    expanded = []
    if n_iter < 6:
        return iters6_betas[:n_iter]
    scale = n_iter // 6
    count = n_iter - 6
    residual = n_iter % 6
    for _ in range(residual):
        expanded.append(iters6_betas[-1])
    for beta in iters6_betas:
        if count == 0:
            expanded.append(beta)
        else:
            expanded += [beta] * scale
            count -= 1
    return expanded

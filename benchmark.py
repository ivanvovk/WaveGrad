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

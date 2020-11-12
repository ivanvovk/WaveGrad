import os
import json
import argparse
import numpy as np

import torch
import torchaudio

from tqdm import tqdm
from datetime import datetime

from model import WaveGrad
from benchmark import compute_rtf
from utils import ConfigWrapper, show_message, str2bool, parse_filelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', required=True,
        type=str, help='configuration file path'
    )
    parser.add_argument(
        '-ch', '--checkpoint_path',
        required=True, type=str, help='checkpoint path'
    )
    parser.add_argument(
        '-ns', '--noise_schedule_path', required=True, type=str,
        help='noise schedule, should be just a torch.Tensor array of shape [n_iter]'
    )
    parser.add_argument(
        '-m', '--mel_filelist', required=True, type=str,
        help='mel spectorgram filelist, files of which should be just a torch.Tensor array of shape [n_mels, T]'
    )
    parser.add_argument(
        '-v', '--verbose', required=False, type=str2bool,
        nargs='?', const=True, default=True, help='verbosity level'
    )
    args = parser.parse_args()

    # Initialize config
    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))

    # Initialize the model
    model = WaveGrad(config)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'], strict=False)

    # Set noise schedule
    noise_schedule = torch.load(args.noise_schedule_path)
    n_iter = noise_schedule.shape[-1]
    init_fn = lambda **kwargs: noise_schedule
    init_kwargs = {'steps': n_iter}
    model.set_new_noise_schedule(init_fn, init_kwargs)

    # Trying to run inference on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Inference
    filelist = parse_filelist(args.mel_filelist)
    rtfs = []
    for mel_path in (tqdm(filelist, leave=False) if args.verbose else filelist):
        with torch.no_grad():
            mel = torch.load(mel_path).unsqueeze(0).to(device)

            start = datetime.now()
            outputs = model.forward(mel, store_intermediate_states=False)
            end = datetime.now()

            outputs = outputs.cpu().squeeze()
            baseidx = os.path.basename(os.path.abspath(mel_path)).split('_')[-1].replace('.pt', '')
            save_path = f'{os.path.dirname(os.path.abspath(mel_path))}/predicted_{baseidx}.wav'
            torchaudio.save(
                save_path, outputs, sample_rate=config.data_config.sample_rate
            )

            inference_time = (end - start).total_seconds()
            rtf = compute_rtf(outputs, inference_time, sample_rate=config.data_config.sample_rate)
            rtfs.append(rtf)

    show_message(f'Done. RTF estimate: {np.mean(rtfs)} ± {np.std(rtfs)}', verbose=args.verbose)

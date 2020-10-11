import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch


def show_message(text, verbose=True, end='\n'):
    if verbose: print(text, end=end)


def parse_filelist(filelist_path):
    with open(filelist_path, 'r') as f:
        filelist = [line.strip() for line in f.readlines()]
    return filelist


def latest_checkpoint_path(dir_path, regex="checkpoint_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_latest_checkpoint(logdir, model, optimizer=None):
    latest_model_path = latest_checkpoint_path(logdir, regex="checkpoint_*.pt")
    print(f'Latest checkpoint: {latest_model_path}')
    d = torch.load(
        latest_model_path,
        map_location=lambda loc, storage: loc
    )
    iteration = d['iteration']
    valid_incompatible_unexp_keys = [
        'betas',
        'alphas',
        'alphas_cumprod',
        'alphas_cumprod_prev',
        'sqrt_alphas_cumprod',
        'sqrt_recip_alphas_cumprod',
        'sqrt_recipm1_alphas_cumprod',
        'posterior_log_variance_clipped',
        'posterior_mean_coef1',
        'posterior_mean_coef2'
    ]
    d['model'] = {
        key: value for key, value in d['model'].items() \
            if key not in valid_incompatible_unexp_keys
    }
    model.load_state_dict(d['model'], strict=False)
    if not isinstance(optimizer, type(None)):
        optimizer.load_state_dict(d['optimizer'])
    return model, optimizer, iteration


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor_to_numpy(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none', cmap='hot')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


class ConfigWrapper(object):
    """
    Wrapper dict class to avoid annoying key dict indexing like:
    `config.sample_rate` instead of `config["sample_rate"]`.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = ConfigWrapper(**v)
            self[k] = v
      
    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def to_dict_type(self):
        return {
            key: (value if not isinstance(value, ConfigWrapper) else value.to_dict_type())
            for key, value in dict(**self).items()
        }

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

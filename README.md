# WaveGrad
Implementation (PyTorch) of Google Brain's WaveGrad vocoder (paper: https://arxiv.org/pdf/2009.00713.pdf).

**Status** (contributing is welcome, share your ideas!):
* **Stable versions: 25-, 50- and 1000-iteration models with linear betas.**
* Conducting experiments on 6-iterations training.
* **Estimated real-time factor (RTF)** for 6-, 25-, 50- and 1000-iteration models (see the table below).
* Uploaded an example of generated audio by current 50-iteration model. Unfortunately, there are some clicking artefacts on the edges of parallel sampling (checkout the last section of README for more details). Using parallel small segment sampling was my quick solution for the issue of when the model sees the mel sequence much longer than a training segment. If you have ideas how to resolve it, please share them by making a new issue. Also there is some hiss background noise. Working on fixing this problems.
* Preparing pretrained checkpoints.

|          Model         |  Stable  |   RTF (NVIDIA 2080 Ti), 22Khz   |
|------------------------|----------|---------------------------------|
| 1000 iterations Base   |   True   |              Bad :)             |
|  50 iterations Base    |   True   |          1.09 ± 0.023           |
|  25 iterations Base    |   True   |          0.55 ± 0.011           |
|  6 iterations Base     |   None   |          0.13 ± 0.005           |

## About

WaveGrad is a conditional model for waveform generation through estimating gradients of the data density. The main concept of vocoder is its relateness to diffusion models based on Langevin dynamics and score matching frameworks. W.r.t. Langevin dynamics WaveGrad achieves super-fast convergence (6 iterations reported in the paper).

## Setup

1. Clone this repo:

```bash
git clone https://github.com/ivanvovk/WaveGrad.git
cd WaveGrad
```

2. Install requirements `pip install -r requirements.txt`

## Train your own model

1. Make filelists of your data like ones included into `filelists` folder.
2. Setup a configuration in `configs` folder.
3. Change config path in `train.sh` and run the script by `sh train.sh`.
4. To track training process run tensorboard by `tensorboard --logdir=logs/YOUR_LOG_FOLDER`.

## Inference and generated audios

Follow instructions provided in Jupyter notebook [`notebooks/inference.ipynb`](notebooks/inference.ipynb). Also there is the code to estimate RTF. Current generated audios are provided in [`generated_samples`](generated_samples/) folder.

## Details, issues and comments

* For Langevin dynamics computation [`Denoising Diffusion Probabilistic Models`](https://github.com/hojonathanho/diffusion) repository has been adopted.
* For **Base 25-, 50- and 1000-iteration LJSpeech WaveGrad** models training succesfully runs on single 12GB GPU. Batch size is modified compared to the paper (256 -> 48, authors trained their model on TPU).
* At some point training might start to behave very weird and crazy (loss explodes), so I introduced learning rate scheduling and gradient clipping.
* Choose betas very carefully. Prefer using standard configs, provided in repository.
* Overall, be careful and tune hyperparameters for your own dataset accurately.
* For the best reconstruction accuracy prefer to use `wavegrad.sample_subregions_parallel(...)` method instead of `wavegrad.sample(...)`. During training the model has seen only the small segments of speech, thus reconstruction accuracy degrades on the longer mel sequences with time (I have determined the reason: because of positional encoding - the model hasn't seen longer positional encoding during training but just the corresponding one for a small segment of speech). Parallel generation method splits given mel into segments of size from training and processes it concurrently. However, it results in some clicking artefacts on the edges of final signals concatenation.

## References

* [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf)
* [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

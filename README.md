# WaveGrad
Implementation (PyTorch) of Google Brain's WaveGrad vocoder (paper: https://arxiv.org/pdf/2009.00713.pdf).

**Status** (contributing is welcome):
* **Stable versions: 25 and 50 iterations** with linear betas.
* Conducting experiments on 6-iterations training.
* Also preparing real-time factor (RTF) benchmarking.
* Preparing pretrained checkpoints.
* Preparing generated audio.

|        Model       | Stable |     RTF     |
|:------------------:|:------:|-------------|
| 50 iterations Base |  True  | In progress |
| 25 iterations Base |  True  | In progress |
| 6 iterations Base  |  False | In progress |

## About

WaveGrad is a conditional model for waveform generation through estimating gradients of the data density. The main concept of vocoder is its relateness to diffusion models based on Langevin dynamics and score matching frameworks.

## Setup

1. Clone this repo

```bash
git clone https://github.com/ivanvovk/WaveGrad.git

cd WaveGrad
```

2. Install requirements `pip install -r requirements.txt`

## Train your own model

1. Make filelists of your data like ones included into `filelists` folder.
2. Setup configuration in `configs` folder
3. Change config path in `train.sh` and run the script `sh train.sh`

## Inference and generated audios

Follow instructions provided in Jupyter notebook [`notebooks/inference.ipynb`](notebooks/inference.ipynb). Generated audios will be provided shortly.

## Details, issues and comments

* For Langevin dynamics computation [`Denoising Diffusion Probabilistic Models`](https://github.com/hojonathanho/diffusion) repository has been adopted.
* **Base 25-and-50-iteration LJSpeech WaveGrad** models succesfully train on single 12GB GPU. Batch size is modified compared to the paper (256->48: authors trained their model on TPU).
* At some point training might start to behave very weird and crazy (loss explodes), so I introduced learning rate scheduling and gradient clipping.
* Choose betas very carefully. Prefer using standard configs, provided in repos.
* Overall, be careful and tune hyperparameters accurately for your own dataset.
* For the best reconstruction accuracy prefer to use `wavegrad.sample_subregions_parallel(...)` method instead of `wavegrad.sample(...)`. During training the model has seen only the small segments of speech, thus because of positional encoding reconstruction accuracy degrades with time. Parallel generation method splits given mel into segments and processes it in parallel. 

## References

* [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf)
* [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

![alt-text-1](generated_samples/denoising.gif "denoising")

# WaveGrad
Implementation (PyTorch) of Google Brain's WaveGrad vocoder (paper: https://arxiv.org/pdf/2009.00713.pdf).

## **Status** (STABLE)

* Model training is stable and supports multi-iteration inference (**6 iterations also work perfect and produce high-fidelity samples**).
* Model training runs on a single 12GB GPU machine.
* **Model produces high-fidelity 22KHz generated samples**. Uploaded samples for a different number of iterations.
* Estimated the real-time factor (RTF) for the model (see the table below). 100- and lower-iteration inference is faster than real-time on NVIDIA RTX 2080 Ti. **6-iteration model is faster than the model reported in the paper**.
* Updated the code with **new grid search utils for finding the best noise schedules**.
* Preparing pretrained checkpoints.

#### Real-time factor (RTF) for 22KHz audio generation and number of parameters

|       Model       |  Stable  | RTF (RTX 2080 Ti) |  RTF (Tesla K80) | CPU (Intel Xeon 2.3GHz) |
|-------------------|----------|-------------------|------------------|-------------------------|
| 1000 iterations   |   True   |    9.59 ± 0.357   |        -         |            -            |
|  100 iterations   |   True   |    0.94 ± 0.046   |   5.85 ± 0.114   |            -            |
|   50 iterations   |   True   |    0.45 ± 0.021   |   2.92 ± 0.055   |            -            |
|   25 iterations   |   True   |    0.22 ± 0.011   |   1.45 ± 0.025   |            -            |
|   12 iterations   |   True   |    0.10 ± 0.005   |   0.69 ± 0.009   |       4.55 ± 0.545      |
|    6 iterations   |   True   |    0.04 ± 0.005   |   0.33 ± 0.005   |       2.09 ± 0.262      |

**Number of parameters**: 15810401

## About

WaveGrad is a conditional model for waveform generation through estimating gradients of the data density. The main concept of vocoder is its relatedness to diffusion models based on Langevin dynamics and score matching frameworks. w.r.t. Langevin dynamics WaveGrad achieves super-fast convergence (6 iterations).

## Setup

1. Clone this repo:

```bash
git clone https://github.com/ivanvovk/WaveGrad.git
cd WaveGrad
```

2. Install requirements `pip install -r requirements.txt`

## Train your own model

1. Make filelists of your audio data like ones included into `filelists` folder.
2. Setup a configuration in `configs` folder.
3. Change config path in `train.sh` and run the script by `sh train.sh`.
4. To track training process run tensorboard by `tensorboard --logdir=logs/YOUR_LOG_FOLDER`.
5. Once model is trained, grid search the best schedule for a needed number of iterations in [`notebooks/inference.ipynb`](notebooks/inference.ipynb).

## Inference, generated audios and pretrained checkpoints

#### Inference, your runtime environment RTF and best schedule grid search

In order to make inference of your model follow the instructions provided in Jupyter Notebook [`notebooks/inference.ipynb`](notebooks/inference.ipynb). Also there is the code to estimate RTF in your runtime environment and to run best schedule grid search.

#### Generated audios

Current generated audios are provided in [`generated_samples`](generated_samples/) folder. Quality degradation between 1000-iteration and 6-iteration inferences is not noticeable if found the best schedule for the latter.

#### Pretrained checkpoints

In progress.

## Important details, issues and comments

* During training WaveGrad uses a default noise schedule with 1000 iterations and linear scale betas from range (1e-6, 0.01). For inference you can set another schedule with less iterations (6 iterations reported in paper are ok for this implementation!). Tune betas carefully, the output quality really highly depends on them.
* **The best practice is to run grid search** function `iters_grid_search(...)` from `benchmark.py` to find the best schedule for your number of iterations.
* **Model training succesfully runs on a single 12GB GPU machine**. Batch size is modified compared to the paper (256 -> 48, as authors trained their model on TPU). After ~10k iterations (1-2 hours) model performs good generation for 50-iteration inference. Total training time is about 1-2 days (for absolute convergence).
* At some point training might start to behave very weird and crazy (loss explodes), so I have introduced learning rate (LR) scheduling and gradient clipping.
* Hop length of your STFT is equal 300 by default (thus total upsampling factor). Other cases are not tested, but you can try. Change `hop_length` in your config and `factors` in `config.model_config` (their product should give your new hop length).
* It is crucial to have `LINEAR_SCALE=5000` (already set by default) flag in `model/linear_modulation.py`. It rescales positional embeddings to have absolute amplitude `1/LINEAR_SCALE`. It is improtant since continuous noise level and sinusoidal positional encodings both have equal range (-1, 1). No rescaling results that the model cannot properly extract noise info from positionally embedded continuous noise level and thus cannot extrapolate on longer mel-spectrogram sequences (longer than training segment: 7200 timepoints by default).

## History of updates

* (NEW) New RTF info for NVIDIA Tesla K80 GPU card (popular in Google Collab service) and CPU Intel Xeon 2.3GHz.
* Huge update. New 6-iteration well generated sample example. New noise schedule setting API. Added the best schedule grid search code.
* Improved training by introducing smarter learning rate scheduler. Obtained high-fidelity synthesis.
* Stable training and multi-iteration inference. 6-iteration noise scheduling is supported.
* Stable training and fixed-iteration inference with significant background static noise left. All positional encoding issues are solved.
* Stable training of 25-, 50- and 1000-fixed-iteration models. Found no linear scaling (C=5000 from paper) of positional encoding (bug).
* Stable training of 25-, 50- and 1000-fixed-iteration models. Fixed positional encoding downscaling. Parallel segment sampling is replaced by full-mel sampling.
* (RELEASE, first on GitHub). Parallel segment sampling and broken positional encoding downscaling. Bad quality with clicks from concatenation from parallel-segment generation.

## References

* Nanxin Chen et al., [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf)
* Jonathan Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
* [Denoising Diffusion Probabilistic Models repository](https://github.com/hojonathanho/diffusion), from which diffusion calculations have been adopted

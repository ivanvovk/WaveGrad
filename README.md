# WaveGrad
Implementation (PyTorch) of Google Brain's WaveGrad vocoder (paper: https://arxiv.org/pdf/2009.00713.pdf).

#### Status (in active development, stay tuned)

* Model training is stable and supports multi-iteration inference (6 iterations also work perfect).
* Model produces high-fidelity 22KHz generated samples.
* Estimated real-time factor (RTF) for the model (see the table below). 100- and lower-iteration inference is faster than real-time on NVIDIA RTX 2080 Ti. 6-iteration model is faster than the model reported in the paper.
* Uploaded generated samples by the current best model for different number of iterations.
* Preparing the best noise schedule search code for you.
* Preparing pretrained checkpoints.

#### Real-time factor (RTF)

|       Model       |  Stable  | RTF (NVIDIA RTX 2080 Ti), 22KHz |
|-------------------|----------|---------------------------------|
| 1000 iterations   |   True   |          9.59 ± 0.357           |
|  100 iterations   |   True   |          0.94 ± 0.046           |
|   50 iterations   |   True   |          0.45 ± 0.021           |
|   25 iterations   |   True   |          0.22 ± 0.011           |
|   12 iterations   |   True   |          0.10 ± 0.005           |
|    6 iterations   |   True   |          0.04 ± 0.005           |

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

1. Make filelists of your data like ones included into `filelists` folder.
2. Setup a configuration in `configs` folder.
3. Change config path in `train.sh` and run the script by `sh train.sh`.
4. To track training process run tensorboard by `tensorboard --logdir=logs/YOUR_LOG_FOLDER`.

## Inference, generated audios and pretrained checkpoints

#### Inference

In order to make inference of your model follow instructions provided in Jupyter Notebook [`notebooks/inference.ipynb`](notebooks/inference.ipynb). Also there is the code to estimate RTF in your runtime environment.

#### Generated audios

Current generated audios are provided in [`generated_samples`](generated_samples/) folder.

#### Pretrained checkpoints

In progress.

## Important details, issues and comments

* During training WaveGrad uses default noise scheduling with 1000 iterations and linear scale betas from range (1e-6, 0.01). For inference you can set another schedulling with less iterations (6 iterations reported in paper are ok for this implementation!). Tune betas carefully, the output quality really highly depends on them.
* **Model training succesfully runs on a single 12GB GPU machine**. Batch size is modified compared to the paper (256 -> 48, authors trained their model on TPU).
* Model converges to acceptable quality in 10-20 thousand iterations (~2 hours).
* At some point training might start to behave very weird and crazy (loss explodes), so I introduced learning rate scheduling and gradient clipping.
* **Hop length of your STFT should always be equal 300** (thus total upsampling factor). Other cases are not supported yet.

## History of updates

* Improved training by introducing smarter learning rate scheduller (**you are here**). Obtained high-fidelity synthesis.
* Stable training and multi-iteration inference. 6-iteration noise schedulling is supported.
* Stable training and fixed-iteration inference with significant background static noise left. All positional encoding issues are solved.
* Stable training of 25-, 50- and 1000-fixed-iteration models. Found no linear scaling (C=5000 from paper) of positional encoding (bug).
* Stable training of 25-, 50- and 1000-fixed-iteration models. Fixed positional encoding downscaling. Parallel segment sampling is replaced by full-mel sampling.
* RELEASE (first on GitHub). Parallel segment sampling and broken positional encoding downscaling. Bad quality with clicks from concatenation from parallel-segment generation.

## References

* [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf)
* [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
* [Denoising Diffusion Probabilistic Models repository](https://github.com/hojonathanho/diffusion), from which diffusion calculations have been adopted

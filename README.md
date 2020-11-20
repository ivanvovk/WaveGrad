![alt-text-1](generated_samples/denoising.gif "denoising")

# WaveGrad
Implementation (PyTorch) of Google Brain's high-fidelity WaveGrad vocoder ([paper](https://arxiv.org/pdf/2009.00713.pdf)). First implementation on GitHub with high-quality generation for 6-iterations.

## **Status**

- [x] Documented API.
- [x] High-fidelity generation.
- [x] Multi-iteration inference support (**stable for low iterations**).
- [x] Stable and fast training with **mixed-precision** support.
- [x] **Distributed training** support.
- [x] Training also successfully runs on a single 12GB GPU with batch size 96.
- [x] CLI inference support.
- [x] Flexible architecture configuration for your own data.
- [x] Estimated RTF on popular GPU and CPU devices (see below).
- [x] 100- and lower-iteration inferences are faster than real-time on RTX 2080 Ti. **6-iteration inference is faster than one reported in the paper**.
- [x] **Parallel grid search for the best noise schedule**.
- [x] Uploaded generated samples for different number of iterations (see `generated_samples` folder).
- [x] [Pretrained checkpoint](https://drive.google.com/file/d/1X_AquK11C0j7U1lLxMDBdK5guabHaoXB/view?usp=sharing) on 22KHz LJSpeech dataset **with noise schedules**.

#### Real-time factor (RTF)

**Number of parameters**: 15.810.401

|       Model       |   Stable  |  RTX 2080 Ti  |   Tesla K80   | Intel Xeon 2.3GHz* |
|-------------------|-----------|---------------|---------------|--------------------|
| 1000 iterations   |   **+**   |     9.59      |       -       |         -          |
|  100 iterations   |   **+**   |     0.94      |     5.85      |         -          |
|   50 iterations   |   **+**   |     0.45      |     2.92      |         -          |
|   25 iterations   |   **+**   |     0.22      |     1.45      |         -          |
|   12 iterations   |   **+**   |     0.10      |     0.69      |        4.55        |
|    6 iterations   |   **+**   |     0.04      |     0.33      |        2.09        |

***Note**: Used an old version of Intel Xeon CPU.

___
## About

WaveGrad is a conditional model for waveform generation through estimating gradients of the data density with WaveNet-similar sampling quality. **This vocoder is neither GAN, nor Normalizing Flow, nor classical autoregressive model**. The main concept of vocoder is based on *Denoising Diffusion Probabilistic Models* (DDPM), which utilize *Langevin dynamics* and *score matching* frameworks. Furthemore, comparing to classic DDPM, WaveGrad achieves super-fast convergence (6 iterations and probably lower) w.r.t. Langevin dynamics iterative sampling scheme.

___
## Installation

1. Clone this repo:

```bash
git clone https://github.com/ivanvovk/WaveGrad.git
cd WaveGrad
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

___
## Training

#### 1 Preparing data

1. Make train and test filelists of your audio data like ones included into `filelists` folder.
2. Make a configuration file* in `configs` folder.

***Note:** if you are going to change `hop_length` for STFT, then make sure that the product of your upsampling `factors` in config is equal to your new `hop_length`.

#### 2 Single and Distributed GPU training

1. Open `runs/train.sh` script and specify visible GPU devices and path to your configuration file. If you specify more than one GPU the training will run in distributed mode.
2. Run `sh runs/train.sh`

#### 3 Tensorboard and logging

To track your training process run tensorboard by `tensorboard --logdir=logs/YOUR_LOGDIR_FOLDER`. All logging information and checkpoints will be stored in `logs/YOUR_LOGDIR_FOLDER`. `logdir` is specified in config file.

#### 4 Noise schedule grid search

Once model is trained, grid search for the best schedule* for a needed number of iterations in [`notebooks/inference.ipynb`](notebooks/inference.ipynb). The code supports parallelism, so you can specify more than one number of jobs to accelerate the search.

***Note**: grid search is necessary just for a small number of iterations (like 6 or 7). For larger number just try Fibonacci sequence `benchmark.fibonacci(...)` initialization: I used it for 25 iteration and it works well. From good 25-iteration schedule, for example, you can build a higher-order schedule by copying elements.

##### Noise schedules for pretrained model

* 6-iteration schedule was obtained using grid search. After, based on obtained scheme, by hand, I found a slightly better approximation.
* 7-iteration schedule was obtained in the same way.
* 12-iteration schedule was obtained in the same way.
* 25-iteration schedule was obtained using Fibonacci sequence `benchmark.fibonacci(...)`.
* 50-iteration schedule was obtained by repeating elements from 25-iteration scheme.
* 100-iteration schedule was obtained in the same way.
* 1000-iteration schedule was obtained in the same way.

___
## Inference

#### CLI

Put your mel-spectrograms in some folder. Make a filelist. Then run this command with your own arguments:

```bash
sh runs/inference.sh -c <your-config> -ch <your-checkpoint> -ns <your-noise-schedule> -m <your-mel-filelist> -v "yes"
```

#### Jupyter Notebook

More inference details are provided in [`notebooks/inference.ipynb`](notebooks/inference.ipynb). There you can also find how to set a noise schedule for the model and make grid search for the best scheme.

___
## Other

#### Generated audios

Examples of generated audios are provided in [`generated_samples`](generated_samples/) folder. Quality degradation between 1000-iteration and 6-iteration inferences is not noticeable if found the best schedule for the latter.

#### Pretrained checkpoints

You can find a pretrained checkpoint file* on LJSpeech (22KHz) via [this](https://drive.google.com/file/d/1X_AquK11C0j7U1lLxMDBdK5guabHaoXB/view?usp=sharing) Google Drive link.

***Note**: uploaded checkpoint is a `dict` with a single key `'model'`.

___
## Important details, issues and comments

* During training WaveGrad uses a default noise schedule with 1000 iterations and linear scale betas from range (1e-6, 0.01). For inference you can set another schedule with less iterations. Tune betas carefully, the output quality really highly depends on it.
* By default model runs in a mixed-precision way. Batch size is modified compared to the paper (256 -> 96) since authors trained their model on TPU.
* After ~10k training iterations (1-2 hours) on a single GPU the model performs good generation for 50-iteration inference. Total training time is about 1-2 days (for absolute convergence).
* At some point training might start to behave weird and crazy (loss explodes), so I have introduced learning rate (LR) scheduling and gradient clipping. If loss explodes for your data, then try to decrease LR scheduler gamma a bit. It should help.
* By default hop length of your STFT is equal 300 (thus total upsampling factor). Other cases are not tested, but you can try. Remember, that total upsampling factor should be still equal to your new hop length.

___
## History of updates

* (**NEW**: 10/24/2020) Huge update. Distributed training and mixed-precision support. More correct positional encoding. CLI support for inference. Parallel grid search. Model size significantly decreased.
* New RTF info for NVIDIA Tesla K80 GPU card (popular in Google Colab service) and CPU Intel Xeon 2.3GHz.
* Huge update. New 6-iteration well generated sample example. New noise schedule setting API. Added the best schedule grid search code.
* Improved training by introducing smarter learning rate scheduler. Obtained high-fidelity synthesis.
* Stable training and multi-iteration inference. 6-iteration noise scheduling is supported.
* Stable training and fixed-iteration inference with significant background static noise left. All positional encoding issues are solved.
* Stable training of 25-, 50- and 1000-fixed-iteration models. Found no linear scaling (C=5000 from paper) of positional encoding (bug).
* Stable training of 25-, 50- and 1000-fixed-iteration models. Fixed positional encoding downscaling. Parallel segment sampling is replaced by full-mel sampling.
* (**RELEASE, first on GitHub**). Parallel segment sampling and broken positional encoding downscaling. Bad quality with clicks from concatenation from parallel-segment generation.

___
## References

* Nanxin Chen et al., [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf)
* Jonathan Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
* [Denoising Diffusion Probabilistic Models repository](https://github.com/hojonathanho/diffusion) (TensorFlow implementation), from which diffusion calculations have been adopted

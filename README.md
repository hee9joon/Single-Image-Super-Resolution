# Single-Image-Super-Resolution

### 0. Overview
This repository contains implementations of **Single Image Super Resolution** models including **EDSR**, **RDN**, **SRGAN** and **ESRGAN**. The criteria for selection of the models are as follows:

| Criterion | Residual | Dense |
|:---:|:---:|:---:|
| **Supervised (CNN-based)** | EDSR | RDN |
| **Unsupervised (GAN-based)** | SRGAN | ESRGAN |

The below lists provides the link to the papers:
- EDSR : [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)
- RDN : [Residual Dense Network for Image Super-Resolution
](https://arxiv.org/abs/1802.08797)
- SRGAN : [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
](https://arxiv.org/abs/1609.04802)
- ESRGAN : [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
](https://arxiv.org/abs/1809.00219)


The dataset used here is **[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)** and 128x128 resolution images are upscaled to 512x512 resolution images.

### 1. Quantitative Evaluation
| Method | PSNR↑ | MSE↓ | SSIM↑ |
|:---:|:---:|:---:|:---:|
| Bicubic | 24.759 ± 2.849 | 0.058 ± 0.040 | 0.710 ± 0.114 |
| EDSR | 25.913 ± 2.800 | 0.050 ± 0.035 | 0.754 ± 0.098 |
| RDN | 26.593 ± 3.144 | 0.043 ± 0.029 | 0.780 ± 0.095 |
| SRGAN | 23.167 ± 2.226 | 0.062 ± 0.036 | 0.688 ± 0.086 |
| ESRGAN | 25.955 ± 2.872 | 0.049 ± 0.033  | 0.755 ± 0.096 |

### 2. Qualitative Evaluation

| Sort | Image 1 | Image 2 | Image 3 |
|:---:|:---:|:---:|:---:|
| Target | <img src = './results/single/Inference_Samples_006_TARGET.png'> | <img src = './results/single/Inference_Samples_020_TARGET.png'> | <img src = './results/single/Inference_Samples_085_TARGET.png'> |
| Bicubic | <img src = './results/single/Inference_Samples_006_BICUBIC.png'> | <img src = './results/single/Inference_Samples_020_BICUBIC.png'> | <img src = './results/single/Inference_Samples_085_BICUBIC.png'> |
| EDSR | <img src = './results/single/Inference_Samples_006_EDSR.png'> | <img src = './results/single/Inference_Samples_020_EDSR.png'> | <img src = './results/single/Inference_Samples_085_EDSR.png'> |
| RDN | <img src = './results/single/Inference_Samples_006_RDN.png'> | <img src = './results/single/Inference_Samples_020_RDN.png'> | <img src = './results/single/Inference_Samples_085_RDN.png'> |
| SRGAN | <img src = './results/single/Inference_Samples_006_SRGAN.png'> | <img src = './results/single/Inference_Samples_020_SRGAN.png'> | <img src = './results/single/Inference_Samples_085_SRGAN.png'> |
| ESRGAN | <img src = './results/single/Inference_Samples_006_ESRGAN.png'> | <img src = './results/single/Inference_Samples_020_ESRGAN.png'> | <img src = './results/single/Inference_Samples_085_ESRGAN.png'> |

### 3. Run the Codes
#### 1) Dataset
```
python download_div2k.py
```

After running the code, the directory should be the same as follows:
```
+---[data]
|    \----[hr_train]
|          +---[0000.png]
|          |...
|          +---[0799.png]
|    \----[lr_train]
|          +---[0000.png]
|          |...
|          +---[0799.png]
|    \---[hr_valid]
|          +---[0000.png]
|          |...
|          +---[0099.png]
|    \---[lr_valid]
|          +---[0000.png]
|          |...
|          +---[0099.png]
+---div2k.py
+---download_div2k.py
|   ...
+---utils.py
```

#### 2) Train
You can train **EDSR**, **RDN**, **SRGAN** or **ESRGAN**.
```
python main.py --phase 'train' --model 'edsr'
```
```
python main.py --phase 'train' --model 'rdn'
```
```
python main.py --phase 'train' --model 'srgan'
```
```
python main.py --phase 'train' --model 'esrgan'
```

Also, you can choose the discriminator type using `--disc_type` option. It supports `fcn`, `conv` and `patch`, which stands for `fully connected layers`, `fully convolutional layers` and `PatchGAN`, respectively.

#### 3) Inference
You can inference each models.
```
python main.py --phase 'inference' --model 'esdr'
```
```
python main.py --phase 'inference' --model 'rdn'
```
```
python main.py --phase 'inference' --model 'srgan'
```
```
python main.py --phase 'inference' --model 'esrgan'
```

#### 4) Generate (Single Image)
You will need to place all the weight files to `./results/weights/`.
```
python main.py --phase 'generate'
```


### Development Environment
```
- Ubuntu 18.04 LTS
- NVIDIA GFORCE RTX 3090
- CUDA 10.2
- torch 1.6.0
- torchvision 0.7.0
- etc
```

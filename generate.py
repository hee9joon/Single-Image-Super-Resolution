import os
import random
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision.utils import save_image

from div2k import get_div2k_loader
from models_cnn import EDSR, RDN
from models_gan import SRGAN, ESRGAN, Discriminator
from utils import make_dirs, denorm

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generate_all(data_loader, device, args):
    """Generate Images using DIV2K Validation set"""

    # Single Results Path #
    if not os.path.exists(args.single_results_path):
        os.makedirs(args.single_results_path)

    # Prepare Networks #
    edsr = EDSR(
        channels=args.channels, 
        features=args.dim, 
        num_residuals=args.num_residuals, 
        scale_factor=args.upscale_factor
        ).to(device)

    rdn = RDN(
        channels=args.channels, 
        features=args.dim, 
        dim=args.dim, 
        num_denses=args.num_blocks, 
        num_layers=args.num_layers, 
        scale_factor=args.upscale_factor
        ).to(device)

    srgan = SRGAN(
        channels=args.channels, 
        ngf=args.dim, 
        scale_factor=args.upscale_factor
        ).to(device)

    esrgan = ESRGAN(
        channels=args.channels, 
        ngf=args.dim, 
        num_denses=args.num_denses, 
        growth_rate=args.growth_rate, 
        scale_ration=args.scale_ration, 
        scale_factor=args.upscale_factor
        ).to(device)

    # Weight Paths #
    edsr_weight_path = os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(edsr.__class__.__name__, args.num_epochs))
    rdn_weight_path = os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(rdn.__class__.__name__, args.num_epochs))
    srgan_weight_path = os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(srgan.__class__.__name__, args.num_epochs))
    esrgan_weight_path = os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(esrgan.__class__.__name__, args.num_epochs))

    # Load State Dict #
    edsr.load_state_dict(torch.load(edsr_weight_path))
    rdn.load_state_dict(torch.load(rdn_weight_path))
    srgan.load_state_dict(torch.load(srgan_weight_path))
    esrgan.load_state_dict(torch.load(esrgan_weight_path))

    # Up-sampling Network #
    up_sampler = torch.nn.Upsample(scale_factor=args.upscale_factor, mode='bicubic').to(device)

    for i, (high, low) in enumerate(data_loader):

        # Prepare Data #
        high = high.to(device)
        low = low.to(device)

        # Forward Data to Networks #
        with torch.no_grad():
            bicubic = up_sampler(low.detach())
            generated_edsr = edsr(low.detach())
            generated_rdn = rdn(low.detach())
            generated_srgan = srgan(low.detach())
            generated_esrgan = esrgan(low.detach())

        # Normalize and Save Images #
        save_image(denorm(high.data), os.path.join(args.single_results_path, 'Inference_Samples_%03d_TARGET.png' % (i+1)))
        save_image(denorm(bicubic.data), os.path.join(args.single_results_path, 'Inference_Samples_%03d_BICUBIC.png' % (i+1)))
        save_image(denorm(generated_edsr.data), os.path.join(args.single_results_path, 'Inference_Samples_%03d_%s.png' % (i+1, edsr.__class__.__name__)))
        save_image(denorm(generated_rdn.data), os.path.join(args.single_results_path, 'Inference_Samples_%03d_%s.png' % (i+1, rdn.__class__.__name__)))
        save_image(denorm(generated_srgan.data), os.path.join(args.single_results_path, 'Inference_Samples_%03d_%s.png' % (i+1, srgan.__class__.__name__)))
        save_image(denorm(generated_esrgan.data), os.path.join(args.single_results_path, 'Inference_Samples_%03d_%s.png' % (i+1, esrgan.__class__.__name__)))
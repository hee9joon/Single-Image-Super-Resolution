import os
import random
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from div2k import get_div2k_loader
from models_cnn import EDSR, RDN
from models_gan import SRGAN, ESRGAN, Discriminator
from train_cnn import train_srcnns
from train_gan import train_srgans
from utils import make_dirs, inference
from generate import generate_all

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):

    # Device Configuration for Multi-GPU Environment #
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Fix Seed for Reproducibility #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Samples and Weights Path #
    paths = [args.samples_path, args.weights_path]
    for path in paths:
        make_dirs(path)

    # Prepare Data Loader #
    train_div2k_loader = get_div2k_loader(sort='train',
                                          batch_size=args.batch_size,
                                          image_size=args.image_size,
                                          upscale_factor=args.upscale_factor,
                                          crop_size=args.crop_size,
                                          patch_size=args.patch_size, 
                                          patch=args.patch, 
                                          flip=args.flip, 
                                          rotate=args.rotate)

    val_div2k_loader = get_div2k_loader(sort='val',
                                        batch_size=args.val_batch_size,
                                        image_size=args.image_size,
                                        upscale_factor=args.upscale_factor,
                                        crop_size=args.crop_size)

    # Prepare Networks #
    if args.model == 'edsr':

        edsr = EDSR(
            channels=args.channels, 
            features=args.dim, 
            num_residuals=args.num_residuals, 
            scale_factor=args.upscale_factor
            ).to(device)

    elif args.model == 'rdn':

        rdn = RDN(
            channels=args.channels, 
            features=args.dim, 
            dim=args.dim, 
            num_denses=args.num_blocks, 
            num_layers=args.num_layers, 
            scale_factor=args.upscale_factor
            ).to(device)

    elif args.model == 'srgan':

        srgan = SRGAN(
            channels=args.channels, 
            ngf=args.dim, 
            scale_factor=args.upscale_factor
            ).to(device)

    elif args.model == 'esrgan':

        esrgan = ESRGAN(
            channels=args.channels, 
            ngf=args.dim, 
            num_denses=args.num_denses, 
            growth_rate=args.growth_rate, 
            scale_ration=args.scale_ration, 
            scale_factor=args.upscale_factor
            ).to(device)
        
    else:
        raise NotImplementedError

    D = Discriminator(
        in_channels=args.channels, 
        ndf=args.dim, 
        linear_dim=args.linear_dim, 
        out_dim=args.out_dim,
        disc_type=args.disc_type
        ).to(device)

    if args.phase == 'train':
        if args.model == 'edsr':
            train_srcnns(train_div2k_loader, val_div2k_loader, edsr, device, args)

        elif args.model == 'rdn':
            train_srcnns(train_div2k_loader, val_div2k_loader, rdn, device, args)

        elif args.model == 'srgan':
            train_srgans(train_div2k_loader, val_div2k_loader, srgan, D, device, args)

        elif args.model == 'esrgan':
            train_srgans(train_div2k_loader, val_div2k_loader, esrgan, D, device, args)

    elif args.phase == 'inference':

        if args.model == 'edsr':
            edsr_weight_path = os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(edsr.__class__.__name__, args.num_epochs))
            edsr.load_state_dict(torch.load(edsr_weight_path))
            inference(val_div2k_loader, edsr, args.upscale_factor, args.num_epochs, args.inference_path, device, save_combined=False)

        elif args.model == 'rdn':
            rdn_weight_path = os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(rdn.__class__.__name__, args.num_epochs))
            rdn.load_state_dict(torch.load(rdn_weight_path))
            inference(val_div2k_loader, rdn, args.upscale_factor, args.num_epochs, args.inference_path, device, save_combined=False)

        elif args.model == 'srgan':
            srgan_weight_path = os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(srgan.__class__.__name__, args.num_epochs))
            srgan.load_state_dict(torch.load(srgan_weight_path))
            inference(val_div2k_loader, srgan, args.upscale_factor, args.num_epochs, args.inference_path, device, save_combined=False)

        elif args.model == 'esrgan':
            esrgan_weight_path = os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(esrgan.__class__.__name__, args.num_epochs))
            esrgan.load_state_dict(torch.load(esrgan_weight_path))
            inference(val_div2k_loader, esrgan, args.upscale_factor, args.num_epochs, args.inference_path, device, save_combined=False)

    elif args.phase == 'generate':
        generate_all(val_div2k_loader, device, args)

    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number configuration')
    parser.add_argument('--seed', type=int, default=42, help='fix seed for reproducibility')
    parser.add_argument('--phase', type=str, default='train', help='train, inference, generate', choices=['train', 'inference', 'generate'])

    parser.add_argument('--model', type=str, default='edsr', choices=['edsr', 'rdn', 'srgan', 'esrgan'])
    parser.add_argument('--disc_type', type=str, default='fcn', choices=['fcn', 'conv', 'patch'])

    parser.add_argument('--channels', type=int, default=3, help='in- and out- channel for models')
    parser.add_argument('--out_dim', type=int, default=1, help='output dimension for discriminator')
    parser.add_argument('--dim', type=int, default=64, help='feature dimension for models')
    parser.add_argument('--scale_factor', type=int, default=4, help='upscale factor')
    parser.add_argument('--num_residuals', type=int, default=16, help='the number of residual blocks')
    parser.add_argument('--num_denses', type=int, default=23, help='the number of dense blocks')
    parser.add_argument('--linear_dim', type=int, default=1000, help='dimension of fully connected layer for discriminator')
    parser.add_argument('--growth_rate', type=int, default=32, help='growth rate')
    parser.add_argument('--num_blocks', type=int, default=16, help='the number of blocks')
    parser.add_argument('--num_layers', type=int, default=8, help='the number of layers')
    parser.add_argument('--scale_ration', type=float, default=0.2, help='scale ration')

    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size for train')
    parser.add_argument('--val_batch_size', type=int, default=1, help='mini-batch size for validation')
    parser.add_argument('--image_size', type=int, default=512, help='image size')
    parser.add_argument('--crop_size', type=int, default=128, help='image crop size')
    parser.add_argument('--patch_size', type=int, default=48, help='image patch size')
    parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
    parser.add_argument('--flip', type=bool, default=True, help='flipping for data augmentation during training')
    parser.add_argument('--patch', type=bool, default=True, help='making patch for data augmentation during training')
    parser.add_argument('--rotate', type=bool, default=True, help='rotation for data augmentation during training')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
    parser.add_argument('--lr_decay_every', type=int, default=25, help='decay learning rate for every default epoch')
    parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

    parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--inference_path', type=str, default='./results/inference/', help='inference path')
    parser.add_argument('--single_results_path', type=str, default='./results/single/', help='single result path')

    parser.add_argument('--num_epochs', type=int, default=100, help='total epoch')
    parser.add_argument('--print_every', type=int, default=100, help='print statistics for every default iteration')
    parser.add_argument('--save_every', type=int, default=5, help='save model weights for every default epoch')

    parser.add_argument('--lambda_adversarial', type=float, default=1e-3, help='lambda for Adversarial Loss used for SRGAN')
    parser.add_argument('--lambda_tv', type=float, default=2e-8, help='lambda for Total Variation Loss used for SRGAN')
    parser.add_argument('--lambda_content', type=float, default=1, help='lambda for Content Loss used for ESRGAN')
    parser.add_argument('--lambda_bce', type=float, default=5e-3, help='lambda for Binary Cross Entropy Loss used for ESRGAN')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    main(args)
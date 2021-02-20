import os
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity

import torch
from torch.nn import init
from torchvision.utils import save_image


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_scheduler(optimizer, args):
    """Learning Rate Scheduler"""
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler


def set_requires_grad(network, requires_grad=False):
    """Prevent a Network from Updating"""
    for param in network.parameters():
        param.requires_grad = requires_grad


def denorm(x):
    """De-normalization"""
    out = (x+1) / 2
    return out.clamp(0, 1)


def PSNR(image_A, image_B):
    """Calculate PSNR Value for a Pair of Images"""
    image_A = np.array(image_A, dtype=np.float32)
    image_B = np.array(image_B, dtype=np.float32)
    mse = np.mean((image_A - image_B) ** 2)
    max_pixel = image_B.max() - image_B.min()
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def MSE(image_A, image_B):
    """Calculate MSE Value for a Pair of Images"""
    image_A = np.array(image_A, dtype=np.float32)
    image_B = np.array(image_B, dtype=np.float32)
    mse = np.sum((image_A - image_B) ** 2)
    mse /= float((image_A.shape[2] * image_A.shape[3]))
    return mse


def SSIM(image_A, image_B):
    """Calculate SSIM Value for a Pair of Images"""
    image_A = np.array(image_A, dtype=np.float32)
    image_B = np.array(image_B, dtype=np.float32)
    image_A = np.transpose(image_A, (0, 2, 3, 1)).squeeze(axis=0)
    image_B = np.transpose(image_B, (0, 2, 3, 1)).squeeze(axis=0)
    ssim = structural_similarity(image_A, image_B, data_range=image_B.max() - image_B.min(), multichannel=True)
    return ssim


def sample_images(data_loader, batch_size, scale_factor, model, epoch, path, device):
    """Save Sample Images for Every Epoch"""

    high, low = next(iter(data_loader))

    high = high.to(device)
    low = low.to(device)

    up_sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bicubic').to(device)
    bicubic = up_sampler(low)

    with torch.no_grad():
        generated = model(low.detach())

    images = [bicubic, generated, high]

    result = torch.cat(images, dim=0)

    save_image(denorm(result.data),
               os.path.join(path, '%s_Samples_Epoch_%03d.png' % (model.__class__.__name__, epoch + 1)), nrow=8 if batch_size > 8 else len(images)
               )

    del images


def inference(data_loader, model, upscale_factor, epoch, path, device, save_combined=True):
    """Inference using DIV2K validation set"""

    # Inference Path #
    results_path = os.path.join(path, '{}_Epoch_{}'.format(model.__class__.__name__, epoch+1))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Lists #
    PSNR_GT_values, PSNR_Gen_values = list(), list()
    MSE_GT_values, MSE_Gen_values = list(), list()
    SSIM_GT_values, SSIM_Gen_values = list(), list()

    # Up-sampling Network #
    up_sampler = torch.nn.Upsample(scale_factor=upscale_factor, mode='bicubic').to(device)

    # Calculate Metric for Ground Truths and Generated Metric and Save Results #
    print("\nInference Results at Epoch {} using {} follow:".format(epoch+1, model.__class__.__name__))
    for i, (high, low) in enumerate(data_loader):

        # Prepare Data #
        high = high.to(device)
        low = low.to(device)

        # Forward Data to Networks #
        with torch.no_grad():
            bicubic = up_sampler(low.detach())
            generated = model(low.detach())

        # Calculate PSNR #
        PSNR_GT_value = PSNR(high.detach().cpu().numpy(), bicubic.detach().cpu().numpy())
        PSNR_Gen_value = PSNR(high.detach().cpu().numpy(), generated.detach().cpu().numpy())

        # Calculate MSE #
        MSE_GT_value = MSE(high.detach().cpu().numpy(), bicubic.detach().cpu().numpy())
        MSE_Gen_value = MSE(high.detach().cpu().numpy(), generated.detach().cpu().numpy())

        # Calculate SSIM #
        SSIM_GT_value = SSIM(high.detach().cpu().numpy(), bicubic.detach().cpu().numpy())
        SSIM_Gen_value = SSIM(high.detach().cpu().numpy(), generated.detach().cpu().numpy())

        # Add items to Lists #
        PSNR_GT_values.append(PSNR_GT_value)
        PSNR_Gen_values.append(PSNR_Gen_value)

        MSE_GT_values.append(MSE_GT_value)
        MSE_Gen_values.append(MSE_Gen_value)

        SSIM_GT_values.append(SSIM_GT_value)
        SSIM_Gen_values.append(SSIM_Gen_value)

        # Normalize and Save Images #
        if save_combined:
            images = [bicubic, generated, high]
            result = torch.cat(images, dim=0)

            save_image(
                denorm(result.data), 
                os.path.join(results_path, '%s_Inference_Epoch_%03d_Samples_%03d.png' % (model.__class__.__name__, epoch+1, i+1)),
                nrow=len(images)
                )
        
        else:
            save_image(
                denorm(generated.data), 
                os.path.join(results_path, '%s_Single_Inference_Epoch_%03d_Samples_%03d.png' % (model.__class__.__name__, epoch+1, i+1)),
                nrow=1
            )

    # Print Statistics #
    print("### PSNR ###")
    print("  Bicubic | Average {:.3f} | SD {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(np.mean(PSNR_GT_values), np.std(PSNR_GT_values), np.max(PSNR_GT_values), np.min(PSNR_GT_values)))
    print("Generated | Average {:.3f} | SD {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(np.average(PSNR_Gen_values), np.std(PSNR_Gen_values), np.max(PSNR_Gen_values), np.min(PSNR_Gen_values)))

    print("### MSE ###")
    print("  Bicubic | Average {:.3f} | SD {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(np.mean(MSE_GT_values), np.std(MSE_GT_values), np.max(MSE_GT_values), np.min(MSE_GT_values)))
    print("Generated | Average {:.3f} | SD {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(np.mean(MSE_Gen_values), np.std(MSE_Gen_values), np.max(MSE_Gen_values), np.min(MSE_Gen_values)))

    print("### SSIM ###")
    print("  Bicubic | Average {:.3f} | SD {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(np.mean(SSIM_GT_values), np.std(SSIM_GT_values), np.max(SSIM_GT_values), np.min(SSIM_GT_values)))
    print("Generated | Average {:.3f} | SD {:.3f} | Maximum {:.3f} | Minimum {:.3f}\n"
          .format(np.mean(SSIM_Gen_values), np.std(SSIM_Gen_values), np.max(SSIM_Gen_values), np.min(SSIM_Gen_values)))
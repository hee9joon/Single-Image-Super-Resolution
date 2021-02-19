import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision.utils import save_image

from losses_gan import PerceptualLoss, TVLoss
from utils import get_lr_scheduler, set_requires_grad, sample_images, inference

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_srgans(train_loader, val_loader, generator, discriminator, device, args):

    # Loss Function #
    criterion_Perceptual = PerceptualLoss(args.model).to(device)

    # For SRGAN #
    criterion_MSE = nn.MSELoss()
    criterion_TV = TVLoss()

    # For ESRGAN #
    criterion_BCE = nn.BCEWithLogitsLoss()
    criterion_Content = nn.L1Loss()

    # Optimizers #
    D_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    G_optim = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    D_optim_scheduler = get_lr_scheduler(args.lr_scheduler, D_optim, args)
    G_optim_scheduler = get_lr_scheduler(args.lr_scheduler, G_optim, args)

    # Lists #
    D_losses, G_losses = list(), list()

    # Train #
    print("Training {} started with total epoch of {}.".format(str(args.model).upper(), args.num_epochs))

    for epoch in range(args.num_epochs):
        for i, (high, low) in enumerate(train_loader):

            discriminator.train()
            if args.model == "srgan":
                generator.train()

            # Data Preparation #
            high = high.to(device)
            low = low.to(device)

            # Initialize Optimizers #
            D_optim.zero_grad()
            G_optim.zero_grad()

            #######################
            # Train Discriminator #
            #######################

            set_requires_grad(discriminator, requires_grad=True)

            # Generate Fake HR Images #
            fake_high = generator(low)

            if args.model == 'srgan':

                # Forward Data #
                prob_real = discriminator(high)
                prob_fake = discriminator(fake_high.detach())

                # Calculate Total Discriminator Loss #
                D_loss = 1 - prob_real.mean() + prob_fake.mean()

            elif args.model == 'esrgan':

                # Forward Data #
                prob_real = discriminator(high)
                prob_fake = discriminator(fake_high.detach())

                # Relativistic Discriminator #
                diff_r2f = prob_real - prob_fake.mean()
                diff_f2r = prob_fake - prob_real.mean()

                # Labels #
                real_labels = torch.ones(diff_r2f.size()).to(device)
                fake_labels = torch.zeros(diff_f2r.size()).to(device)

                # Adversarial Loss #
                D_loss_real = criterion_BCE(diff_r2f, real_labels)
                D_loss_fake = criterion_BCE(diff_f2r, fake_labels)

                # Calculate Total Discriminator Loss #
                D_loss = (D_loss_real + D_loss_fake).mean()

            # Back Propagation and Update #
            D_loss.backward()
            D_optim.step()

            ###################
            # Train Generator #
            ###################

            set_requires_grad(discriminator, requires_grad=False)

            if args.model == 'srgan':

                # Adversarial Loss #
                prob_fake = discriminator(fake_high).mean()
                G_loss_adversarial = torch.mean(1 - prob_fake)
                G_loss_mse = criterion_MSE(fake_high, high)

                # Perceptual Loss #
                lambda_perceptual = 6e-3
                G_loss_perceptual = criterion_Perceptual(fake_high, high)

                # Total Variation Loss #
                G_loss_tv = criterion_TV(fake_high)

                # Calculate Total Generator Loss #
                G_loss = args.lambda_adversarial * G_loss_adversarial + G_loss_mse + lambda_perceptual * G_loss_perceptual + args.lambda_tv * G_loss_tv

            elif args.model == 'esrgan':

                # Forward Data #
                prob_real = discriminator(high)
                prob_fake = discriminator(fake_high)

                # Relativistic Discriminator #
                diff_r2f = prob_real - prob_fake.mean()
                diff_f2r = prob_fake - prob_real.mean()

                # Labels #
                real_labels = torch.ones(diff_r2f.size()).to(device)
                fake_labels = torch.zeros(diff_f2r.size()).to(device)

                # Adversarial Loss #
                G_loss_bce_real = criterion_BCE(diff_f2r, real_labels)
                G_loss_bce_fake = criterion_BCE(diff_r2f, fake_labels)

                G_loss_bce = (G_loss_bce_real + G_loss_bce_fake).mean()

                # Perceptual Loss #
                lambda_perceptual = 1e-2
                G_loss_perceptual = criterion_Perceptual(fake_high, high)

                # Content Loss #
                G_loss_content = criterion_Content(fake_high, high)

                # Calculate Total Generator Loss #
                G_loss = args.lambda_bce * G_loss_bce + lambda_perceptual * G_loss_perceptual + args.lambda_content * G_loss_content

            # Back Propagation and Update #
            G_loss.backward()
            G_optim.step()

            # Add items to Lists #
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % args.print_every == 0:
                print("{} | Epoch [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | G Loss {:.4f}"
                      .format(str(args.model).upper(), epoch + 1, args.num_epochs, i + 1, len(train_loader), np.average(D_losses), np.average(G_losses)))

                # Save Sample Images #
                sample_images(val_loader, args.batch_size, args.scale_factor, generator, epoch, args.samples_path, device)

        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights and Inference #
        if (epoch+1) % args.save_every == 0:
            torch.save(generator.state_dict(), os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(generator.__class__.__name__, epoch + 1)))
            inference(val_loader, generator, args.upscale_factor, epoch, args.inference_path, device)
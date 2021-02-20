import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision.utils import save_image

from utils import get_lr_scheduler, sample_images, inference

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_srcnns(train_loader, val_loader, model, device, args):

    # Loss Function #
    criterion = nn.L1Loss()

    # Optimizers #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_scheduler = get_lr_scheduler(optimizer=optimizer, args=args)

    # Lists #
    losses = list()

    # Train #
    print("Training {} started with total epoch of {}.".format(str(args.model).upper(), args.num_epochs))

    for epoch in range(args.num_epochs):
        for i, (high, low) in enumerate(train_loader):

            # Data Preparation #
            high = high.to(device)
            low = low.to(device)

            # Forward Data #
            generated = model(low)

            # Calculate Loss #
            loss = criterion(generated, high)

            # Initialize Optimizer #
            optimizer.zero_grad()

            # Back Propagation and Update #
            loss.backward()
            optimizer.step()

            # Add items to Lists #
            losses.append(loss.item())

            # Print Statistics #
            if (i+1) % args.print_every == 0:
                print("{} | Epoch [{}/{}] | Iterations [{}/{}] | Loss {:.4f}"
                      .format(str(args.model).upper(), epoch+1, args.num_epochs, i+1, len(train_loader), np.average(losses)))

                # Save Sample Images #
                sample_images(val_loader, args.batch_size, args.upscale_factor, model, epoch, args.samples_path, device)

        # Adjust Learning Rate #
        optimizer_scheduler.step()

        # Save Model Weights and Inference #
        if (epoch+1) % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.weights_path, '{}_Epoch_{}.pkl'.format(model.__class__.__name__, epoch+1)))
            inference(val_loader, model, args.upscale_factor, epoch, args.inference_path, device)
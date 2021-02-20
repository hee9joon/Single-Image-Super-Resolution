import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class DIV2K(Dataset):
    def __init__(self, sort, original_size, crop_size, upscale_factor, patch_size, patch, flip, rotate):
        
        self.sort = sort
        self.train_hr_path = './data/hr_train/'
        self.train_lr_path = './data/lr_train/'

        self.val_hr_path = './data/hr_valid/'
        self.val_lr_path = './data/lr_valid/'

        self.original_size = original_size
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size

        self.train_images = sorted(os.listdir(self.train_hr_path))
        self.val_images = sorted(os.listdir(self.val_hr_path))

        self.patch = patch
        self.flip = flip
        self.rotate = rotate
        
    def __len__(self):
        return len(self.train_images) if self.sort == 'train' else len(self.val_images)
        
    def __getitem__(self, index):
        
        if self.sort == 'train':
            hr_image = Image.open(os.path.join(self.train_hr_path, self.train_images[index])).convert("RGB")
            lr_image = Image.open(os.path.join(self.train_lr_path, self.train_images[index])).convert("RGB")

            hr_image = hr_image.resize((self.original_size, self.original_size))
            lr_image = lr_image.resize((self.crop_size, self.crop_size))

            hr_image = np.array(hr_image)
            lr_image = np.array(lr_image)

            hr_image = (hr_image / 127.5) - 1.0
            lr_image = (lr_image / 127.5) - 1.0

            if self.patch:
                hr_image, lr_image = self._get_patch(hr_image, lr_image)

            if self.flip:
                hr_image, lr_image = self._flipping(hr_image, lr_image)
            
            if self.rotate:
                hr_image, lr_image = self._rotating(hr_image, lr_image)
            
            hr_image = hr_image.transpose(2, 0, 1).astype(np.float32)
            lr_image = lr_image.transpose(2, 0, 1).astype(np.float32)

        elif self.sort == 'val':
            hr_image = Image.open(os.path.join(self.val_hr_path, self.val_images[index])).convert("RGB")
            lr_image = Image.open(os.path.join(self.val_lr_path, self.val_images[index])).convert("RGB")

            hr_image = hr_image.resize((self.original_size, self.original_size))
            lr_image = lr_image.resize((self.crop_size, self.crop_size))

            hr_image = np.array(hr_image)
            lr_image = np.array(lr_image)

            hr_image = (hr_image / 127.5) - 1.0
            lr_image = (lr_image / 127.5) - 1.0
            
            hr_image = hr_image.transpose(2, 0, 1).astype(np.float32)
            lr_image = lr_image.transpose(2, 0, 1).astype(np.float32)
        
        return hr_image, lr_image

    def _get_patch(self, hr_image, lr_image):

        height, width, _ = lr_image.shape

        lr_width = random.randrange(0, width - self.patch_size + 1)
        lr_height = random.randrange(0, height - self.patch_size + 1)

        hr_width = lr_width * self.upscale_factor
        hr_height = lr_height * self.upscale_factor

        hr_patch = hr_image[hr_height: hr_height + self.upscale_factor * self.patch_size, hr_width: hr_width + self.upscale_factor * self.patch_size]
        lr_patch = lr_image[lr_height: lr_height + self.patch_size, lr_width: lr_width + self.patch_size]

        return hr_patch, lr_patch

    def _flipping(self, hr_image, lr_image):

        horizontal_flip = random.choice([0, 1])
        vertical_flip = random.choice([0, 1])
            
        if horizontal_flip:
            hr_image = np.fliplr(hr_image)
            lr_image = np.fliplr(lr_image)
        
        if vertical_flip:
            hr_image = np.flipud(hr_image)
            lr_image = np.flipud(lr_image)

        return hr_image, lr_image

    def _rotating(self, hr_image, lr_image):
        rotating = random.choice([0, 1])
        
        if rotating:
            hr_image = hr_image.transpose(1, 0, 2)
            lr_image = lr_image.transpose(1, 0, 2) 

        return hr_image, lr_image



def get_div2k_loader(sort, batch_size, image_size, upscale_factor, crop_size, patch_size=None, patch=False, flip=False, rotate=False):
    """Prepare DIV2K Loader"""

    if sort == 'train':
        dataset = DIV2K(sort, image_size, crop_size, upscale_factor, patch_size, patch, flip, rotate)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif sort == 'val':
        dataset = DIV2K(sort, image_size, crop_size, upscale_factor, patch_size, patch, flip, rotate)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    else:
        raise NotImplementedError

    return data_loader
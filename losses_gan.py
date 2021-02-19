import torch
import torch.nn as nn
from torchvision.models import vgg19


class TVLoss(nn.Module):
    """Total Variation Loss for SRGAN"""
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        height_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :height - 1, :]), 2).sum()
        width_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :width - 1]), 2).sum()
        out = 2 * (height_tv / count_h + width_tv / count_w) / batch_size
        out = self.tv_loss_weight * out
        return out

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class PerceptualLoss(nn.Module):
    """Perception Loss using VGG19 for SRGAN and ESRGAN"""
    def __init__(self, model):
        super(PerceptualLoss, self).__init__()

        self.model = model

        # Set Different Activation Function #
        if self.model == "srgan":
            relu = 31
        elif self.model == "esrgan":
            relu = 35
        else:
            raise NotImplementedError

        # Modify VGG19 #
        vgg = vgg19(pretrained=True)
        model = nn.Sequential(*list(vgg.features)[:relu])
        model = model.eval()

        # Freeze VGG19 (As it will not be trained) #
        for param in model.parameters():
            param.requires_grad = False

        self.freezed_vgg = model

        # Loss Function #
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, source, target):

        # Forward Data #
        source_feature = self.freezed_vgg(source)
        target_feature = self.freezed_vgg(target)

        # Calculate Perceptual Loss #
        if self.model == "srgan":
            perceptual_loss = self.mse_loss(source_feature, target_feature)
        elif self.model == "esrgan":
            perceptual_loss = self.mae_loss(source_feature, target_feature)
        else:
            raise NotImplementedError

        return perceptual_loss
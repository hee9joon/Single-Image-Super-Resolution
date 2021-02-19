import torch
import torch.nn as nn


##########################################################
# Enhanced Deep (Residual Networks for) Super Resolution #
##########################################################

class MeanShift(nn.Conv2d):
    """Mean Shift"""
    def __init__(self, rgb_mean, std, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) / torch.Tensor(std)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False


class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.main(x)
        out *= 0.1
        out += x
        return out


class Upscale(nn.Module):
    """Up-Scaling using Convolutoinal Layer and Pixel Shuffle Layer"""
    def __init__(self, features):
        super(Upscale, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        out = self.main(x)
        return out


class EDSR(nn.Module):
    """Enhanced Deep (Residual Networks for) Super Resolution"""
    def __init__(self, channels, features, num_residuals, scale_factor, rgb_mean=(0.5, 0.5, 0.5), std=(1, 1, 1)):
        super(EDSR, self).__init__()

        # Mean Shift #
        self.subtract_mean = MeanShift(rgb_mean, std, -1)
        self.add_mean = MeanShift(rgb_mean, std, +1)

        # Convolutional Layers #
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Residual Blocks #
        blocks = list()
        for _ in range(num_residuals):
            blocks.append(ResidualBlock(features))
        self.residual_blocks = nn.Sequential(*blocks)

        # Up Scaling #
        scales = list()
        for _ in range(int(scale_factor/2)):
            scales.append(Upscale(features))
        self.upscale_blocks = nn.Sequential(*scales)

    def forward(self, x):
        x = self.subtract_mean(x)
        x = self.conv1(x)
        out = self.residual_blocks(x)
        out = self.conv2(out)
        out += x
        out = self.upscale_blocks(out)
        out = self.conv3(out)
        out = self.add_mean(out)
        return out


##########################
# Deep Residual Networks #
##########################

class ConvBlock(nn.Module):
    """Convolutional Block"""
    def __init__(self, channels, features, kernel_size=3):
        super(ConvBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.main(x)
        out = torch.cat((x, out), dim=1)
        return out


class RDB(nn.Module):
    """Residual Deep Block"""
    def __init__(self, channels, features, num_layers):
        super(RDB, self).__init__()

        blocks = list()

        for _ in range(num_layers):
            blocks.append(ConvBlock(channels, features, kernel_size=3))
            channels += features

        blocks.append(nn.Conv2d(channels, features, kernel_size=1, stride=1, padding=0, bias=False))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        out += x
        return out


class RDN(nn.Module):
    """Deep Residual Network for Super Resolution"""
    def __init__(self, channels, features, dim, num_denses, num_layers, scale_factor):
        super(RDN, self).__init__()

        self.num_denses = num_denses

        # Convolutional Layers #
        self.conv1 = nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Global Feature Fusion #
        self.gff = nn.Sequential(
            nn.Conv2d(dim * num_denses, features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # Dense Blocks #
        self.dense_blocks = nn.ModuleList()
        self.dense_blocks.append(RDB(features, dim, num_layers))
        for _ in range(num_denses):
            self.dense_blocks.append(RDB(dim, dim, num_layers))

        # Up Scaling #
        scales = list()
        for _ in range(int(scale_factor/2)):
            scales.append(Upscale(features))
        self.upscale_blocks = nn.Sequential(*scales)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)

        x = out_conv2
        local_features = []
        for i in range(self.num_denses):
            x = self.dense_blocks[i](x)
            local_features.append(x)

        out = torch.cat(local_features, dim=1)
        out_gff = self.gff(out)
        out_gff += out_conv1

        out_upscale = self.upscale_blocks(out_gff)
        out = self.conv3(out_upscale)
        return out
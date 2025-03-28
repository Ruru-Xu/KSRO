import numpy as np
import scipy.signal

import torch
import torch.nn as nn

from actor_critic_modules.fft_conv import FFTConv2d
from actor_critic_modules.unet import UnetModel
import torch.nn.functional as F

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv1(x))
        return x * attention


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = self.global_avg_pool(x).view(b, c)
        fc_out = self.sigmoid(self.fc2(F.relu(self.fc1(avg_pool)))).view(b, c, 1, 1)
        return x * fc_out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class Kspace_Net_MT(nn.Module):

    def __init__(self, act_dim, image_shape, dropout, feature_dim, mt_shape):
        super().__init__()
        self.image_shape = image_shape
        self.dropout = dropout
        self.act_dim = act_dim
        self.aux_shape = 0
        self.mt_shape = mt_shape

        self.fft_conv1 = FFTConv2d(in_channels=1, out_channels=1, kernel_size=9, stride=1, bias=False)
        self.fft_conv2 = FFTConv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, bias=False)
        self.fft_conv3 = FFTConv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, bias=False)

        elementwise_affine = True

        self.layernorm = nn.LayerNorm(
            elementwise_affine=elementwise_affine, normalized_shape=self.image_shape
        )

        # Initialize UNet as the backbone
        self.unet_backbone = UnetModel(in_chans=3, out_chans=64, chans=64, num_pool_layers=4, drop_prob=dropout)
        # Hybrid pooling: global and local context
        self.global_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.local_pool = nn.AdaptiveAvgPool2d((64, 64))

        # Trunk layer for processed UNet output
        self.trunk = nn.Sequential(
            nn.Linear(64 * (32 * 32 + 64 * 64), feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, feature_dim),  # Add another linear layer
            nn.Tanh()
        )

        # Policy layer for action logits
        self.policy_layer = nn.Sequential(
            nn.Linear(feature_dim + sum(self.mt_shape), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.act_dim)
        )

        self.trunk.apply(weight_init)
        self.policy_layer.apply(weight_init)
    def forward(self, input_dict):
        kspace = input_dict['kspace'] * 1000 # Input k-space data: [batch, 1, 320, 320]
        mt = input_dict['mt']  # Metadata: [batch]

        # Pass through FFT Conv layer
        out_complex = torch.cat((self.fft_conv1(kspace), self.fft_conv2(kspace), self.fft_conv3(kspace)), dim=1)
        out_mag = out_complex.abs()
        out_mag = self.layernorm(out_mag)  # Shape: [batch, 1, 320, 320]

        # Extract features using UNet
        unet_features = self.unet_backbone(out_mag)  # Shape: [batch, 64, 320, 320]

        # Hybrid pooling for global and local contexts
        global_features = self.global_pool(unet_features)  # Shape: [batch, 64, 32, 32]
        local_features = self.local_pool(unet_features)  # Shape: [batch, 64, 128, 128]

        # Flatten and combine features
        global_flat = global_features.view(global_features.size(0), -1)  # Shape: [batch, 64*32*32]
        local_flat = local_features.view(local_features.size(0), -1)  # Shape: [batch, 64*128*128]
        combined_features = torch.cat((global_flat, local_flat), dim=-1)  # Shape: [batch, 64*(32*32 + 128*128)]

        # Pass through trunk layer
        h = self.trunk(combined_features)  # Shape: [batch, feature_dim]

        # Metadata processing and fusion
        mt_vec = torch.nn.functional.one_hot(mt, num_classes=self.mt_shape[0]).float()  # Shape: [batch, mt_shape]
        if len(mt_vec.shape) == 1:
            out = torch.cat((h, mt_vec.repeat(unet_features.shape[0], 1)), dim=-1)
        else:
            out = torch.cat((h, mt_vec), dim=-1)

        # Compute action logits
        action_logits = self.policy_layer(out)  # Shape: [batch, act_dim]

        return action_logits


class Kspace_Net_Critic_MT(nn.Module):

    def __init__(self, image_shape, dropout, feature_dim, mt_shape):
        super().__init__()

        self.image_shape = image_shape
        self.dropout = dropout
        self.aux_shape = 0
        self.mt_shape = mt_shape

        self.fft_conv1 = FFTConv2d(in_channels=1, out_channels=1, kernel_size=9, stride=1, bias=False)
        self.fft_conv2 = FFTConv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, bias=False)
        self.fft_conv3 = FFTConv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, bias=False)

        elementwise_affine = True

        self.layernorm = nn.LayerNorm(
            elementwise_affine=elementwise_affine, normalized_shape=self.image_shape
        )

        # Initialize UNet as the backbone
        self.unet_backbone = UnetModel(in_chans=3, out_chans=64, chans=64, num_pool_layers=4, drop_prob=dropout)
        # Hybrid pooling: global and local context
        self.global_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.local_pool = nn.AdaptiveAvgPool2d((64, 64))

        # Trunk layer for processed UNet output
        self.trunk = nn.Sequential(
            nn.Linear(64 * (32 * 32 + 64 * 64), feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, feature_dim),  # Add another linear layer
            nn.Tanh()
        )

        # Critic layer for value prediction
        self.critic_layer = nn.Sequential(
            nn.Linear(feature_dim + sum(self.mt_shape), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        self.critic_layer.apply(weight_init)

    def forward(self, input_dict):
        kspace = input_dict['kspace'] * 1000
        mt = input_dict['mt']

        # Pass through FFT Conv layer
        out_complex = torch.cat((self.fft_conv1(kspace), self.fft_conv2(kspace), self.fft_conv3(kspace)), dim=1)
        out_mag = out_complex.abs()
        out_mag = self.layernorm(out_mag)  # Shape: [batch, 1, 320, 320]

        # Extract features using UNet
        unet_features = self.unet_backbone(out_mag)  # Shape: [batch, 64, 320, 320]

        # Hybrid pooling for global and local contexts
        global_features = self.global_pool(unet_features)  # Shape: [batch, 64, 32, 32]
        local_features = self.local_pool(unet_features)  # Shape: [batch, 64, 128, 128]

        # Flatten and combine features
        global_flat = global_features.view(global_features.size(0), -1)  # Shape: [batch, 64*32*32]
        local_flat = local_features.view(local_features.size(0), -1)  # Shape: [batch, 64*128*128]
        combined_features = torch.cat((global_flat, local_flat), dim=-1)  # Shape: [batch, 64*(32*32 + 128*128)]

        # Pass through trunk layer
        h = self.trunk(combined_features)  # Shape: [batch, feature_dim]

        # Metadata processing and fusion
        mt_vec = torch.nn.functional.one_hot(mt, num_classes=self.mt_shape[0]).float()  # Shape: [batch, mt_shape]
        if len(mt_vec.shape) == 1:
            out = torch.cat((h, mt_vec.repeat(unet_features.shape[0], 1)), dim=-1)
        else:
            out = torch.cat((h, mt_vec), dim=-1)
        value = self.critic_layer(out).squeeze()
        return value


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


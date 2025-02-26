import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    """
    A Convolutional Block with two convolution layers, instance normalization, and ReLU activation.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.inorm1 = nn.InstanceNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.inorm2 = nn.InstanceNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.relu(self.inorm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.inorm2(self.conv2(x)))
        return self.dropout(x)

class UnetModel(nn.Module):
    """
    Optimized U-Net model with flexible layers.
    """

    def __init__(self, in_chans, out_chans, chans=64, num_pool_layers=4, drop_prob=0.1):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_pool_layers = num_pool_layers

        # Down-sampling path
        self.down_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        for _ in range(num_pool_layers - 1):
            self.down_layers.append(ConvBlock(chans, chans * 2, drop_prob))
            chans *= 2

        # Bottleneck layer
        self.bottleneck = ConvBlock(chans, chans, drop_prob)

        # Up-sampling path
        self.up_layers = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_layers.append(ConvBlock(chans * 2, chans // 2, drop_prob))
            chans //= 2
        self.up_layers.append(ConvBlock(chans * 2, chans, drop_prob))

        # Final layer
        self.final_conv = nn.Conv2d(chans, out_chans, kernel_size=1)

    def forward(self, x):
        # Down-sampling path
        enc_features = []
        for layer in self.down_layers:
            x = layer(x)
            enc_features.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Up-sampling path with skip connections
        for i, layer in enumerate(self.up_layers):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self._pad_to_match(x, enc_features[-i-1])
            x = torch.cat([x, enc_features[-i-1]], dim=1)
            x = layer(x)

        return self.final_conv(x)

    def _pad_to_match(self, x, target):
        """Pad tensor x to have the same shape as target."""
        diff_y = target.size(2) - x.size(2)
        diff_x = target.size(3) - x.size(3)
        return F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

def build_reconstruction_model():
    return UnetModel(in_chans=1, out_chans=1, chans=64, num_pool_layers=4, drop_prob=0.1).cuda()

## models
import math
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


# build discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.Lrelu1 = nn.LeakyReLU(0.2)

        self.Conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.Lrelu2 = nn.LeakyReLU(0.2)

        self.Conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BatchNorm3 = nn.BatchNorm2d(128)
        self.Lrelu3 = nn.LeakyReLU(0.2)

        self.Conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.BatchNorm4 = nn.BatchNorm2d(128)
        self.Lrelu4 = nn.LeakyReLU(0.2)

        self.Conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BatchNorm5 = nn.BatchNorm2d(256)
        self.Lrelu5 = nn.LeakyReLU(0.2)

        self.Conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.BatchNorm6 = nn.BatchNorm2d(256)
        self.Lrelu6 = nn.LeakyReLU(0.2)

        self.Conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BatchNorm7 = nn.BatchNorm2d(512)
        self.Lrelu7 = nn.LeakyReLU(0.2)

        self.Conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.BatchNorm8 = nn.BatchNorm2d(512)
        self.Lrelu8 = nn.LeakyReLU(0.2)

        self.adaptivePool = nn.AdaptiveAvgPool2d(1)
        self.Conv9 = nn.Conv2d(512, 1024, kernel_size=1)
        self.Lrelu9 = nn.LeakyReLU(0.2)
        self.Conv10 = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.Lrelu1(self.Conv1(x))
        out = self.Lrelu2(self.BatchNorm2(self.Conv2(out)))
        out = self.Lrelu3(self.BatchNorm3(self.Conv3(out)))
        out = self.Lrelu4(self.BatchNorm4(self.Conv4(out)))
        out = self.Lrelu5(self.BatchNorm5(self.Conv5(out)))
        out = self.Lrelu6(self.BatchNorm6(self.Conv6(out)))
        out = self.Lrelu7(self.BatchNorm7(self.Conv7(out)))
        out = self.Lrelu8(self.BatchNorm8(self.Conv8(out)))
        out = self.Conv10(self.Lrelu9(self.Conv9(self.adaptivePool(out))))
        out.view(batch_size)
        out = torch.sigmoid(out)
        return out
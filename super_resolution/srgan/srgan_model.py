## models
import math
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super(ResidualBlock, self).__init__()
        self.con1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(c)
        self.prelu = nn.PReLU()
        self.con2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = self.con1(x)
        out = self.b1(out)
        out = self.prelu(out)
        out = self.con2(out)
        out = self.b2(out)

        return x + out


class UpsampleBLock(nn.Module):
    def __init__(self, in_c, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c * up_scale ** 2, kernel_size=3, padding=1)
        self.shufflePixel = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.shufflePixel(out)
        out = self.prelu(out)
        return out


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_number = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.b2 = ResidualBlock(64)
        self.b3 = ResidualBlock(64)
        self.b4 = ResidualBlock(64)
        self.b5 = ResidualBlock(64)
        self.b6 = ResidualBlock(64)
        self.b7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        b8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_number)]
        b8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.b8 = nn.Sequential(*b8)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        b6 = self.b6(b5)
        b7 = self.b7(b6)
        b8 = self.b8(b1 + b7)
        out = (torch.tanh(b8) + 1) / 2

        return out


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
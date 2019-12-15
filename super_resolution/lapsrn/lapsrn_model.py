import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
import torch.distributed as dist
import numpy as np
import math

'''
My contribution:
I directly used publicly available functions upsample_filt() and bilinear_upsample_weights()
    for intializing a bilinear kernel for upsampling and creating weights matrix for 
    transposed convolution
    
I basically write all the model blocks since I found the published code has something
    inconsistent with the original paper (It has an one extra conv layer in feature extraction;
    also when doing 8x SR, the level 3 feature extraction block may only have 5 conv layers).
'''

#  bilinear kernel for upsampling
# (cr: https://github.com/BUPTLdy/Pytorch-LapSRN)
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# Create weights matrix for transposed convolution
# (cr: https://github.com/BUPTLdy/Pytorch-LapSRN)
def bilinear_upsample_weights(filter_size, weights):
    f_out = weights.size(0)
    f_in = weights.size(1)
    weights = np.zeros((f_out, f_in, 4, 4), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(f_out):
        for j in range(f_in):
            weights[i, j, :, :] = upsample_kernel
    return torch.Tensor(weights)


class FEBlock(nn.Module):
    def __init__(self, levels):
        super(FEBlock, self).__init__()
        if levels == 1:
            self.convinitial = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        else:
            self.convinitial = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.convtrans_fe = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.convtrans_fe.weight.data.copy_(bilinear_upsample_weights(4, self.convtrans_fe.weight))

    def forward(self, x, levels):
        out = self.leakyrelu(self.convinitial(x))
        out = self.leakyrelu(self.conv1(out))
        out = self.leakyrelu(self.conv2(out))
        out = self.leakyrelu(self.conv3(out))
        out = self.leakyrelu(self.conv4(out))
        out = self.leakyrelu(self.conv5(out))
        if levels == 3:
            out = self.leakyrelu(self.convtrans_fe(out))
            return out
        else:
            out = self.leakyrelu(self.conv6(out))
            out = self.leakyrelu(self.conv7(out))
            out = self.leakyrelu(self.conv8(out))
            out = self.leakyrelu(self.conv9(out))
            out = self.leakyrelu(self.conv10(out))
            out = self.leakyrelu(self.convtrans_fe(out))
            return out


class IRBlock(nn.Module):
    def __init__(self):
        super(IRBlock, self).__init__()
        self.conv_ir = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.convtrans_ir = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)
        self.convtrans_ir.weight.data.copy_(bilinear_upsample_weights(4, self.convtrans_ir.weight))

    def forward(self, input, outfromconv):
        residual = self.convtrans_ir(input)
        conv_out = self.conv_ir(outfromconv)
        SR = residual + conv_out
        return SR


class LapSRN(nn.Module):
    def __init__(self):
        super(LapSRN, self).__init__()
        self.FE1 = FEBlock(levels=1)
        self.FE2 = FEBlock(levels=2)
        # self.FE3 = FEBlock(levels=3)
        self.IR1 = IRBlock()
        self.IR2 = IRBlock()
        # self.IR3 = IRBlock()

    def forward(self, LR):
        outFromConv1 = self.FE1(LR, levels=1)
        SR2x = self.IR1(LR, outFromConv1)
        outFromConv2 = self.FE2(outFromConv1, levels=2)
        SR4x = self.IR2(SR2x, outFromConv2)
        # outFromConv3 = self.FE3(outFromConv2, levels=3)
        # SR8x = self.IR3(SR4x, outFromConv3)
        return SR2x, SR4x
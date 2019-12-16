from __future__ import print_function
import os
from os.path import exists, join, basename
from os import makedirs, remove
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
import numpy as np

test_folder = "project_image/SR_testing_datasets/Set14"
if not exists ("outputs"):
    makedirs ("outputs")
save_folder = "outputs"

def cropImage(img, times): 
    width, height = img.size 
    subset_width = width - (width % times)
    subset_height = height - (height % times)
    width_diff = width - subset_width
    height_diff = height - subset_height
    left = width_diff / 2
    right = width - (width_diff / 2)
    up =  height_diff / 2
    down = height - (height_diff / 2)
    out_img = img.crop((left, up, right, down))
    return out_img

def YCbCr2RGB(sr, cb, cr):
    sr_img_y = sr.data[0].numpy()
    sr_img_y = sr_img_y * 255.0
    sr_img_y = sr_img_y.clip(0, 255)
    sr_img_y = np.uint8(sr_img_y[0])
    sr_img_y = Image.fromarray(sr_img_y, mode='L')
    sr_img_cb = cb.resize(sr_img_y.size, Image.BICUBIC)
    sr_img_cr = cr.resize(sr_img_y.size, Image.BICUBIC)
    sr_img_merge = Image.merge('YCbCr', [sr_img_y, sr_img_cb, sr_img_cr])
    sr_img_merge_rgb = sr_img_merge.convert('RGB')
    return sr_img_merge_rgb
    
img_paths = glob(test_folder + '/*.png')
for epoch in range(0, 501, 50):
    model = torch.load('models/model_epoch_' + str(epoch) + '.pth').cuda()
    for img_path in img_paths:
        img_name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path)
        img = cropImage(img.convert('YCbCr'), 4)
        y, cb, cr = img.split()
        LR = y.resize((int(y.size[0] / 4), int(y.size[1] / 4)), Image.BICUBIC)
        LR = Variable(ToTensor()(LR)).view(1, -1, LR.size[1], LR.size[0])
        LR = LR.cuda()
        SR2x, SR4x = model(LR)
        SR2x = SR2x.cpu()
        SR4x = SR4x.cpu()
        # SR8x = SR8x.cpu()
        SR2x = YCbCr2RGB(SR2x, cb, cr)
        SR4x = YCbCr2RGB(SR4x, cb, cr)
        # SR8x = YCbCr2RGB(SR8x, cb, cr)
        img = img.convert('RGB')
        SR4x.save(save_folder + '/' +  img_name + '_' + str(epoch) + '_sr4.png')
        if epoch == 0:
            y = y.resize((int(y.size[0] / 4), int(y.size[1] / 4)), Image.BICUBIC)
            y = y.resize((int(y.size[0] * 4), int(y.size[1] * 4)), Image.BICUBIC)
            y = Image.merge('YCbCr', [y, cb, cr]).convert('RGB')
            y.save(save_folder + '/' +  img_name + '_' + str(epoch) + '_lr.png')
            # img = Image.open(image_path)
            # img = img.resize((int(img.size[0] / 4), int(img.size[1] / 4)), Image.BICUBIC)
            # img = img.resize((int(img.size[0] * 4), int(img.size[1] * 4)), Image.BICUBIC)
            # img.save(save_folder + '/' +  img_name + '_' + str(epoch) + '_lr.png')

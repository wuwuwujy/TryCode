from __future__ import print_function
from os.path import exists, join, basename
from os import makedirs, remove
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

'''
I used the centerCrop function from published code. 
'''

test_folder = "project_image/SR_testing_datasets/Set14"
if not exists ("outputs"):
    makedirs ("outputs")
save_folder = "outputs"

def centeredCrop(img): # cr: https://github.com/BUPTLdy/Pytorch-LapSRN/
    width, height = img.size  # Get dimensions
    new_width = width - width % 4
    new_height = height - height % 4
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))

def YCbCr2RGB(sr, cb, cr):
    sr_img_y = sr.data[0].numpy()
    sr_img_y = sr_img_y * 255.0
    sr_img_y = sr_img_y.clip(0, 255)
    sr_img_y = Image.fromarray(np.uint8(sr_img_y[0]), mode='L')
    sr_img_cb = cb.resize(sr_img_y.size, Image.BICUBIC)
    sr_img_cr = cr.resize(sr_img_y.size, Image.BICUBIC)
    sr_img_merge = Image.merge('YCbCr', [sr_img_y, sr_img_cb, sr_img_cr])
    return sr_img_merge.convert('RGB')
    
img_paths = glob(test_folder + '/*.png')
for epoch in range(0, 500, 50):
    model = torch.load('checkpoint_folder/model_epoch_' + str(epoch) + '.pth').cuda()
    for img_path in img_paths:
        img_name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path)
        img = centeredCrop(img.convert('YCbCr'))
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
        # save_image(SR2x, SR4x, SR8x, img, img_name)
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

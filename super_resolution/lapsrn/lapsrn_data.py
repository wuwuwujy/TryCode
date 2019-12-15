import os
from os import listdir, makedirs, remove
import torch.utils.data as data
import torchvision
from os.path import join, exists, join, basename
from PIL import Image
import numpy as np
import random
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Resize, RandomResizedCrop, ColorJitter, RandomAffine, RandomGrayscale

'''
Code referenced to https://github.com/BUPTLdy/Pytorch-LapSRN
What I modify: 
load_img(), DatasetFromFolder()

What I re-write: 
SR4x_transform() - add multiple image augmentation
get_training_set() - provide multiple choices for different datasets
get_testing_set() - provide multiple choices for different datasets

What I add: 
SR4x_transform_4test - just used centercrop as what the original paper did
'''

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, cb, cr = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, LR_transform=None, SR2x_transform=None,
                 SR4x_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)]
        self.LR_transform = LR_transform
        self.SR2x_transform = SR2x_transform
        self.SR4x_transform = SR4x_transform
        # self.SR8x_transform = SR8x_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        # SR8x = self.SR8x_transform(input)
        SR4x = self.SR4x_transform(input)
        SR2x = self.SR2x_transform(SR4x)
        LR = self.LR_transform(SR4x)
        to_tensor = torchvision.transforms.ToTensor()
        SR4x = to_tensor(SR4x)
        return LR, SR2x, SR4x

    def __len__(self):
        return len(self.image_filenames)


crop_size = 128


def LR_transform(crop_size):
    return Compose([
        Resize(crop_size // 4),
        ToTensor(),
    ])


def SR2x_transform(crop_size):
    return Compose([
        Resize(crop_size // 2),
        ToTensor(),
    ])

def SR4x_transform(crop_size):
    try:
        ts = Compose([
            RandomCrop((crop_size, crop_size)),#, pad_if_needed=False, padding_mode='reflect'),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(hue=.05, saturation=.05),
            RandomAffine(30),
            RandomGrayscale(),
        ])
    except:
        ts = Compose([
            Resize(crop_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(hue=.05, saturation=.05),
            RandomAffine(30),
            RandomGrayscale(),
        ])
    return ts
    
def SR4x_transform_4test(crop_size):
        return Compose([
        CenterCrop(crop_size),
        RandomHorizontalFlip(),
    ])
    



def get_training_set(Set_train):
    if Set_train == "T91":
        folder_path = "project_image/SR_training_datasets/T91"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(crop_size),
                                 SR2x_transform=SR2x_transform(crop_size),
                                 SR4x_transform=SR4x_transform(crop_size))

    elif Set_train == "General100":
        folder_path = "project_image/SR_training_datasets/General100"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(crop_size),
                                 SR2x_transform=SR2x_transform(crop_size),
                                 SR4x_transform=SR4x_transform(crop_size))
    elif Set_train == "BSDS200":
        folder_path = "project_image/SR_training_datasets/BSDS200"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(crop_size),
                                 SR2x_transform=SR2x_transform(crop_size),
                                 SR4x_transform=SR4x_transform(crop_size))
    else:
        print("Train folder not found")


def get_test_set(Set):
    if Set == 5:
        folder_path = "project_image/SR_testing_datasets/Set5"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(crop_size),
                                 SR2x_transform=SR2x_transform(crop_size),
                                 SR4x_transform=SR4x_transform_4test(crop_size))
    elif Set == 14:
        folder_path = "project_image/SR_testing_datasets/Set14"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(crop_size),
                                 SR2x_transform=SR2x_transform(crop_size),
                                 SR4x_transform=SR4x_transform_4test(crop_size))
    elif Set == 109:
        folder_path = "project_image/SR_testing_datasets/Manga109"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(crop_size),
                                 SR2x_transform=SR2x_transform(crop_size),
                                 SR4x_transform=SR4x_transform_4test(crop_size))

    elif Set == "BSDS100":
        folder_path = "project_image/SR_testing_datasets/BSDS100"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(crop_size),
                                 SR2x_transform=SR2x_transform(crop_size),
                                 SR4x_transform=SR4x_transform_4test(crop_size))
    elif Set == "Urban100":
        folder_path = "project_image/SR_testing_datasets/Urban100"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(crop_size),
                                 SR2x_transform=SR2x_transform(crop_size),
                                 SR4x_transform=SR4x_transform_4test(crop_size))

    elif Set == "historical":
        folder_path = "project_image/SR_testing_datasets/historical"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(crop_size),
                                 SR2x_transform=SR2x_transform(crop_size),
                                 SR4x_transform=SR4x_transform_4test(crop_size))
    else:
        print("Folder not found")
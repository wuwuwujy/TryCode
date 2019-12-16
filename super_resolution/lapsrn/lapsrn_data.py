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

def open_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, cb, cr = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, folder_path, LR_transform=None, SR2x_transform=None,
                 SR4x_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.img_names = [join(folder_path, names) for names in listdir(folder_path)]
        self.LR_transform = LR_transform
        self.SR2x_transform = SR2x_transform
        self.SR4x_transform = SR4x_transform
        # self.SR8x_transform = SR8x_transform

    def __getitem__(self, index):
        input = open_img(self.img_names[index])
        # SR8x = self.SR8x_transform(input)
        SR4x = self.SR4x_transform(input)
        SR2x = self.SR2x_transform(SR4x)
        LR = self.LR_transform(SR4x)
        SR4x = torchvision.transforms.ToTensor(SR4x)
        return LR, SR2x, SR4x

    def __len__(self):
        return len(self.img_names)


subset = 128


def LR_transform(subset):
    ts = Compose([
        Resize(subset // 4),
        ToTensor(),
    ])
    return ts


def SR2x_transform(subset):
    ts = Compose([
        Resize(subset // 2),
        ToTensor(),
    ])
    return ts

def SR4x_transform(subset):
    try:
        ts = Compose([
            RandomCrop((subset, subset)),#, pad_if_needed=False, padding_mode='reflect'),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(hue=.05, saturation=.05),
            RandomAffine(30),
            RandomGrayscale(),
        ])
    except:
        ts = Compose([
            Resize(subset),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(hue=.05, saturation=.05),
            RandomAffine(30),
            RandomGrayscale(),
        ])
    return ts
    
def SR4x_transform_4test(subset):
    ts = Compose([
        CenterCrop(subset),
        RandomHorizontalFlip(),
    ])
    return ts
    



def get_training_set(Set_train):
    if Set_train == "T91":
        folder_path = "project_image/SR_training_datasets/T91"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(subset),
                                 SR2x_transform=SR2x_transform(subset),
                                 SR4x_transform=SR4x_transform(subset))

    elif Set_train == "General100":
        folder_path = "project_image/SR_training_datasets/General100"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(subset),
                                 SR2x_transform=SR2x_transform(subset),
                                 SR4x_transform=SR4x_transform(subset))
    elif Set_train == "BSDS200":
        folder_path = "project_image/SR_training_datasets/BSDS200"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(subset),
                                 SR2x_transform=SR2x_transform(subset),
                                 SR4x_transform=SR4x_transform(subset))
    else:
        print("Train folder not found")


def get_test_set(Set):
    if Set == 5:
        folder_path = "project_image/SR_testing_datasets/Set5"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(subset),
                                 SR2x_transform=SR2x_transform(subset),
                                 SR4x_transform=SR4x_transform_4test(subset))
    elif Set == 14:
        folder_path = "project_image/SR_testing_datasets/Set14"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(subset),
                                 SR2x_transform=SR2x_transform(subset),
                                 SR4x_transform=SR4x_transform_4test(subset))
    elif Set == 109:
        folder_path = "project_image/SR_testing_datasets/Manga109"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(subset),
                                 SR2x_transform=SR2x_transform(subset),
                                 SR4x_transform=SR4x_transform_4test(subset))

    elif Set == "BSDS100":
        folder_path = "project_image/SR_testing_datasets/BSDS100"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(subset),
                                 SR2x_transform=SR2x_transform(subset),
                                 SR4x_transform=SR4x_transform_4test(subset))
    elif Set == "Urban100":
        folder_path = "project_image/SR_testing_datasets/Urban100"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(subset),
                                 SR2x_transform=SR2x_transform(subset),
                                 SR4x_transform=SR4x_transform_4test(subset))

    elif Set == "historical":
        folder_path = "project_image/SR_testing_datasets/historical"
        return DatasetFromFolder(folder_path,
                                 LR_transform=LR_transform(subset),
                                 SR2x_transform=SR2x_transform(subset),
                                 SR4x_transform=SR4x_transform_4test(subset))
    else:
        print("Folder not found")

import os
from os import listdir, makedirs, remove
from os.path import exists, join, basename
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import time
from PIL import Image
import numpy as np
import random
from six.moves import urllib
import tarfile
from math import log10
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader
from project_model import LapSRN
from project_data import get_training_set, get_test_set
from glob import glob
import numpy as np

'''
This file is I created based on the lapsrn_main.py for testing models on 
    other datasets. E.g. we may use set5 as the testing set during training
    and later we want to test performance on set14.
'''

batch_size = 64
test_batch_size = 1
num_epochs = 50
lr = 1e-5
num_workers = 4

test_set = get_test_set(5)  # Set = 5, 14, 109, BSDS100, Urban100, historical
testing_data_loader = DataLoader(dataset=test_set, num_workers=num_workers, batch_size=test_batch_size, shuffle=False)

criterion = nn.MSELoss()
criterion = criterion.cuda()
# optimizer = optim.Adam(model.parameters(), weight_decay = 1e-5)

for epoch in range(0, 500, 50):
    # start_time = time.time()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    model = torch.load('checkpoint_folder/model_epoch_' + str(epoch) + '.pth').cuda()
    epoch_psnr2x = 0
    epoch_psnr4x = 0
    # epoch_psnr8x = 0
    for batch in testing_data_loader:
        LR, SR2x_target, SR4x_target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        LR = LR.cuda()
        SR2x_target = SR2x_target.cuda()
        SR4x_target = SR4x_target.cuda()
        # SR8x_target = SR8x_target.cuda()

        SR2x, SR4x = model(LR)
        mse2x = criterion(SR2x, SR2x_target)
        mse4x = criterion(SR4x, SR4x_target)
        # mse8x = criterion(SR8x, SR8x_target)
        psnr2x = 10 * log10(1 / mse2x.item())
        psnr4x = 10 * log10(1 / mse4x.item())
        # psnr8x = 10 * log10(1 / mse8x.item())
        epoch_psnr2x = epoch_psnr2x + psnr2x
        epoch_psnr4x = epoch_psnr4x + psnr4x
        # epoch_psnr8x = epoch_psnr8x + psnr8x
    print("Epoch {}, Avg. SR4x_PSNR: {:.4f} dB".format(epoch, epoch_psnr4x / len(testing_data_loader)))
    # end_time = time.time()
    # print(end_time - start_time)
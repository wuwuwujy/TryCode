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
from six.moves import urllib
import tarfile
from math import log10
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader
from lapsrn_model import LapSRN
from lapsrn_data import get_training_set, get_test_set
from glob import glob

batch_size = 64
test_batch_size = 1
num_epochs = 2500
lr = 1e-5
num_workers = 4

train_set = get_training_set("BSDS200")
test_set = get_test_set(5)  # Set = 5, 14, 109, BSDS100, Urban100, historical
training_data_loader = DataLoader(dataset = train_set, num_workers = num_workers, batch_size = batch_size, shuffle = True)
testing_data_loader = DataLoader(dataset = test_set, num_workers = num_workers, batch_size = test_batch_size, shuffle = False)
print(len(training_data_loader))

def Charbonnier(predict, target): 
    loss = torch.sqrt(torch.pow((predict - target), 2) + 1e-6) # set epsilon=1e-3
    mean_loss = torch.mean(loss)
    return mean_loss  

def getPSNR(mse):
    PSNR = 10 * log10(1 / mse2x.item())
    return PSNR

    
model = LapSRN()
getMSE = nn.MSELoss()
model = model.cuda()
getMSE = getMSE.cuda()
optimizer = optim.Adam(model.parameters(), weight_decay = 1e-5)

if not exists("models"):
        makedirs("models")
        
for epoch in range(num_epochs):
    start_time = time.time()
    # training
    for i in range(5):
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            LR = Variable(batch[0])
            SR2x_target  = Variable(batch[1])
            SR4x_target = Variable(batch[2])
            # SR8x_target = Variable(batch[3])
            
            LR = LR.cuda()
            SR2x_target = SR2x_target.cuda()
            SR4x_target = SR4x_target.cuda()
            # SR8x_target = SR8x_target.cuda()

            optimizer.zero_grad()
            SR2x, SR4x = model(LR)

            loss2x = Charbonnier(SR2x, SR2x_target)
            loss4x = Charbonnier(SR4x, SR4x_target)
            # loss8x = Charbonnier(SR8x, SR8x_target)
            loss = loss2x + loss4x # + loss8x
            
            if loss.data.item() < 10:
                epoch_loss += loss.data.item()
            else:
                epoch_loss += 10
                
            loss.backward()
            if(epoch>0):
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if 'step' in state.keys():
                            if(state['step']>=1024):
                                state['step'] = 1000
            optimizer.step()
        print("Epoch {}, Loop{} Loss {:.4f}".format(epoch, i, epoch_loss / len(training_data_loader)))
        
    # test
    epoch_psnr2x = 0
    epoch_psnr4x = 0
    # epoch_psnr8x = 0
    for batch in testing_data_loader:
        LR = Variable(batch[0])
        SR2x_target  = Variable(batch[1])
        SR4x_target = Variable(batch[2])
        # SR8x_target = Variable(batch[3])
        
        LR = LR.cuda()
        SR2x_target = SR2x_target.cuda()
        SR4x_target = SR4x_target.cuda()
        # SR8x_target = SR8x_target.cuda()

        SR2x, SR4x = model(LR)
        mse2x = getMSE(SR2x, SR2x_target)
        mse4x = getMSE(SR4x, SR4x_target)
        # mse8x = getMSE(SR8x, SR8x_target)
        
        psnr2x = getPSNR(mse2x)
        psnr4x = getPSNR(mse4x)
        # psnr8x = getPSNR(mse8x)
        
        epoch_psnr2x = epoch_psnr2x + psnr2x
        epoch_psnr4x = epoch_psnr4x + psnr4x
        # epoch_psnr8x = epoch_psnr8x + psnr8x
    print("Epoch {}, Avg. SR4x_PSNR: {:.4f}".format(epoch, epoch_psnr4x / len(testing_data_loader)))
    
    if epoch % 50 == 0:
        model_save_path = "models/model_epoch_{}.pth".format(epoch)
        torch.save(model, model_save_path)
    end_time = time.time()
    print(end_time - start_time)
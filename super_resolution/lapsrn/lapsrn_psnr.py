import os
from os import listdir, makedirs, remove
from os.path import exists, join, basename
from PIL import Image
import numpy as np
from math import log10

img_names = ["butterfly", "barbara"]
set_names = ["Set5", "Set14"]

def getPSNR(img1, img2):
    dif = (np.array(img1, dtype = np.float32) - np.array(img2, dtype = np.float32)) 
    dif = dif * dif
    psnr = 10 * log10( 255*255/np.mean( dif ))
    return psnr

all_psnr = []    

for i in range(2):
    all_psnr.append(img_names[i])
    print(img_names[i])
    original_img_path = "project_image/SR_testing_datasets/" + set_names[i] + "/" + img_names[i] + ".png"
    original_img = Image.open(original_img_path)
    bicubic_img_path = "outputs/" + img_names[i] + "_0_lr.png"
    bicubic_img = Image.open(bicubic_img_path)
    all_psnr.append(getPSNR(original_img, bicubic_img))
    print(getPSNR(original_img, bicubic_img))
    
    for epoch in range(0, 500, 50):
        epoch_img_path = "outputs/" + img_names[i] + "_" + str(epoch) + "_sr4.png"
        epoch_img = Image.open(epoch_img_path)
        all_psnr.append(getPSNR(original_img, epoch_img))
        print(getPSNR(original_img, epoch_img))

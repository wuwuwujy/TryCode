## test and visulization
import os
from math import log10

import numpy as np
import pandas as pd
import torch
# import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import pytorch_ssim
from srgan_data import Test_Dataset
from srgan_model import Generator, Discriminator

upscale_factor = 4
v_set = 14 #or 5
t_set = 5 #or 14
IMAGE_NAME = "Set5_003.png" #or Set14_002.png
UPSCALE_FACTOR = upscale_factor
test_data_dir = "image_srgan/SRF_4/Set" + str(t_set)
image_path = test_data_dir + '/data/' + IMAGE_NAME
test_set = Test_Dataset(test_data_dir, upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

out_path = "images_set"+str(t_set)
if not os.path.exists(out_path):
  os.makedirs(out_path)

model_path = "epochs_" + str(v_set)


for i in range(1,41):
  if v_set==5:
    j = i*25
  if v_set==14:
    j = i*50
  MODEL_NAME = 'netG_epoch_4_'+str(j)+'.pth'
  model = Generator(UPSCALE_FACTOR).eval()
  if torch.cuda.is_available():
    model = model.cuda()
  model.load_state_dict(torch.load(model_path + "/" + MODEL_NAME))
  #output image for each epoch
  image = Image.open(image_path)
  image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
  image = image.cuda()
  out = model(image)
  out_img = ToPILImage()(out[0].data.cpu())
  out_img.save(out_path + "/" + 'out' + str(j) + '_' + IMAGE_NAME)

  for image_name, lr_image, hr_restore_img, hr_image in test_loader:
    image_name = image_name[0]
    lr_image = Variable(lr_image, volatile=True)
    hr_image = Variable(hr_image, volatile=True)
    if torch.cuda.is_available():
      lr_image = lr_image.cuda()
      hr_image = hr_image.cuda()

    sr_image = model(lr_image)
    mse = ((hr_image - sr_image) ** 2).data.mean()
    psnr = 10 * log10(1 / mse)
    ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
    print("epoch:{} image:{} psnr:{:.4f} ssim:{:.4f}".format(j,image_name,psnr,ssim))
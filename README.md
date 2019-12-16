## [Super-Resolution](https://github.com/wuwuwujy/TryCode/tree/master/super_resolution)
1. [LapSRN](https://github.com/wuwuwujy/TryCode/tree/master/super_resolution/lapsrn)
   - Put all scripts along with the folder `project_image` in the same directory
   - Run [`lapsrn_main.py`](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/lapsrn/lapsrn_main.py) for training. Test dataset was set to Set5 by default.
   - Run [`lapsrn_test.py`](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/lapsrn/lapsrn_test.py) to test PSNR on Set14 by default.
   - Run [`lapsrn_visualization.py`](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/lapsrn/lapsrn_visualization.py) to get super-resolution images during the training process.
   - Run [`lapsrn_psnr.py`](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/lapsrn/lapsrn_psnr.py) to get PSNR values for super-resolution images during the training process.
   The scripts are for 4x super-resolution, for 8x script see `LapSRN_8x.zip`
2. SRGAN
   - Before running [`srgan_train.py`](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/srgan/srgan_train.py) file, specify validation dataset (v_set) and test dataset (t_set) by typing 5 or 14.
   - After running srgan_strain, use [`srgan_test.py`](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/srgan/srgan_test.py) to get psnr and ssim for testing dataset. In this step, specify v_set, t_set and IMAGE_NAME for choosing validation dataset, test dataset and the image want to output. Please change the path if you want to test other datasets.
3. SkipNet
   -The skip net based super resolution is written in jupyter notebook
   -Run super_resolution/sr-skipnet cells to run skip net image super resolutoin algorithm

## Inpainting
1. [ResNet](https://github.com/wuwuwujy/TryCode/blob/master/inpainting_with_resnet.ipynb)
   
   Open the ipython file and run every cell, it will load the libraries, define the model, load the images, and start to train on this image. You can load the images you like by modify the image path. There are two different ways that can choose, one is to add an existing mask with English words, the other is inpainting 50% of missing pixels.
   
 ## Denoising
 1. Skip Net. The skip net image denoising is written in jupyter notebook denoising.ipynb. Open the ipython file and run every cell, it will load the libraries, define the model, load the images, and start to train on this image. You can load the images you like by modify the image path.
 
   

   

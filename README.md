## [Denoising](https://github.com/wuwuwujy/TryCode/blob/master/denoising_project_script.py)

Running Denoising Script:

The file denoising_project_script.py is the file used to run the code. 
It is already set up to automatically loop through all of the images created in the paper.
It is only dependent on the "utils" folder and the "data" folder, in which the input images are kept. 
The script will create an output folder for the developed images.

## Super-Resolution
1. LapSRN
   1. Put all scripts along with the folder `project_image` in the same directory
   2. Run [lapsrn_main.py](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/lapsrn/lapsrn_main.py) for training. Test dataset was set to Set5 by default.
   3. Run [lapsrn_test.py](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/lapsrn/lapsrn_test.py) to test PSNR on Set14 by default.
   4. Run [lapsrn_visualization.py](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/lapsrn/lapsrn_visualization.py) to get super-resolution images during the training process.
   5. Run [lapsrn_psnr.py](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/lapsrn/lapsrn_psnr.py) to get PSNR values for super-resolution images during the training process.
   The scripts are for 4x super-resolution, for 8x script see `LapSRN_8x.zip`
2. SRGAN
   1. Before running [srgan_train](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/srgan/srgan_train.py) file, please specify validation dataset (v_set) and test dataset (t_set) by typing 5 or 14.
   2. After running srgan_strain, use [srgan_test](https://github.com/wuwuwujy/TryCode/blob/master/super_resolution/srgan/srgan_test.py) to get psnr and ssim for testing dataset. In this step, please specify v_set, t_set and IMAGE_NAME. Please change the path if you want to test other datasets.
3. DeepPrior

## Inpainting
1. [ResNet](https://github.com/wuwuwujy/TryCode/blob/master/inpainting_with_resnet.ipynb)
   
   Open the ipython file and run every cell, it will load the libraries, define the model, load the images, and start to train on this image. You can load the images you like by modify the image path. There are two different ways that can choose, one is to add an existing mask with English words, the other is inpainting 50% of missing pixels.
   

   

## [Denoising](https://github.com/wuwuwujy/TryCode/blob/master/denoising_project_script.py)

Running Denoising Script:

The file denoising_project_script.py is the file used to run the code. 
It is already set up to automatically loop through all of the images created in the paper.
It is only dependent on the "utils" folder and the "data" folder, in which the input images are kept. 
The script will create an output folder for the developed images.

## Super-Resolution
1. LapSRN
2. SRGAN
   1. Before running [srgan_train](https://github.com/wuwuwujy/TryCode/blob/master/srgan_train.py) file, please specify validation dataset (v_set) and test dataset (t_set) by typing 5 or 14.
   2. After running srgan_strain, use [srgan_test](https://github.com/wuwuwujy/TryCode/blob/master/srgan_test.py) to get psnr and ssim for testing dataset. In this step, please specify v_set and t_set, and IMAGE_NAME.
   

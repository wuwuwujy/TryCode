from __future__ import print_function
import matplotlib.pyplot as plt

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
#from models import *

import torch
import torch.optim
import time
from skimage.measure import compare_psnr
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
sigma = 25
sigma_ = sigma/255.

####################################################
# Filepath to denoising images
path = 'data/denoising'

# List of image names in filepath

fname_list = ['justin.png','snail.jpg', 'F16_GT.png', 'peppers.png', 'lena.png','baboon.png','house.png', 'kodak1.png', 'kodak3.png', 'kodak12.png']

#####################################################


def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos', show=False):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    if show:
        plt.show()
    
    return grid

 
def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module


class Concat(nn.Module):
    def __init__(self, dim, skip, deeper):
        super(Concat, self).__init__()
        self.dim = dim
        self.layer1 = skip
        self.layer2 = deeper
    def forward(self, input):
        inputs = []
        inputs.append(self.layer1(input))
        inputs.append(self.layer2(input))
        return torch.cat(inputs, dim=self.dim)

def act(activation_method = 'LeakyReLU'):
    if activation_method=='ReLU':
        return nn.ReLU()
    elif activation_method =='LeakyReLU':
        return nn.LeakyReLU()
    elif activation_method=='sigmoid':
        return nn.Sigmoid()
    else:
        return nn.Tanh()

def conv(in_f, out_f, kernel_size, stride=1): 
    res = []
    res.append(nn.ReflectionPad2d(int((kernel_size - 1) / 2)))
    res.append(nn.Conv2d(in_f, out_f, kernel_size, stride, padding=0))
    return nn.Sequential(*res)


def skip(c_in, c_out, c_down, c_up, c_skip, k_down, k_up, k_skip, upsample_mode, act_fun, ):
    model = nn.Sequential()
    model_tmp = model
    input_depth = c_in
    for i in range(len(c_down)):
        temp_layers = []
        temp_layers.append(nn.ReflectionPad2d(int((k_down - 1) / 2)))
        temp_layers.append(nn.Conv2d(input_depth, c_down[i], k_down, 2))
        temp_layers.append(nn.BatchNorm2d(c_down[i]))
        temp_layers.append(nn.LeakyReLU())
        
        temp_layers.append(nn.ReflectionPad2d(int((k_down - 1) / 2)))
        temp_layers.append(nn.Conv2d(c_down[i], c_down[i], k_down, 1))
        temp_layers.append(nn.BatchNorm2d(c_down[i]))
        temp_layers.append(nn.LeakyReLU())
        deeper_main = nn.Sequential()
        
        if i < len(c_down)-1:
            temp_layers.append(deeper_main)
            temp_layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
            if c_skip[i] != 0:
                skip_layers = []
                skip_layers.append(conv(input_depth, c_skip[i], k_skip))
                skip_layers.append(nn.BatchNorm2d(c_skip[i]))
                skip_layers.append(act(act_fun))
                model_tmp.add(Concat(1, nn.Sequential(*skip_layers), nn.Sequential(*temp_layers)))
            else:
                model_tmp.add(nn.Sequential(*temp_layers))
            model_tmp.add(nn.BatchNorm2d(c_skip[i] + c_up[i + 1] ))
            model_tmp.add(conv(c_skip[i] + c_up[i + 1], c_up[i], k_up, 1))
            
        else:#last layer
            temp_layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
            if c_skip[i] != 0:
                skip_layers = []
                skip_layers.append(conv(input_depth, c_skip[i], k_skip))
                skip_layers.append(nn.BatchNorm2d(c_skip[i]))
                skip_layers.append(act(act_fun))
                model_tmp.add(Concat(1, nn.Sequential(*skip_layers), nn.Sequential(*temp_layers)))
            else:
                model_tmp.add(nn.Sequential(*temp_layers))
            model_tmp.add(nn.BatchNorm2d(c_skip[i] +c_down[i]))
            model_tmp.add(conv(c_skip[i] + c_down[i], c_up[i], k_up, 1))

        model_tmp.add(nn.BatchNorm2d(c_up[i]))
        model_tmp.add(act(act_fun))
        model_tmp.add(conv(c_up[i], c_up[i], 1))
        model_tmp.add(nn.BatchNorm2d(c_up[i]))
        model_tmp.add(act(act_fun))
        input_depth = c_down[i]
        model_tmp = deeper_main

    model.add(conv(c_up[0], c_out, 1))
    model.add(nn.ReLU())

    return model





if not os.path.exists('outputs'):
    os.makedirs('outputs')

if not os.path.exists('outputs/denoising'):
    os.makedirs('outputs/denoising')
start = time.time()
for img_name in fname_list:
    print("Beginning loop for", img_name)
    fname = path + '/' + img_name
    if img_name == 'snail.jpg':
        img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_noisy_np = pil_to_np(img_noisy_pil)
        # As we don't have ground truth
        img_pil = img_noisy_pil
        img_np = img_noisy_np
        
        
        img_out = plot_image_grid([img_np], 4, 5, show=False);
        img_out =  np.moveaxis(img_out, 0, -1)
        img_out =(img_out*255).astype(np.uint8)
        im = Image.fromarray(img_out, 'RGB')
        im.save("outputs/denoising/" + str(img_name)[:-4] + '_input.png')
        plt.close()

    else:
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        img_np = img_np[:3, :, :]
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
        

        img_out = plot_image_grid([img_np, img_noisy_np], 4, 6);
        img_out =  np.moveaxis(img_out, 0, -1)
        img_out =(img_out*255).astype(np.uint8)
        im = Image.fromarray(img_out, 'RGB')
        im.save("outputs/denoising/" + str(img_name)[:-4] + '_input.png')
        plt.close()


    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    reg_noise_std = 1./30. # set to 1./20. for sigma=50
    LR = 0.01

    OPTIMIZER='adam' # 'LBFGS'
    show_every = 100
    exp_weight=0.99

    if fname == 'data/denoising/snail.jpg':
        num_iter = 2400
        input_depth = 3
        figsize = 5 
        
        net = skip(
                    input_depth, 3, 
                    c_down = [8, 16, 32, 64, 128], 
                    c_up   = [8, 16, 32, 64, 128],
                    c_skip = [0, 0, 0, 4, 4], 
                    k_down = 3,
                    k_up = 3,
                    k_skip = 1,
                    upsample_mode='bilinear',
                    act_fun='LeakyReLU')

        net = net.type(dtype)

    elif (fname == 'data/denoising/F16_GT.png'):
        num_iter = 5000
        input_depth = 32 
        figsize = 4 
        
        net = skip(
            input_depth, 3, 
            c_down =  [128, 128, 128, 128, 128], 
            c_up   = [128, 128, 128, 128, 128],
            c_skip = [0, 0, 0, 4, 4], 
            k_down = 3,
            k_up = 3,
            k_skip = 1,
            upsample_mode='bilinear',
            act_fun='LeakyReLU')

        net = net.type(dtype)

    elif (fname == 'data/denoising/kodak1.png') or (fname == 'data/denoising/kodak3.png') or (fname == 'data/denoising/kodak12.png'):
        num_iter = 5000
        input_depth = 32 
        figsize = 4 
        
        net = skip(
            input_depth, 3, 
            c_down =  [128, 128, 128, 128, 128], 
            c_up   = [128, 128, 128, 128, 128],
            c_skip = [0, 128, 128, 128, 128], 
            k_down = 3,
            k_up = 3,
            k_skip = 1,
            upsample_mode='bilinear',
            act_fun='LeakyReLU')

        net = net.type(dtype)


    else:
        num_iter = 5000
        input_depth = 3
        figsize = 5 
        net = skip(
            input_depth, 3, 
            c_down =  [128, 128, 128, 128, 128], 
            c_up   = [128, 128, 128, 128, 128],
            #c_skip = [0, 0, 0, 4, 4], 
            c_skip = [128, 128, 128, 128, 128],
            k_down = 3,
            k_up = 3,
            k_skip = 1,
            upsample_mode='bilinear',
            act_fun='LeakyReLU')

        net = net.type(dtype)
        
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0

    if not os.path.exists('outputs/denoising/' + str(img_name)[:-4]):
        os.makedirs('outputs/denoising/' + str(img_name)[:-4])

    f= open('outputs/denoising/' + str(img_name)[:-4] + "/log.txt","w+")

    i = 0
    def closure():
        
        global i, out_avg, psrn_noisy_last, last_net, net_input
        
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        net.cuda()
        out = net(net_input)
        
        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
                
        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()
            
        
        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
        
        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        f.write('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f \n' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm))


        if  i % show_every == 0:
            out_np = torch_to_np(out)
            img_out = plot_image_grid([np.clip(out_np, 0, 1), np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1, show=False);
            img_out =  np.moveaxis(img_out, 0, -1)
            img_out =(img_out*255).astype(np.uint8)
            im = Image.fromarray(img_out, 'RGB')
            im.save("outputs/denoising/" + str(img_name)[:-4] + '/' + str(img_name)[:-4] + '-' + str(i) +'.png')
            plt.close()
            
        
        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5: 
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy
                
        i += 1

        return total_loss


    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)
    f.close()

    out_np = torch_to_np(net(net_input).cuda())
    img_out = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);
    img_out =  np.moveaxis(img_out, 0, -1)
    img_out =(img_out*255).astype(np.uint8)
    im = Image.fromarray(img_out, 'RGB')
    im.save("outputs/denoising/" + str(img_name)[:-4] + '/' + str(img_name)[:-4] + '-' + '_final.png')
    plt.close()
end = time.time()
hrs_elapsed = (end - start)/3600

print("Hours elapsed:", hrs_elapsed)
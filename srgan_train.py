# Train
import os
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_ssim

from srgan_data import Train_Dataset, Val_Dataset, display_transform
from srgan_loss import GeneratorLoss
from srgan_model import Generator, Discriminator

crop_size = 88
upscale_factor = 4
v_set = 14  # choose set5 or set14 as validation set
if v_set == 5:
    batch_Size = 5
    num_epochs = 1000
    v = 25
if v_set == 14:
    batch_Size = 12 # exclude 3 and 6 because they are black and white
    num_epochs = 2000
    v = 50

train_data_dir = "project_image/SR_training_datasets/BSDS200"
val_data_dir = "image_srgan/SRF_4/Set" + str(v_set) + "/target"

epoch_path = "epochs" + "_" + str(v_set)
if not os.path.exists(epoch_path):
    os.makedirs(epoch_path)

if __name__ == '__main__':
    train_set = Train_Dataset(train_data_dir, crop_size=crop_size, upscale_factor=upscale_factor)
    val_set = Val_Dataset(val_data_dir, upscale_factor=upscale_factor)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    G = Generator(upscale_factor)
    D = Discriminator()
    G_criterion = GeneratorLoss().cuda()
    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
    G_optimizer = optim.Adam(G.parameters())
    D_optimizer = optim.Adam(D.parameters())
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, num_epochs + 1):
        running_results = {'batch_sizes': 0, 'd_loss': 0.0, 'g_loss': 0.0, 'd_score': 0.0, 'g_score': 0.0}

        G.train()
        D.train()
        for data, target in train_loader:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            # update D: maximize d loss
            real_img = Variable(target).cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = G(z)

            D.zero_grad()
            real_out = D(real_img).mean()
            fake_out = D(fake_img).mean()

            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)

            if (epoch > 1):
                for group in D_optimizer.param_groups:
                    for p in group['params']:
                        state = D_optimizer.state[p]
                        if 'step' in state.keys():
                            if (state['step'] >= 1024):
                                state['step'] = 1000

            D_optimizer.step()

            # Update G: minimize g loss
            G.zero_grad()
            g_loss = G_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            if (epoch > 1):
                for group in G_optimizer.param_groups:
                    for p in group['params']:
                        state = G_optimizer.state[p]
                        if 'step' in state.keys():
                            if (state['step'] >= 1024):
                                state['step'] = 1000
            fake_img = G(z)
            fake_out = D(fake_img).mean()

            G_optimizer.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

        # validation part
        G.eval()
        out_path = 'training_results/SRF_' + str(upscale_factor) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with torch.no_grad():
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            sum_val_psnr = 0.0
            sum_val_ssim = 0.0
            for val_lr, val_hr_restore, val_hr in val_loader:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr.cuda()
                hr = val_hr.cuda()
                sr = G(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))

                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                sum_val_psnr += valing_results['psnr']
                sum_val_ssim += valing_results['ssim']

        # save model parameters:
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)

        if epoch % v == 0 and epoch != 0:
            torch.save(G.state_dict(), epoch_path + "/" + 'netG_epoch_%d_%d.pth' % (upscale_factor, epoch))
            torch.save(D.state_dict(), epoch_path + "/" + 'netD_epoch_%d_%d.pth' % (upscale_factor, epoch))
            # save loss, scores, psnr, ssim
            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
            results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])

            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])

            avg_psnr = sum_val_psnr / batch_Size
            avg_ssim = sum_val_ssim / batch_Size
            avg_d_score = running_results['d_score'] / running_results['batch_sizes']
            avg_g_score = running_results['g_score'] / running_results['batch_sizes']
            avg_d_loss = running_results['d_loss'] / running_results['batch_sizes']
            avg_g_loss = running_results['g_loss'] / running_results['batch_sizes']
            print(
                "Epoch {:.0f} psnr {:.8f} ssim {:.8f} d_loss {:.8f} g_loss {:.8f} d_score {:.8f} g_score {:.8f} ".format(
                    epoch, avg_psnr, avg_ssim, avg_d_loss, avg_g_loss, avg_d_score, avg_g_score))

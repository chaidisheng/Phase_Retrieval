#!/usr/bin/python
# coding:utf-8
# author: chaidisheng

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.utils import batch_PSNR, batch_SSIM
import os
import cv2
import glob
import time
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.backends import cudnn
from utilities.log import init_logging
from Algorithm.DIP_RED.dip_red import DIP_RED
from utils.measurement_utils import wgn, SNR
from utils.figure import show_detail, show_double_detail
from utils.measurement_utils import code_diffraction_forward as AA
from utils.measurement_utils import gaussion_forward as gaussion

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # physics specify


def normalize(data):
    """normlize image"""
    return data / 255.0


def proj(image, *options):
    return np.minimum(np.maximum(image, options[0]), options[1])


def test(args):
    # make save direction
    print('Start to test on {}'.format(args.test_data))
    out_dir = args.save_dir + args.test_data.split('/')[-1]+'/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', args.test_data, '*.png'))
    files_source.sort()

    # log info file
    logger = init_logging(out_dir)
    logger.info("Algorithm parameters are {}".format(args))

    # process data
    name, PSNR, SSIM, TIME = [], [], [], []
    averaged_psnr, averaged_ssim, averaged_time = 0, 0, 0
    for files in files_source:
        Img = cv2.imread(files)
        # show_detail(Img[:, :, 0:1], logdir, [180, 150], [240, 250], 2, 1, 0)
        Img = normalize(np.float32(Img[:, :, 0]))
        original_image = Img
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img).to(args.device)

        # initial array
        # x_init = 0.5 * torch.ones_like(ISource)
        x_init = torch.rand_like(ISource)
        A = lambda x: AA(x, sampling_rate=args.sampling_rate, seed=args.random_seed)
        # A = lambda x: gaussion(x, sampling_rate=args.sampling_rate, seed=args.random_seed)

        # set noise model
        pre_measure = torch.sqrt(torch.sum(A(ISource[:]) ** 2, dim=3, keepdim=True))
        gaussian_noise, sigma_w = wgn(ISource.repeat(1, args.sampling_rate, 1, 1), args.snr, seed=args.random_seed)
        measure = pre_measure + gaussian_noise.view(pre_measure.shape)  # 5.65
        print('SNR is: {0:.2f}'.format(SNR(pre_measure, measure).item()))

        options = dict()
        start_time = time.time()
        out_image = DIP_RED(A, measure[:], ISource, x_init, **options)
        stop_time = time.time()
        test_time = stop_time-start_time
        TIME.append(test_time)
        averaged_time += test_time
        psnr, ssim= batch_PSNR(out_image, ISource, 1.), batch_SSIM(out_image, ISource, 1.)
        PSNR.append(psnr)
        SSIM.append(ssim)
        averaged_psnr += psnr
        averaged_ssim += ssim
        print("%s PSNR: %.2fdb, SSIM: %.2f, Consuming Time: % .2fs. " % (files, psnr, ssim, test_time))

        # save images
        filename = files.split('/')[-1].split('.')[0]  # get the name of image file
        name.append(filename)
        # figure = plt.figure("restoration image")
        plt.figure("restoration image")
        plt.subplot(1, 2, 1)
        plt.imshow(torch.squeeze(ISource).cpu(), cmap='gray')
        plt.title("inf db")
        plt.subplot(1, 2, 2)
        plt.imshow(torch.squeeze(out_image).cpu(), cmap='gray')
        plt.title(str(format(psnr, '.2f'))+'db')
        plt.show()

        # plt.figure("concatenate image")
        # plt.imshow(np.concatenate((torch.squeeze(ISource), torch.squeeze(out_image)), axis=1), cmap='gray')
        # plt.title("Original-Restoration")
        # plt.show()

        org_dir = out_dir+str(files.split('/')[2].split('.')[0])+'.png'
        logdir = out_dir+str(files.split('/')[2].split('.')[0])+'-'+str(format(psnr, '.2f'))+'db'+'.png'
        logdir_double = out_dir+str(files.split('/')[2].split('.')[0])+'_'+str(format(psnr, '.2f'))+'db'+'.png'
        list_ = [np.expand_dims(original_image, 2),
                 np.expand_dims(proj(torch.squeeze(out_image).cpu().numpy(), 0., 1.), 2)]
        width, length = *[[40, 100] if ISource.size()[3] == 256 else [180, 150]], \
                        *[[100, 160] if ISource.size()[3] == 256 else [240, 250]]
        xy = [160, 40] if ISource.size()[3] == 256 else [250, 180]
        yx = [120, 136] if ISource.size()[3] == 256 else [200, 392]
        show_detail(np.expand_dims(original_image, 2), org_dir, width, length, 2, 1, 0, xy, yx)
        show_double_detail(logdir_double, width, length, 2, 1, 0, xy, yx, *list_)
        show_detail(np.expand_dims(proj(out_image.squeeze().cpu().numpy(), 0., 1.), 2),
                    logdir, width, length, 2, 1, 0, xy, yx)

    name.append("Average")
    averaged_psnr /= len(files_source)
    PSNR.append(averaged_psnr)
    averaged_ssim /= len(files_source)
    SSIM.append(averaged_ssim)
    averaged_time /= len(files_source)
    TIME.append(averaged_time)

    print("\nOn the test data Average PSNR: %.2fdb, Average SSIM: %.2f, Average Consuming Time: %.2fs. " %
          (averaged_psnr, averaged_ssim, averaged_time))

    pd.DataFrame({'name': np.array(name), 'psnr': np.round(np.array(PSNR), 2), 'ssim': np.round(np.array(SSIM), 2),
                  'test_time': np.round(np.array(TIME), 2)}).to_csv(out_dir+'/metrics.csv', index=True)


if __name__ == "__main__":
    # python3 train_full_realsn.py --sigma=15./255 or python3 train_full_realsn.py --sigma 15./255
    parser = argparse.ArgumentParser(description="DIP_Test")
    parser.add_argument("--save_dir", type=str, default="./result/PnP_DIP/", help='result of log files')
    parser.add_argument("--test_data", type=str, default='TestImages', help='test on Set12 or TestImages')
    parser.add_argument("--sigma", nargs='+', type=float, default=3, help='noise level used denoiser')
    # seed = random.randint(1, 1000)
    parser.add_argument("--random_seed", type=int, default=0, help='random seed of measurements')
    parser.add_argument("--sampling_rate", type=int, default=1, help='number of  measurements')
    parser.add_argument("--snr", type=float, default=15, help='signal-to-noise ratio')
    parser.add_argument("--max_iters", type=int, default=100, help='max iterations')
    parser.add_argument("--verbose", type=bool, default=True, help='log info running')
    parser.add_argument("--gpu", type=int, default=0, help='GPU index')
    opt = parser.parse_args()

    # show software version
    print("PyTorch version : ", torch.__version__)  # PyTorch version
    print("CUDA version : ", torch.version.cuda)  # Corresponding CUDA version
    print("cuDNN version : ", torch.backends.cudnn.version())  # Corresponding cuDNN version
    if torch.cuda.is_available():
        print("GPU type : ", torch.cuda.get_device_name(0))  # GPU type

    # set gpu id, CUDA_LAUNCH_BLOCKING = 1
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % opt.gpu

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print('CUDA available: {}'.format(torch.cuda.is_available()))

    # torch.device('cuda:0') or torch.device('cpu')
    opt.device = torch.device("cuda:%d" % opt.gpu if torch.cuda.is_available() else "cpu")

    # clear cache of gpu
    torch.cuda.empty_cache()

    # specify random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if torch.cuda.is_available():
        print("Returns the current %d random  seed  of the current GPU." % torch.cuda.initial_seed())
    # print parameters
    print("\n### running PnP model ###")
    print("> Parameters:")
    for p, v in zip(opt.__dict__.keys(), opt.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test(opt)
#!/usr/bin/python
# coding:utf-8
# author: chaidisheng

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.utils import batch_PSNR, batch_SSIM
import os
import time
import cv2
import glob
import random
import argparse
import torch
import torch.nn as nn
from torch.backends import cudnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities.log import init_logging
from utils.figure import show_detail, show_double_detail
from utils.measurement_utils import wgn, SNR, poisson
from Algorithm.Proximal_Gradient.prDeep import prDeep
from Algorithm.Proximal_Gradient.PnP_DAE import prDAE
from Algorithm.Proximal_Gradient.SGD_DAE import SGD_DAE
from utils.measurement_utils import code_diffraction_forward as AA
from utils.measurement_utils import code_diffraction_backward as AAt

# set gpu id, CUDA_LAUNCH_BLOCKING = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def normalize(data):
    """normlize image"""
    return data/255.


def proj(image, *options):
    return np.minimum(np.maximum(image, options[0]), options[1])


def test(args):
    # make save direction
    print('Start to test on {}'.format(args.test_data))
    out_dir = args.save_dir+args.test_data.split('/')[-1]+'/'
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
        Img = normalize(np.float32(Img[:, :, 0]))
        original_image = Img
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img).to(args.device)

        # initial array
        x_init = 0.5 * torch.ones_like(ISource)
        # x_init = torch.rand_like(ISource)

        # forward operator or backward operator
        A = lambda x: AA(x, sampling_rate=args.sampling_rate, seed=args.random_seed)
        At = lambda x: AAt(x, x_init.shape, sampling_rate=args.sampling_rate)

        filename = files.split('/')[-1].split('.')[0]  # get the name of image file
        print("\n%s image of phase retrieval is begin!" % filename)
        name.append(filename)

        # set noise model
        pre_measure = torch.sqrt(torch.sum(A(ISource[:])**2, 3, keepdim=True))
        gaussian_noise, sigma_w = wgn(ISource, args.snr)  # measure = gaussian(pre_measure, 5.65)
        measure = pre_measure + gaussian_noise.view(pre_measure.shape)
        print('SNR is: {0:.2f}'.format(SNR(pre_measure, measure).item()))

        # fbs: 9-0.36 27-0.9 81-10.0
        # fasta: 9-0.36
        # non-blind lambda=1.8 mu=0.15
        # alpha = 81
        # num_trials = 4
        # for iters in range(num_trials):
        #     sigma = 50
        #     out_image = fbs(measure[:], x_init, ISource.shape, 1, 4.0, 0.20, 1.0/2, 200, sigma, seed)
        #     sigma = 40
        #     out_image = fbs(measure, out_image, ISource.shape, 1, 4.0, 0.20, 1.0/2, 200, sigma, seed)
        #     if alpha < 50:
        #         sigma = 20
        #         out_image = fbs(measure, out_image, ISource.shape, 1, 4.0, 0.20, 1.0/2, 200, sigma, seed)
        #         sigma = 10
        #         out_image = fbs(measure, out_image, ISource.shape, 1, 4.0, 0.20, 1.0/2, 200, sigma, seed)
        # lambda = 4.0 mu = 0.21 sigma = 10
        # out_image = fbs(measure[:], x_init, ISource.shape, 1, 4.0, 0.1, 1.0/2, 200, 10, seed)
        # out_image = fasta(measure[:], x_init, ISource.shape, 1, 4.0, 0.21, 1.0/2.0, 200, 10, seed)

        # hyperparameters: fasts_options(red), prox_options(fasta)
        # prox_options = (args.lambd*(sigma_w)**2, sigma_w*255, args.sigma, args.prox_iters)
        # options = dict(max_iters=args.max_iters, tol=args.tol, record_objective=False, adaptive=False,
        #                accelerate=False, backtrack=False, restart=False, verbose=args.verbose, ori_img=ISource)

        prox_options = (args.lambd, args.sigma, args.prox_iters)
        options = dict(max_iters=args.max_iters, tol=args.tol, record_objective=False, adaptive=False,
                       accelerate=False, backtrack=False, restart=False, verbose=args.verbose, ori_img=ISource)

        # prox_options = list()
        # options = dict(max_iters=args.max_iters,  lambd=args.lambd, sigma_w=sigma_w, sigma_hat = args.sigma,
        #                verbose=args.verbose, ori_img=ISource)

        # prDeep solver
        Prior_Method = dict(PrDeep=prDeep, prDAE=prDAE, sgd_dae=SGD_DAE)
        try:
            start_time = time.time()
            out_image, _ = Prior_Method[args.Prior_Method](A, At, measure[:], x_init, args.noise_type,
                                                           args.denoiser, *prox_options, **options)
            stop_time = time.time()
        except KeyError:
            print("Priors Method Fault!")

        test_time = stop_time-start_time
        TIME.append(test_time)
        averaged_time += test_time
        psnr, ssim = batch_PSNR(out_image, ISource, 1.), batch_SSIM(out_image, ISource, 1.)
        PSNR.append(psnr)
        SSIM.append(ssim)
        averaged_psnr += psnr
        averaged_ssim += ssim
        print("%s PSNR: %.2fdb, SSIM: %.2f, Consuming Time: % .2fs. " % (files, psnr, ssim, test_time))

        # fig = plt.figure("compare image")
        # plt.subplot(1, 2, 1)
        # plt.imshow(torch.squeeze(ISource).cpu(), cmap='gray')
        # plt.title("")
        # plt.xlabel("")
        # plt.ylabel("")
        # plt.subplot(1, 2, 2)
        # plt.imshow(torch.squeeze(out_image).cpu(), cmap='gray')
        # plt.show()

        fig = plt.figure("restoration image")
        ax0 = fig.add_subplot(121)
        ax0.set_xlabel('inf db')
        ax0.imshow(torch.squeeze(ISource).cpu(), cmap='gray')
        ax1 = fig.add_subplot(122)
        ax1.set_xlabel(str(format(psnr, '.2f')) + 'db')
        ax1.imshow(torch.squeeze(out_image).cpu(), cmap='gray')
        plt.show()

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
    parser = argparse.ArgumentParser(description="PrDeep_Test")
    parser.add_argument("--Prior_Method", type=str, default='prDAE', help='prDeep or prDAE or sgd_dae')
    parser.add_argument("--save_dir", type=str, default="./result/PrDAE/", help='result of log files')
    parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or TestImages')
    parser.add_argument("--lambd", type=float, default=6.875*4, help='relative weight of red')
    parser.add_argument("--prox_iters", type=int, default=1, help='inner iters of red')
    parser.add_argument("--sigma", nargs='+', type=int, default=25, help='noise level of used denoiser')
    parser.add_argument("--denoiser", type=str, default="realsn_DnCNN_dae", help='method of used denoiser')
    parser.add_argument("--noise_type", type=bool, default=False, help='noise type of used denoiser')
    # seed = random.randint(1, 1000)
    parser.add_argument("--random_seed", type=int, default=0, help='random seed of measurements')
    parser.add_argument("--sampling_rate", type=int, default=1, help='number of  measurements')
    parser.add_argument("--snr", type=float, default=15, help='signal-to-noise ratio')
    parser.add_argument("--max_iters", type=int, default=100, help='max iterations')
    parser.add_argument("--tol", type=float, default=1e-7, help='stop rule of algorithm')
    parser.add_argument("--verbose", type=bool, default=True, help='log info running')
    parser.add_argument("--gpu", type=int, default=0, help='GPU index')
    opt = parser.parse_args()

    # show software version
    print("PyTorch version : ", torch.__version__)  # PyTorch version
    print("CUDA version : ", torch.version.cuda)  # Corresponding CUDA version
    print("cuDNN version : ", torch.backends.cudnn.version())  # Corresponding cuDNN version
    if torch.cuda.is_available():
        print("GPU type : ", torch.cuda.get_device_name(0))  # GPU type

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
    print("\n### running Prior model ###")
    print("> Parameters:")
    for p, v in zip(opt.__dict__.keys(), opt.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test(opt)
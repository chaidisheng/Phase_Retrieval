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
import scipy.io as io
from torch.backends import cudnn
from utils.log import init_logging
from Algorithm.ADMM.PnP_ADMM import PnP_ADMM
from Algorithm.ADMM.PnP_DRS import PnP_DRS
from Algorithm.ADMM.PnP_Multi_ADMM import PnP_Multi_ADMM
from Algorithm.ADMM.PnP_Stochastic_ADMM import PnP_Stochastic_ADMM
from Algorithm.ADMM.PnP_Combination_ADMM import PnP_Combination_ADMM
from Algorithm.Half_Quadratic_Splitting.half_quadratic_splitting import PnP_HQS, half_quadratic_splitting_
from Algorithm.Primal_Dual_Splitting.PnP_Pock_Chambolle import PnP_PDS, primal_dual_splitting
from Algorithm.Proximal_Gradient.PnP_FBS import PnP_FBS
from Algorithm.Denoised_Approximate_Message_Passing.PnP_AMP import PnP_AMP
from Algorithm.ADMM.PnP_CEOP import PnP_CEOP
from Algorithm.ADMM.PnP_CE import PnP_CE
from utils.measurement_utils import wgn, SNR
from utils.figure import show_detail, show_double_detail
from utils.measurement_utils import code_diffraction_forward as AA
from utils.measurement_utils import code_diffraction_backward as AAt
from utils.measurement_utils import code_diffraction_backward_complex as Complex_AAt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # physics specify


def normalize(data):
    """normalize image"""
    return data / 255.0


def proj(image, *options):
    return np.minimum(np.maximum(image, options[0]), options[1])


def save_mat(tensor):
    # save mat file
    out = tensor.squeeze().numpy()
    image = np.mat(out)
    io.savemat('image.mat', {'matrix': image})


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
    averaged_psnr, averaged_ssim, averaged_time = (0, 0, 0)
    for files in files_source:
        Img = cv2.imread(files)
        Img = normalize(np.float32(Img[:, :, 0]))
        original_image = Img
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img).to(args.device)

        # initial array
        # x_init = 0.5 * torch.ones_like(ISource)
        x_init = torch.rand_like(ISource)

        # forward operator or backward operator
        A = lambda x: AA(x, sampling_rate=args.sampling_rate, seed=args.random_seed)
        At = lambda x: AAt(x, x_init.shape, sampling_rate=args.sampling_rate)
        Complex_At = lambda x: Complex_AAt(x, x_init.shape, sampling_rate=args.sampling_rate)

        # save images file name
        filename = files.split('/')[-1].split('.')[0]  # get the name of image file
        print("\n%s image of phase retrieval is begin!" % filename)
        hist_logdir = out_dir+filename+'_hist'
        name.append(filename)

        # set noise model
        pre_measure = torch.sqrt(torch.sum(A(ISource[:]) ** 2, dim=3, keepdim=True))
        gaussian_noise, sigma_w = wgn(ISource, args.snr, seed=args.random_seed)
        measure = pre_measure+gaussian_noise.view(pre_measure.shape)  # 5.65
        print('SNR is: {0:.2f}'.format(SNR(pre_measure, measure).item()))

        # DnCnn: blind: 0.01=25.5, non-blind: 0.004=16
        # options = dict(lambd=0.01, max_iters=50, beta=0.5, verbose=True)
        # start_time = time.time()
        # out_image = PnP_ADMM(measure[:], A, At, 'DnCNN', x_init, **options)
        # stop_time = time.time()

        # options = dict(lambd=[0.001, 0.005], NUM_CHANNELS=2, max_iters=50, verbose=True)
        # out_image = PnP_Multi_ADMM(measure[:], A, At, 'DnCNN', x_init, **options)

        # options = dict(lambd=[0.01, 0.01], max_iters=50, beta=0.5, verbose=True)
        # out_image = PnP_Stochastic_ADMM(measure[:], A, At, ['DnCNN', 'BM3D'], x_init, **options)

        # options = dict(lambd=[0.001, 0.005], max_iters=50, verbose=True)
        # out_image = PnP_Combination_ADMM(measure[:], A, At, 'DnCNN', x_init, **options)

        # options = dict(lambd=[0.003, 0.01], beta=0.5, numbers=3, max_iters=50, verbose=True)
        # out_image = PnP_CEOP(measure[:], A, At, 'DnCNN', x_init, **options)

        # options = dict(lambd=0.01, max_iters=50, verbose=True)
        # out_image = PnP_DRS(measure[:], A, At, 'DnCNN', x_init, **options)

        # options = dict(lambd=0.01, rho_scale=1.0, max_iters=1, max_inner_iters=70, verbose=True)
        # out_image = PnP_HQS(measure[:], A, At, 'DnCNN', x_init, **options)

        # options = dict(lambd=0.01, max_iters=50, verbose=True)
        # out_image = PnP_PDS(measure[:], A, At, 'DnCNN', x_init, **options)

        # options = dict(max_iters=30, tol=1e-7, sigma_w=sigma_w, sigma_hat=[10./255, 20./255], record_objective=False,
        #               adaptive=False, accelerate=False, backtrack=False, restart=False)
        # out_image = PnP_FBS(measure[:], A, At, 'DnCNN', x_init, **options)

        # options = dict(max_iters=50, eps=1e1, blind=False, denoiser='DnCNN')
        # out_image = PnP_AMP(measure[:], A, Complex_At, x_init, **options)

        # Algorithm parameters
        # args.sigma[0] if sigma_w < 0.10 else args.sigma[1]
        # options = dict(lambd=args.sigma, max_iters=args.max_iters,
        #                beta=args.beta, verbose=args.verbose, logdir=hist_logdir, ori_img=ISource,)

        # options = dict(sigma_hat=args.sigma, max_iters=args.max_iters, tol=1e-7, record_objective=False,
        #                adaptive=False, accelerate=False, backtrack=False, restart=False, logdir=hist_logdir,
        #                verbose=args.verbose, ori_img=ISource,)

        options = dict(lambd=torch.tensor(args.sigma), max_iters=args.max_iters, ori_img=ISource,
                       noise_level=sigma_w, beta=args.beta, verbose=args.verbose,)

        PnP_Method = dict(PnP_ADMM=PnP_ADMM, PnP_FBS=PnP_FBS, PnP_CE=PnP_CE)
        try:
            start_time = time.time()
            out_image = PnP_Method[args.PnP_Method](measure[:], A, At, args.noise_type,
                                                    args.denoiser, x_init, **options)
            stop_time = time.time()
        except KeyError:
            print("PnP Method Fault!")

        test_time = stop_time-start_time
        TIME.append(test_time)
        averaged_time += test_time
        psnr, ssim = batch_PSNR(out_image, ISource, 1.), batch_SSIM(out_image, ISource, 1.)
        PSNR.append(psnr)
        SSIM.append(ssim)
        averaged_psnr += psnr
        averaged_ssim += ssim
        print("%s PSNR: %.2fdb, SSIM: %.2f, Consuming Time: % .2fs. " % (files, psnr, ssim, test_time))

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
    parser = argparse.ArgumentParser(description="PnP_Test")
    parser.add_argument("--PnP_Method", type=str, default='PnP_CE', help='PnP_ADMM or PnP_FBS or PnP_CE')
    parser.add_argument("--save_dir", type=str, default="./result/PnP_CE_blind_25/", help='result of log files')
    parser.add_argument("--test_data", type=str, default='TestImages', help='test on Set12 or TestImages')
    parser.add_argument("--sigma", nargs='+', type=float, default=[15./255, 25./255],
                        help='noise level of used denoiser')
    parser.add_argument("--denoiser", type=str, default="DnCNN", help='method of used denoiser')
    parser.add_argument("--noise_type", type=bool, default=True, help='noise type of used denoiser')
    # seed = random.randint(1, 1000)
    parser.add_argument("--random_seed", type=int, default=0, help='random seed of measurements')
    parser.add_argument("--sampling_rate", type=int, default=1, help='number of  measurements')
    parser.add_argument("--snr", type=float, default=25, help='signal-to-noise ratio')
    parser.add_argument("--max_iters", type=int, default=60, help='max iterations')
    parser.add_argument("--beta", type=float, default=0.5, help='averaged operator parameter')
    parser.add_argument("--verbose", type=bool, default=True, help='log info running')
    parser.add_argument("--gpu", type=int, default=0, help='GPU index')
    opt = parser.parse_args()

    # show software version
    print("PyTorch version : ", torch.__version__)  # PyTorch version
    print("CUDA version : ", torch.version.cuda)  # Corresponding CUDA version
    print("cuDNN version : ", torch.backends.cudnn.version())  # Corresponding cuDNN version
    if torch.cuda.is_available():
        print("Cuda current device : ", torch.cuda.current_device())  # Corresponding CUDA serial number
        print("GPU type : ", torch.cuda.get_device_name(0))  # GPU type

    # specify gpu number
    if opt.gpu is None:
        raise ValueError("please provide gpu id")
    opt.device = torch.device("cuda:%d" % opt.gpu if torch.cuda.is_available() else "cpu")  # logic specify

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False  # default: False
        torch.backends.cudnn.benchmark = False  # default: False
        torch.backends.cudnn.deterministic = True  # default: True
        print('CUDA available: {}'.format(torch.cuda.is_available()))

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

    # test model
    test(opt)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    DIP-RED.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        4/16/20 6:33 PM
# @torch: tensor.method(in-place) or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import queue
import numpy as np
import torch
import torch.optim
from torch import Tensor
import threading
from threading import Thread  # for running the denoiser in parallel
from Algorithm.DIP_RED.models.skip import skip  # our network
from denoise.denoise import denoise
from utils.utils import *  # auxiliary functions
from utils.data import Data  # class that holds img, psnr, time
from skimage.restoration import denoise_nl_means
from skimage.measure import compare_ssim
from skimage.measure.simple_metrics import compare_psnr


def get_network_and_input(img_shape, input_depth=32, pad='reflection',
                          upsample_mode='bilinear', use_interpolate=True, align_corners=False,
                          act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4,
                          num_scales=5, downsample_mode='stride', INPUT='noise'):  # 'meshgrid'
    r""" Getting the relevant network and network input (based on the image shape and input depth)
        We are using the same default params as in DIP article
        img_shape - the image shape (ch, x, y)
    """
    n_channels = img_shape[0]
    net = skip(input_depth, n_channels,
               num_channels_down=[skip_n33d] * num_scales if isinstance(skip_n33d, int) else skip_n33d,
               num_channels_up=[skip_n33u] * num_scales if isinstance(skip_n33u, int) else skip_n33u,
               num_channels_skip=[skip_n11] * num_scales if isinstance(skip_n11, int) else skip_n11,
               upsample_mode=upsample_mode, use_interpolate=use_interpolate, align_corners=align_corners,
               downsample_mode=downsample_mode, need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)
    net_input = get_noise(input_depth, INPUT, img_shape[1:]).detach()
    return net, net_input


def train_via_admm(net, net_input, denoiser_function, A, y, InitEst=None, clean_img=None,
                   # y is the stochastic CDP measurement./
                   plot_array={}, algorithm_name="", admm_iter=200, save_path="",  # path to save params
                   LR=0.001,  # learning rate
                   sigma_f=3, update_iter=1, method='fixed_point',  # method: 'fixed_point' or 'grad' or 'mixed'
                   lambd=.5, rho=1., LR_x=None, noise_factor=0.01,  # LR_x needed only if method!=fixed_point
                   ):
    r""" training the network using
        ## Must Params ##
        net                 - the network to be trained
        net_input           - the network input
        denoiser_function   - an external denoiser function, used as black box, this function
                              must get numpy noisy image, and return numpy denoised image
        A                   - forward operator
        y                   - the noisy image
        sigma               - the noise level (int 0-255)

        # optional params #
        org_img             - the original image if exist for psnr compare only, or None (default)
        plot_array          - prints params at the begging of the training and plot images at the required indices
        admm_iter           - total number of admm epoch
        LR                  - the lr of the network in admm (step 2)
        sigma_f             - the sigma to send the denoiser function
        update_iter         - denoised image updated every 'update_iter' iteration
        method              - 'fixed_point' or 'grad' or 'mixed'
        algorithm_name      - the name that would show up while running, just to know what we are running ;)

        # equation params #
        beta                - regularization parameter (lambda in the article)
        mu                  - ADMM parameter
        LR_x                - learning rate of the parameter x, needed only if method!=fixed point
        # more
        noise_factor       - the amount of noise added to the input of the network
        threshold          - when the image become close to the noisy image at this psnr
        increase_reg       - we going to increase regularization by this amount
        threshold_step     - and keep increasing it every step
    """
    # get optimizer and loss function:
    mse = torch.nn.MSELoss().to(InitEst.device) # using MSE loss
    # additional noise added to the input:
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    # x update method:
    if method == 'fixed_point':
        swap_iter = admm_iter+1
        LR_x = None
    elif method == 'grad':
        swap_iter = -1
    elif method == 'mixed':
        swap_iter = admm_iter // 2
    else:
        assert False, "method can be 'fixed_point' or 'grad' or 'mixed' only "

    # optimizer and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # using ADAM opt

    x, u = InitEst.clone(), torch.zeros_like(InitEst)
    f_x = x.clone()

    img_queue = queue.Queue()
    denoiser_thread = Thread(target=lambda q, f, f_args: q.put(f(*f_args)),
                             args=(img_queue, denoiser_function, [x.clone(), sigma_f, False, 'NLM']))
    denoiser_thread.start()
    for i in range(1, 1+admm_iter):
        # step 1, update network:
        optimizer.zero_grad()
        net_input = net_input_saved + (noise.normal_() * noise_factor)
        out = net(net_input)
        out_np = torch_to_np(out)

        # loss:
        loss_y = mse(torch.sum(A(out)**2, dim=3, keepdim=True), y)
        loss_x = mse(out, x - u)
        total_loss = loss_y + rho * loss_x
        total_loss.backward()
        optimizer.step()

        # step 2, update x using a denoiser and result from step 1
        if i % update_iter == 0:  # the denoiser work in parallel
            denoiser_thread.join()
            f_x = img_queue.get()
            denoiser_thread = Thread(target=lambda q, f, f_args: q.put(f(*f_args)),
                                     args=(img_queue, denoiser_function, [x.clone(), sigma_f, False, 'NLM']))
            # print(denoiser_thread.getName(), threading.active_count())
            denoiser_thread.start()

        if i < swap_iter:
            x = 1. / (lambd + rho) * (lambd * f_x + rho * (out.detach() + u))
        else:
            x = x - LR_x * (lambd * (x - f_x) + rho * (x - out.detach() - u))

        x = torch.clamp(x , 0., 1., out=x)  # making sure that image is in bounds

        # step 3, update u
        u = u + out.detach() - x

        # show psnrs:
        if clean_img is not None:
            psnr_net = compare_psnr(torch_to_np(clean_img), out_np, 1)
            psnr_x_u = compare_psnr(torch_to_np(clean_img), torch_to_np(x - u), 1)
            print('\r', algorithm_name, '%04d/%04d Loss %f' % (i, admm_iter, total_loss.item()),
                  'psnrs: net: %.2f x-u: %.2f' % (psnr_net, psnr_x_u), end='')
            if plot_array:  # plot graphs only if plotting images, since it time-consuming
                psnr_x_u = compare_psnr(torch_to_np(clean_img), torch_to_np(x - u), 1)
                psnr_net = compare_psnr(torch_to_np(clean_img), out_np, 1)
                if i in plot_array:  # plot images
                    u_ = torch_to_np(u)
                    tmp_dict = {'Clean': Data(torch_to_np(clean_img)),
                                'CDP': Data(torch_to_np(y[:, :, 0:InitEst.shape[3]**2, :].view(InitEst.shape))),
                                'Net': Data(out_np, psnr_net),
                                'x-u': Data(torch_to_np(x - u), psnr_x_u),
                                'u': Data((u_- np.min(u_)) / (np.max(u_) - np.min(u_)))
                                }
                    plot_dict(tmp_dict)
        else:
            print('\r', algorithm_name, 'iteration %04d/%04d Loss %f' % (i, admm_iter, total_loss.item()), end='')

    if denoiser_thread.is_alive():
        denoiser_thread.join()  # joining the thread
    return x - u


def DIP_RED(A, y, clean_img, InitEst=None, **options):
    # Deep Learning Powered by RED, Our Generic Function
    # The RED engine with Neural Network
    # you may test it with any neural net, and any denoiser
    # you may try it with different denoisers
    plot_checkpoints = {1, 10, 50, 100, 250, 500, 2000, 3500, 5000}
    net, net_input = get_network_and_input(img_shape=InitEst.shape[1:])
    net, net_input = net.to(InitEst.device), net_input.to(InitEst.device)
    solution = train_via_admm(net, net_input, denoise, A, y, InitEst, plot_array=plot_checkpoints,
                              algorithm_name="DIP-RED(NLM)", clean_img=clean_img)
    data_dict = dict()
    data_dict["DIP_NLM"] = Data(torch_to_np(solution), compare_psnr(torch_to_np(clean_img), torch_to_np(solution)))
    plot_dict(data_dict)
    return solution
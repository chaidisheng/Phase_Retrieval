#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    SGS_DAE.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        9/10/20 1:20 AM
# @torch: tensor.method(in-place) or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
from torch import Tensor
from denoise.denoise import denoise
from skimage.measure import compare_psnr
from utils.utils import *


def proj(tensor, *options):
    # type: (Tensor, tuple or float) -> Tensor
    r"""tensor belongs to [options[0], options[1]]
    Args:
        :tensor: input tensor of shape(N, C, H, W)
        :*options: box bound for [a, b]
    :return: projection on box
    """
    return torch.min(torch.max(tensor, torch.tensor(options[0]).to(tensor.device)),
                     torch.tensor(options[1]).to(tensor.device))


def SGD_DAE(A, At, y, x0, noise_type, denoiser_method, *prox_options, **options):
    r"""Implements stochastic gradient descent (SGD)"""

    # Hyperparameters: check preconditions, fill missing optional entries on 'options'
    if not isinstance(A, (int, float)):
        assert not isinstance(At, (int, float)), "If A is a function handle, then At must be a handle as well"

    if isinstance(A, (int, float)):
        At = lambda x: torch.matmul(A.permute(0, 1, 3, 2), x)
        A = lambda x: torch.matmul(A, x)

    # if user didn't pass this arg, then create it
    if not options:
        options = dict()

    lambd = options.get('lambd', 6.875)
    mu = options.get('mu', 0.9)
    alpha = options.get('alpha', 0.1)
    sigma_hat = options.get('sigma_hat', 25)
    sigma_w = options.get('sigma_w', 1.0)
    max_iters = options.get('max_iters', 300)
    verbose = options.get('verbose', True)
    ori_img = options.get('ori_img', None)
    sigma_eta = torch.sqrt(torch.tensor(2.)) * sigma_hat
    x, step = x0, torch.zeros_like(x0)
    sigma_hat /= 255
    sigma_eta /= 255
    relative_weight = lambd / (sigma_eta ** 2) / (lambd / (sigma_eta ** 2)+1 / (sigma_w ** 2))
    print(relative_weight)
    # output information
    if verbose:
        print("Running SGD DAE")
        print("iter \t PSNR \n")

    #
    for i in range(max_iters):

        input = x.detach().clone()

        input.requires_grad = True
        noise = sigma_hat * torch.randn_like(input)
        if input.grad is not None:
            input.grad.zero_()
        denoiser_f = denoise(input+noise, sigma_hat * 255, noise_type, denoiser_method)
        prior_err = input - denoiser_f
        denoiser_f.backward(-prior_err, retain_graph=False)
        denoiser_Jacobian = input.grad.data.detach().clone()
        with torch.no_grad():  # this can save much memory
            prior_err = prior_err+denoiser_Jacobian

        data_err = At(A(x)-y * (A(x) / torch.sqrt(torch.sum(A(x) ** 2, 3, keepdim=True))))

        # sum the gradients
        err = relative_weight * prior_err+(1-relative_weight) * data_err

        # update
        step = mu * step-alpha * err
        x = x+step
        x = proj(x, 0., 1.)
        psnr = compare_psnr(torch_to_np(x), torch_to_np(ori_img), data_range=1)
        print("iter={} psnr={:.2f}".format(i, psnr))

    return x, options

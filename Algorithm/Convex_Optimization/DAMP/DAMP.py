#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    DAMP.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        4/12/20 10:20 PM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from denoise.denoise import denoise


def proj(tensor, *options):
    # type: (Tensor, tuple) -> Tensor
    r"""tensor belongs to [options[0], options[1]]
    Args:
        :tensor: input tensor of shape(N, C, H, W)
        :*options: box bound for [a, b]
    :return: projection on box
    """
    return torch.min(torch.max(tensor, torch.tensor(options[0])), torch.tensor(options[1]))


def DAMP(y, A, At, InitEst, **options):
    r"""TODO: Docstring for DAMP.
    this function implements D-AMP based on any denoiser present in the denoise function
    Args:
        :y: the measurements.
        :A: function handle that projects onto M. Or a matrix M
        :At: function handle that projects onto M'.
        :options: hyper-parameters and initial value of unknown image
        :returns:  the recovered signal.
    Examples:
        >>>
    """
    # default hyperparameters
    eps = options.get('eps', 1e-1)
    verbose = options.get('verbose', False)
    blind = options.get('blind', False)
    denoiser = options.get('denoiser', 'DnCNN')
    max_iters = options.get('max_iters', 1000)
    bound = options.get('bound', [0., 1.])

    # initial variables
    x = torch.zeros_like(InitEst)
    z = y.clone().repeat(1, 1, 1, 2)
    m = y.shape[2]

    for iter in range(max_iters):
        pseudo_data = At(z)[:, :, :, :, 0] + x
        sigma_hat = 1. / np.sqrt(m) * torch.norm(z, p='fro', dim=(2, 3), keepdim=True) * 255.
        x = denoise(proj(pseudo_data, *bound), sigma_hat, blind=blind, denoise_method=denoiser)

        eta = torch.randn_like(x)
        epsilon = torch.max(pseudo_data) / 1000 + eps
        div = torch.einsum('...ii->...', [eta.permute(0, 1, 3, 2).matmul((denoise(proj(pseudo_data + epsilon*eta,\
                            *bound), sigma_hat, blind=blind, denoise_method=denoiser) - x) / epsilon)])
        z = y - torch.sum(A(x) ** 2, dim=3, keepdim=True) + 1. / m * z * div
    return x
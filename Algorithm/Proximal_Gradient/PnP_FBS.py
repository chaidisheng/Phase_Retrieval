#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    PnP_FBS.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        4/11/20 8:49 PM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from utils.figure import make_hist
from denoise.denoise import denoise
from Algorithm.Convex_Optimization.FBS.fasta import fasta


def proj(tensor, *options):
    # type: (Tensor, tuple) -> Tensor
    r"""tensor belongs to [options[0], options[1]]
    Args:
        :tensor: input tensor of shape(N, C, H, W)
        :*options: box bound for [a, b]
    :return: projection on box
    """
    return torch.min(torch.max(tensor, torch.tensor(options[0], device=tensor.device)),
                     torch.tensor(options[1], device=tensor.device))


def denoise_operator(tensor, sigma, blind, denoiser):
    # type: (Tensor, int, bool, str) -> Tensor
    r"""TODO: Docstring for prox_gn.
    Args:
    :tensor: input tensor of shape(N, C, H, W)
    :mu: parameter of proximal operator
    :lambd: trade-off 'g' ang 'g'
    :returns: solution of proximal operator

    """
    bound = [0., 1.]
    tensor = proj(tensor, *bound)
    v = denoise(tensor, sigma, blind=blind, denoise_method=denoiser)
    lipschitz.append(tensor), _lipschitz.append(v)
    return v


def PnP_FBS(y, A, At, noise_type, denoiser, x0, **options):
    # type: (Tensor, object or float, object or float, bool, str, Tensor, dict) -> Tensor
    r"""prDeep"""
    # Hyperparameters: check preconditions, fill missing optional entries on 'options'
    if not isinstance(A, (int, float)):
        assert not isinstance(At, (int, float)), "If A is a function handle, then At must be a handle as well!"

    if isinstance(A, (int, float)):
        At = lambda x: torch.matmul(A.permute(0, 1, 3, 2), x)
        A = lambda x: torch.matmul(A, x)

    # if user didn't pass this arg, then create it
    if not options:
        options = dict()

    # hyper-parameters about denoiser
    sigma_w = options.get('sigma_w', 1.0)
    sigma_hat = options.get('sigma_hat', list())
    logdir = options.get('logdir', './')
    # lipschitz constant compute
    global lipschitz, _lipschitz
    lipschitz, _lipschitz = list(), list()

    # define ingredients for FASTA
    # note: fasta solves min f(Ax) + lambda*g(x)

    # f(z) = 1/(2*sigma_w^2)||abs(z) - y||^2
    f = lambda z: 1. / 2 * torch.norm(torch.sqrt(torch.sum(z ** 2, dim=3, keepdim=True))-y) ** 2
    subgradient = lambda z: (z-y * (z / torch.sqrt(torch.sum(z ** 2, dim=3, keepdim=True))))

    # compute the weights
    p = torch.exp(-(torch.tensor(sigma_hat) - sigma_w) ** 2 / (2 * (5 / 255) ** 2))
    p = p / torch.sum(p)
    # print(p)

    # denoising : NLM, BM3D, DnCNN, ResUnet
    denoised = lambda noisy, mu: denoise_operator(noisy, sigma_hat, blind=noise_type, denoiser=denoiser)
    # denoised = denoise(noisy, sigma_hat[1] * 255, blind=noise_type, denoise_method=denoiser) +
    # denoise(noisy, sigma_hat[0] * 255, blind=noise_type, denoise_method=denoiser)

    # g(x) = None
    g = lambda x: 0.

    # call solver
    solution, outputs, _ = fasta(A, At, f, subgradient, g, denoised, x0, **options)
    lipschitz_constant = [torch.norm(_lipschitz[i+1]-_lipschitz[i], p='fro', dim=(2, 3), keepdim=True) /
                          torch.norm(lipschitz[i+1]-lipschitz[i], p='fro', dim=(2, 3), keepdim=True)
                          for i in range(len(lipschitz)-1)]
    lipschitz_constant = [torch.squeeze(lipschitz_constant[i]).cpu().numpy()
                          for i in range(len(lipschitz_constant))]
    interval = (max(lipschitz_constant)-min(lipschitz_constant)) / 10.
    bins = np.arange(max(lipschitz_constant), min(lipschitz_constant)-interval, -interval, dtype=np.float32)[::-1]
    # bins = np.arange(min(lipschitz_constant), max(lipschitz_constant) + interval, interval, dtype=np.float32)
    make_hist(lipschitz_constant, bins, [min(lipschitz_constant), max(lipschitz_constant)], logdir)
    return solution

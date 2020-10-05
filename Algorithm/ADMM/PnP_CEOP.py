#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    PnP_CEOP.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        3/26/20 9:13 PM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from denoise.denoise import denoise
from Algorithm.Convex_Optimization.Consensus_Optimization.consensus_optimization import solve


def proj(tensor, *options):
    # type: (Tensor, tuple) -> Tensor
    r"""tensor belongs to [options[0], options[1]]
    Args:
        :tensor: input tensor of shape(N, C, H, W)
        :*options: box bound for [a, b]
    :return: projection on box
    """
    return torch.min(torch.max(tensor, torch.tensor(options[0])), torch.tensor(options[1]))


def prox_fn_sd(tensor, *args, **kwargs):
    # type: (Tensor, tuple, dict) -> Tensor
    r"""TODO: Docstring for prox_fn_sd.
    Args:
        :tensor: input tensor of shape(N, C, H, W)
        :*args: hyperparameters of inexact minimization
        :**kwargs: parameters of proximal operator or pseudo
    :returns: solution of proximal operator or pseudo

    """
    mu = args[0]  # 0 < mu < 2./L(Lipschitz constant of gradient of f)
    sigma_w = args[1]  # log-likelihood variance
    y = args[2]  # measurement value
    ForwardFunc = args[3]  # linear operator
    BackwardFunc = args[4]  # adjoint operator
    inner_iters = args[5]  # inner iters

    lambd = kwargs.get('lambd', 1.0)  # trade-off parameters  of proximal operator
    x = kwargs.get('x', None)  # unknown image

    # SD: steepest descent
    for iter in range(inner_iters):
        v = ForwardFunc(x)
        abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
        subgradient = 1. / sigma_w ** 2 * BackwardFunc(v-y * (v / abs_v))+1. / lambd * (x-tensor)
        x = x-mu * subgradient
    return x


def prox_fn_fbs(tensor, *args, **kwargs):
    # type: (Tensor, tuple, dict) -> Tensor
    r"""TODO: Docstring for prox_fn_fbs.
    Args:
        :tensor: input tensor of shape(N, C, H, W)
        :*args: hyperparameters of inexact minimization
        :**kwargs: parameters of proximal operator or pseudo
    :returns: solution of proximal operator or pseudo

    """
    mu = args[0]  # 0< mu <2./L(Lipschitz constant of gradient of f)
    sigma_w = args[1]  # log-likelihood variance
    y = args[2]  # measurement value
    ForwardFunc = args[3]  # linear operator
    BackwardFunc = args[4]  # adjoint operator
    inner_iters = args[5]  # inner iters

    lambd = kwargs.get('lambd', 1.0)  # trade-off parameters  of proximal operator
    x = kwargs.get('x', None)  # unknown image

    # forward-backward operator
    for iter in range(inner_iters):
        v = ForwardFunc(x)
        abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
        subgradient = 1. / sigma_w ** 2 * BackwardFunc(v-y * (v / abs_v))
        x = (1. / lambd * mu * tensor+x-mu * subgradient) / (1.0+1. / lambd * mu)
    return x


def prox_fn_fp(tensor, *args, **kwargs):
    # type: (Tensor, tuple, dict) -> Tensor
    r"""TODO: Docstring for prox_fn_fp.
    Args:
        :tensor: input tensor of shape(N, C, H, W)
        :*args: hyperparameters of inexact minimization
        :**kwargs: parameters of proximal operator or pseudo
    :returns: solution of proximal operator or pseudo

    """
    beta = args[0]  # beta average
    sigma_w = args[1]  # log-likelihood variance
    y = args[2]  # measurement value
    ForwardFunc = args[3]  # linear operator
    BackwardFunc = args[4]  # adjoint operator
    inner_iters = args[5]  # inner iters

    x = kwargs.get('x', None)  # unknown image
    lambd = kwargs.get('lambd', 1.0)  # trade-off parameters of proximal operator

    # the fixed point
    for iter in range(inner_iters):
        v = ForwardFunc(x)
        abs_v = torch.sqrt(torch.sum(v ** 2, dim=3, keepdim=True))
        x = (1 - beta) * x + beta * (BackwardFunc(y * (v / abs_v)) + 1. / lambd * sigma_w ** 2 *
                                 tensor) / (1 + 1. / lambd * sigma_w ** 2)
    return x


def prox_gn(tensor, mu, lambd, denoiser, prox_g='random'):
    # type: (Tensor, float, float, str, str) -> Tensor
    r"""TODO: Docstring for prox_gn.
    Args:
    :tensor: input tensor of shape(N, C, H, W)
    :mu: parameter of proximal operator
    :lambd: trade-off 'g' ang 'g'
    :returns: solution of proximal operator

    """
    bound = [0., 1.]
    tensor = proj(tensor, *bound)
    sigma = np.sqrt(lambd * mu) * 255
    v = denoise(tensor, sigma, blind=True, denoise_method=denoiser)

    return v


def PnP_CEOP(y, ForwardFunc, BackwardFunc, denoiser, InitEst, **options):
    # type: (Tensor, object, object, str, Tensor, dict) -> Tensor
    r"""TODO: Docstring for PnP_ADMM.
    Args:
        :y: measurement
        :ForwardFunc: linear operator
        :BackwardFunc: adjoint linear operator
        :denoiser: denoise method
        :InitEst: initial value
        :**options: parameters of pnp admm
    :returns: restoration image

    """

    # start algorithm
    print("\nPlug-and-Play ADMM --- General")
    print("Denoiser = %s" % denoiser)

    # set hyperparameters
    lambd = options.get('lambd', 1.0)
    mu = options.get('mu', 0.2)
    sigma_w = options.get('sigma_w', 1.0)
    beta = options.get('beta', 1.0)
    inner_iters = options.get('inner_iters', 1)

    # proximal operator: inverse step and denoise step
    args = (beta, sigma_w, y, ForwardFunc, BackwardFunc, inner_iters)
    prox_f = lambda x, options: prox_fn_fp(x, *args, **options)
    prox_g = lambda x, mu: prox_gn(x, mu, lambd[0], denoiser)
    prox_g_ = lambda x, mu: prox_gn(x, mu, lambd[1], denoiser)
    prox_fns = (prox_f, prox_g, prox_g_)

    # admm solve
    solution = solve(InitEst, *prox_fns, **options)

    return solution

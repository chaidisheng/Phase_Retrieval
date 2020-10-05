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
import sys
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from denoise.denoise import denoise
from Algorithm.Convex_Optimization.Consensus_Optimization.Consensus_Equilibrium import solve


def proj(tensor, *options):
    # type: (Tensor, tuple) -> Tensor
    r"""tensor belongs to [options[0], options[1]]
    Args:
        :tensor: input tensor of shape(N, C, H, W)
        :*options: box bound for [a, b]
    :return: projection on box
    """
    return torch.min(torch.max(tensor, torch.tensor(options[0]).to(tensor.device)),
                     torch.tensor(options[1]).to(tensor.device))


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
    y = args[2]  # measurement value
    ForwardFunc = args[3]  # linear operator
    BackwardFunc = args[4]  # adjoint operator
    inner_iters = args[5]  # inner iters

    lambd = kwargs.get('lambd', 1.0)  # trade-off parameters  of proximal operator
    x = kwargs.get('x', torch.tensor(None))  # unknown image

    # SD: steepest descent
    for iter in range(inner_iters):
        v = ForwardFunc(x)
        abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
        subgradient = BackwardFunc(v-y * (v / abs_v))+1. / lambd * (x-tensor)
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
        subgradient = BackwardFunc(v-y * (v / abs_v))
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
    func_name = sys._getframe().f_code.co_name
    # print("%s run ..." % func_name)
    beta = args[0]  # beta average
    y = args[1]  # measurement value
    ForwardFunc = args[2]  # linear operator
    BackwardFunc = args[3]  # adjoint operator
    inner_iters = args[4]  # inner iters

    x = kwargs.get('x', None)  # unknown image
    lambd = kwargs.get('lambd', 1.0)  # trade-off parameters of proximal operator

    # the fixed point
    for iter in range(inner_iters):
        v = ForwardFunc(x)
        abs_v = torch.sqrt(torch.sum(v ** 2, dim=3, keepdim=True))
        x = (1-beta) * x+beta * (BackwardFunc(y * (v / abs_v))+1. / lambd * tensor) / (1+1. / lambd)
        # x = (BackwardFunc(y * (v / abs_v))+1. / lambd * tensor) / (1+1. / lambd)
    return x


def prox_gn(tensor, sigma, blind, denoiser):
    # type: (Tensor, float, bool, str) -> Tensor
    r"""TODO: Docstring for prox_gn.
    Args:
    :tensor: input tensor of shape(N, C, H, W)
    :mu: parameter of proximal operator
    :lambd: trade-off 'g' ang 'g'
    :returns: solution of proximal operator

    """
    func_name = sys._getframe().f_code.co_name
    # print("%s run ..." % func_name)
    # bound = [0., 1.]
    # tensor = proj(tensor, *bound)
    v = denoise(tensor, sigma * 255., blind=blind, denoise_method=denoiser)

    return v


def prox_gn_red(tensor, sigma, blind, denoiser):
    # type: (Tensor, float, bool, str) -> Tensor
    r"""TODO: Docstring for prox_gn.
    Args:
    :tensor: input tensor of shape(N, C, H, W)
    :temp: input tensor of shape(N, C, H, W)
    :sigma: parameter of proximal operator
    :denoiser: denoiser
    :returns: solution of proximal operator

    """
    func_name = sys._getframe().f_code.co_name
    # print("%s run ..." % func_name)
    lambd = 10
    # bound = [0., 1.]
    # tensor = proj(tensor, *bound)
    denoiser_f = denoise(tensor, sigma * 255., blind=blind, denoise_method=denoiser)
    v = lambd*denoiser_f + tensor
    v /= (1 + lambd)

    return v


def prox_gn_dae(tensor, sigma, blind, denoiser):
    # type: (Tensor, float, bool, str) -> Tensor
    r"""TODO: Docstring for prox_gn.
    Args:
    :tensor: input tensor of shape(N, C, H, W)
    :temp: input tensor of shape(N, C, H, W)
    :sigma: parameter of proximal operator
    :denoiser: denoiser
    :returns: solution of proximal operator

    """
    func_name = sys._getframe().f_code.co_name
    # print("%s run ..." % func_name)
    # bound = [0., 1.]
    # tensor = proj(tensor, *bound)
    lambd = 1.
    x = tensor.detach().clone()
    x.requires_grad = True
    noise = sigma*torch.randn_like(x)
    if x.grad is not None:
        x.grad.zero_()
    denoiser_f = denoise(x + noise, sigma * 255., blind=blind, denoise_method=denoiser)
    denoiser_f.backward(x - denoiser_f, retain_graph=False)
    denoiser_Jacobian = x.grad.data.detach().clone()
    with torch.no_grad():  # this can save much memory
        v = lambd*(denoiser_f + denoiser_Jacobian) + tensor
        v /= (1. + lambd)

    return v


def PnP_CE(y, ForwardFunc, BackwardFunc, noise_type, denoiser, InitEst, **options):
    # type: (Tensor, object, object, bool, str, Tensor, dict) -> Tensor
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
    print("\nPlug-and-Play CE --- General")
    print("Denoiser = %s" % denoiser)

    # set hyperparameters
    numel = len(options['lambd'])
    beta = options.get('beta', 0.5)
    inner_iters = options.get('inner_iters', 1)

    # proximal operator: inverse step and denoise step
    args = (beta, y, ForwardFunc, BackwardFunc, inner_iters)
    prox_f = lambda x, vargs: prox_fn_fp(x, *args, **vargs)

    prox_g = lambda x, sigma: prox_gn(x, sigma, noise_type, denoiser)
    prox_fns = [prox_f]+[prox_g] * numel

    # prox_g = lambda x, sigma: prox_gn_dae(x, sigma, noise_type, denoiser)
    # prox_fns = [prox_f]+[prox_g] * numel

    # CE solve
    solution = solve(InitEst, *prox_fns, **options)

    return solution

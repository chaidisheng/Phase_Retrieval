#!/usr/bin/python
# coding:utf-8
# author: chaidisheng

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from utils.figure import make_hist
from Algorithm.Convex_Optimization.ADMM.admm import solve
from denoise.denoise import denoise


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
    sigma_w = args[1]  # log-likelihood variance
    y = args[2]  # measurement value
    ForwardFunc = args[3]  # linear operator
    BackwardFunc = args[4]  # adjoint operator
    inner_iters = args[5]  # inner iters

    lambd = kwargs.get('lambd', 1.0)  # trade-off parameters  of proximal operator
    x = kwargs.get('x', None)  # unknown image
    x = torch.tensor(x)

    # SD: steepest descent
    for iter in range(inner_iters):
        v = ForwardFunc(x)
        abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
        subgradient = 1. / sigma_w ** 2 * BackwardFunc(v-y * (v / abs_v))+1. / lambd * (x - tensor)
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
    NUM_CHANNELS = kwargs.get('NUM_CHANNELS', 1.0)  # numbers of denoiser
    lambd = kwargs.get('lambd', 1.0)  # trade-off parameters of proximal operator

    # the fixed point
    for iter in range(inner_iters):
        v = ForwardFunc(x)
        abs_v = torch.sqrt(torch.sum(v ** 2, dim=3, keepdim=True))
        x = (1 - beta) * x + beta * (BackwardFunc(y * (v / abs_v)) + 1. / lambd * sigma_w ** 2 * \
            torch.sum(tensor, dim=1, keepdim=True)) / (1 + 1. / lambd * NUM_CHANNELS * sigma_w ** 2)
    return x


def prox_gn(tensor, sigma, blind, denoiser):
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


def PnP_ADMM(y, ForwardFunc, BackwardFunc, noise_type, denoiser, InitEst, **options):
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
    print("Plug-and-Play ADMM --- General")
    print("Denoiser = %s" % denoiser)

    # set hyperparameters
    lambd = options.get('lambd', 20)
    mu = options.get('mu', 0.2)
    sigma_w = options.get('sigma_w', 1.0)
    beta = options.get('beta', 0.5)
    inner_iters = options.get('inner_iters', 1)
    logdir = options.get('logdir', './')
    # lipschitz constant compute
    global lipschitz, _lipschitz
    lipschitz, _lipschitz = list(), list()

    # proximal operator: inverse step and denoise step
    args = (beta, sigma_w, y, ForwardFunc, BackwardFunc, inner_iters)
    prox_f = lambda x, kwargs: prox_fn_fp(x, *args, **kwargs)
    prox_g = lambda x: prox_gn(x, lambd, noise_type, denoiser)
    prox_fn = (prox_f, prox_g)

    # admm solve
    solution = solve(InitEst, *prox_fn, **options)
    lipschitz_constant = [torch.norm(_lipschitz[i+1] - _lipschitz[i], p='fro', dim=(2, 3), keepdim=True) /
                          torch.norm(lipschitz[i+1] - lipschitz[i] , p='fro', dim=(2, 3), keepdim=True)
                          for i in range(len(lipschitz) - 1)]
    lipschitz_constant = [torch.squeeze(lipschitz_constant[i]).cpu().numpy()
                          for i in range(len(lipschitz_constant))]
    interval = (max(lipschitz_constant) - min(lipschitz_constant)) / 10.
    bins = np.arange(max(lipschitz_constant), min(lipschitz_constant) - interval, -interval, dtype=np.float32)[::-1]
    # bins = np.arange(min(lipschitz_constant), max(lipschitz_constant) + interval, interval, dtype=np.float32)
    make_hist(lipschitz_constant, bins, [min(lipschitz_constant), max(lipschitz_constant)], logdir)
    return solution


def PnP_ADMM_Chan(y, ForwardFunc, BackwardFunc, denoiser, InitEst, **options):
    r"""TODO: Docstring for PnP_ADMM.
    optional problem: minimzie f(Ax) + g(x)
    :arg1 y:
    :arg2 ForwardFunc: ForwardFunc operator
    :arg3 BackwardFunc: BackwardFunc operator
    :arg4 denoiser:
    :arg5 InitEst:
    :arg6 options: options dictionary
    :returns: restoration image

    """
    # set hyperparameters
    maxiter = options.get('maxiter', 50)
    lambd = options.get('lambd', 0.01)
    alpha = options.get('alpha', 1.0)
    beta = options.get('beta', 0.99)
    rho = options.get('rho', 1.0)
    gamma = options.get('gamma', 1.0)
    eta = options.get('eta', 1.0)
    sigma_w = options.get('sigma_w', 1.0)
    tol = options.get('tol', 1e-4)
    verbose = options.get('verbose', True)
    mu = options.get('mu', 0.2)
    sub_method = options.get('sub_method', 'SD')
    inner_iters = options.get('inner_iters', 1)

    # initialize variables
    x = InitEst
    shape = InitEst.shape
    z = torch.ones_like(InitEst)
    u = torch.zeros_like(InitEst)
    N = shape[2] * shape[3]
    bound = [0., 1.]
    residual = [torch.tensor(float('inf'))]

    # main loop
    if verbose:
        print("Plug-and-Play ADMM --- General \n")
        print("Denoiser = %s \n\n" % denoiser)
        print("iter \t ||x-xold|| \t ||z-zold|| \t ||u-uold|| \n")

    iter = 1
    while residual[-1] > tol and iter <= maxiter:
        # store x, v, u from previous iteration for psnr residual calculation
        x_old = x
        z_old = z
        u_old = u
        for inner_iter in range(inner_iters):
            # inversion step
            if sub_method is 'SD':
                # SD: steepest descent
                v = ForwardFunc(x)
                abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
                subgradient = 1. / sigma_w ** 2 * BackwardFunc(v-y * (v / abs_v))+rho * (x-z+u)
                x = x-mu * subgradient
            elif sub_method is 'PGM':
                v = ForwardFunc(x)
                abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
                subgradient = BackwardFunc(v-y * (v / abs_v))
                x = (rho * mu * (z-u)+x-mu * subgradient) / (1.0+rho * mu)
            else:  # the fixed point
                print('sub_method is %s.' % sub_method)
                v = ForwardFunc(x)
                abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
                x = (1-beta) * x+beta * (BackwardFunc(y * (v / abs_v))+rho * (z-u)) / (1+rho)

        # relaxation
        x_hat = alpha * x+(1.0-alpha) * z_old

        # denoising step
        ztilde = x_hat+u
        ztilde = proj(ztilde, *bound)
        sigma = np.sqrt(lambd / rho) * 255
        z = denoise(ztilde, sigma)

        # update lagrangian multiplier
        u = u+(x-z)

        # update rho
        rho = rho * gamma

        # calculate residual
        residualx = (1. / np.sqrt(N)) * torch.norm(x-x_old, p='fro', dim=(2, 3), keepdim=True)
        residualz = (1. / np.sqrt(N)) * torch.norm(z-z_old, p=2, dim=(2, 3), keepdim=True)
        residualu = (1. / np.sqrt(N)) * torch.norm(u-u_old, p='fro', dim=(2, 3), keepdim=True)

        residual.append(residualx+residualz+residualu)

        if verbose:
            print("%3g \t %3.5e \t %3.5e \t %3.5e \n" % (iter, residualx, residualz, residualu))

        iter = iter+1
    return z

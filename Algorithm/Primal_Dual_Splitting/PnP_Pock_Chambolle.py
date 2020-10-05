#!/usr/bin/python
# coding:utf-8
# author: chaidisheng

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from denoise.denoise import denoise
from Algorithm.Convex_Optimization.PDS.pock_chambolle import solve


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
    sigma_w = args[1] # log-likelihood variance
    y = args[2] # measurement value
    ForwardFunc = args[3] # linear operator
    BackwardFunc = args[4] # adjoint operator
    inner_iters = args[5] # inner iters

    gamma = kwargs.get('gamma', 1.0)  # trade-off parameters  of proximal operator
    x = kwargs.get('x', None)  # unknown image

    # SD: steepest descent
    for iter in range(inner_iters):
        v = ForwardFunc(x)
        abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
        subgradient = 1. / sigma_w ** 2 * BackwardFunc(v - y * (v / abs_v)) + 1. / gamma * (x - tensor)
        x = x - mu * subgradient
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
    sigma_w = args[1] # log-likelihood variance
    y = args[2] # measurement value
    ForwardFunc = args[3] # linear operator
    BackwardFunc = args[4] # adjoint operator
    inner_iters = args[5] # inner iters

    gamma = kwargs.get('gamma', 1.0)  # trade-off parameters  of proximal operator
    x = kwargs.get('x', None)  # unknown image

    # forward-backward operator
    for iter in range(inner_iters):
        v = ForwardFunc(x)
        abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
        subgradient = 1. / sigma_w ** 2 *BackwardFunc(v - y * (v / abs_v))
        x = (1. / gamma * mu * tensor + x - mu * subgradient) / (1.0 + 1. / gamma * mu)
    return x


def prox_fn_fp(tensor, tau, tensor_, *args):
    # type: (Tensor, float, Tensor, tuple) -> Tensor
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
    inner_iters = args[5] # inner iters

    # the fixed point
    for iter in range(inner_iters):
        v = ForwardFunc(tensor_)
        abs_v = torch.sqrt(torch.sum(v ** 2, 3, keepdim=True))
        x = (1 - beta) * tensor_ + beta * (tau / sigma_w ** 2 * BackwardFunc(y * (v / abs_v)) + tensor) \
            / (1 + tau / sigma_w ** 2)
    return x


def prox_gn(tensor, mu, lambd):
    # type: (Tensor, float, float) -> Tensor
    r"""TODO: Docstring for prox_gn.
    Args:
    :tensor: input tensor of shape(N, C, H, W)
    :mu: parameter of proximal operator
    :lambd: trade-off 'g' ang 'g'
    :returns: solution of proximal operator

    """
    bound = [0., 1.]
    tensor = proj(tensor, *bound)
    sigma = np.sqrt(lambd / mu) * 255
    v = denoise(tensor, sigma)

    return v


def PnP_PDS(y, ForwardFunc, BackwardFunc, denoiser, InitEst=None, **options):
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
    # mu = options.get('mu', 0.2)
    sigma_w = options.get('sigma_w', 1.0)
    beta = options.get('beta', 1.0)
    inner_iters = options.get('inner_iters', 1)

    # proximal operator: inverse step and denoise step
    args = (beta, sigma_w, y, ForwardFunc, BackwardFunc, inner_iters)
    prox_f = lambda x, tau_, x_old: prox_fn_fp(x, tau_, x_old, *args)
    prox_g = lambda x, mu_: prox_gn(x, mu_, lambd)

    # pds solve
    solution = solve(InitEst, prox_f, prox_g, **options)

    return solution


def primal_dual_splitting(y, ForwardFunc, BackwardFunc, denoiser, InitEst, **options):
    r"""TODO: Docstring for PnP_ADMM.

    optional problem: minimzie f(Ax) + g(x)
    :ForwardFunc: ForwardFunc operator
    :BackwardFunc: BackwardFunc operator
    :denoiser: denoiser method with standard variance sigma
    :InitEst: initial value
    :**options: options dictionary
    :returns: restoration image

    """
    # set hyperparameters
    max_iters = options.get('maxiter', 60)
    lambd = options.get('lambd', 0.01)
    beta = options.get('beta', 1.0)
    eta = options.get('eta', 1.0)
    sigma_w = options.get('sigma_w', 1.0)
    tol = options.get('tol', 1e-4)
    verbose = options.get('verbose', True)
    mu = options.get('mu', 1.0)
    nv = options.get('nv', 0.2)
    theta = options.get('theta', 1.0)
    sub_method = options.get('sub_method', 'FP')
    inner_iters = options.get('inner_iters', 1)

    # initialize variables
    x = InitEst
    x_hat = InitEst
    u = torch.zeros_like(InitEst)
    shape = InitEst.shape
    N = shape[2]*shape[3]
    bound = [0., 1.]
    residual = [torch.tensor(float('inf'))]

    # main loop
    if verbose:
        print("Plug-and-Play ADMM --- General \n")
        print("Denoiser = %s \n\n" % denoiser)
        print("iter \t ||x-xold|| \t ||xhat-xhat_old|| \t ||u-uold|| \n")

    iter = 1
    while residual[-1] > tol and iter <= max_iters:
        # store x, v, u from previous iteration for psnr residual calculation
        x_old = x
        xhat_old = x_hat
        u_old = u

        # update lagrangian multiplier
        u_half = u + mu*x_hat
        sigma = np.sqrt(lambd/mu)*255
        u_half_ = proj(u_half / mu, *bound)
        u = u_half - mu*denoise(u_half_, sigma)

        # inversion step
        for inner_iter in range(inner_iters):
            print('sub_method is %s.' % sub_method)
            v = ForwardFunc(x)
            abs_v = torch.sqrt(torch.sum(v**2, 3, keepdim=True))
            x = (1 - beta)*x + beta*(nv*BackwardFunc(y*(v/abs_v)) + (x - nv*u))/(1 + nv)

        # under relaxation
        x_hat = x + theta*(x - x_old)

        # calculate residual
        residualx = (1./np.sqrt(N))*torch.norm(x - x_old, p='fro', dim=(2, 3), keepdim=True)
        residualx_hat = (1./np.sqrt(N))*torch.norm(x_hat - xhat_old, p=2, dim=(2, 3), keepdim=True)
        residualu = (1./np.sqrt(N))*torch.norm(u - u_old, p='fro', dim=(2, 3), keepdim=True)

        residual.append(residualx + residualx_hat + residualu)

        if verbose:
            print("%3g \t %3.5e \t %3.5e \t %3.5e \n" % (iter, residualx, residualx_hat, residualu))

        iter = iter + 1
    return x
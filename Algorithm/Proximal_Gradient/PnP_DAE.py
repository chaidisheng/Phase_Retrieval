#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    PnP_DAE.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        9/8/20 9:49 PM
# @torch: tensor.method(in-place) or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
from torch import Tensor

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
    return torch.min(torch.max(tensor, torch.tensor(options[0]).to(tensor.device)),
                     torch.tensor(options[1]).to(tensor.device))


def proximal_dae(z, mu, denoised, *prox_options):  # **options -> dict
    r"""proximal_red"""
    lambd = prox_options[0]
    sigma_hat = prox_options[1]
    prox_iters = prox_options[2]

    x = z

    # bound = [0., 1.]
    # x = proj(x, *bound)

    for iters in range(prox_iters):
        # monte carlo approximate
        # epsilon = 1e-3
        # x = (z + mu*lambd/2.*denoise(x) + mu*lambd/2.*(denoise((1. + epsilon)*x) - denoise(x))
        # /epsilon)/(1.0 + mu*lambd)

        x.requires_grad = True
        noise = sigma_hat/255*torch.randn_like(x)
        if x.grad is not None:
            x.grad.zero_()
        denoiser_f = denoised(x + noise)
        denoiser_f.backward(x-denoiser_f, retain_graph=False)
        denoiser_Jacobian = x.grad.data.detach().clone()
        with torch.no_grad():  # this can save much memory
            x = (z + lambd*mu*(denoiser_Jacobian + denoiser_f))/(1.0 + lambd*mu)
        z = x
    return x


def prDAE(A, At, y, x0, noise_type, denoiser_method, *prox_options, **options):
    r"""prDeep"""

    # Hyperparameters: check preconditions, fill missing optional entries on 'options'
    if not isinstance(A, (int, float)):
        assert not isinstance(At, (int, float)), "If A is a function handle, then At must be a handle as well"

    if isinstance(A, (int, float)):
        At = lambda x: torch.matmul(A.permute(0, 1, 3, 2), x)
        A = lambda x: torch.matmul(A, x)

    # if user didn't pass this arg, then create it
    if not options:
        options = dict()

    lambd = prox_options[0]
    sigma_hat = prox_options[1]

    # define ingredients for FASTA
    # note: fasta solves min f(Ax) + lambda*g(x)

    # f(z) = 1/(2*sigma_w^2)||abs(z) - y||^2
    f = lambda z: torch.norm(torch.sqrt(torch.sum(z**2, dim=3, keepdim=True)) - y)**2
    subgradient = lambda z: (z - y*(z/torch.sqrt(torch.sum(z**2, dim=3, keepdim=True))))

    # denoising : NLM, BM3D, DnCNN, realsn_DnCNN
    denoised = lambda noisy: denoise(noisy, sigma_hat, noise_type, denoiser_method)

    # g(x) = lambda/2*x'(x - f(x))
    g = lambda x: lambd/2.*torch.norm(x - denoised(x))**2

    # proximal operator: prox_mu*g(v) = argmin mu*g(x) + 1/2*||x - v||^2
    prox_dae = lambda z, mu: proximal_dae(z, mu, denoised, *prox_options)

    # call solver
    solution, outputs, _ = fasta(A, At, f, subgradient, g, prox_dae, x0, **options)
    return solution, outputs


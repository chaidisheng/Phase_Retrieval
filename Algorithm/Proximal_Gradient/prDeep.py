#!/usr/bin/python3
# coding:utf-8
# author: chaidisheng
# torch: tensor.method torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
from denoise.denoise import denoise
from Algorithm.Convex_Optimization.FBS.fasta import fasta


def proximal_red(z, mu, denoised, *prox_options):  # **options -> dict
    r"""proximal_red"""
    lambd = prox_options[0]
    prox_iters = prox_options[3]
    x = z
    for iters in range(prox_iters):
        # monte carlo approximate
        # epsilon = 1e-3
        # x = (z + mu*lambd/2.*denoise(x) + mu*lambd/2.*(denoise((1. + epsilon)*x) - denoise(x))
        # /epsilon)/(1.0 + mu*lambd)
        x = (z + lambd*mu*denoised(x))/(1.0 + lambd*mu)
        z = x
    return x


def prDeep(A, At, y, x0, noise_type, denoiser_method, *prox_options, **options):
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
    sigma_w = prox_options[1]
    sigma_hat = prox_options[2]

    # define ingredients for FASTA
    # note: fasta solves min f(Ax) + lambda*g(x)

    # f(z) = 1/(2*sigma_w^2)||abs(z) - y||^2
    f = lambda z: 1./(2*sigma_w**2)*torch.norm(torch.sqrt(torch.sum(z**2, dim=3, keepdim=True)) - y)**2
    subgradient = lambda z: 1./(sigma_w**2)*(z - y*(z/torch.sqrt(torch.sum(z**2, dim=3, keepdim=True))))

    # denoising : DnCNN, KSVD, TV
    denoised = lambda noisy: denoise(noisy, sigma_hat, noise_type, denoiser_method)

    # g(x) = lambda/2*x'(x - f(x))
    g = lambda x: lambd/2.*torch.einsum('...ii->...', x.permute(0, 1, 3, 2).matmul(x - denoised(x)))

    # proximal operator: prox_mu*g(v) = argmin mu*g(x) + 1/2*||x - v||^2
    prox_red = lambda z, mu: proximal_red(z, mu, denoised, *prox_options)

    # call solver
    solution, outputs, _ = fasta(A, At, f, subgradient, g, prox_red, x0, **options)
    return solution, outputs

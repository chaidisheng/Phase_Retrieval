#!/usr/bin/python
# coding:utf-8
# author: chaidisheng

from __future__ import print_function
import torch
import torch.nn as nn
from denoise import denoise
from utils.utils import code_diffraction_forward as A
from utils.utils import code_diffraction_backward as At


def fasta(x, x_init, *parameters):
    """
    Fast Adaptive Shrinkage/Thresholding
    a handy forward-backward solver

    (1) adaptive:
    (2) accelerated: None
    (3) backtracking line search
    (4) restart: None

    """
    # Hyperparameters
    shape = parameters[0]
    # v_est = 0.5 * torch.ones(shape)
    v_est = x_init
    sampling_rate = parameters[1]
    lambd = parameters[2]
    mu = parameters[3]
    beta = parameters[4]
    max_iters = parameters[5]
    sigma = parameters[6]
    seed = parameters[7]

    # f = lambda u: 1./2*torch.norm(torch.sum(A(u, sampling_rate, seed)**2, dim=3, keepdim=True) - x).pow(2)
    f = lambda u: 1./2*torch.einsum('...ij->...', (torch.sum(A(u, sampling_rate, seed)**2, 3, keepdim=True) - x)**2)
    for iters in range(max_iters):

        while True:

            z = A(v_est, sampling_rate, seed)
            abs_z = torch.sqrt(torch.sum(z**2, 3, keepdim=True))
            subgradient = At(z - x * (z/abs_z), shape, sampling_rate)
            v_hat = v_est - mu*subgradient
            f_v_hat = denoise(v_hat, sigma)

            v = (v_hat + lambd * mu * f_v_hat) / (1.0 + lambd * mu)
            # linear = subgradient.reshape(-1, shape[0] * shape[1] * shape[2] * shape[3])
            # affine = torch.matmul(linear, (v - v_est).reshape(shape[0] * shape[1] * shape[2] * shape[3], -1))
            subgradient_t = subgradient.permute(0, 1, 3, 2)  # transpose einsum
            affine = torch.matmul(subgradient_t, v - v_est)
            affine = torch.einsum('...ii->...', affine)
            # if f(v) <= f(v_est) + affine.squeeze() + 1./(2*mu)*torch.norm(v - v_est).pow(2):
            if f(v) <= f(v_est) + affine + 1./(2*mu)*torch.einsum('...ij->...', (v - v_est)**2):
                break
            mu = beta * mu

        v_est = v

    y = v_est
    return y

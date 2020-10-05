#!/usr/bin/python
# coding:utf-8
# author: chaidisheng

from __future__ import print_function
import torch
from denoise import denoise
from utils.utils import code_diffraction_forward as A
from utils.utils import code_diffraction_backward as At


def fbs(x, x_init, *parameters):
    """
    FBS Method: a handy forward-backward solver
    (1) adaptive
    (2) accelerated
    (3) backtracking line search
    (4) restart

    """
    # Hyperparameters
    shape = parameters[0]
    # v_est = 0.5*torch.ones(shape).to(x.device)
    v_est = x_init
    sampling_rate = parameters[1]
    lambd = parameters[2]
    mu = parameters[3]
    beta = parameters[4]
    max_iters = parameters[5]
    sigma = parameters[6]
    seed = parameters[7]

    for iters in range(max_iters):

        z = A(v_est, sampling_rate, seed)
        abs_z = torch.sqrt(torch.sum(z**2, dim=3, keepdim=True))
        subgradient = At(z - x*(z/abs_z), shape, sampling_rate)
        v_hat = v_est - mu*subgradient
        f_v_hat = denoise(v_hat, sigma)
        v_est = (v_hat + lambd*mu*f_v_hat)/(1.0 + lambd*mu)

    y = v_est
    return y

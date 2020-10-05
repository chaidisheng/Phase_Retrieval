#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    half_quadratic_splitting.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        2/24/20 5:46 AM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


def conv2d(input, weight, stride=(1, 1), padding=(0, 0), groups=1):
    # type: (Tensor, Tensor, tuple, tuple, int) -> Tensor
    r"""TODO: Docstring for conv2d.
    Args:
        :input: input tensor of shape(N, C, iH, iW)
        :weight: filter kernel of shape(out_channels, in_channels/groups, kH, kW)
        :stride: the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
        :padding: keep shape of : Default input.shape//2
        :groups: split input into groups, \text{in\_channels}in_channels should be divisible by the number of
        groups. Default: 1
    :returns: tensor image keep shape of input

    """
    return F.conv2d(input, weight, stride=stride, padding=padding, groups=groups)


def solve(InitEst=None, *prox_fns, **options):
    # type: (Tensor, tuple, dict) -> Tensor
    r"""TODO: Docstring for solve.

    :prox_fns: TODO
    :options: TODO
    :returns: TODO

    """
    # default hyperparameters
    rho = options.get('rho', 1.0)
    rho_max = options.get('rho_max', 2 ** 8)
    rho_scale = options.get('rho_scale', np.sqrt(2.0) * 2.0)
    eps_abs = options.get('eps_abs', 1e-3)
    eps_rel = options.get('eps_rel', 1e-3)
    verbose = options.get('verbose', False)
    max_iters = options.get('max_iters', 1000)
    max_inner_iters = options.get('max_inner_iters', 100)

    # proximal operator
    f = options.get('f', lambda x: 0.)
    g = options.get('g', lambda x: 0.)
    pseudo_prox_fn = prox_fns[0]
    prox_gn = prox_fns[1]

    # linear operator: using 2D convolution implements
    kernel = torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]])
    kernel_t = torch.rot90(kernel, 2, (2, 3))
    D, Dt = options.get('D', kernel), options.get('Dt', kernel_t)
    prox_options = dict()
    prox_options['D'] = D
    prox_options['Dt'] = Dt

    # initial variables
    x = InitEst.clone()
    z = torch.zeros_like(InitEst)
    Dx = torch.zeros_like(InitEst)

    # convergence information
    objval = []
    eps_x, eps_z = [], []
    shape = InitEst.size()
    N = shape[2] * shape[3]

    # output information
    if verbose:
        print('%3s\t%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n' %
              ('iter', 'inner_iter', 'rho', 'eps_x', 'eps abs', 'eps_z', 'eps rel', 'objective'))

    # half quadratic splitting solver
    iter = 0
    while iter < max_iters and rho < rho_max:
        # store x, z from previous iteration for psnr residual calculation
        x_old = x
        z_old = z
        for inner_iter in range(max_inner_iters):
            # update x
            prox_options['gamma'] = 1. / rho
            prox_options['x'] = x
            x = pseudo_prox_fn(z, prox_options)

            # update z
            Dx = conv2d(x, D, padding=(D.size()[2] // 2, D.size()[3] // 2))
            z = prox_gn(Dx, 1. / rho)

            # record objective
            objval.append(f(x)+g(Dx))

            # calculate residual
            eps_x.append((1. / np.sqrt(N)) * torch.norm(x-x_old, p='fro', dim=(2, 3), keepdim=True))
            eps_z.append((1. / np.sqrt(N)) * torch.norm(z-z_old, p=2, dim=(2, 3), keepdim=True))

            # output information
            if verbose:
                print('%3d\t%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n' %
                      (iter, inner_iter, rho, eps_x[-1], eps_abs, eps_z[-1], eps_rel, objval[-1]))

            if all(eps_x[-1] < eps_abs and eps_z[-1] < eps_rel):
                break

        # update rho
        rho = np.minimum(rho * rho_scale, rho_max)

        iter += 1
    # restoration image
    solution = z
    return solution

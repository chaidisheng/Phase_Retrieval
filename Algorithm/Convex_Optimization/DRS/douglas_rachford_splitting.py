#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    douglas_rachford_splitting.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        2/21/20 9:23 AM
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
    r"""TODO: Docstring for drs solve.
    This method solves the problem
        minimize f(x) + g(Dx)
    Where D is a matrix, and both f and g are closed proper convex functions.
    The algorithm is an alternating direction method of multipliers.
    The user supplies function handles that evaluate 'f' and 'g'.  The user
    also supplies a function that evaluates the proximal operator of 'f' and 'g',
    which is given by
                 prox_fn(v, lambda) = argmin lambda*f(x)+.5||x-v||^2. or
                 pseudo_prox_fn(v, D, Dt, lambda) = argmin lambda*f(x)+.5||Dx-v||^2.
                 prox_gn(v, lambda) = argmin lambda*g(x)+.5||x-v||^2.
    Args:
    :*prox_fns: The users suppliers proximal operator of 'f' and 'g'.
    :**options: hyper-parameters and initial value of unknown image.
    :returns: restoration image and convergence information.

    """

    # default hyperparameters
    gamma = options.get('rho', 1.0)
    alpha = options.get('alpha', 2.0)
    theta = options.get('theta', 1.0)
    eps_abs = options.get('eps_abs', 1e-3)
    eps_rel = options.get('eps_rel', 1e-3)
    tol = options.get('tol', 1e-4)
    verbose = options.get('verbose', False)
    max_iters = options.get('max_iters', 1000)

    # proximal operator
    f = options.get('f', lambda x: 0.)
    g = options.get('g', lambda x: 0.)
    prox_fn = prox_fns[0]
    pseudo_prox_gn = prox_fns[1]
    prox_options = dict()

    # Default linear operator: using 2D convolution implements
    kernel = torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]])
    kernel_t = torch.rot90(kernel, 2, (2, 3))
    D, Dt = options.get('D', kernel), options.get('Dt', kernel_t)

    # initial variables
    x_half = InitEst.clone()
    x = InitEst.clone()
    z = torch.zeros_like(InitEst)

    # convergence information
    objval,residual = [], [torch.tensor(float('inf'))]
    r_norm, s_norm = [], []
    eps_pri, eps_dual = [], []
    shape = InitEst.size()
    N = shape[2]*shape[3]

    if verbose:
        print("iter \t ||x_1/2-x_1/2_old|| \t ||x-xold|| \t ||z-zold|| \n")

    # douglas-rachford splitting solver
    for iter in range(max_iters):
        # store x, v, u from previous iteration for psnr residual calculation
        x_half_old = x_half
        x_old = x
        z_old = z

        # update x^(k + 1/2)
        prox_options['gamma'] = gamma
        prox_options['x'] = x
        x_half = prox_fn(z, prox_options)

        # update x with relaxation: alpha belongs to (0, 2)
        z_half = alpha*x_half - z
        x = pseudo_prox_gn(z_half, gamma, D, Dt)
        Dx = conv2d(x, D, padding=(D.size()[2]//2, D.size()[3]//2))

        # update z
        z = z + theta*(x - x_half)

        # stooping criteria
        objval.append(f(x) + g(Dx))

        # calculate residual
        residualx_half = (1./np.sqrt(N))*torch.norm(x_half - x_half_old, p='fro', dim=(2, 3), keepdim=True)
        residualx = (1./np.sqrt(N))*torch.norm(x - x_old, p=2, dim=(2, 3), keepdim=True)
        residualz = (1./np.sqrt(N))*torch.norm(z - z_old, p='fro', dim=(2, 3), keepdim=True)

        residual.append(residualx_half + residualx + residualz)

        if residual[-1] <= tol:
            break

        # output information
        if verbose:
            print("%3g \t %3.5e \t %3.5e \t %3.5e \n" % (iter, residualx_half, residualx, residualz))

    # restoration image
    solution = x

    return solution

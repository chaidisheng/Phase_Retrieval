#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence
# @Filename:    admm.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        2/21/20 4:46 AM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from skimage.measure import compare_psnr
from utils.utils import *
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
    r"""TODO: Docstring for admm solve.
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
    rho = options.get('rho', 1.0)
    alpha = options.get('alpha', 1.0)
    gamma = options.get('gamma', 1.0)
    eps_abs = options.get('eps_abs', 1e-4)
    eps_rel = options.get('eps_rel', 1e-3)
    verbose = options.get('verbose', False)
    max_iters = options.get('max_iters', 1000)
    NUM_CHANNELS = options.get('NUM_CHANNELS', 1)
    ori_img = options.get('ori_img', None)

    # objective function
    fn = [options.get('f' + str(i), lambda tensor: 0.) for i in range(NUM_CHANNELS + 1)]

    # proximal operator
    pseudo_prox_fn = prox_fns[0]
    prox_gn = prox_fns[1]

    # Default linear operator: using 2D convolution implements
    kernel = torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]]).to(InitEst.device)
    kernel_t = torch.rot90(kernel, 2, (2, 3))
    # kernel = kernel.expand(filters_size)
    kernel_t = kernel_t.expand(1, NUM_CHANNELS, 3, 3).to(InitEst.device)
    # kernel_t = kernel_t.expand_as(kernel)
    D, Dt = options.get('D', kernel), options.get('Dt', kernel_t)
    prox_options = dict(D=D, Dt=Dt, NUM_CHANNELS=NUM_CHANNELS)

    # initial variables
    shape = InitEst.size()
    x = InitEst.clone()
    z = InitEst.clone().expand(shape[0], NUM_CHANNELS, shape[2], shape[3])
    u = torch.zeros_like(InitEst).expand(shape[0], NUM_CHANNELS, shape[2], shape[3])

    # convergence information
    objval = []
    r_norm, s_norm = [], []
    eps_pri, eps_dual = [], []
    N = shape[2] * shape[3]

    # output information
    if verbose:
        print('%3s\t%6s\t%10s\t%10s\t%10s\t%10s\t%10s\n' % \
              ('iter', 'r_norm', 'eps_pri', 's_norm', 'eps_dual', 'objective', 'PSNR'))

    # alternating direction method of multipliers solver
    for iter in range(max_iters):
        # update x
        # prox_options['lambd'] = 1. / rho
        # prox_options['x'] = x
        # x = pseudo_prox_fn(z - u, prox_options)

        # update x
        x = prox_gn(z - u)

        # update z with over-relaxation: alpha belongs to (1, 2)
        # z_old, Dx = z, conv2d(x, D, padding=(D.size()[2] // 2, D.size()[3] // 2))
        # Ax_hat = alpha * Dx + (1 - alpha) * z_old
        # z = prox_gn(Ax_hat + u, 1. / rho)

        # update z with over-relaxation: alpha belongs to (1, 2)
        z_old, Dx = z, conv2d(x, D, padding=(D.size()[2] // 2, D.size()[3] // 2))
        Ax_hat = alpha * Dx + (1 - alpha) * z_old
        prox_options['lambd'] = 1. / rho
        prox_options['x'] = z
        z = pseudo_prox_fn(Ax_hat + u, prox_options)

        # update u
        u = u + Ax_hat - z

        # update rho
        rho = gamma * rho

        # stopping criteria
        objval.append(sum([fn[0](x)] + [fn[i](Dx) for i in range(1, NUM_CHANNELS + 1)]))
        DTs = -rho * conv2d(z - z_old, Dt, padding=(Dt.size()[2] // 2, Dt.size()[3] // 2))

        r_norm.append((torch.norm(torch.sum(Dx - z, dim=1, keepdim=True), p='fro', dim=(2, 3), keepdim=True)))
        s_norm.append(torch.norm(DTs, p='fro', dim=(2, 3), keepdim=True))

        eps_pri.append(np.sqrt(N) * eps_abs + eps_rel * torch.max(torch.norm(Dx, p=2, dim=(2, 3), keepdim=True),\
                                    torch.norm(torch.sum(-z,  dim=1, keepdim=True), p=2, dim=(2, 3), keepdim=True)))
        DTu = conv2d(u, Dt, padding=(Dt.size()[2] // 2, Dt.size()[3] // 2))
        eps_dual.append(np.sqrt(N) * eps_abs + eps_rel * torch.norm(rho * DTu))

        if all(sum(r_norm[-1]) < eps_pri[-1] and s_norm[-1] < eps_dual[-1]):
            break

        # output information
        psnr = compare_psnr(torch_to_np(x), torch_to_np(ori_img), data_range=1)
        if verbose:
            print('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\t%10.2f' % \
                  (iter, r_norm[-1], eps_pri[-1], s_norm[-1], eps_dual[-1], objval[-1], psnr))

    # restoration image
    solution = x
    return solution

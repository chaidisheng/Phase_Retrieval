#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    consensus_optimization.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        3/24/20 6:58 AM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    gamma = options.get('gamma', 1.0)
    eps_abs = options.get('eps_abs', 1e-4)
    eps_rel = options.get('eps_rel', 1e-2)
    verbose = options.get('verbose', False)
    max_iters = options.get('max_iters', 1000)
    numbers = options.get('numbers', 2)
    prox_options = dict()

    # objective function
    fn = [options.get('f'+str(i), lambda tensor: 0.) for i in range(numbers+1)]

    # initial variables
    shape = InitEst.size()
    x = InitEst.clone().expand(shape[0], numbers, shape[2], shape[3])
    x_bar = torch.zeros_like(InitEst)
    u = torch.zeros_like(InitEst).expand(shape[0], numbers, shape[2], shape[3])

    # convergence information
    objval = list()
    r_norm, s_norm = [], []
    eps_pri, eps_dual = [], []
    N = shape[2] * shape[3]

    # output information
    if verbose:
        print('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n' % \
              ('iter', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective'))

    # alternating direction method of multipliers solver
    for iter in range(max_iters):
        # update x
        x_bar_old = x_bar
        prox_options['x'] = x[:, 0:1, :, :]
        prox_options['lambd'] = 1. / rho
        x_list = [prox_fns[0](x_bar - u[:, 0:1, :, :], prox_options)] + \
            [prox_fns[i](x_bar - u[:, i:i+1, :, :], 1. / rho) for i in range(1, numbers)]
        x = torch.cat(x_list, dim=1)

        # update x_bar
        x_bar = 1. / numbers * torch.sum(x, dim=1, keepdim=True)

        # update u
        u = u + x - x_bar

        # update rho
        rho = gamma * rho

        # stopping criteria
        objval.append(sum([fn[0](x)]+[fn[i](x) for i in range(1, numbers+1)]))
        r_norm.append(torch.sqrt(torch.sum(torch.norm(x - x_bar, p='fro', dim=(2, 3), keepdim=True) ** 2, dim=1,\
                                           keepdim=True)))
        s_norm.append(np.sqrt(numbers) * rho * torch.norm(x_bar - x_bar_old, p='fro', dim=(2, 3), keepdim=True))

        eps_pri.append(np.sqrt(N) * eps_abs + eps_rel * torch.max(torch.norm(x[:, 0:1, :, :], p=2, dim=(2, 3),\
                                            keepdim=True), torch.norm(x_bar, p=2, dim=(2, 3), keepdim=True)))
        eps_dual.append(np.sqrt(N) * eps_abs + eps_rel * torch.norm(rho * u))

        if all(sum(r_norm[-1]) < eps_pri[-1] and s_norm[-1] < eps_dual[-1]):
            break

        # output information
        if verbose:
            print('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n' % \
                  (iter, r_norm[-1].numpy(), eps_pri[-1], s_norm[-1], eps_dual[-1], objval[-1]))

    # restoration image
    solution = x[:, 0:1, :, :]
    return solution


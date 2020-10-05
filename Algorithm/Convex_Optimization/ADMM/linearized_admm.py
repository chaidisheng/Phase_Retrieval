#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence
# @Filename:    linearized_admm.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        2/21/20 4:46 AM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import warnings


def conv2d(input, weight, stride=(1, 1), padding=(0, 0), groups=1):
    # type: (Tensor, Tensor, tuple, tuple, float) -> Tensor
    r"""TODO: Docstring for conv2d.

    :input: input tensor of shape(N, C, iH, iW)
    :weight: filter kernel of shape(out_channels, in_channels/groups, kH, kW)
    :stride: the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
    :padding: keep shape of : Default input.shape//2
    :groups: split input into groups, \text{in\_channels}in_channels should be divisible by the number of
     groups. Default: 1
    :returns: tensor image keep shape of input

    """
    return F.conv2d(input, weight, stride=stride, padding=padding, groups=groups)


def est_CompGraph_norm(x0, K, tol=1e-3, try_fast_norm=True):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    """Estimates operator norm for L = ||K|| or Lipschitz constant.

    Parameters
    ----------
    tol : float
        Accuracy of estimate if not trying for upper bound.
    try_fast_norm : bool
        Whether to try for a fast upper bound.

    Returns
    -------
    float
        Estimate of ||K||.
    """
    if try_fast_norm:
        print("Not Implemented!")
    else:
        x1, x2 = torch.randn_like(x0), torch.randn_like(x0)
        Kx, Ky = conv2d(x1, K, padding=(K.size()[2]//2, K.size()[3]//2)), \
                conv2d(x2, K, padding=(K.size()[2]//2, K.size()[3]//2))
        L = torch.norm(Ky - Kx, p='fro', dim=(2, 3), keepdim=True)
        L /= torch.norm(x2 - x1, p='fro', dim=(2, 3), keepdim=True)
        L = torch.max(L, torch.tensor(1e-6).to(x0.device))

    return L


def est_params_lin_admm(x0, K, lamb=None, verbose=True, scaled=False, try_fast_norm=False):
    # type: (Tensor, Tensor, float, bool, bool, bool) -> Tensor
    # Select lambda
    lamb = 1.0 if lamb is None else np.maximum(lamb, 1e-5)

    # Warn user
    if lamb > 1.0:
        warnings.warn("Large lambda value given by user.")

    # Estimate Lipschitz constant and compute mu
    if scaled:
        L = 1.
    else:
        x0 = torch.randn_like(x0)
        L = est_CompGraph_norm(x0, K, try_fast_norm)
    mu = lamb / (L**2)

    if verbose:
        print("Estimated params [lambda = %3.3f | mu = %3.3f | L_est = %3.4f]" % (lamb, mu, L))

    return lamb, mu


def solve(*prox_fns, **options):
    # type: (tuple, dict) -> Tensor
    r"""TODO: Docstring for linearized-admm solve.
        This method solves the problem
            minimize f(x) + g(Dx)
        Where D is a matrix, and both f and g are closed proper convex functions.
        The algorithm is an preconditioned alternating direction method of multipliers.
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
    x0 = options.get('x0', None)
    rho = options.get('rho', 1.0)
    eps_abs = options.get('eps_abs', 1e-3)
    eps_rel = options.get('eps_rel', 1e-3)
    verbose = options.get('verbose', False)
    max_iters = options.get('max_iters', 1000)

    # proximal operator
    f = options.get('f', lambda x: 0.)
    g = options.get('g', lambda x: 0.)
    prox_fn = prox_fns[0]
    prox_gn = prox_fns[1]

    # linear operator: using 2D convolution implements
    kernel = torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]])
    kernel_t = torch.rot90(kernel, 2, (2, 3))
    D, Dt = options.get('D', kernel), options.get('Dt', kernel_t)

    # evalute mu and rho
    rho, mu = est_params_lin_admm(x0, D, lamb=1.0 / rho, verbose=False,)

    # initial variables
    x = x0.clone()
    z = x0.clone()
    u = torch.zeros_like(x0)

    # convergence information
    objval = []
    r_norm, s_norm = [], []
    eps_pri, eps_dual = [], []
    shape = x0.size()
    N = shape[2]*shape[3]

    # output information
    if verbose:
        print('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n' % ('iter', \
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective'))

    # linearized alternating direction method of multipliers solver
    for iter in range(max_iters):
        # update x
        Dx = conv2d(x, D, padding=(D.size()[2]//2, D.size()[3]//2))
        x = x - mu*rho*conv2d(Dx - z + u, Dt, padding=(Dt.size()[2]//2, Dt.size()[3]//2))
        x = prox_fn(x, mu)

        # update z
        z_old = z
        z = prox_gn(z + u, 1./rho)

        # update u
        u = u + Dx - z

        # stooping criteria
        objval.append(f(x) + g(Dx))

        DTs = -rho*conv2d(z - z_old, Dt, padding=(Dt.size()[2] // 2, Dt.size()[3] // 2))
        r_norm.append(torch.norm(Dx - z, p='fro', dim=(2, 3), keepdim=True))
        s_norm.append(torch.norm(DTs, p='fro', dim=(2, 3), keepdim=True))

        eps_pri.append(np.sqrt(N)*eps_abs + eps_rel*torch.max(torch.norm(Dx, p=2, dim=(2, 3), keepdim=True), \
                                                         torch.norm(-z, p=2, dim=(2, 3), keepdim=True)))
        DTu = conv2d(u, Dt, padding=(Dt.size()[2]//2, Dt.size()[3]//2))
        eps_dual.append(np.sqrt(N)*eps_abs + eps_rel*torch.norm(rho*DTu))

        if all(r_norm[-1] < eps_pri[-1] and s_norm[-1] < eps_dual[-1]):
            break

        # output information
        if verbose:
            print('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n' % (iter, \
            r_norm[-1], eps_pri[-1], s_norm[-1], eps_dual[-1], objval[-1]))

    # restoration image
    solution = x

    return solution

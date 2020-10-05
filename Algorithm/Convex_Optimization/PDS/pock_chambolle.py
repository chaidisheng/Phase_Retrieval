#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    pock_chambolle.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        2/24/20 10:26 PM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

from torch import Tensor


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


def est_CompGraph_norm(x0, K, tol=1e-3, try_fast_norm=False):
    """Estimates operator norm for L = ||K||.

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
        L = None
        print("Not Implemented!")
    else:
        x1, x2 = torch.randn_like(x0), torch.randn_like(x0)
        Kx, Ky = conv2d(x1, K, padding=(K.size()[2]//2, K.size()[3]//2)), \
                conv2d(x2, K, padding=(K.size()[2]//2, K.size()[3]//2))
        L = torch.norm(Ky - Kx, p='fro', dim=(2, 3), keepdim=True)
        L /= torch.norm(x2 - x1, p='fro', dim=(2, 3), keepdim=True)
        L = torch.max(L, torch.tensor(1e-6).to(x0.device))
    return L


def est_params_lin_admm(x0, K, mu=None, verbose=True, scaled=False, try_fast_norm=False):

    # Select lambda
    mu = 1.0 if mu is None else np.maximum(mu, 1e-5)

    # Warn user
    if mu > 1.0:
        warnings.warn("Large lambda value given by user.")

    # Estimate Lipschitz constant and compute mu
    if scaled:
        L = 1.
    else:
        L = est_CompGraph_norm(x0, K, 1e-3, try_fast_norm)
    tau = 2.0 / (mu * L**2) / 10.

    if verbose:
        print("Estimated params [lambda = %3.3f | mu = %3.3f | L_est = %3.4f]" % (tau, mu, L))

    return tau, mu


def solve(InitEst=None, *prox_fns, **options):
    r"""TODO: Docstring for pds solve.
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
        :InitEst: initial value, default None.
        :*prox_fns: The users suppliers proximal operator of 'f' and 'g'.
        :**options: hyper-parameters and initial value of unknown image.
    :returns: restoration image and convergence information.

    """
    # default hyperparameters
    theta = options.get('theta', 1.0)
    eps_abs = options.get('eps_abs', 1e-4)
    eps_rel = options.get('eps_rel', 1e-4)
    verbose = options.get('verbose', False)
    max_iters = options.get('max_iters', 1000)

    # proximal operator
    f = options.get('f', lambda x : 0.)
    g = options.get('g', lambda x: 0.)
    prox_fn = prox_fns[0]
    prox_conjugate_gn = prox_fns[1]

    # linear operator: using 2D convolution implements
    kernel = torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]])
    kernel_t = torch.rot90(kernel, 2, (2, 3))
    D, Dt = options.get('D', kernel), options.get('Dt', kernel_t)

    # evalute parameters of proximal operator 'f' and 'g'
    tau, mu = est_params_lin_admm(InitEst.clone(), D, mu=None, verbose=False, scaled=False, try_fast_norm=False)

    # initial variables
    x = InitEst.clone()
    x_bar = InitEst.clone()
    y = torch.zeros_like(InitEst)
    u = torch.zeros_like(InitEst)
    z = torch.zeros_like(InitEst)

    # buffers
    Dx_bar = torch.zeros_like(InitEst)
    Dx = torch.zeros_like(InitEst)
    DTy = torch.zeros_like(InitEst)
    DTu = torch.zeros_like(InitEst)

    prev_x = x.clone()
    prev_Dx = Dx.clone()
    prev_z = z.clone()
    prev_u = y.clone()

    # convergence information
    objval = []
    r_norm, s_norm = [], []
    eps_pri, eps_dual = [], []
    shape = InitEst.size()
    N = shape[2]*shape[3]

    # output information
    if verbose:
        print('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n' % ('iter', \
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective'))

    # primal dual splitting method of multipliers solver
    for iter in range(max_iters):
        # store from previous iteration for residual calculation
        prev_Dx = Dx
        prev_u = u
        prev_z = z

        # update dual variable of lagrangian multipliers y
        Dx_bar = conv2d(x_bar, D, padding=(D.size()[2] // 2, D.size()[3] // 2))
        y_half = y + mu * Dx_bar
        y = y_half - mu * prox_conjugate_gn(y_half / mu, mu)

        # update x
        x_old = x
        DTy = conv2d(y, Dt, padding=(Dt.size()[2] // 2, Dt.size()[3] // 2))
        x_half = x - tau * DTy
        x = prox_fn(x_half, tau, x_old)

        # update x_bar
        x_bar = x + theta*(x - x_old)

        """ Old convergence check
        # Very basic convergence check.
        r_x = (1./np.sqrt(N))*torch.norm(x - x_old, p='fro', dim=(2, 3), keepdim=True)
        r_xbar = (1./np.sqrt(N))*torch.norm( x_bar - xbar_old, p=2, dim=(2, 3), keepdim=True)
        r_u = (1./np.sqrt(N))*torch.norm(u - u_old, p='fro', dim=(2, 3), keepdim=True)
        error = r_x + r_xbar + r_u
        """
        # stopping criteria
        Dx = conv2d(x, D, padding=(D.size()[2] // 2, D.size()[3] // 2))

        u = 1./mu * y + theta * (Dx - prev_Dx)
        DTu = conv2d(u, Dt, padding=(Dt.size()[2] // 2, Dt.size()[3] // 2))
        z = prev_u + prev_Dx - 1./mu * y

        objval.append(f(x) + g(Dx))
        r_norm.append(torch.norm(prev_Dx - z, p='fro', dim=(2, 3), keepdim=True))
        s_norm.append(torch.norm(mu * (z - prev_z), p='fro', dim=(2, 3), keepdim=True))

        eps_pri.append(np.sqrt(N)*eps_abs + eps_rel*torch.max(torch.norm(prev_Dx, p=2, dim=(2, 3), keepdim=True), \
                                                              torch.norm(z, p=2, dim=(2, 3), keepdim=True)))

        eps_dual.append(np.sqrt(N)*eps_abs + eps_rel*torch.norm(DTu, p=2, dim=(2, 3), keepdim=True) / mu)

        if all(r_norm[-1] < eps_pri[-1] and s_norm[-1] < eps_dual[-1]):
            break

        # output information
        if verbose:
            print('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n' % \
                  (iter, r_norm[-1], eps_pri[-1], s_norm[-1], eps_dual[-1], objval[-1]))
    # restoration image
    solution = x
    return solution

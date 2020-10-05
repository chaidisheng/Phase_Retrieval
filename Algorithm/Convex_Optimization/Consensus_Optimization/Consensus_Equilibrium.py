#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    Consensus_Equilibrium.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        3/29/20 6:18 AM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import torch
from threading import Thread
import time, queue, threading
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import compare_psnr
from utils.utils import *
from torch import Tensor


def solve(InitEst=None, *prox_fns, **options):
    # type: (Tensor, list, dict) -> Tensor
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
    lambd = options.get('lambd', torch.tensor(list()))
    noise_level = options.get('noise_level', torch.tensor(1.0))
    rho = options.get('rho', 1.0)
    gamma = options.get('gamma', 0.5)
    verbose = options.get('verbose', False)
    max_iters = options.get('max_iters', 10)
    ori_img = options.get('ori_img', None)
    prox_options = dict()

    # compute the weights
    p1 = torch.exp(-(lambd-noise_level) ** 2 / (2 * (5 / 255) ** 2))
    p1 = p1 / torch.sum(p1)
    p = torch.cat([torch.sum(p1, dim=0, keepdim=True), p1])
    p = p / torch.sum(p)
    print("Weights of PnP-CE are :", p)

    # initialization
    num_denoiser = len(prox_fns)-1
    psnrFv = np.zeros((max_iters+1, num_denoiser+1))  # individual denoiser
    psnrzhat = np.zeros((max_iters+1, 1))  # CE solution

    zhat_init = torch.cat([InitEst.clone()] * (num_denoiser+1), dim=1, out=None)
    # stochastic initializer
    Fv = torch.ones_like(zhat_init)  # Fv = torch.rand_like(zhat_init)
    w = zhat_init
    p = p.view(1, num_denoiser+1, 1, 1).to(InitEst.device)
    zhat = torch.sum(p * w, dim=1, keepdim=True)

    # parallel processing
    threads, dequeue = [], [queue.Queue(1) for _ in range(num_denoiser+1)]

    # output information
    if verbose:
        print("Running Consensus Equilibrium")
        print("iter \t inv \t D1 \t D2 \t D3 \t D4 \t D5 \t CE \n")

    # ==== CE main routine ====
    # consensus equilibrium solver
    for iter in range(max_iters):
        # === update v ===
        Gw = zhat
        Gw = Gw.repeat(1, num_denoiser+1, 1, 1)
        v = 2 * Gw-w

        # === update w ===
        prox_options['lambd'] = 1. / rho
        prox_options['x'] = Fv[:, 0:1, :, :]  # it also is Fv[:, [0], :, :]

        start = time.time()
        # non parallel and distribution
        # Fv[:, 0:1, :, :] = prox_fns[0](v[:, 0:1, :, :], prox_options)
        # for i in range(1, num_denoiser+1):
        #     Fv[:, i:i+1, :, :] = prox_fns[i](v[:, i:i+1, :, :], lambd[i-1]/rho)
        # Fv = torch.cat(Fv, dim=1)
        # print("One time consuming: %.2f" % (time.time()-start))

        # multithreading
        t = Thread(target=lambda q, f, f_args: q.put(f(*f_args)),
                   args=(dequeue[0], prox_fns[0], [v[:, 0:1, :, :], prox_options]))
        threads.append(t)
        t.start()
        t.join()
        for i in range(1, num_denoiser+1):
            t = Thread(target=lambda q, f, f_args: q.put(f(*f_args)),  # Fv[:, [i], :, :]
                       args=(dequeue[i], prox_fns[i], [v[:, i:i+1, :, :], lambd[i-1] / rho]))
            threads.append(t)
            t.start()
            # t.join()

        for t in threads:
            # print(t.getName(), threading.active_count())
            t.join()

        Fv_queue = list()
        for i in range(num_denoiser+1):
            while not dequeue[i].empty():
                Fv_queue.append(dequeue[i].get())

        Fv = torch.cat(Fv_queue, dim=1)
        print("One time consuming: %.2f" % (time.time()-start))
        w = (1 - gamma) * w + gamma * (2 * Fv - v)

        # === Compute zhat ===
        zhat = torch.sum(p * w, dim=1, keepdim=True)
        print('%3g \t' % iter, end='')
        for i in range(0, num_denoiser+1):
            psnrFv[iter+1, i] = compare_psnr(torch_to_np(Fv[:, i:i+1, :, :]), torch_to_np(ori_img), data_range=1)
            print('%3.2f \t' % psnrFv[iter+1, i], end='')

        psnrzhat[iter+1] = compare_psnr(torch_to_np(zhat), torch_to_np(ori_img), data_range=1)
        print('%3.2f' % psnrzhat[iter+1])

        # joining the thread
        for t in threads:
            if t.is_alive():
                t.join()

    return zhat

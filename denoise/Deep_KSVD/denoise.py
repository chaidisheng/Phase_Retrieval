#!/usr/bin/python
# coding:utf-8
# author: chaidisheng

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import torch
import torch.nn as nn
from scipy import linalg
import denoise.Deep_KSVD.Deep_KSVD as Deep_KSVD
from torch.autograd import Variable

def denoise(noisy, sigma, blind=True):
    """LKSVD denoise operator"""
    if not blind:
        # non-blind noise
        if sigma >= 55:
            logdir = 'denoise/Deep_KSVD/model/train_60'
        elif sigma >= 50:
            logdir = 'denoise/Deep_KSVD/model/train_55'
        elif sigma >= 45:
            logdir = 'denoise/Deep_KSVD/model/train_50'
        elif sigma >= 40:
            logdir = 'denoise/Deep_KSVD/model/train_45'
        elif sigma >= 35:
            logdir = 'denoise/Deep_KSVD/model/train_40'
        elif sigma >= 30:
            logdir = 'denoise/Deep_KSVD/model/train_35'
        elif sigma >= 25:
            logdir = 'denoise/Deep_KSVD/model/train_30'
        elif sigma >= 20:
            logdir = 'denoise/Deep_KSVD/model/train_25'
        elif sigma >= 15:
            logdir = 'denoise/Deep_KSVD/model/train_20'
        elif sigma >= 10:
            logdir = 'denoise/Deep_KSVD/model/train_15'
        else:
            logdir = "denoise/Deep_KSVD/model/train_10"
    else:
        # blind noise
        if sigma >= 80:
            logdir = 'denoise/Deep_KSVD/model/train_80_100'
        elif sigma >= 60:
            logdir = 'denoise/Deep_KSVD/model/train_60_80'
        elif sigma >= 40:
            logdir = 'denoise/Deep_KSVD/model/train_40_60'
        elif sigma >= 20:
            logdir = 'denoise/Deep_KSVD/model/train_20_40'
        elif sigma >= 10:
            logdir = 'denoise/Deep_KSVD/model/train_10_20'
        else:
            logdir = 'denoise/Deep_KSVD/model/train_0_10'

    # Rescaling in [-1, 1]:
    mean = 1. / 2
    std = 1. / 2
    noisy = (noisy - mean)/std
    # Overcomplete Discrete Cosinus Transform:
    patch_size = 8
    m = 16
    Dict_init = Deep_KSVD.Init_DCT(patch_size, m)
    Dict_init = Dict_init.to(noisy.device)
    #print(Dict_init.shape)

    # Squared Spectral norm:
    c_init = linalg.norm(Dict_init.cpu(), ord=2) ** 2  # Dict_init
    c_init = torch.FloatTensor((c_init,))
    c_init = c_init.to(noisy.device)

    # Average weight:
    w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
    w_init = w_init.to(noisy.device)

    # Deep-KSVD:
    D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = 64, 128, 64, 32, 1, 7, -1, 1
    model = Deep_KSVD.DenoisingNet_MLP_3(
        patch_size,
        D_in,
        H_1,
        H_2,
        H_3,
        D_out_lam,
        T,
        min_v,
        max_v,
        Dict_init,
        c_init,
        w_init,
        w_init,
        w_init,
        noisy.device,
    )
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), logdir, 'model.pth'), map_location='cpu'))
    model.to(noisy.device)

    model.eval()
    # noisy = Variable(noisy)
    with torch.no_grad():  # this can save much memory
        # output = torch.clamp(model(noisy), 0., 1.)
        output = model(noisy)
        output = output*std + mean
    return output

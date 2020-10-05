#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    denoise.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        9/7/20 7:48 PM
# @torch: tensor.method(in-place) or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import torch
import torch.nn as nn
from .model.full_realsn_models import DnCNN
from torch.autograd import Variable


def denoise(noisy, sigma, blind=True):
    r"""DncNN denoise operator"""

    if not blind:
        # non-blind noise
        if sigma > 45:
            logdir = 'denoise/realsn_DnCNN/logs/realsn_DnCNN-50'
        elif sigma > 40:
            logdir = 'denoise/realsn_DnCNN/logs/realsn_DnCNN-45'
        elif sigma > 35:
            logdir = 'denoise/realsn_DnCNN/logs/realsn_DnCNN-40'
        elif sigma > 30:
            logdir = 'denoise/realsn_DnCNN/logs/realsn_DnCNN-35'
        elif sigma > 25:
            logdir = 'denoise/realsn_DnCNN/logs/realsn_DnCNN-30'
        elif sigma > 20:
            logdir = 'denoise/realsn_DnCNN/logs/realsn_DnCNN-25'
        elif sigma > 15:
            logdir = 'denoise/realsn_DnCNN/logs/realsn_DnCNN-20'
        elif sigma > 10:
            logdir = 'denoise/realsn_DnCNN/logs/realsn_DnCNN-15'
        else:
            logdir = "denoise/realsn_DnCNN/logs/realsn_DnCNN-10"
    else:
        # blind noise: the number of layers are 17.
        logdir = 'denoise/realsn_DnCNN/logs/realsn_DnCNN-1_50'

    # Build model
    net = DnCNN(channels=1, num_of_layers=17, lip=1.0, no_bn=False)
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # model = nn.DataParallel(net, device_ids=[0, 1])
    model = nn.DataParallel(net, device_ids=[noisy.device])
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), logdir, 'net.pth'),  map_location=noisy.device))
    model.eval()
    noisy = Variable(noisy)
    with torch.no_grad():  # this can save much memory
        output = torch.clamp(noisy - model(noisy), 0., 1.)
    return output

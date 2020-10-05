#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    denoise.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        9/13/20 3:50 AM
# @torch: tensor.method(in-place) or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import torch
import torch.nn as nn
from .residual_unet import ResUnet
from torch.autograd import Variable


def denoise(noisy, sigma, blind=True):
    r"""DncNN denoise operator"""

    if not blind:
        # non-blind noise
        logdir = "./"
    else:
        # blind noise
        if sigma >= 20:
            logdir = 'denoise/ResUnet/logs/u-net-20_40'
        elif sigma >= 10:
            logdir = 'denoise/ResUnet/logs/u-net-10_20'
        elif sigma >= 0:
            logdir = 'denoise/DnCNN/logss/DnCNN-B-0_10'
        else:
            logdir = 'denoise/ResUnet/logs/u-net-1_50'

    # Build model
    net = ResUnet(num_input_channels=1, num_output_channels=1, feature_scale=4)
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # model = nn.DataParallel(net, device_ids=[0, 1])
    model = nn.DataParallel(net, device_ids=[noisy.device])
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), logdir, 'net.pth'),  map_location=noisy.device))
    model.eval()
    noisy = Variable(noisy)
    with torch.no_grad():  # this can save much memory
        output = torch.clamp(model(noisy), 0., 1.)
    return output


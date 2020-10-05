#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: ??
# @license: Apache Licence 
# @Filename:    denoise_red.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        9/7/20 8:39 PM
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
        if sigma > 5:
            logdir = 'denoise/realsn_DnCNN/logs_red/realsn_DnCNN-7'
        elif sigma > 3:
            logdir = 'denoise/realsn_DnCNN/logs_red/realsn_DnCNN-5'
        else:
            logdir = 'denoise/realsn_DnCNN/logs_red/realsn_DnCNN-3'

    # Build model
    net = DnCNN(channels=1, num_of_layers=10, lip=1.0, no_bn=True)
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


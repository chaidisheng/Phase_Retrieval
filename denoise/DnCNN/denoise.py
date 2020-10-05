#! usr/bin/python
# coding:utf-8
# author: chaidisheng

from __future__ import print_function
import os
import torch
import torch.nn as nn
from .models import DnCNN
from torch.autograd import Variable


def denoise(noisy, sigma, blind=True):
    r"""DncNN denoise operator"""

    if not blind:
        # non-blind noise
        if sigma > 55:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-60'
        elif sigma > 50:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-55'
        elif sigma > 45:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-50'
        elif sigma > 40:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-45'
        elif sigma > 35:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-40'
        elif sigma > 30:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-35'
        elif sigma > 25:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-30'
        elif sigma > 20:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-25'
        elif sigma > 15:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-20'
        elif sigma > 10:
            logdir = 'denoise/DnCNN/logs/DnCNN-S-15'
        else:
            logdir = "denoise/DnCNN/logs/DnCNN-S-10"
    else:
        # blind noise: the number of layers are 17.
        if sigma >= 80:
            logdir = 'denoise/DnCNN/logss/DnCNN-B-80_100'
        elif sigma >= 60:
            logdir = 'denoise/DnCNN/logss/DnCNN-B-60_80'
        elif sigma >= 40:
            logdir = 'denoise/DnCNN/logss/DnCNN-B-40_60'
        elif sigma >= 20:
            logdir = 'denoise/DnCNN/logss/DnCNN-B-20_40'
        elif sigma >= 10:
            logdir = 'denoise/DnCNN/logss/DnCNN-B-10_20'
        elif sigma >= 0:
            logdir = 'denoise/DnCNN/logss/DnCNN-B-0_10'
        else:
            logdir = 'denoise/DnCNN/logss/DnCNN-1_50'

    # Build model
    net = DnCNN(channels=1, num_of_layers=17)
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

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    PnP_AMP.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        4/13/20 6:42 AM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from Algorithm.Convex_Optimization.DAMP.DAMP import DAMP


def PnP_AMP(y, A, At, x0, **options):
    # type: (Tensor, object or float, object or float, Tensor, dict) -> Tensor
    r"""prDeep"""
    # Hyperparameters: check preconditions, fill missing optional entries on 'options'
    if not isinstance(A, (int, float)):
        assert not isinstance(At, (int, float)), "If A is a function handle, then At must be a handle as well!"

    if isinstance(A, (int, float)):
        At = lambda x: torch.matmul(A.permute(0, 1, 3, 2), x)
        A = lambda x: torch.matmul(A, x)

    # if user didn't pass this arg, then create it
    if not options:
        options = dict()

    solution = DAMP(y, A, At, x0, **options)
    return solution
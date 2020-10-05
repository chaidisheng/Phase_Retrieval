#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 1.0
# @license: Apache Licence
# @Filename:    ll.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        2/17/20 3:54 AM

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

"""Proximal Operators

This "library" contains sample implementations of various proximal operators in
Matlab. These implementations are intended to be pedagogical, not the most
performant.

This code is associated with the paper 
*[Proximal Algorithms](http://www.stanford.edu/~boyd/papers/prox_algs.html)* 
by Neal Parikh and Stephen Boyd.

The torch functions include the following examples:

* Projection onto an affine set
* Projection onto a box
* Projection onto the consensus set (averaging)
* Projection onto the exponential cone
* Projection onto the nonnegative orthant
* Projection onto the second-order cone
* Projection onto the semidefinite cone
* Proximal operator of a generic function (via CVX)
* Proximal operator of the *l1* norm
* Proximal operator of the max function
* Proximal operator of a quadratic function
* Proximal operator of a generic scalar function (vectorized)
* Proximal operator of an orthogonally invariant matrix function
* Precomposition of a proximal operator

"""
# Authors
"""
* [Neal Parikh](http://cs.stanford.edu/~npparikh)
* [Eric Chu](http://www.stanford.edu/~echu508)
* [Stephen Boyd](http://www.stanford.edu/~boyd)
* [chaidishneg pytorch](https://github.com/chaidisheng)

"""


def prox_l0(tensor, lambd):
    """ proximal operator of L0 penalty. """
    shrink = torch.full_like(tensor, lambd)
    zeros = torch.zeros_like(tensor)
    return torch.where(tensor > shrink, tensor, zeros)


def prox_l1(tensor, lambd):
    """ proximal operator of L1 norm. """
    return tensor*torch.max(torch.ones_like(tensor) - lambd/torch.abs(tensor), torch.zeros_like(tensor))


def soft_shrink(tensor, lambd):
    """ soft_shrink. """
    return torch.max(tensor - lambd, torch.tensor(0.)) - torch.max(-tensor - lambd, torch.tensor(0.))


def prox_sum_square(tensor, lambd):
    """ pro_sum_square.
    (1/2)||.||_2^2 with parameter lambd.
    """
    return (1./(1. + lambd))*tensor


def prox_l2(tensor, lambd):
    """ proximal operator of l2 norm.
    ||.||_2 with parameter lambd.
    """
    return tensor*torch.max(1. - lambd/torch.sqrt(torch.sum(tensor**2, (2, 3), keepdim=True)), torch.zeros_like(tensor))


def project_affine(arg1):
    """TODO: Docstring for project_affine.

    :arg1: TODO
    :returns: TODO

    """
    pass

def project_box(arg1):
    """TODO: Docstring for project_box.

    :arg1: TODO
    :returns: TODO

    """
    pass


def project_consensus(arg1):
    """TODO: Docstring for project_consensus.

    :arg1: TODO
    :returns: TODO

    """
    pass


def project_exp(arg1):
    """TODO: Docstring for project_exp.

    :arg1: TODO
    :returns: TODO

    """
    pass


def project_graph(arg1):
    """TODO: Docstring for project_graph.

    :arg1: TODO
    :returns: TODO

    """
    pass


def project_pos(arg1):
    """TODO: Docstring for project_pos.

    :arg1: TODO
    :returns: TODO

    """
    pass


def project_sdc(arg1):
    """TODO: Docstring for project_sdc.

    :arg1: TODO
    :returns: TODO

    """
    pass


def project_soc(arg1):
    """TODO: Docstring for project_soc.

    :arg1: TODO
    :returns: TODO

    """
    pass


def prox_cvx(arg1):
    """TODO: Docstring for prox_cvx.

    :arg1: TODO
    :returns: TODO

    """
    pass


def prox_matrix(arg1):
    """TODO: Docstring for prox_matrix.
    
    :arg1: TODO
    :returns: TODO
    
    """
    pass


def prox_max(arg1):
    """TODO: Docstring for prox_max.
    
    :arg1: TODO
    :returns: TODO
    
    """
    pass
    
    
def prox_precompose(arg1):
    """TODO: Docstring for prox_precompose.
    
    :arg1: TODO
    :returns: TODO
    
    """
    pass


def prox_quad(arg1):
    """TODO: Docstring for prox_quad.
    
    :arg1: TODO
    :returns: TODO
    
    """
    pass


def prox_separable(arg1):
    """TODO: Docstring for prox_separable.
    
    :arg1: TODO
    :returns: TODO
    
    """
    pass
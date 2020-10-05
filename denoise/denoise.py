#! usr/bin/python
# coding:utf-8
# author: chaidisheng

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import numpy as np
from bm3d import bm3d
from denoise.DnCNN.denoise import denoise as DnCNN
# from denoise.Deep_Image_Prior.denoise import denoise as DIP  but DIP is slow
from denoise.Deep_KSVD.denoise import denoise as LKSVD
from denoise.realsn_DnCNN.denoise import denoise as realsn_DnCNN
from denoise.realsn_DnCNN.denoise_red import denoise as realsn_DnCNN_red
from denoise.realsn_DnCNN.denoise_dae import denoise as realsn_DnCNN_dae
from denoise.ResUnet.denoise import denoise as ResUnet
from skimage.restoration import denoise_nl_means


def non_local_means(noisy_np_img, sigma, fast_mode=True):
    """ get a numpy noisy image
        returns a denoised numpy image using Non-Local-Means
    """
    sigma = sigma / 255.
    h = 0.6 * sigma if fast_mode else 0.8 * sigma
    patch_kw = dict(h=h,                   # Cut-off distance, a higher h results in a smoother image
                    sigma=sigma,           # sigma provided
                    fast_mode=fast_mode,   # If True, a fast version is used. If False, the original version is used.
                    patch_size=5,          # 5x5 patches (Size of patches used for denoising.)
                    patch_distance=6,      # 13x13 search area
                    multichannel=False)
    denoised_img = []
    mini_batch = noisy_np_img.shape[0]
    n_channels = noisy_np_img.shape[1]
    for n in range(mini_batch):
        for c in range(n_channels):
            denoise_fast = denoise_nl_means(noisy_np_img[n, c, :, :].numpy(), **patch_kw)
            denoised_img += [denoise_fast]
    return np.array(denoised_img, dtype=np.float32)


def denoise(noisy, sigma, blind=True, denoise_method='DnCNN'):
    r"""denoise method"""
    if denoise_method == 'DnCNN':
        return DnCNN(noisy, sigma, blind)
    elif denoise_method == 'realsn_DnCNN':
        return realsn_DnCNN(noisy, sigma, blind)
    elif denoise_method == 'realsn_DnCNN_red':
        return realsn_DnCNN_red(noisy, sigma, blind)
    elif denoise_method == 'realsn_DnCNN_dae':
        return realsn_DnCNN_dae(noisy, sigma, blind)
    elif denoise_method == 'ResUnet':
        return ResUnet(noisy, sigma, blind)
    # elif denoise_method == 'DIP':
    #    return DIP(noisy, sigma, blind)
    elif denoise_method == 'NLM':
        return torch.tensor(non_local_means(noisy.cpu(), sigma), dtype=torch.float32,
                            device=noisy.device).view(noisy.size())
    elif denoise_method == 'BM3D':
        shape = noisy.size()
        return torch.tensor(bm3d(noisy.cpu().view(shape[2], shape[3], shape[1]), sigma / 255.),
                            dtype=torch.float32, device=noisy.device).view(noisy.size())
    else:
        return LKSVD(noisy, sigma, blind)

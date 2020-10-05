#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence
# @Filename:    fff.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        2/21/20 4:46 AM
# @torch: tensor.method or torch.method(tensor)

from __future__ import print_function
import torch
import random
import numpy as np
import torch.nn as nn
import scipy.io as io

def save_mat(tensor):
    # save mat file
    out = tensor.squeeze().numpy()
    image = np.mat(out)
    io.savemat('image.mat', {'matrix': image})


def fft2(tensor, order=2):
    r""" torch.fft """
    return torch.fft(tensor, order)


def ifft2(amplitude, order=2):
    r""" torch.ifft """
    return torch.ifft(amplitude, order)


def conjugate(tensor):
    r""" complex tensor conjugate """
    return torch.cat((tensor[:, :, :, 0:1], torch.neg(tensor[:, :, :, 1:2])), 3)


def complex_sign(tensor):
    r""" complex sign """
    return torch.sqrt(torch.sum(tensor ** 2, 3, keepdim=True))


def SNR(*tensor):
    r"""Compute SNR"""
    absolute = torch.sqrt(torch.sum(tensor[0] ** 2, dim=3, keepdim=True))
    snr = torch.norm(absolute, dim=2, keepdim=True)
    snr /= torch.norm(absolute - tensor[1], dim=2, keepdim=True)
    return 20 * torch.log10(snr)


def _SNR(*tensor):
    r"""Compute SNR"""
    absolute = torch.sqrt(torch.sum(tensor[0] ** 2, dim=2, keepdim=True))
    snr = torch.norm(absolute, dim=1, keepdim=True)
    snr /= torch.norm(absolute - tensor[1], dim=1, keepdim=True)
    return 20 * torch.log10(snr)


def trace(tensor):  # torch.einsum()
    r"""trace of tensor"""
    pass


def four_mask(tensor):
    r""" four mask """
    return np.random.choice([1, -1, complex('j'), complex('-j')])


def gaussion_forward(tensor, sampling_rate=1, seed=0):
    r"""Generate Gaussian Measurement Forward Matrix"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    N, C, H, W = tensor.shape
    m, n = int(sampling_rate * H * W), H * W
    global matrix_real, matrix_imag
    matrix_real = torch.tensor(1.0 / np.sqrt(2.0) * 1.0 / np.sqrt(m)) * torch.randn([N, C, m, n]).to(tensor.device)
    matrix_imag = torch.tensor(1.0 / np.sqrt(2.0) * 1.0 / np.sqrt(m)) * torch.randn([N, C, m, n]).to(tensor.device)
    measure_real = torch.matmul(matrix_real, tensor.view([N, C, H * W, 1]))
    measure_imag = torch.matmul(matrix_imag, tensor.view([N, C, H * W, 1]))
    measure = torch.cat((measure_real, measure_imag), 3)

    return measure  # NxCxmx2


def gaussion_backward_pinv(amplitude, shape, sampling_rate=1):
    r"""Generator Gaussion Backward Pinv Matrix"""

    N, C, H, W = shape
    m, n = sampling_rate * H * W, H * W
    # matrix_pinv = torch.pinverse(matrix);
    # measure_pinv = matrix_pinv*amplitude(:)
    # svd method

    # return measure_pinv


def gaussion_backward_hermition(amplitude, shape, sampling_rate=1):
    r"""Generator Gaussion Backward Hermition Matrix"""

    N, C, H, W = shape
    m, n = sampling_rate * H * W, H * W
    matrix_real_t = torch.transpose(matrix_real, 2, 3)
    matrix_imag_t = torch.neg(torch.transpose(matrix_imag, 2, 3))
    real_t = torch.matmul(matrix_real_t, amplitude[:, :, :, 0:1])  # real -> real
    imag_t = torch.matmul(matrix_imag_t, amplitude[:, :, :, 1:2])  # imag -> imag
    measure_t = real_t-imag_t

    return measure_t  # NxCxnx1


def code_diffraction_forward_3D(tensor, sampling_rate=1, seed=0):
    r"""Generate Coded 3D Diffraction Pattern Measurement Forward Matrix"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    C, H, W = tensor.shape
    m, n = int(sampling_rate * H * W), H * W

    # undersampling
    under_rand = torch.rand([C, n, 1]).to(tensor.device)
    sign_under_real = torch.cos(2.0 * torch.tensor(np.pi) * under_rand)
    sign_under_imag = torch.sin(2.0 * torch.tensor(np.pi) * under_rand)
    sign_under = torch.cat((sign_under_real, sign_under_imag), 2)

    # oversampling
    over_rand = torch.rand(C, m, 1).to(tensor.device)
    sign_over_real = torch.cos(2.0 * torch.tensor(np.pi) * over_rand)
    sign_over_imag = torch.sin(2.0 * torch.tensor(np.pi) * over_rand)
    sign_over = torch.cat((sign_over_real, sign_over_imag), 2)

    if m < n:  # undersampling
        # Euler Form
        print("forward undersampling begins running!")
        # every batch image has same random sample
        list_sample = [i for i in range(n-1)]
        inds = random.sample(list_sample, m-1)
        inds = [j+1 for j in inds]
        inds.insert(0, 0)
        I = torch.eye(n).view([1, 1, n, n]).expand([C, n, n])
        sub_sample = I[:, :, inds, :]
        measure_fft = fft2(sign_under * tensor.view([C, n, 1]).view([C, H, W, 2])).view([C, n, 2])
        # 1.0/np.sqrt(n)*np.sqrt(float(n)/m)
        measure = torch.tensor(1.0) / torch.tensor(np.sqrt(m)) * torch.matmul(sub_sample, measure_fft)  # NxCxmx2
        del measure_fft
        return measure  # NxCxmx2

    elif round(m / n) != m / n:

        print("Oversampled coded diffraction patterns need m/n to be an integer.")
        return -1

    else:  # oversampling
        print("forward oversampling begins running!")
        tensor_copy = tensor.view(C, n, 1)
        tensor_copy = tensor_copy.repeat(1, 1, sampling_rate, 1)
        measure_fft = fft2((sign_over * tensor_copy).view(C, sampling_rate, H, W, 2))
        measure = torch.tensor(1.0) / torch.tensor(np.sqrt(n)) * measure_fft.view(C, m, 2)
        del tensor_copy
        del measure_fft

        return measure  # Cxmx2
    # return -1


def code_diffraction_forward(tensor, sampling_rate=1, seed=0):
    r"""Generate Coded Diffraction Pattern Measurement Forward Matrix"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    N, C, H, W = tensor.shape
    m, n = int(sampling_rate * H * W), H * W

    # undersampling
    global under_rand, sign_under_real, sign_under_imag, sign_under
    under_rand = torch.rand([N, C, n, 1]).to(tensor.device)
    sign_under_real = torch.cos(2.0 * torch.tensor(np.pi) * under_rand)
    sign_under_imag = torch.sin(2.0 * torch.tensor(np.pi) * under_rand)
    sign_under = torch.cat((sign_under_real, sign_under_imag), 3)

    # oversampling
    global over_rand, sign_over_real, sign_over_imag, sign_over
    over_rand = torch.rand(N, C, m, 1).to(tensor.device)
    sign_over_real = torch.cos(2.0 * torch.tensor(np.pi) * over_rand)
    sign_over_imag = torch.sin(2.0 * torch.tensor(np.pi) * over_rand)
    # sign_over_real = [np.random.choice([0, 1, -1]) for _ in range(N*C*m*1)]
    # sign_over_imag = [np.random.choice([1, -1]) if sign_over_real[i]==0 else 0 for i in range(N*C*m*1) ]
    # sign_over_real = torch.tensor(sign_over_real).view(N, C, m, 1).to(tensor.device)
    # sign_over_imag = torch.tensor(sign_over_imag).view(N, C, m, 1).to(tensor.device)
    sign_over = torch.cat((sign_over_real, sign_over_imag), dim=3)

    if m < n:  # undersampling
        # Euler Form
        # print("forward undersampling begins running!")
        # every batch image has same random sample
        list_sample = [i for i in range(n-1)]
        inds = random.sample(list_sample, m-1)
        inds = [j+1 for j in inds]
        inds.insert(0, 0)
        I = torch.eye(n).view([1, 1, n, n]).expand([N, C, n, n])
        sub_sample = I[:, :, inds, :]
        measure_fft = fft2(sign_under * tensor.view([N, C, n, 1]).view([N, C, H, W, 2])).view([N, C, n, 2])
        # 1.0/np.sqrt(n)*np.sqrt(float(n)/m)
        measure = torch.tensor(1.0) / torch.tensor(np.sqrt(m)) * torch.matmul(sub_sample, measure_fft)  # NxCxmx2
        del measure_fft
        return measure  # NxCxmx2

    elif round(m / n) != m / n:

        print("Oversampled coded diffraction patterns need m/n to be an integer.")
        return -1

    else:  # oversampling
        # print("forward oversampling begins running!")
        tensor_copy = tensor.view(N, C, n, 1)
        tensor_copy = tensor_copy.repeat(1, 1, sampling_rate, 1)
        measure_fft = fft2((sign_over * tensor_copy).view(N, C, sampling_rate, H, W, 2))
        measure = torch.tensor(1.0) / torch.tensor(np.sqrt(n)) * measure_fft.view(N, C, m, 2)
        del tensor_copy
        del measure_fft

        return measure  # NxCxmx2
    # return -1


def code_diffraction_backward(amplitude, shape, sampling_rate=1):
    r"""Generator Code Diffraction Backward Matrix"""

    N, C, H, W = shape
    m, n = int(sampling_rate * H * W), H * W

    if m < n:  # undersampling
        # Euler Form
        print("backward undersampling begins running!")
        list_sample = [i for i in range(n-1)]
        inds = random.sample(list_sample, m-1)
        inds = [i+1 for i in inds]
        inds.insert(0, 0)
        I = torch.eye(n).view([1, 1, n, n]).expand([N, C, n, n])
        sub_sample = I[:, :, inds, :]
        sub_sample_t = torch.transpose(sub_sample, 2, 3)
        amp_sub_real = torch.matmul(sub_sample_t, amplitude[:, :, :, 0:1]).view(N, C, H, W)
        amp_sub_imag = torch.matmul(sub_sample_t, amplitude[:, :, :, 1:2]).view(N, C, H, W)
        measure_fft = ifft2(torch.stack((amp_sub_real, amp_sub_imag), 4).view([N, C, n, 2]))
        measure_t = torch.tensor(np.sqrt(n) * np.sqrt(float(n) / m)) * conjugate(sign_under) * measure_fft
        measure_t = measure_t[:, :, :, 0:1]-measure_t[:, :, :, 1:2]
        measure_t = measure_t.view(N, C, H, W)
        del amp_sub_real, amp_sub_imag, measure_fft
        return measure_t  # NxCxmx2

    elif round(m / n) != m / n:

        print("Oversampled coded diffraction patterns need m/n to be an integer.")
        return -1

    else:

        # print("backward oversampling begins running!"
        measure_ifft = ifft2(amplitude.view(N, C, sampling_rate, H, W, 2))
        measure_ifft = measure_ifft.view(N, C, m, 2)
        measure_conj = conjugate(sign_over) * measure_ifft
        measure_conj = measure_conj[:, :, :, 0:1]-measure_conj[:, :, :, 1:2]
        measure_sum = torch.sum(measure_conj.view(N, C, sampling_rate, H, W), 2, keepdim=True)
        measure_t = torch.squeeze(measure_sum, 2)
        measure_t = torch.tensor(np.sqrt(n)) * measure_t.view(N, C, H, W)
        del measure_sum
        del measure_ifft
        del measure_conj

        return measure_t  # NxCxnx2
    # return -1


def code_diffraction_backward_complex(amplitude, shape, sampling_rate=1):
    r"""Generator Code Diffraction Backward Complex Matrix"""

    N, C, H, W = shape
    m, n = int(sampling_rate * H * W), H * W

    if m < n:  # undersampling

        # Euler Form
        print("backward undersampling begins running!")
        list_sample = [i for i in range(n-1)]
        inds = random.sample(list_sample, m-1)
        inds = [i+1 for i in inds]
        inds.insert(0, 0)
        I = torch.eye(n).view([1, 1, n, n]).expand([N, C, n, n])
        sub_sample = I[:, :, inds, :]
        sub_sample_t = torch.transpose(sub_sample, 2, 3)
        amp_sub_real = torch.matmul(sub_sample_t, amplitude[:, :, :, 0:1]).view(N, C, H, W)
        amp_sub_imag = torch.matmul(sub_sample_t, amplitude[:, :, :, 1:2]).view(N, C, H, W)
        measure_fft = ifft2(torch.stack((amp_sub_real, amp_sub_imag), 4).view([N, C, n, 2]))
        measure_t = torch.tensor(np.sqrt(n) * np.sqrt(float(n) / m)) * conjugate(sign_under) * measure_fft
        measure_t = measure_t[:, :, :, 0:1]-measure_t[:, :, :, 1:2]
        measure_t = measure_t.view(N, C, H, W)
        del amp_sub_real, amp_sub_imag, measure_fft
        return measure_t  # NxCxmx2

    elif round(m / n) != m / n:

        print("Oversampled coded diffraction patterns need m/n to be an integer.")
        return -1

    else:

        print("backward oversampling begins running!")
        measure_ifft = ifft2(amplitude.view(N, C, sampling_rate, H, W, 2))
        measure_ifft = measure_ifft.view(N, C, m, 2)
        re_measure_conj = conjugate(sign_over) * measure_ifft
        im_measure_conj = conjugate(sign_over)[:, :, :, 0:1] * measure_ifft[:, :, :, 1:2] + \
                          conjugate(sign_over)[:, :, :, 1:2] * measure_ifft[:, :, :, 0:1]
        re_measure_conj = re_measure_conj[:, :, :, 0:1] - re_measure_conj[:, :, :, 1:2]
        measure_conj = torch.cat([re_measure_conj, im_measure_conj], dim=3)
        measure_sum = torch.sum(measure_conj.view(N, C, sampling_rate, H, W, 2), dim=2, keepdim=True)
        measure_t = torch.squeeze(measure_sum, dim=2)
        measure_t = torch.tensor(np.sqrt(n)) * measure_t.view(N, C, H, W, 2)  # NxCxHxWx2
        del measure_sum
        del measure_ifft
        del measure_conj

        return measure_t  # NxCxnx2
    # return -1


def fourier_forward(tensor, sampling_rate=1):
    r"""Generate Fourier Forward Measurement Matrix"""

    N, C, H, W = tensor.shape
    m, n = int(sampling_rate * H * W), H * W

    if m <= n:

        print("Undersampled fourier measurements not supported!")
        return -1

    else:
        # every batch image has same mask
        pad_number = int((np.sqrt(m)-np.sqrt(n)) * 1 / 2)
        I = torch.eye(m).view([1, 1, m, m]).expand([N, C, m, m])
        padarray = nn.ZeroPad2d(pad_number)
        mask = padarray(torch.ones(H, W)).bool()
        over_sample = I[:, :, :, mask.view(-1)]
        tensor_sample = torch.matmul(over_sample, tensor.view([N, C, n, 1]))
        view_sample = tensor_sample.view([N, C, int(np.sqrt(m / n) * H), int(np.sqrt(m / n) * W), 1])

        measure_fft = fft2(torch.cat((view_sample, torch.zeros_like(view_sample)), 4))
        measure = torch.tensor(1.0 / np.sqrt(m) * np.sqrt(float(n) / m)) * measure_fft.view([N, C, m, 2])

        measure_matrix = measure.view([N, C, int(np.sqrt(m)), int(np.sqrt(m)), 2])

        # F(x*x) = |F(x)|.^2
        tensor_square = tensor.view(N, C, n, 1)
        measure_square = fft2(torch.cat((tensor_square, torch.zeros_like(tensor_square)), 3))
        measure_square = torch.tensor(1.0 / np.sqrt(m) * np.sqrt(float(n) / m)) * measure_square.pow(2)
        del tensor_sample
        del view_sample
        del measure_fft

        return measure  # NxCxmx2
    # return 1


def fourier_backward(amplitude, shape, sampling_rate=1):
    r"""Generate Fourier Backward Measurement Matrix"""

    N, C, H, W = shape
    m, n = int(sampling_rate * H * W), H * W

    if m <= n:

        print('Undersampled Fourier measurements not supported!')
        return -1

    else:

        pad_number = int((np.sqrt(m)-np.sqrt(n)) * 1 / 2)
        I = torch.eye(m).view([1, 1, m, m]).expand([N, C, m, m])
        padarray = nn.ZeroPad2d(pad_number)
        mask = padarray(torch.ones((H, W))).bool()
        over_sample = I[:, :, :, mask.view(-1)]
        amplitude = amplitude.view([N, C, int(np.sqrt(m / n) * H), int(np.sqrt(m / n) * W), 1])
        measure_ifft = ifft2(amplitude)
        over_sample_t = over_sample.permute(0, 1, 3, 2)
        measure_t = torch.matmul(over_sample_t, measure_ifft.view((N, C, m, 2)))
        measure_t = torch.tensor(np.sqrt(m) * np.sqrt(float(n) / m)) * measure_t[:, :, :, 0:1]

        measure_matrix_t = measure_t.view(N, C, H, W)

        amplitude_square = amplitude.view(N, C, m, 2)
        measure_square = ifft2(amplitude_square)
        measure_square = torch.tensor(np.sqrt(m) * np.sqrt(float(n) / m)) * measure_square.pow(2)
        del measure_ifft

        return measure_t  # NxCxnx2
    # return 1


def poisson(tensor, alpha=9. / 255, seed=0):
    r"""poisson noise model for CDP"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    intensity_noise = alpha * tensor * torch.randn(tensor.shape).to(tensor.device)
    measurement = tensor ** 2+intensity_noise
    # python3
    # measurement = torch.sqrt(measurement*(measurement > 0).type_as(measurement))
    # python2
    measurement = torch.sqrt(measurement * (measurement > 0).type_as(measurement))
    error = measurement-tensor
    sigma = torch.std(error, dim=2, keepdim=True)
    print("sigma is: {0:.2f}.".format(sigma.item()))
    return measurement, sigma


def gaussian(tensor, snr=10.0, seed=0):
    r"""gaussian noise model for CDP"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    noise = torch.randn(tensor.shape).to(tensor.device)
    noise = noise * torch.norm(tensor, dim=2, keepdim=True) / torch.norm(noise, dim=2, keepdim=True) / snr
    measurement = tensor + noise
    sigma = torch.std(noise, dim=2, keepdim=True)
    print("Sigma of measurement is: {0:.3f}.".format(sigma.item()))
    return measurement


def wgn(tensor, snr=15, seed=0):
    r"""white gaussian moise"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    snr = 10 ** (snr / 10.0)
    power = torch.mean(tensor.mul(tensor))
    power /= snr
    sigma = torch.sqrt(power)
    print("Sigma of measurement is: {0:.3f}.".format(sigma.item()))
    noise = sigma * torch.randn(tensor.shape).to(tensor.device)
    return noise, sigma


def rician(tensor, SNR=10.0, seed=0):
    r"""rician noise model for CDP"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    noise = torch.tensor(1.0 / np.sqrt(2)) * torch.randn(tensor.shape)
    noise = noise * torch.norm(tensor, dim=2, keepdim=True) / torch.norm(noise, dim=2, keepdim=True) / SNR
    measurement = torch.sqrt(torch.sum((tensor+noise) ** 2, dim=3, keepdim=True))
    sigma = torch.std(noise, dim=2, keepdim=True)
    print("sigma is: {0:.2f}.".format(sigma.item()))
    return measurement


def check_adjoint(A, At, x, seed=0):
    r""" check At = A' """

    torch.manual_seed(seed)
    x = torch.randn(x.size())
    Ax = A(x)
    y = torch.randn(Ax.shape)
    Aty = At(y, x.shape)

    Ax_real = Ax[:, :, :, 0:1].permute(0, 1, 3, 2)
    Ax_imag = torch.neg(Ax[:, :, :, 1:2]).permute(0, 1, 3, 2)
    y_real = y[:, :, :, 0:1]
    y_imag = y[:, :, :, 1:2]
    inner_real_1 = torch.matmul(Ax_real, y_real)-torch.matmul(Ax_imag, y_imag)
    inner_imag_1 = torch.matmul(Ax_real, y_imag)+torch.matmul(Ax_imag, y_real)
    print(inner_imag_1)
    inner_Product_1 = torch.cat((inner_real_1, inner_imag_1), dim=3)

    inner_Product_2 = torch.einsum('...ii->...', [torch.matmul(x.permute(0, 1, 3, 2), Aty)])  # trace: matrix
    inner_Product = inner_Product_1[:, :, :, 0:1]-inner_Product_2
    inner_Product = torch.cat((inner_Product, inner_Product_1[:, :, :, 1:2]), dim=3)

    error = torch.sqrt(torch.sum(inner_Product ** 2, dim=3, keepdim=True))
    print(error)
    error = error / torch.max(torch.sqrt(torch.sum(inner_Product_1 ** 2, dim=3, keepdim=True)),
                              torch.abs(inner_Product_2))
    print(error)
    assert error < 1e-9, "At is not the adjoint of A, check the definitions of these operators!"
    return -1


def test(seed=0):
    r"""test"""
    torch.manual_seed(seed)
    x = torch.randn(1, 1, 16, 16)
    check_adjoint(code_diffraction_forward, code_diffraction_backward, x)


if __name__ == '__main__':
    test()

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    figure.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        3/8/20 9:31 PM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import cv2
import glob
import torch
import numpy as np
import scipy.special
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import array

from denoise.denoise import denoise


def normalize(data):
    r"""normlize image"""
    return data / 255.0


def draw_rectangle(image, leftuppoint, rightbottompoint, linewidth):
    # type: (array, list, list, int) -> array
    r""" draw rectangle
    Examples:
        >>> Image = draw_rectangle(image, [10, 20], [50, 60], 1)
    """
    if image.shape[2] == 1:
        Image = np.concatenate([image]*3, axis=2)
    else:
        Image = image

    if not linewidth:
        linewidth = 1

    uprow, bottomrow = leftuppoint[0], rightbottompoint[0]
    leftcolumn, rightcolumn = leftuppoint[1], rightbottompoint[1]
    # set rectangle color
    rectangle = [255./255, 0., 0.]
    for i in range(3):
        Image[uprow:uprow + linewidth, leftcolumn:rightcolumn + 1, i] = rectangle[i]
        Image[bottomrow:bottomrow + linewidth, leftcolumn:rightcolumn + 1, i] = rectangle[i]
        Image[uprow:bottomrow + 1, leftcolumn:leftcolumn + linewidth, i] = rectangle[i]
        Image[uprow:bottomrow + 1, rightcolumn:rightcolumn + linewidth, i] = rectangle[i]

    return Image


def show_enlarge_rectangle(image, leftuppoint, rightbottompoint, enlargement, linewidth, gap):
    # type: (array, list, list, float, int, int) -> array
    r"""TODO: Docstring for show_enlarge_rectangle.
    Args:
        :image, leftuppoint, rightbottompoint, enlargement, linewidth, gap: TODO
        :returns: TODO
    Examples:
        >>> Image = show_enlarge_rectangle(image, [10, 20], [50, 60], 1.5, 1.)
    """

    if image.shape[2] == 1:
        Image = np.concatenate([image] * 3, axis=2)
    else:
        Image = image

    if not linewidth:
        linewidth = 1

    if not enlargement:
        enlargement = 1.5

    if not gap:
        gap = 1

    Image = draw_rectangle(Image, leftuppoint,  rightbottompoint,  linewidth)
    uprow, bottomrow = leftuppoint[0], rightbottompoint[0]
    leftcolumn, rightcolumn = leftuppoint[1], rightbottompoint[1]
    patch = [Image[uprow + linewidth:bottomrow - linewidth + 1, \
            leftcolumn + linewidth:rightcolumn - linewidth + 1, i:i+1] for i in range(Image.shape[2])]
    patch = np.concatenate(patch, axis=2)
    width, height, channels = patch.shape
    # size = int(enlargement*height), int(enlargement*width)
    Interpolation_Method = cv2.INTER_AREA  # 'bicubic'
    enlarged = [np.expand_dims(cv2.resize(src=patch[:, :, i], dsize=(0, 0), fx=enlargement, fy=enlargement,
                                          interpolation=Interpolation_Method), 2) for i in range(channels)]
    enlarged = np.concatenate(enlarged, axis=2)
    m, n, c = enlarged.shape
    row, _, _ = Image.shape
    enlargedshowstartrow = row - gap - linewidth
    enlargedshowstartcolumn = gap + linewidth + 1
    for j in range(c):
        Image[enlargedshowstartrow-m+1:enlargedshowstartrow+1, enlargedshowstartcolumn-1-1:enlargedshowstartcolumn+n-1-1, j]\
            = enlarged[:, :, j]
    Point1 = [enlargedshowstartrow-m+1-linewidth, enlargedshowstartcolumn-linewidth-1-1]
    Point2 = [enlargedshowstartrow+1, enlargedshowstartcolumn+n-1-1]
    Image = draw_rectangle(Image, Point1, Point2, linewidth)
    return Image


def show_detail(image, logdir, leftuppoint, rightbottompoint, enlargement, linewidth, gap, xy, yx):
    # type: (array, str, list, list, float, int, int, list, list) -> array
    r"""TODO: Docstring for show_detail.
    Args:
        :image, leftuppoint, rightbottompoint, enlargement, linewidth, gap: TODO
        :returns: TODO
    Examples:
        >>> show_detail(image, logdir, [180, 150], [240, 250], 2, 1, 0)
    """
    Image = show_enlarge_rectangle(image, leftuppoint, rightbottompoint, enlargement, linewidth, gap)
    height, width, channels = Image.shape
    fig = plt.figure("Rectangle Image")
    plt.imshow(Image, cmap='gray', aspect='auto')
    # plt.colorbar()
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    plt.annotate('a', xy=xy)  # 256x256: [160, 40], 512x512: [250, 180]
    plt.annotate('a', xy=yx)  # 256x256: [120, 136], 512x512: [200, 392]
    fig.set_size_inches(width / 100.0, height / 100.0)
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(logdir, dpi=100)
    # plt.savefig('result/' + str(result) + '.png', dpi = 300)
    # plt.savefig('result/' + str(result) + '.png', bbox_inches='tight')
    plt.show()


def show_double_detail(logdir, leftuppoint, rightbottompoint, enlargement, linewidth, gap,  xy, yx, *image):
    # type: (str, list, list, float, int, int, list, list, list) -> array
    r"""TODO: Docstring for show_detail.
    Args:
        :image, leftuppoint, rightbottompoint, enlargement, linewidth, gap: TODO
        :returns: TODO
    Examples:
        >>> show_enlarge_rectangle(logdir, [180, 150], [240, 250], 2, 1, 0, *image)
    """
    Image = list()
    for i in range(len(image)):
        Image.append(show_enlarge_rectangle(image[i], leftuppoint, rightbottompoint, enlargement, linewidth, gap))
    Image = np.concatenate(Image, axis=1)
    height, width, channels = Image.shape
    fig = plt.figure("concatenate image")
    plt.imshow(Image, cmap='gray', aspect='auto')
    # plt.colorbar()
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    plt.annotate('a', xy=xy)  # 256x256: [160, 40], 512x512: [250, 180]
    plt.annotate('a', xy=yx)  # 256x256: [120, 136], 512x512: [200, 392]
    fig.set_size_inches(width / 100.0, height / 100.0)
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(logdir, dpi=100)
    # plt.savefig('result/' + str(result) + '.png', dpi = 300)
    # plt.savefig('result/' + str(result) + '.png', bbox_inches='tight')
    plt.show()


def show_compare_psnr(*psnr):
    # type: (tuple) -> None
    r"""TODO: Docstring for show_compare_psnr.
    Args:
        :*psnr: TODO
        :returns: TODO
    Examples:
        >>>
    """
    length, x_axis = len(psnr), len(psnr[0])
    # color = list(filter(lambda x: x is not None, []))
    color, label = ['green', 'red'], ['PnP-ADMM', 'PnP-FBS']
    plt.figure("compare_psnr")
    plt.title("")
    for i in range(length):
        plt.plot(psnr[i], color=color[i], label=label[i])
        plt.legend(loc='best',)  # edgecolor='blue'
    plt.xlabel("Number of iterations(#IT.)")
    plt.ylabel("PSNR(dB)")
    plt.savefig('compare_psnr.png')
    plt.show()


def show_compare_histogram(**kwargs):
    # type: (dict) -> None
    r"""TODO: Docstring for show_compare_histogram.
    Args:
        :arg1: TODO
        :returns: TODO
    Examples:
        >>>
    """
    # 一个字典，value是四季最大阵风的风速值，key是年份
    # max_lst_of_all = {}
    # max_lst_of_all[2010] = [29.7, 34.3, 29.7, 26.3]
    # max_lst_of_all[2011] = [36.0, 30.2, 27.3, 30.9]
    # max_lst_of_all[2012] = [27.3, 32.3, 40.4, 27.8]
    # max_lst_of_all[2013] = [35.9, 29.9, 40.1, 33.3]
    # max_lst_of_all[2014] = [26.3, 30.6, 28.6, 34.3]
    # max_lst_of_all[2015] = [33.1, 27.0, 25.4, 30.7]
    # max_lst_of_all[2016] = [41.3, 31.3, 41.1, 38.0]
    # max_lst_of_all[2017] = [27.5, 31.2, 43.2, 41.2]

    fig = plt.figure()
    color = ['lightskyblue', 'lime', 'red', 'gold']  # 指定bar的颜色
    for key in kwargs.keys():
        print(kwargs[key])
        # 一年有四季，此行指定四季对应的bar的位置，比如2010年：2009.7,2009.9,2010.1,2010.3
        x = np.arange(key-0.3, key+0.31, 0.2)
        y = kwargs[key]  # 此行决定了bar的高度(风速值）
        # bar_width = 0.2
        for x1, y1, c1 in zip(x, y, color):  # 遍历以上三者，每一次生成一条bar
            plt.bar(x1, y1, width=0.2, color=c1)
    # 我试过这里不能直接生成legend，解决方法就是自己定义，创建legend
    labels = ['winter', 'spring', 'summer', 'autumn']  # legend标签列表，上面的color即是颜色列表
    # 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # 下面一行中bbox_to_anchor指定了legend的位置
    ax.legend(handles=patches, bbox_to_anchor=(0.95, 1.12), ncol=4)  # 生成legend
    plt.show()


def nMSE(*error):
    # type: (tuple) -> None
    r"""TODO: Docstring for nMSE.
    Args:
        :*error: TODO
        :returns: TODO
    Examples:
        >>>
    """
    length, x_axis = len(error), len(error[0])
    # color = list(filter(lambda x: x is not None, []))
    color, label = ['green', 'red'], ['PnP-ADMM', 'PnP-FBS']
    plt.figure("compare_nMSE")
    plt.title("")
    for i in range(length):
        plt.plot(error[i], color=color[i], label=label[i])
        plt.legend(loc='best',)  # edgecolor='blue'
    plt.xlabel("Number of iterations(#IT.)")
    plt.ylabel("nMSE")
    plt.savefig('compare_nMSE.png')
    plt.show()


def power_iteration_(A, num_simulations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def power_iteration(x, f, num_simulations, tol):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector

    # global spectral_radius
    h_k = torch.rand_like(x)
    h_k /= torch.norm(h_k, p=2, dim=(2, 3), keepdim=True)

    iter = 0
    spectral_radius = 0.
    error = torch.tensor(float('inf'))
    while iter <= num_simulations and error >=tol:
        # calculate the matrix-by-vector product f(x_k)
        h_k1 = f(x + h_k) - f(x)
        h_k1 /= torch.norm(h_k1, p=2, dim=(2, 3), keepdim=True)

        # calculate the norm
        h_k1 = h_k1.permute(0, 1, 3, 2)
        schur_prod = torch.matmul(h_k1, h_k)
        schur_prod = torch.einsum('...ii->...', [schur_prod])

        # h_k = h_k.permute(0, 1, 3, 2)
        # schur_prod_ = torch.matmul(h_k, h_k)
        # schur_prod_ = torch.einsum('...ii->...', [schur_prod_])
        # compute spectral radius
        # spectral_radius = schur_prod / schur_prod_
        spectral_radius = schur_prod

        # stopping criteria
        error = torch.norm(h_k1 - spectral_radius * h_k, p=2, dim=(2, 3), keepdim=True)
        print(error)
        h_k = h_k1
        iter += 1
    return spectral_radius


def make_hist(x, bins, range, logdir):
    # type: (array, array, list, str) -> None
    r"""TODO: Docstring for make_hist.
    Args:
        :x: input array
        :bins: numbers of groups
        :range: show range
        :returns: histogram
    Examples:
        >>> make_hist(x, bins=8, range=[-1, 1])
    """
    frequency_each, _, _ = plt.hist(x, bins, range, color='fuchsia', edgecolor='black', alpha=0.5, density=True)
    label = r'$\epsilon=$' + str(format(max(x), '.2f'))
    plt.vlines(max(x), 0, max(frequency_each), colors='r', linestyles='dashed', label=label)
    plt.legend(loc='best')
    # plt.xlim([0, 1.2])
    # plt.ylim([0, 1])
    plt.savefig(logdir, dpi=100)
    plt.show()


def make_plot(title, hist, edges, x, pdf, cdf):
    r"""绘图函数"""
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend="PDF")
    p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend="CDF")

    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color = "white"
    return p


def test_histogram():
    # 正态分布
    mu, sigma = 0, 0.5
    measured = np.random.normal(mu, sigma, 1000)
    hist, edges = np.histogram(measured, density=True, bins=50)
    x = np.linspace(-2, 2, 1000)
    # 拟合曲线
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x-mu) ** 2 / (2 * sigma ** 2))
    cdf = (1+scipy.special.erf((x-mu) / np.sqrt(2 * sigma ** 2))) / 2
    p1 = make_plot("Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf)

    # 对数正态分布
    mu, sigma = 0, 0.5
    measured = np.random.lognormal(mu, sigma, 1000)
    hist, edges = np.histogram(measured, density=True, bins=50)
    x = np.linspace(0.0001, 8.0, 1000)
    pdf = 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x)-mu) ** 2 / (2 * sigma ** 2))
    cdf = (1+scipy.special.erf((np.log(x)-mu) / (np.sqrt(2) * sigma))) / 2
    p2 = make_plot("Log Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf)

    # 伽玛分布
    k, theta = 7.5, 1.0
    measured = np.random.gamma(k, theta, 1000)
    hist, edges = np.histogram(measured, density=True, bins=50)
    x = np.linspace(0.0001, 20.0, 1000)
    pdf = x ** (k-1) * np.exp(-x / theta) / (theta ** k * scipy.special.gamma(k))
    cdf = scipy.special.gammainc(k, x / theta)
    p3 = make_plot("Gamma Distribution (k=7.5, θ=1)", hist, edges, x, pdf, cdf)

    # 韦伯分布
    lam, k = 1, 1.25
    measured = lam * (-np.log(np.random.uniform(0, 1, 1000))) ** (1 / k)
    hist, edges = np.histogram(measured, density=True, bins=50)
    x = np.linspace(0.0001, 8, 1000)
    pdf = (k / lam) * (x / lam) ** (k-1) * np.exp(-(x / lam) ** k)
    cdf = 1-np.exp(-(x / lam) ** k)
    p4 = make_plot("Wei bull Distribution (λ=1, k=1.25)", hist, edges, x, pdf, cdf)
    # 显示
    show(gridplot([p1, p2, p3, p4], ncols=2, plot_width=400, plot_height=400, toolbar_location=None))


def test_power():
    r"""power iteration"""
    # spectral radius of matrix
    print(power_iteration_(np.array([[0.5, 0.5], [0.2, 0.8]]), 10))

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', 'Set12', '*.png'))
    files_source.sort()
    denoised = lambda tensor: denoise(tensor, sigma=5, blind=False, denoise_method='LKSVD')
    # process data
    for files in files_source:
        Img = cv2.imread(files)
        Img = normalize(np.float32(Img[:, :, 0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        spectral_radius = power_iteration(ISource, denoised, 20, 1e-5)
        print("spectral_radius is %.4f" % spectral_radius)

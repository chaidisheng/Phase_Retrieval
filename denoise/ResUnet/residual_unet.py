#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    residual_unet.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        9/11/20 3:02 AM
# @torch: tensor.method(in-place) or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
from torchsummary import summary


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride, upsample_mode='deconv'):
        super(Upsample, self).__init__()
        if upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                input_dim, output_dim, kernel_size=kernel, stride=stride
            )
        else:
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.upsample(x)


class ResUnet(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=3, feature_scale=4):
        super(ResUnet, self).__init__()

        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        self.input_layer = nn.Sequential(
            nn.Conv2d(num_input_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(num_input_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1]+filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], num_output_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)

        # Bridge
        x5 = self.bridge(x4)

        # Decode
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)
        x7 = self.up_residual_conv1(x6)

        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)
        x9 = self.up_residual_conv2(x8)

        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)
        x11 = self.up_residual_conv3(x10)

        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)
        x13 = self.up_residual_conv4(x12)

        output = self.output_layer(x13)

        return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUnet(num_input_channels=1, num_output_channels=1, feature_scale=4).to(device)
    print(model)
    summary(model, input_size=(1, 64, 64), batch_size=-1, device=str(device))
    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    print("Trainable parameters are: ", num_parameters)
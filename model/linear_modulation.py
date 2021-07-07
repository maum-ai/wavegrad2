#This code is adopted from
#https://github.com/ivanvovk/WaveGrad
import math

import torch

from model.base import BaseModule
from model.layers import Conv1dWithInitialization




class PositionalEncoding(BaseModule):
    def __init__(self, n_channels, linear_scale):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels
        self.linear_scale = linear_scale

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(half_dim)
        exponents = 1e-4 ** exponents
        exponents = self.linear_scale * noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


class FeatureWiseLinearModulation(BaseModule):
    def __init__(self, in_channels, out_channels, input_dscaled_by, linear_scale):
        super(FeatureWiseLinearModulation, self).__init__()
        self.signal_conv = torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.LeakyReLU(0.2)
        ])
        self.positional_encoding = PositionalEncoding(in_channels, linear_scale)
        self.scale_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.shift_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x, noise_level):
        outputs = self.signal_conv(x)
        outputs = outputs + self.positional_encoding(noise_level).unsqueeze(-1)
        scale, shift = self.scale_conv(outputs), self.shift_conv(outputs)
        return scale, shift


class FeatureWiseAffine(BaseModule):
    def __init__(self):
        super(FeatureWiseAffine, self).__init__()

    def forward(self, x, scale, shift):
        outputs = scale * x + shift
        return outputs

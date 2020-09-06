import math

import torch

from model.base import BaseModule
from model.layers import Conv1dWithInitialization


DEFAULT_MAX_PE_LENGTH=500000
CRITICAL_MAX_PE_LENGTH=2000000


class PositionalEncoding(BaseModule):
    def __init__(self, n_channels, max_len):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels
        self.max_len = max_len
        pe = self.build_matrix(self.n_channels, self.max_len)
        self.register_buffer('pe', pe)

    def build_matrix(self, n_channels, max_len):
        pe = torch.zeros(max_len, n_channels, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = (torch.arange(0, n_channels, 2).float() * (-math.log(10000.0) / n_channels)).exp()
        pe[:, 0::2] = (position * div_term).sin()
        pe[:, 1::2] = (position * div_term).cos()
        return pe.transpose(0, 1)

    def rescale_to_new_max_len(self, new_max_len):
        if new_max_len > CRITICAL_MAX_PE_LENGTH:
            raise RuntimeError(
                f'Rescaling PE to {new_max_len}, '
                f'which is more than set CRITICAL_MAX_PE_LENGTH={CRITICAL_MAX_PE_LENGTH}.'
            )
        self.pe = self.build_matrix(self.n_channels, new_max_len).to(self.pe)
        self.max_len = new_max_len

    def forward(self, noise_level, length):
        if noise_level.shape[-1] > self.max_len:
            print(
                'Warning! Given the sequence longer than supports PE block. '
                f'Running max length rescaling from {self.max_len} to {length}. '
                'Next time inference will be faster.'
            )
            self.rescale_to_new_max_len(length)
        batch_size = noise_level.shape[0]
        outputs = noise_level[..., None] + self.pe[:, :length].repeat(batch_size, 1, 1)
        return outputs


class FeatureWiseLinearModulation(BaseModule):
    def __init__(self, in_channels, out_channels, input_dscaled_by):
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
        self.positional_encoding = PositionalEncoding(
            in_channels, DEFAULT_MAX_PE_LENGTH//input_dscaled_by
        )
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
        outputs = outputs + self.positional_encoding(noise_level, length=x.shape[-1])
        scale, shift = self.scale_conv(outputs), self.shift_conv(outputs)
        return scale, shift


class FeatureWiseAffine(BaseModule):
    def __init__(self):
        super(FeatureWiseAffine, self).__init__()

    def forward(self, x, scale, shift):
        outputs = scale * x + shift
        return outputs

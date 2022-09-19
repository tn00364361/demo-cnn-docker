
import logging
from functools import partial

import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, c0: int = 32, kernel_size: int = 3):
        super().__init__()

        self.in_channels = in_channels

        _conv2d = partial(
            nn.Conv2d,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            padding_mode='circular',
            bias=False
        )

        # [n, in_channels, 32, 32]
        # -> [n, c0, 16, 16] -> [n, 2 * c0, 8, 8]
        # -> [n, 4 * c0, 4, 4] -> [n, 8 * c0, 2, 2]
        self.layers = nn.ModuleList()
        c_in = self.in_channels
        for k in range(4):
            c_out = 2**k * c0
            self.layers.append(_conv2d(c_in, c_out))
            self.layers.append(nn.BatchNorm2d(c_out))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout2d())

            c_in = c_out

        # [n, 8 * c0, 2, 2] -> [n, 32 * c0] -> [n, 10]
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(32 * c0, 10))

    def forward(self, x):
        y = x.reshape(-1, self.in_channels, 32, 32)
        for layer in self.layers:
            y = layer(y)

        return y


if __name__ == '__main__':
    logging.warning('This file is not intended to be executed.')


import logging
from functools import partial

import torch


class SimpleCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 1, c0: int = 32, kernel_size: int = 3):
        super().__init__()

        self.in_channels = in_channels

        _conv2d = partial(
            torch.nn.Conv2d,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            padding_mode='circular',
            bias=False
        )

        # [n, in_channels, 32, 32] -> [n, in_channels * 4, 16, 16]
        # -> [n, c0, 8, 8] -> [n, 2 * c0, 4, 4] -> [n, 4 * c0, 2, 2]
        self.layers = torch.nn.ModuleList()
        c_in = self.in_channels * 4
        for k in range(3):
            c_out = 2**k * c0
            self.layers.append(_conv2d(c_in, c_out))
            self.layers.append(torch.nn.BatchNorm2d(c_out))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout2d())
            c_in = c_out

        # [n, 8 * c0, 2, 2] -> [n, 32 * c0] -> [n, 10]
        self.layers.append(torch.nn.Flatten())
        self.layers.append(torch.nn.Linear(16 * c0, 10))

        self.haar_basis = 0.5 * torch.tensor([
            [+1, +1, +1, +1],
            [+1, +1, -1, -1],
            [+1, -1, +1, -1],
            [+1, -1, -1, +1]
        ], dtype=torch.float32, requires_grad=False).unsqueeze(0)

    def forward(self, x):
        if self.haar_basis != x.device:
            self.haar_basis = self.haar_basis.to(x.device)

        h2, w2 = x.shape[-2] // 2, x.shape[-1] // 2

        y = x.reshape(-1, self.in_channels, h2, 2, w2, 2).permute(0, 3, 5, 1, 2, 4)
        y = torch.matmul(
            self.haar_basis,
            y.reshape(-1, 4, self.in_channels * h2 * w2)
        )
        y = y.reshape(-1, self.in_channels * 4, h2, w2)
        for layer in self.layers:
            y = layer(y)

        return y


if __name__ == '__main__':
    logging.warning('This file is not intended to be executed.')

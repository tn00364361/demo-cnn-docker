
import logging
import torch


class ConvBN2d(torch.nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3
        ) -> None:
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=kernel_size // 2,
            padding_mode='circular',
            bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class DWT2(torch.nn.Module):
    def __init__(self) -> None:
        """
        @brief      2D discrete wavelet transform w/ Haar basis
        """
        super().__init__()

        self.haar_basis = 0.5 * torch.tensor([
            [+1, +1, +1, +1],
            [+1, +1, -1, -1],
            [+1, -1, +1, -1],
            [+1, -1, -1, +1]
        ], dtype=torch.float32, requires_grad=False).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.haar_basis != x.device:
            self.haar_basis = self.haar_basis.to(x.device)

        n, c, h, w = x.shape
        h2, w2 = h // 2, w // 2

        y = x.reshape(n, c, h2, 2, w2, 2).permute(0, 3, 5, 1, 2, 4)
        y = torch.matmul(self.haar_basis, y.reshape(n, 4, c * h2 * w2))

        return y.reshape(n, c * 4, h2, w2)


class SimpleCNN(torch.nn.Module):
    def __init__(self, c0: int = 32) -> None:
        super().__init__()

        # [n, 1, 32, 32] -> [n, 4, 16, 16]
        self.dwt2 = DWT2()

        # [n, 4, 16, 16] -> [n, c0, 8, 8] -> [n, 2 * c0, 4, 4] -> [n, 4 * c0, 2, 2]
        self.layers = torch.nn.ModuleList()
        c_in = 4
        for k in range(3):
            c_out = 2**k * c0
            self.layers.append(ConvBN2d(c_in, c_out))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout2d())
            c_in = c_out

        # [n, 4 * c0, 2, 2] -> [n, 16 * c0] -> [n, 10]
        self.layers.append(torch.nn.Flatten())
        self.layers.append(torch.nn.Linear(16 * c0, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwt2(x)

        # regular convolutional layers
        for layer in self.layers:
            y = layer(y)

        return y


if __name__ == '__main__':
    logging.warning('This file is not intended to be executed.')


import logging
import torch


class ConvBN2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode='circular',
            bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.activation = torch.nn.ReLU(inplace=True)
        self.main_block = torch.nn.Sequential(
            ConvBN2d(in_channels, out_channels),
            self.activation,
            torch.nn.Dropout2d(),
            ConvBN2d(out_channels, out_channels)
        )
        if in_channels == out_channels:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = ConvBN2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (n, in_channels, h, w) -> (n, out_channels, h, w)

        return self.activation(self.main_block(x) + self.shortcut(x))


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
        # (n, c, h, w) -> (n, 4 * c, h // 2, w // 2)
        if self.haar_basis != x.device:
            self.haar_basis = self.haar_basis.to(x.device)

        n, c, h, w = x.shape
        h2, w2 = h // 2, w // 2

        y = x.reshape(n, c, h2, 2, w2, 2).permute(0, 3, 5, 1, 2, 4)
        y = torch.matmul(self.haar_basis, y.reshape(n, 4, c * h2 * w2))

        return y.reshape(n, c * 4, h2, w2)


class SimpleCNN(torch.nn.Module):
    def __init__(self, c0: int = 16, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()

        self.dwt = DWT2()
        self.layers = torch.nn.Sequential(
            self.dwt,
            ResidualBlock(in_channels * 4, c0),     # (n, c0, 16, 16)
            self.dwt,
            ResidualBlock(4 * c0, 2 * c0),          # (n, 2 * c0, 8, 8)
            self.dwt,
            ResidualBlock(4 * 2 * c0, 4 * c0),      # (n, 4 * c0, 4, 4)
            self.dwt,
            ResidualBlock(4 * 4 * c0, 8 * c0),      # (n, 8 * c0, 2, 2)
            torch.nn.Flatten(),                     # (n, 32 * c0)
            torch.nn.Linear(32 * c0, num_classes)   # (n, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (n, in_channels, 32, 32) -> (n, num_classes)
        return self.layers(x)


if __name__ == '__main__':
    logging.warning('This file is not intended to be executed.')

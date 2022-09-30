
import logging
import torch


class MyCustomBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList()

        self.layers.append(DWT2())
        self.layers.append(torch.nn.Conv2d(
            in_channels * 4,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular',
            bias=False
        ))
        self.layers.append(torch.nn.BatchNorm2d(out_channels))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Dropout2d())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (n, in_channels, h, w) -> (n, out_channels, h // 2, w // 2)
        for layer in self.layers:
            x = layer(x)

        return x


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
    def __init__(self, c0: int = 16, num_classes: int = 10) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList()

        self.layers.append(MyCustomBlock(1, c0))            # (n, c0, 16, 16)
        self.layers.append(MyCustomBlock(1 * c0, 2 * c0))   # (n, 2 * c0, 8, 8)
        self.layers.append(MyCustomBlock(2 * c0, 4 * c0))   # (n, 4 * c0, 4, 4)
        self.layers.append(MyCustomBlock(4 * c0, 8 * c0))   # (n, 8 * c0, 2, 2)

        # (n, 8 * c0, 2, 2) -> (n, 32 * c0) -> (n, num_classes)
        self.layers.append(torch.nn.Flatten())
        self.layers.append(torch.nn.Linear(32 * c0, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


if __name__ == '__main__':
    logging.warning('This file is not intended to be executed.')

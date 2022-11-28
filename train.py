#! /usr/bin/env python3
from time import perf_counter
import argparse
import logging

import torch
from torch.utils.data import DataLoader
import torchvision as tv
import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs. (default: 50)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size in each training step. (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate. (default: 1e-3)')
    parser.add_argument('--num_workers', '-w', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--fashion', dest='fashion', action='store_true',
                        help='If True, train on FashionMNIST; otherwise use CIFAR10.')
    args = parser.parse_args()

    # TODO: set the correct log path
    logging.basicConfig(filename='./logs/test.log', filemode='w', level=logging.DEBUG)
    logging.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.fashion:
        mnist = tv.datasets.FashionMNIST
        in_channels = 1
        train_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),               # PILImage, images of range [0, 1].
            tv.transforms.Pad(2),                   # 28x28 -> 32x32
            tv.transforms.Normalize(0.5, 0.5)       # normalize
        ])
        test_transform = train_transform
    else:
        mnist = tv.datasets.CIFAR10
        in_channels = 3
        train_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            # tv.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.Normalize(0.5, 0.5)
        ])
        test_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(0.5, 0.5)
        ])

    train_ds = mnist(root=args.data_dir, download=True, transform=train_transform, train=True)
    test_ds = mnist(root=args.data_dir, download=True, transform=test_transform, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    logging.info(f'# training samples = {len(train_ds):d}')
    logging.info(f'# test samples = {len(test_ds):d}')

    # Using multiple GPUs
    model = torch.nn.DataParallel(
        utils.SimpleCNN(in_channels=in_channels),
        device_ids=range(torch.cuda.device_count())
    ).to(device)

    logging.info(f'Loaded model:\n{str(model):s}')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    t_start = perf_counter()
    for epoch in range(args.num_epochs):
        total_loss = torch.tensor(0, dtype=torch.float32).to(device)

        model.train()
        num_correct, num_samples = 0, 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            y = model(images)
            loss = criterion(y, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(y, 1)
            num_samples += labels.size(0)
            num_correct += (pred == labels).sum().item()

            total_loss += loss

        avg_loss = total_loss / len(train_loader)
        train_acc = num_correct / num_samples

        model.eval()
        num_correct, num_samples = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                y = model(images)
                _, pred = torch.max(y, 1)
                num_samples += labels.size(0)
                num_correct += (pred == labels).sum().item()

        test_acc = num_correct / num_samples

        # time elapsed
        mm, ss = divmod(perf_counter() - t_start, 60)

        logging.info(' '.join([
            f'[{epoch + 1:2d}/{args.num_epochs:2d}]',
            f'Time elapsed: {int(mm):02d}:{ss:05.2f}',
            f'{avg_loss = :.4f} {train_acc = :.4f} {test_acc = :.4f}'
        ]))

    logging.info('Finished Training')

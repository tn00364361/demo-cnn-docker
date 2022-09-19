#! /usr/bin/env python3
from time import perf_counter
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs. (default: 50)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size in each training step. (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate. (default: 1e-3)')
    parser.add_argument('--fashion', dest='fashion', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(filename='./logs/test.log', filemode='w', level=logging.DEBUG)
    logging.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),          # PILImage, images of range [0, 1].
        transforms.Pad(2),
        transforms.Normalize(0.5, 0.5)  # normalize
    ])

    if args.fashion:
        mnist = torchvision.datasets.FashionMNIST
    else:
        mnist = torchvision.datasets.MNIST

    train_ds = mnist(root='./data', download=True, transform=transform, train=True)
    test_ds = mnist(root='./data', download=True, transform=transform, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logging.info(f'# training samples = {len(train_ds):d}')
    logging.info(f'# test samples = {len(test_ds):d}')

    # Using multiple GPUs
    model = nn.DataParallel(
        utils.SimpleCNN(),
        device_ids=range(torch.cuda.device_count())
    ).to(device)

    logging.info(f'Loaded model:\n{str(model):s}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    t_start = perf_counter()
    for epoch in range(args.num_epochs):
        total_loss = torch.tensor(0, dtype=torch.float32).to(device)
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
            f'{avg_loss = :.4f}',
            f'{train_acc = :.4f}',
            f'{test_acc = :.4f}'
        ]))

    logging.info('Finished Training')

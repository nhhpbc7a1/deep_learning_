#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Huấn luyện StudentCNN (models.student_cnn) — CIFAR-10/100, ImageFolder, Stanford Dogs.
Tách biệt train_pdpnet.py; dùng cho giữa kỳ / cuối kỳ với cùng kiến trúc StudentCNN.
"""
import argparse
import os
import sys

import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms
import torchvision.datasets

from models.cross_entropy import LabelSmoothingCrossEntropy
from models.datasets import StanfordDogs
from models.student_cnn import StudentCNN
import writeLogAcc as wA


def get_args():
    parser = argparse.ArgumentParser(
        description='Train StudentCNN on CIFAR or ImageFolder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        choices=['cifar10', 'cifar100', 'dogs', 'imagefolder'],
        default='imagefolder',
        help='cifar10/cifar100: 32x32; imagefolder: train/ và val/ theo lớp.',
    )
    parser.add_argument('-r', '--data-root', type=str, default='./data_custom')
    parser.add_argument('--download', action='store_true', help='(CIFAR / dogs) tải dataset nếu cần.')
    parser.add_argument('-g', '--gpu-id', default=0, type=int, help='GPU id, -1 = CPU.')
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-l', '--learning-rate', default=0.1, type=float)
    parser.add_argument(
        '-s', '--schedule', nargs='+', default=[100, 150, 180], type=int,
        help='Epoch giảm LR (MultiStepLR).',
    )
    parser.add_argument('-m', '--momentum', default=0.9, type=float)
    parser.add_argument('-w', '--weight-decay', default=5e-4, type=float,
                        help='Weight decay (CIFAR thường 5e-4 với BN+SGD).')
    parser.add_argument(
        '--image-size',
        type=int,
        default=None,
        help='32 (CIFAR / ảnh nhỏ) hoặc 224 (ImageFolder lớn).',
    )
    parser.add_argument('--n-class', type=int, default=None)
    parser.add_argument(
        '--run-tag',
        type=str,
        default=None,
        help='Thư mục con dưới checkpoints/StudentCNN_<dataset>/ (vd: D_224, D_32).',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout trước FC.',
    )
    return parser.parse_args()


def get_device(args):
    if args.gpu_id >= 0 and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(args.gpu_id))
    return torch.device('cpu')


def build_transforms(dataset_name, train, image_size):
    if dataset_name in ('cifar10', 'cifar100'):
        if train:
            return torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if image_size <= 32:
        if train:
            return torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(40, 40)),
                torchvision.transforms.RandomCrop(image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(image_size, image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])
    if train:
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(256, 256)),
            torchvision.transforms.RandomCrop(image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])


def get_data_loaders(args, image_size):
    if args.dataset in ('cifar10', 'cifar100'):
        dataset_class = (
            torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' else torchvision.datasets.CIFAR100
        )
        train_tf = build_transforms(args.dataset, True, image_size)
        val_tf = build_transforms(args.dataset, False, image_size)
        if args.download:
            dataset_class(root=args.data_root, train=True, download=True, transform=train_tf)
        train_ds = dataset_class(root=args.data_root, train=True, download=True, transform=train_tf)
        val_ds = dataset_class(root=args.data_root, train=False, download=True, transform=val_tf)
        n_class = 10 if args.dataset == 'cifar10' else 100
        if args.n_class is not None and args.n_class != n_class:
            print('Warning: --n-class ignored for CIFAR (using {}).'.format(n_class))
    elif args.dataset == 'dogs':
        train_tf = build_transforms('dogs', True, image_size)
        val_tf = build_transforms('dogs', False, image_size)
        if args.download:
            StanfordDogs(root=args.data_root, train=True, download=True, transform=train_tf)
        train_ds = StanfordDogs(root=args.data_root, train=True, download=True, transform=train_tf)
        val_ds = StanfordDogs(root=args.data_root, train=False, download=True, transform=val_tf)
        n_class = len(train_ds.unique_class_names)
        if args.n_class is not None and args.n_class != n_class:
            print('Warning: --n-class ignored for dogs (using {}).'.format(n_class))
    else:
        train_dir = os.path.join(args.data_root, 'train')
        val_dir = os.path.join(args.data_root, 'val')
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise FileNotFoundError(
                'Missing {} or {} (ImageFolder).'.format(train_dir, val_dir)
            )
        train_tf = build_transforms('imagefolder', True, image_size)
        val_tf = build_transforms('imagefolder', False, image_size)
        train_ds = torchvision.datasets.ImageFolder(train_dir, train_tf)
        val_ds = torchvision.datasets.ImageFolder(val_dir, val_tf)
        if train_ds.classes != val_ds.classes:
            raise ValueError('Train and val class names do not match.')
        n_class = len(train_ds.classes)
        if args.n_class is not None and args.n_class != n_class:
            raise ValueError('--n-class ({}) != so lop trong data ({}).'.format(args.n_class, n_class))
        print('Classes ({}): {}'.format(n_class, train_ds.classes))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(args.gpu_id >= 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(args.gpu_id >= 0),
    )
    return train_loader, val_loader, n_class


def resolve_image_size(args):
    if args.image_size is not None:
        return args.image_size
    if args.dataset in ('cifar10', 'cifar100'):
        return 32
    return 224


def calculate_accuracy(output, target):
    with torch.no_grad():
        batch_size = output.shape[0]
        prediction = torch.argmax(output, dim=1)
        return torch.sum(prediction == target).item() / batch_size


def run_epoch(train_mode, data_loader, model, criterion, optimizer, n_epoch, args, device):
    if train_mode:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    batch_count = len(data_loader)
    losses = []
    accs = []
    for n_batch, (images, target) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        loss_item = loss.item()
        losses.append(loss_item)
        accs.append(calculate_accuracy(output, target))

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (n_batch % 10) == 0:
            tag = 'train' if train_mode else ' val '
            name = 'train' if train_mode else 'val'
            print('[{}]  epoch {}/{},  batch {}/{},  loss_{}={:.5f},  acc_{}={:.2f}%'.format(
                tag, n_epoch + 1, args.epochs, n_batch + 1, batch_count,
                name, loss_item, name, 100.0 * accs[-1]))

    return (sum(losses) / len(losses), sum(accs) / len(accs))


def main():
    args = get_args()
    print('Command: {}'.format(' '.join(sys.argv)))
    image_size = resolve_image_size(args)
    device = get_device(args)
    print('Using device {}, image_size={}'.format(device, image_size))

    train_loader, val_loader, n_class = get_data_loaders(args, image_size)

    model = StudentCNN(image_size=image_size, n_class=n_class, dropout=args.dropout)
    model = model.to(device)

    print(model)
    n_params = sum(p.data.nelement() for p in model.parameters())
    print('Number of model parameters: {}'.format(n_params))

    tag = 'StudentCNN_{}'.format(args.dataset)
    pathout = os.path.join('./checkpoints', tag)
    if args.run_tag:
        pathout = os.path.join(pathout, args.run_tag)
    os.makedirs(pathout, exist_ok=True)
    filenameLOG = os.path.join(pathout, 'train_log.txt')

    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.schedule, gamma=0.1,
    )

    acc_val_max = None
    acc_val_argmax = None

    for n_epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print('Starting epoch {}/{},  learning_rate={}'.format(n_epoch + 1, args.epochs, lr))

        loss_train, acc_train = run_epoch(
            True, train_loader, model, train_loss_fn, optimizer, n_epoch, args, device,
        )
        loss_val, acc_val = run_epoch(
            False, val_loader, model, criterion, None, n_epoch, args, device,
        )

        if acc_val_max is None or acc_val > acc_val_max:
            acc_val_max = acc_val
            acc_val_argmax = n_epoch
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'n_class': n_class,
                    'image_size': image_size,
                    'arch': 'StudentCNN',
                    'dropout': args.dropout,
                },
                os.path.join(pathout, 'checkpoint_best_{:.2f}.pth'.format(100.0 * acc_val_max)),
            )

        scheduler.step()

        line = (
            'Epoch {}/{} summary:  loss_train={:.5f},  acc_train={:.2f}%,  loss_val={:.2f},  '
            'acc_val={:.2f}% (best: {:.2f}% @ epoch {})'
        ).format(
            n_epoch + 1, args.epochs, loss_train, 100.0 * acc_train,
            loss_val, 100.0 * acc_val, 100.0 * acc_val_max, acc_val_argmax + 1,
        )
        print('=' * len(line))
        print(line)
        print('=' * len(line))
        wA.writeLogAcc(filenameLOG, line)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
        sys.exit(0)
    except Exception as e:
        print('Error: {}'.format(e))
        sys.exit(1)

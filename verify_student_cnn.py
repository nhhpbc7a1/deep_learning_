#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify StudentCNN checkpoint (train_student_cnn.py)."""
import argparse
import os
import sys

import torch
import torch.nn as nn

from models.student_cnn import StudentCNN
from train_student_cnn import (
    get_device,
    get_data_loaders,
    resolve_image_size,
    run_epoch,
)


def get_args():
    p = argparse.ArgumentParser(description='Verify StudentCNN checkpoint.')
    p.add_argument('-c', '--checkpoint', required=True, type=str)
    p.add_argument(
        '-d', '--dataset',
        type=str,
        choices=['cifar10', 'cifar100', 'dogs', 'imagefolder'],
        required=True,
    )
    p.add_argument('-r', '--data-root', type=str, default='./data_custom')
    p.add_argument('--download', action='store_true')
    p.add_argument('-g', '--gpu-id', default=0, type=int)
    p.add_argument('-j', '--workers', default=4, type=int)
    p.add_argument('-b', '--batch-size', default=32, type=int)
    p.add_argument('-n', '--n-class', default=None, type=int)
    p.add_argument('--image-size', default=None, type=int)
    p.add_argument('--dropout', default=None, type=float)
    return p.parse_args()


def main():
    args_eval = get_args()
    device = get_device(args_eval)

    ckpt_path = os.path.abspath(args_eval.checkpoint)
    if not os.path.isfile(ckpt_path):
        print('Khong tim thay: {}'.format(ckpt_path))
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' not in ckpt:
        print('Checkpoint thieu model_state_dict.')
        sys.exit(1)

    n_class = ckpt.get('n_class', args_eval.n_class)
    image_size = ckpt.get('image_size', None)
    dropout = ckpt.get('dropout', 0.3)
    if args_eval.image_size is not None:
        image_size = args_eval.image_size
    if args_eval.dropout is not None:
        dropout = args_eval.dropout
    if n_class is None:
        print('Thieu n_class; truyen --n-class.')
        sys.exit(1)
    if image_size is None:
        image_size = resolve_image_size(
            argparse.Namespace(
                dataset=args_eval.dataset,
                image_size=None,
            )
        )
        print('Warning: dung image_size mac dinh {}'.format(image_size))

    ns = argparse.Namespace(
        dataset=args_eval.dataset,
        data_root=args_eval.data_root,
        download=args_eval.download,
        workers=args_eval.workers,
        batch_size=args_eval.batch_size,
        gpu_id=args_eval.gpu_id,
        n_class=args_eval.n_class,
        epochs=1,
    )
    train_loader, val_loader, n_class_data = get_data_loaders(ns, image_size)
    if n_class_data != n_class:
        print('Warning: n_class checkpoint {} != dataset {}'.format(n_class, n_class_data))

    model = StudentCNN(image_size=image_size, n_class=n_class, dropout=dropout)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    loss_val, acc_val = run_epoch(
        False, val_loader, model, criterion, None, 0, ns, device,
    )
    print('--- Verify StudentCNN ---')
    print('File: {}'.format(ckpt_path))
    print('image_size={}, n_class={}, dropout={}, dataset={}'.format(
        image_size, n_class, dropout, args_eval.dataset))
    print('Val/Test loss={:.5f}, acc={:.2f}%'.format(loss_val, 100.0 * acc_val))


if __name__ == '__main__':
    main()

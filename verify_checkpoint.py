#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nạp checkpoint train_pdpnet (checkpoint_best_*.pth) và đánh giá trên tập val/test
(không train) để giảng viên / báo cáo xác minh kết quả.
"""
import argparse
import os
import sys

import torch

from models.ModelBT2 import PDPNet

# Tái sử dụng pipeline dữ liệu giống train_pdpnet
from train_pdpnet import (
    get_device,
    get_data_loaders,
    resolve_image_size,
    run_epoch,
)


def get_eval_args():
    p = argparse.ArgumentParser(description='Verify PDPNet checkpoint (eval only).')
    p.add_argument('-c', '--checkpoint', required=True, type=str, help='File .pth (có model_state_dict, n_class, image_size).')
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
    p.add_argument('-n', '--n-class', default=None, type=int, help='Chỉ dùng khi checkpoint thiếu metadata (không khuyến nghị).')
    p.add_argument('--image-size', default=None, type=int, help='Ghi đè image_size từ checkpoint nếu cần.')
    return p.parse_args()


def main():
    args_eval = get_eval_args()
    device = get_device(args_eval)

    ckpt_path = os.path.abspath(args_eval.checkpoint)
    if not os.path.isfile(ckpt_path):
        print('Không tìm thấy file: {}'.format(ckpt_path))
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' not in ckpt:
        print('Checkpoint không có key model_state_dict.')
        sys.exit(1)

    n_class = ckpt.get('n_class', args_eval.n_class)
    image_size = ckpt.get('image_size', None)
    if args_eval.image_size is not None:
        image_size = args_eval.image_size
    if n_class is None:
        print('Thiếu n_class trong checkpoint; truyền --n-class.')
        sys.exit(1)
    if image_size is None:
        image_size = resolve_image_size(
            argparse.Namespace(
                dataset=args_eval.dataset,
                image_size=None,
            )
        )
        print('Warning: checkpoint không có image_size, dùng mặc định {}'.format(image_size))

    # Namespace tương thích get_data_loaders
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
        print(
            'Cảnh báo: số lớp checkpoint ({}) != số lớp dataset ({}). Kiểm tra --dataset / data-root.'.format(
                n_class, n_class_data,
            )
        )

    model = PDPNet(image_size=image_size, n_class=n_class)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    loss_val, acc_val = run_epoch(
        False, val_loader, model, criterion, None, 0, ns, device,
    )
    print('--- Verify checkpoint ---')
    print('File: {}'.format(ckpt_path))
    print('image_size={}, n_class={}, dataset={}'.format(image_size, n_class, args_eval.dataset))
    print('Val/Test loss={:.5f}, acc={:.2f}%'.format(loss_val, 100.0 * acc_val))


if __name__ == '__main__':
    main()

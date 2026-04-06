#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tai bo anh hoa 5 lop (TensorFlow tutorial), giai nen va chia train/val (mac dinh 80/20).
Chay 1 lenh:

  python download_flowers_dataset.py

Hoac:

  python download_flowers_dataset.py -o ./data_custom --val-ratio 0.2

Nguon: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
"""
import argparse
import os
import random
import shutil
import sys
import tarfile
import tempfile
import urllib.request

URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'


def get_args():
    p = argparse.ArgumentParser(description='Download TF Flowers (5 classes) and split train/val.')
    p.add_argument(
        '-o', '--output',
        default='./data_custom',
        help='Thu muc goc: se tao train/ va val/ ben trong.',
    )
    p.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Ty le anh cho val (0-1).',
    )
    p.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed chia ngau nhien.',
    )
    p.add_argument(
        '--keep-extract',
        action='store_true',
        help='Giu thu muc giai nen tam (debug).',
    )
    return p.parse_args()


def download_file(url, dest_path):
    print('Downloading: {}'.format(url))
    def reporthook(block, block_size, total):
        if total < 0:
            return
        done = block * block_size
        pct = min(100.0, 100.0 * done / total)
        sys.stdout.write('\r  {:.1f}%'.format(pct))
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print()


def main():
    args = get_args()
    out_root = os.path.abspath(args.output)
    val_ratio = args.val_ratio
    if not (0 < val_ratio < 1):
        print('val-ratio must be between 0 and 1.')
        sys.exit(1)

    train_dir = os.path.join(out_root, 'train')
    val_dir = os.path.join(out_root, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tgz = os.path.join(tmp, 'flower_photos.tgz')
        download_file(URL, tgz)

        print('Extracting...')
        with tarfile.open(tgz, 'r:gz') as tar:
            # Python 3.12+: filter tránh cảnh báo / hành vi mặc định 3.14
            if sys.version_info >= (3, 12):
                tar.extractall(path=tmp, filter='data')
            else:
                tar.extractall(path=tmp)

        # Sau giai nen: flower_photos/<class_name>/*.jpg
        src_root = os.path.join(tmp, 'flower_photos')
        if not os.path.isdir(src_root):
            print('Unexpected archive layout: missing flower_photos/')
            sys.exit(1)

        rng = random.Random(args.seed)
        classes = sorted([d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))])
        print('Classes ({}): {}'.format(len(classes), classes))

        for cls in classes:
            src_cls = os.path.join(src_root, cls)
            files = [
                f for f in os.listdir(src_cls)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
            ]
            rng.shuffle(files)
            n_val = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
            val_files = set(files[:n_val])
            train_files = [f for f in files if f not in val_files]

            dt = os.path.join(train_dir, cls)
            dv = os.path.join(val_dir, cls)
            os.makedirs(dt, exist_ok=True)
            os.makedirs(dv, exist_ok=True)

            for f in train_files:
                shutil.copy2(os.path.join(src_cls, f), os.path.join(dt, f))
            for f in val_files:
                shutil.copy2(os.path.join(src_cls, f), os.path.join(dv, f))

            print('  {}: {} train, {} val'.format(cls, len(train_files), len(val_files)))

        if args.keep_extract:
            keep = os.path.join(out_root, '_flower_photos_extracted')
            if os.path.isdir(keep):
                shutil.rmtree(keep)
            shutil.copytree(src_root, keep)
            print('Kept extract at: {}'.format(keep))

    print('Done. ImageFolder root: {}'.format(out_root))
    print('Train: {}'.format(train_dir))
    print('Val:   {}'.format(val_dir))


if __name__ == '__main__':
    main()

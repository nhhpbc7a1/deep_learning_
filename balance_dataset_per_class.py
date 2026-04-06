#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Can bang ~N anh moi lop (mac dinh 500): gop anh tu train/ va val/, chon ngau nhien N roi chia lai train/val.

Vi du:

  python balance_dataset_per_class.py -i ./data_custom -o ./data_custom_500 --per-class 500 --val-ratio 0.2 --seed 42 --force

Neu lop co it hon N anh: lay het va in canh bao.
"""
import argparse
import os
import random
import shutil
import sys

IMG_EXT = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


def list_image_paths(class_dir):
    if not os.path.isdir(class_dir):
        return []
    out = []
    for f in os.listdir(class_dir):
        if f.lower().endswith(IMG_EXT):
            p = os.path.join(class_dir, f)
            if os.path.isfile(p):
                out.append(p)
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description='Cap ~N images per class, re-split train/val.')
    p.add_argument('-i', '--input', required=True, help='Thu muc co train/ va val/.')
    p.add_argument('-o', '--output', required=True, help='Thu muc moi.')
    p.add_argument('--per-class', type=int, default=500, help='Toi da moi lop (mac dinh 500).')
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force', action='store_true', help='Ghi de output neu da ton tai.')
    args = p.parse_args()

    root = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output)
    train_in = os.path.join(root, 'train')
    val_in = os.path.join(root, 'val')
    if not os.path.isdir(train_in) or not os.path.isdir(val_in):
        print('Input must contain train/ and val/.')
        sys.exit(1)

    vr = args.val_ratio
    if not (0 < vr < 1):
        print('val-ratio must be in (0, 1).')
        sys.exit(1)

    classes = sorted(
        d for d in os.listdir(train_in)
        if os.path.isdir(os.path.join(train_in, d)) and os.path.isdir(os.path.join(val_in, d))
    )
    if not classes:
        print('No matching class folders.')
        sys.exit(1)

    if os.path.exists(out_root):
        if not args.force:
            print('Output exists: {}. Use --force to overwrite.'.format(out_root))
            sys.exit(1)
        shutil.rmtree(out_root)

    rng = random.Random(args.seed)
    n_pc = args.per_class

    for cls in classes:
        pool = list_image_paths(os.path.join(train_in, cls)) + list_image_paths(os.path.join(val_in, cls))
        # Trung ten file (hiem): uu tien giu 1
        by_name = {}
        for path in pool:
            by_name[os.path.basename(path)] = path
        pool = list(by_name.values())
        rng.shuffle(pool)

        if len(pool) < n_pc:
            print('Warning: class "{}" has only {} images (cap {}). Using all.'.format(
                cls, len(pool), n_pc))
            chosen = pool
        else:
            chosen = pool[:n_pc]

        rng.shuffle(chosen)
        n = len(chosen)
        n_val = max(1, int(round(n * vr))) if n >= 2 else 0
        if n_val >= n:
            n_val = n - 1
        val_paths = chosen[:n_val] if n_val else []
        train_paths = chosen[n_val:] if n_val else chosen

        dt = os.path.join(out_root, 'train', cls)
        dv = os.path.join(out_root, 'val', cls)
        os.makedirs(dt, exist_ok=True)
        os.makedirs(dv, exist_ok=True)

        for path in train_paths:
            shutil.copy2(path, os.path.join(dt, os.path.basename(path)))
        for path in val_paths:
            shutil.copy2(path, os.path.join(dv, os.path.basename(path)))

        print('{}: {} images -> {} train, {} val'.format(cls, n, len(train_paths), len(val_paths)))

    print('Done. Output: {}'.format(out_root))


if __name__ == '__main__':
    main()

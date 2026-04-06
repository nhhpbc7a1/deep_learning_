#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Đọc file log kiểu train_pdpnet / writeLogAcc (dòng 'Epoch ... summary: loss_train=...')
và vẽ đồ thị loss + accuracy (train/val). Có thể xuất CSV cho báo cáo.
"""
import argparse
import csv
import os
import re
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('Install matplotlib: pip install matplotlib', file=sys.stderr)
    sys.exit(1)


LINE_RE = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s+summary:\s+'
    r'loss_train=([\d.]+),\s+acc_train=([\d.]+)%,\s+'
    r'loss_val=([\d.]+),\s+acc_val=([\d.]+)%',
    re.IGNORECASE,
)


def parse_log(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            ep, total_ep, lt, at, lv, av = m.groups()
            rows.append({
                'epoch': int(ep),
                'epochs_total': int(total_ep),
                'loss_train': float(lt),
                'acc_train': float(at) / 100.0,
                'loss_val': float(lv),
                'acc_val': float(av) / 100.0,
            })
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = ['epoch', 'loss_train', 'acc_train', 'loss_val', 'acc_val']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({
                'epoch': r['epoch'],
                'loss_train': r['loss_train'],
                'acc_train': r['acc_train'],
                'loss_val': r['loss_val'],
                'acc_val': r['acc_val'],
            })


def plot_rows(rows, out_path, title=None):
    if not rows:
        raise ValueError('Không parse được dòng hợp lệ nào từ log.')

    xs = [r['epoch'] for r in rows]
    lt = [r['loss_train'] for r in rows]
    lv = [r['loss_val'] for r in rows]
    at = [r['acc_train'] for r in rows]
    av = [r['acc_val'] for r in rows]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax0.plot(xs, lt, label='Train', color='#1f77b4')
    ax0.plot(xs, lv, label='Val/Test', color='#ff7f0e')
    ax0.set_ylabel('Loss')
    ax0.legend(loc='upper right')
    ax0.grid(True, alpha=0.3)
    ax0.set_title(title or 'Loss vs epoch')

    ax1.plot(xs, at, label='Train acc', color='#2ca02c')
    ax1.plot(xs, av, label='Val/Test acc', color='#d62728')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title or 'Accuracy vs epoch')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Vẽ đồ thị từ file log train (Epoch summary).')
    p.add_argument('--log', '-l', required=True, help='Đường dẫn file .txt log')
    p.add_argument('--out', '-o', default=None, help='File PNG đầu ra (mặc định: cùng thư mục log, tên plots.png)')
    p.add_argument('--csv', default=None, help='Tùy chọn: xuất bảng số ra CSV')
    p.add_argument('--title', '-t', default=None, help='Tiêu đề đồ thị')
    args = p.parse_args()

    rows = parse_log(args.log)
    out = args.out
    if out is None:
        base = os.path.splitext(os.path.abspath(args.log))[0]
        out = base + '_plots.png'

    plot_rows(rows, out, title=args.title)
    print('Saved: {} ({} epochs)'.format(out, len(rows)))

    if args.csv:
        write_csv(rows, args.csv)
        print('Saved CSV: {}'.format(args.csv))


if __name__ == '__main__':
    main()

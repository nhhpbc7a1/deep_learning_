# -*- coding: utf-8 -*-
"""In kien truc StudentCNN + torchsummary (32 hoac 224)."""
import argparse
import sys

try:
    from torchsummary import summary
except ImportError:
    print('pip install torchsummary')
    sys.exit(1)

import torch

from models.student_cnn import StudentCNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-size', type=int, default=32, choices=[32, 224])
    p.add_argument('--n-class', type=int, default=10)
    p.add_argument('--gpu', action='store_true', help='Use .cuda() for summary')
    args = p.parse_args()

    model = StudentCNN(image_size=args.image_size, n_class=args.n_class)
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
        dev = 'cuda'
    else:
        dev = 'cpu'

    print(model)
    n_params = sum(p.data.nelement() for p in model.parameters())
    print('Parameters: {}'.format(n_params))

    size = (3, args.image_size, args.image_size)
    if dev == 'cuda':
        summary(model, size)
    else:
        print('torchsummary needs CUDA; model input size:', size)


if __name__ == '__main__':
    main()

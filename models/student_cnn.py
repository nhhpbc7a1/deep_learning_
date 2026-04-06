# -*- coding: utf-8 -*-
"""
StudentCNN — CNN tương đối đơn giản (Conv + BatchNorm + ReLU), hai backbone:
  - ảnh nhỏ (<=32, CIFAR / midterm 32x32): vài tầng conv stride-2, AdaptiveAvgPool, FC.
  - ảnh lớn (224, ImageFolder): stem 7x7 stride 2 + MaxPool, rồi các khối tương tự.

Thiết kế hướng tới baseline ổn định: BN, GAP thay flatten đầy đặc trưng, dropout nhẹ trước FC.
"""
import torch
import torch.nn as nn


class StudentCNN(nn.Module):
    """
    Args:
        image_size: 32 (CIFAR / custom nhỏ) hoặc 224 (custom lớn).
        n_class: số lớp phân loại.
        dropout: dropout trước lớp FC (mặc định 0.3).
    """

    def __init__(self, image_size, n_class, dropout=0.3):
        super(StudentCNN, self).__init__()
        self.image_size = image_size
        self.n_class = n_class

        if image_size <= 32: #????
            self.features = self._make_small_image_backbone()
            feat_dim = 512
        else:
            self.features = self._make_large_image_backbone()
            feat_dim = 512

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(feat_dim, n_class)
        self._init_weights()

    def _make_small_image_backbone(self):
        """32x32: hai conv mỗi độ phân giải trước khi downsample (VGG-lite)."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

    def _make_large_image_backbone(self):
        """224x224: stem kiểu ImageNet nhẹ + giảm kích thước dần tới 7x7."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

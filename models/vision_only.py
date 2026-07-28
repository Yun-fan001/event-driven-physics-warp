"""
基线 B: Vision-Only
仅用视觉帧预测事件，证明力觉模态的必要性
"""

from __future__ import annotations
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """轻量级 CNN 编码单帧图像"""
    def __init__(self, in_channels: int = 3, feat_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),           # 16→8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(64 * 4 * 4, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class VisionOnly(nn.Module):
    def __init__(self, num_classes, feat_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()  # 【修复】必须调用父类初始化
        self.encoder = VisionEncoder(feat_dim=feat_dim)  # 【修复】补上缺失的图像编码器
        
        # 【公平性调整】将 bidirectional=False，保证因果性（与 Event-TCN 对齐）
        self.lstm = nn.LSTM(
            feat_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=False,  # 改 False
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        # 由于双向改单向，输入维度从 hidden_dim*2 改回 hidden_dim
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            logits: (B, T, num_classes)
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.encoder(x)  # (B*T, feat_dim)
        feat = feat.reshape(B, T, -1)
        h, _ = self.lstm(feat)  # (B, T, hidden_dim*2)
        logits = self.classifier(h)  # (B, T, num_classes)
        return logits
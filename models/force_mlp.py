"""
基线 A: Force-MLP
路径: 连续力回归 → 阈值判断事件
模拟传统"先恢复物理量再推理"的 pipeline
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ForceMLP(nn.Module):
    """
    输入: 力觉序列 (B, T, 3)  或 力+速度 (B, T, 6)
    输出: 每帧连续力预测 (B, T, 3)，外部通过阈值转为离散事件
    """
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.force_head = nn.Linear(hidden_dim, 3)  # 预测三维力

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) 力觉序列
        Returns:
            force_pred: (B, T, 3) 连续力预测
        """
        B, T, _ = x.shape
        x = x.reshape(B * T, -1)
        h = self.encoder(x)
        force_pred = self.force_head(h)
        force_pred = force_pred.reshape(B, T, 3)
        return force_pred
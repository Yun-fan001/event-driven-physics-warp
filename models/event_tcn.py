"""
主模型: Event-TCN (Temporal Convolutional Network)
输入: 多模态序列 (视觉特征 + 力觉 + 本体感知)
输出: 离散事件分类
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """因果卷积: 确保输出 t 时刻只依赖输入 ≤ t 时刻"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        # 截断前向填充，保持因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class ResidualBlock(nn.Module):
    """TCN 残差块: 两个因果卷积 + 残差连接"""
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.activation(self.norm2(self.conv2(out)))
        out = self.dropout(out)
        return out + residual


class EventTCN(nn.Module):
    """
    Event-Driven TCN for Contact State Classification
    
    架构:
    1. 视觉编码器 (CNN) → 视觉特征
    2. 模态融合投影 → 统一维度
    3. TCN 时序编码 (多层空洞因果卷积)
    4. 分类头 → 每帧事件类别
    
    输入:
        vision: (B, T, C, H, W)  视觉帧序列
        force:  (B, T, F)        力觉序列 (F=3 or 6)
        proprio: (B, T, P)       本体感知 (位置等, P=1~3)
    输出:
        logits: (B, T, num_classes)
    """
    def __init__(
        self,
        num_classes: int,
        vision_feat_dim: int = 64,
        force_dim: int = 3,
        proprio_dim: int = 1,
        hidden_dim: int = 128,
        tcn_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vision_feat_dim = vision_feat_dim

        # 视觉编码 (vision_feat_dim=0 时跳过，用于 force_tcn 消融)
        if vision_feat_dim > 0:
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.vision_fc = nn.Linear(64 * 4 * 4, vision_feat_dim)
        else:
            self.vision_encoder = None
            self.vision_fc = None

        # 模态融合投影
        total_input_dim = vision_feat_dim + force_dim + proprio_dim
        self.fusion_proj = nn.Linear(total_input_dim, hidden_dim)

        # TCN: 多层残差块，空洞率指数增长
        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                ResidualBlock(hidden_dim, kernel_size, dilation, dropout)
            )

        # 分类头
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        vision: torch.Tensor | None,
        force: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        if vision is None:
            B, T = force.shape[:2]
        else:
            B, T = vision.shape[:2]

        # 1. 视觉编码 (vision_feat_dim=0 或 vision=None 时跳过)
        if self.vision_feat_dim > 0 and vision is not None:
            v = vision.reshape(B * T, *vision.shape[2:])
            v_feat = self.vision_encoder(v)
            v_feat = v_feat.flatten(1)
            v_feat = self.vision_fc(v_feat)
            v_feat = v_feat.reshape(B, T, -1)
            # 2. 模态拼接 (含视觉)
            x = torch.cat([v_feat, force, proprio], dim=-1)
        else:
            # 力觉-only: 跳过视觉编码
            x = torch.cat([force, proprio], dim=-1)

        # 3. 融合投影
        x = self.fusion_proj(x)  # (B, T, hidden_dim)

        # 3. TCN: (B, T, C) → (B, C, T) → TCN → (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)  # (B, hidden_dim, T)
        for layer in self.tcn_layers:
            x = layer(x)
        x = x.transpose(1, 2)  # (B, T, hidden_dim)

        # 4. 分类
        logits = self.classifier(x)  # (B, T, num_classes)
        return logits
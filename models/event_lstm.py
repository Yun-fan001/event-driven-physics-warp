"""
降级方案: Event-LSTM (Bi-LSTM 多模态时序模型)
Plan B: 当 Event-TCN 训练不稳定时降级为此模型。
与 Event-TCN 使用相同的输入接口 (视觉+力觉+本体)，但用时序 LSTM 替代因果卷积。

注意: Bi-LSTM 使用了未来帧信息，不保证因果性，不适合在线部署。
      论文中需将此作为 discussion 点说明。
"""

from __future__ import annotations
import torch
import torch.nn as nn


class EventLSTM(nn.Module):
    """
    Event-Driven Bi-LSTM for Contact State Classification (TCN 降级方案)

    架构:
    1. 视觉编码器 (CNN) → 视觉特征
    2. 模态融合投影 → 统一维度
    3. Bi-LSTM 时序编码 (多层，双向)
    4. 分类头 → 每帧事件类别

    输入:
        vision: (B, T, C, H, W)  视觉帧序列
        force:  (B, T, F)        力觉序列 (F=6: 力3+速度3)
        proprio: (B, T, P)       本体感知 (位置等, P=1~2)
    输出:
        logits: (B, T, num_classes)
    """
    def __init__(
        self,
        num_classes: int,
        vision_feat_dim: int = 64,
        force_dim: int = 6,
        proprio_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vision_feat_dim = vision_feat_dim

        # 视觉编码 (vision_feat_dim=0 时跳过，用于 force_lstm 消融)
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

        # Bi-LSTM: 双向以获得完整时序上下文
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 分类头 (Bi-LSTM 输出维度 ×2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

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

        # 1. 视觉编码
        if self.vision_feat_dim > 0 and vision is not None:
            v = vision.reshape(B * T, *vision.shape[2:])
            v_feat = self.vision_encoder(v)
            v_feat = v_feat.flatten(1)
            v_feat = self.vision_fc(v_feat)
            v_feat = v_feat.reshape(B, T, -1)
            x = torch.cat([v_feat, force, proprio], dim=-1)
        else:
            # 力觉-only 模式
            x = torch.cat([force, proprio], dim=-1)

        # 2. 融合投影
        x = self.fusion_proj(x)  # (B, T, hidden_dim)

        # 3. Bi-LSTM 时序编码
        x, _ = self.lstm(x)  # (B, T, hidden_dim*2)

        # 4. 分类
        logits = self.classifier(x)  # (B, T, num_classes)
        return logits
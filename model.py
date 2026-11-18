"""
model.py
--------
Architecture:

    Input image
        -> EfficientNetV2-B0 backbone (pretrained, progressively unfrozen)  [local features]
        -> Group-Wise Transformer block(s)                                  [global context, cheap]
        -> Dual Attention module (Triplet Attention or CBAM)                [spatial + channel focus]
        -> Global Average Pool -> Dropout -> Dense -> GELU -> Dense -> Softmax
"""

import torch
import torch.nn as nn
import timm

import config


# ---------------------------------------------------------------------------
# 1. Backbone
# ---------------------------------------------------------------------------
class EfficientNetV2Backbone(nn.Module):
    """Wraps timm's EfficientNetV2-B0, returning the last conv feature map
    (before pooling/classifier) instead of logits.

    Progressive freezing: the first `freeze_stages` blocks (the early
    Fused-MBConv stages) start frozen to preserve generalized edge/texture
    weights. Call `unfreeze_all()` from the training loop once you reach the
    late-stage fine-tuning phase.
    """

    def __init__(self, freeze_stages: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.out_channels = self.backbone.num_features  # 1280 for effnetv2-b0

        self.freeze_stages = freeze_stages
        self._freeze_early_layers()

    def _freeze_early_layers(self):
        stages = list(self.backbone.blocks.children())
        for stage in stages[: self.freeze_stages]:
            for p in stage.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        """Called mid-training (progressive fine-tuning) to unfreeze
        everything, including the early Fused-MBConv stages."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.backbone.forward_features(x)  # shape: (B, 1280, H', W')


# ---------------------------------------------------------------------------
# 2. Group-Wise Transformer block (EGWT-style, lightweight global context)
# ---------------------------------------------------------------------------
class GroupWiseMultiHeadAttention(nn.Module):
    """Splits the channel dimension into `num_groups` groups and runs a small
    multi-head self-attention independently (with shared weights) within each
    group, over the flattened spatial tokens."""

    def __init__(self, channels: int, num_groups: int = 4, num_heads: int = 4):
        super().__init__()
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        self.num_groups = num_groups
        self.group_dim = channels // num_groups

        self.mha = nn.MultiheadAttention(embed_dim=self.group_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.norm = nn.LayerNorm(self.group_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)

        groups = torch.chunk(tokens, self.num_groups, dim=-1)

        out_groups = []
        for g in groups:
            attn_out, _ = self.mha(g, g, g)
            out_groups.append(self.norm(attn_out + g))

        out = torch.cat(out_groups, dim=-1)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out


class GroupWiseMLP(nn.Module):
    """Group-wise feed-forward block (channel-split MLP), paired with the
    attention block above to form one transformer block."""

    def __init__(self, channels: int, num_groups: int = 4, expansion: int = 2):
        super().__init__()
        assert channels % num_groups == 0
        self.num_groups = num_groups
        group_dim = channels // num_groups
        hidden = group_dim * expansion

        self.fc1 = nn.ModuleList([nn.Linear(group_dim, hidden) for _ in range(num_groups)])
        self.fc2 = nn.ModuleList([nn.Linear(hidden, group_dim) for _ in range(num_groups)])
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        groups = torch.chunk(tokens, self.num_groups, dim=-1)

        out_groups = []
        for i, g in enumerate(groups):
            y = self.fc2[i](self.act(self.fc1[i](g)))
            out_groups.append(y)

        out = torch.cat(out_groups, dim=-1)
        out = self.norm(out + tokens)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out


class GroupWiseTransformerBlock(nn.Module):
    """One full block: Group-Wise MHSA + Group-Wise MLP."""

    def __init__(self, channels: int, num_groups: int = 4, num_heads: int = 4):
        super().__init__()
        self.attn = GroupWiseMultiHeadAttention(channels, num_groups, num_heads)
        self.mlp = GroupWiseMLP(channels, num_groups)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# 3. Dual attention: Triplet Attention (default) and CBAM (alternative)
# ---------------------------------------------------------------------------
class ZPool(nn.Module):
    """Concatenate max-pool and avg-pool across the channel dimension."""

    def forward(self, x):
        return torch.cat([torch.max(x, dim=1, keepdim=True)[0],
                           torch.mean(x, dim=1, keepdim=True)], dim=1)


class AttentionGate(nn.Module):
    """One branch of Triplet Attention: Z-pool -> conv -> sigmoid gate."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.zpool = ZPool()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        y = self.zpool(x)
        y = self.bn(self.conv(y))
        scale = torch.sigmoid(y)
        return x * scale


class TripletAttention(nn.Module):
    """Triplet Attention (Misra et al., 2021): three branches capturing
    cross-dimensional interactions (C-H, C-W, H-W), averaged together."""

    def __init__(self):
        super().__init__()
        self.gate_hw = AttentionGate()
        self.gate_cw = AttentionGate()
        self.gate_ch = AttentionGate()

    def forward(self, x):
        out_hw = self.gate_hw(x)

        x_cw = x.permute(0, 2, 1, 3).contiguous()
        out_cw = self.gate_cw(x_cw)
        out_cw = out_cw.permute(0, 2, 1, 3).contiguous()

        x_ch = x.permute(0, 3, 2, 1).contiguous()
        out_ch = self.gate_ch(x_ch)
        out_ch = out_ch.permute(0, 3, 2, 1).contiguous()

        return (out_hw + out_cw + out_ch) / 3.0


class CBAM(nn.Module):
    """Alternative dual-attention option: channel attention + spatial attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * ca

        sa_input = torch.cat([torch.mean(x, dim=1, keepdim=True),
                               torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        sa = torch.sigmoid(self.spatial_conv(sa_input))
        x = x * sa
        return x


# ---------------------------------------------------------------------------
# 4. Classification head
# ---------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    """GAP -> Dropout -> Dense -> GELU -> Dropout -> Dense -> (Softmax via loss).

    Softmax itself is not applied here — it's baked into
    FocalLossWithLabelSmoothing at train time (via log_softmax) and should
    be applied explicitly at inference.
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, feats):
        return self.net(self.pool(feats))  # raw logits


# ---------------------------------------------------------------------------
# 5. Full model
# ---------------------------------------------------------------------------
class PlantDiseaseNet(nn.Module):
    def __init__(self,
                 num_classes: int = config.NUM_CLASSES,
                 num_groups: int = config.NUM_GROUPS,
                 num_heads: int = config.TRANSFORMER_HEADS,
                 transformer_depth: int = config.TRANSFORMER_DEPTH,
                 attention_type: str = config.ATTENTION_TYPE,
                 dropout: float = config.DROPOUT,
                 freeze_stages: int = config.FREEZE_BACKBONE_STAGES):
        super().__init__()

        self.backbone = EfficientNetV2Backbone(freeze_stages=freeze_stages)
        c = self.backbone.out_channels

        self.transformer_blocks = nn.Sequential(*[
            GroupWiseTransformerBlock(c, num_groups=num_groups, num_heads=num_heads)
            for _ in range(transformer_depth)
        ])

        if attention_type == "triplet":
            self.attention = TripletAttention()
        elif attention_type == "cbam":
            self.attention = CBAM(c)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.head = ClassificationHead(c, num_classes, dropout=dropout)

    def unfreeze_backbone(self):
        """Call from the training loop at the scheduled epoch to begin
        late-stage fine-tuning of the full backbone (progressive unfreezing)."""
        self.backbone.unfreeze_all()

    def forward(self, x, return_features: bool = False):
        feats = self.backbone(x)                # (B, 1280, H', W')
        feats = self.transformer_blocks(feats)  # global context
        feats = self.attention(feats)           # dual attention refinement
        logits = self.head(feats)

        if return_features:
            return logits, feats
        return logits


def build_model():
    model = PlantDiseaseNet()
    return model.to(config.DEVICE)


if __name__ == "__main__":
    m = build_model()
    dummy = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
    out = m(dummy)
    print("Output shape:", out.shape)  # expect (2, NUM_CLASSES)
    print("Backbone stage count:", len(list(m.backbone.backbone.blocks.children())))
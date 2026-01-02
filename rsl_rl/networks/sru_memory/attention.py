#  Copyright 2025 ETH Zurich
#  Created by Fan Yang, Robotic Systems Lab, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

"""Cross-attention module for fusing image features with proprioceptive info.

Architecture follows the paper: Self-attention → Cross-attention → SRU → MLP.
"""

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import PositionalEncodingPermute3D


class CrossAttentionFuseModule(nn.Module):
    """Cross-Attention Module for combining volumetric features with external embeddings.

    Combines self-attention over volumetric features with cross-attention using
    an external info embedding (proprioceptive information).

    Supports a list of 2D feature maps of varying spatial sizes, which are
    zero-padded, stacked along a new depth dimension, and processed with a 3D
    positional encoding. Padding positions are masked out during attention.

    Args:
        image_dim: Number of channels in the given features.
        info_dim: Dimension of the info embedding.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        image_dim: int,
        info_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        assert image_dim % num_heads == 0, "image_dim must be divisible by num_heads"

        expand_dim = image_dim * 2
        self.image_dim = image_dim
        self.info_dim = info_dim
        self.num_heads = num_heads

        # Info projection: 2-layer MLP with ELU
        self.info_proj = nn.Sequential(
            nn.Linear(info_dim, expand_dim),
            nn.ELU(inplace=True),
            nn.Linear(expand_dim, image_dim),
            nn.ELU(inplace=True),
        )

        # Positional encoding for 2D feature maps
        self.position_embedding = PositionalEncodingPermute3D(image_dim)

        # Self-attention sub-layer
        self.norm1 = nn.LayerNorm(image_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Feed-forward sub-layer
        self.norm2 = nn.LayerNorm(image_dim)
        self.ffn = nn.Sequential(
            nn.Linear(image_dim, expand_dim),
            nn.ELU(inplace=True),
            nn.Linear(expand_dim, image_dim),
            nn.ELU(inplace=True),
        )

        # Cross-attention sub-layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        img: Union[torch.Tensor, List[torch.Tensor]],
        info: torch.Tensor,
    ) -> torch.Tensor:
        """Process image features with proprioceptive info.

        Args:
            img: Tensor of shape (B, C, H, W) or list of Tensors [(B, C, H_i, W_i), ...]
            info: Tensor of shape (B, info_dim)

        Returns:
            Tensor of shape (B, image_dim)
        """
        # 1. pad & stack if list
        if isinstance(img, torch.Tensor):
            B, C, H, W = img.shape
            feats = img.unsqueeze(2)  # (B,C,1,H,W)
            mask = None
        else:
            views = img
            B, C = views[0].shape[:2]
            H_max = max(v.shape[2] for v in views)
            W_max = max(v.shape[3] for v in views)
            padded, masks = [], []
            for v in views:
                _, _, h, w = v.shape
                pad = (0, W_max - w, 0, H_max - h)
                vp = F.pad(v, pad)
                padded.append(vp)
                m = torch.zeros((B, h, w), dtype=torch.bool, device=v.device)
                masks.append(F.pad(m, pad, value=True))
            feats = torch.stack(padded, dim=2)  # (B,C,D,H_max,W_max)
            mask = torch.stack(masks, dim=1)  # (B,D,H_max,W_max)

        # 2. add 3D pos-embed
        feats = feats + self.position_embedding(feats)

        # 3. flatten to (B, N, C)
        B, C, D, H, W = feats.shape
        x = feats.view(B, C, D * H * W).permute(0, 2, 1)

        # 4. build padding mask
        key_mask = mask.view(B, D * H * W) if mask is not None else None

        # 5. self-attention (pre-norm + residual)
        x_norm = self.norm1(x)
        sa, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=key_mask, need_weights=False)
        x = x + sa

        # 6. feed-forward (pre-norm + residual)
        x = x + self.ffn(self.norm2(x))

        # 7. cross-attention with info query
        q = self.info_proj(info).unsqueeze(1)  # (B,1,C)
        ca, _ = self.cross_attn(q, x, x, key_padding_mask=key_mask, need_weights=False)

        return ca.squeeze(1)  # (B, C)

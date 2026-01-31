"""
EMA Codebook + Engram (content-aware position) for Stage 4.
Plan: cluster_ids = ema_codebook(_x4) -> Engram(_x4, position_ids=cluster_ids) -> knowledge_tokens.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class EMACodebook(nn.Module):
    """
    Content-aware clustering: hidden_states -> cluster_ids (for Engram position_ids).
    Codebook (K, C) updated by EMA; no grad through codebook into backbone.
    """

    def __init__(self, K: int, C: int, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.K = K
        self.C = C
        self.momentum = momentum
        self.eps = eps
        codebook = torch.randn(K, C)
        codebook = F.normalize(codebook, dim=-1)
        self.register_buffer("codebook", codebook)

    def forward(self, hidden_states: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        hidden_states: (B, L, C)
        Returns: cluster_ids (B, L), long, values in [0, K-1]
        """
        feat = hidden_states.detach()
        feat = F.normalize(feat, dim=-1, eps=self.eps)
        B, L, C = feat.shape
        feat_flat = feat.reshape(-1, C)
        dist = torch.cdist(feat_flat, self.codebook, p=2)
        cluster_ids = dist.argmin(dim=1)
        cluster_ids = cluster_ids.reshape(B, L)

        if update and self.training:
            with torch.no_grad():
                for k in range(self.K):
                    mask = (cluster_ids == k)
                    if mask.any():
                        mean_k = feat_flat[mask.reshape(-1)].mean(dim=0)
                        mean_k = F.normalize(mean_k.unsqueeze(0), dim=-1, eps=self.eps).squeeze(0)
                        self.codebook[k] = F.normalize(
                            self.momentum * self.codebook[k] + (1.0 - self.momentum) * mean_k,
                            dim=-1, eps=self.eps
                        )

        return cluster_ids


class EngramVision(nn.Module):
    """
    Engram: hash position_ids (cluster_ids) -> embedding; gate by hidden_states -> knowledge_tokens.
    position_ids (B, L) in [0, K-1]; output same shape as x (B, L, C).
    """

    def __init__(self, dim: int, vocab_size: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.gate_q = nn.Linear(dim, dim)
        self.gate_k = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C), position_ids: (B, L) long, values in [0, K-1] (or hashed)
        Returns: knowledge_tokens (B, L, C)
        """
        B, L, C = x.shape
        if position_ids.max() >= self.vocab_size:
            position_ids = position_ids.clamp(max=self.vocab_size - 1)
        emb = self.embed(position_ids)
        q = self.gate_q(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.gate_k(emb).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        gate = (q * k).sum(dim=-1) * self.scale
        gate = gate.sigmoid()
        gate = gate.mean(dim=1).unsqueeze(-1)
        out = gate * emb
        return out


class EngramInject(nn.Module):
    """Learnable alpha: out = x + alpha * engram, alpha = exp(log_alpha). Init -2.0 -> alpha ~ 0.135."""

    def __init__(self, init_alpha: float = -2.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor, engram: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.log_alpha)
        return x + alpha * engram

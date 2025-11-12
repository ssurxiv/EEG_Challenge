# -*- coding: utf-8 -*-
# =========================
# DIR utilities and losses
# =========================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Constants
# -------------------------
EPS = 1e-12


# -------------------------
# Small MLP block
# -------------------------
class MLP(nn.Module):
    """2-layer MLP with GELU and dropout"""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        dropout: float = 0.1,
        last_act: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.last_act = last_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        if self.last_act is not None:
            x = self.last_act(x)
        return x


# -------------------------
# Grouping helpers
# -------------------------
@torch.no_grad()
def compute_quantile_edges(
    y_train_t: torch.Tensor,
    G: int,
    ignore_nan: bool = True,
) -> np.ndarray:
    """
    Compute G+1 quantile edges for grouping a continuous target.

    Args
        y_train_t  target tensor [N] or [N, 1] on any device
        G          number of groups
        ignore_nan remove NaNs before computing edges

    Returns
        edges      numpy array shape [G+1]
    """
    y_np = y_train_t.view(-1).detach().cpu().numpy()
    if ignore_nan:
        y_np = y_np[~np.isnan(y_np)]
    if y_np.size == 0:
        raise ValueError("y_train_t has no valid values after NaN filtering")

    uniq = np.unique(y_np)
    if uniq.size < G:
        # If many duplicates, place edges uniformly over unique values
        ranks = np.linspace(0.0, 1.0, G + 1)
        edges = np.interp(ranks, np.linspace(0.0, 1.0, uniq.size), uniq)
    else:
        edges = np.quantile(y_np, np.linspace(0.0, 1.0, G + 1))

    # Ensure strictly increasing edges to avoid empty bins
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.float64("inf"))
    return edges


@torch.no_grad()
def make_group_id_quantile(
    y: torch.Tensor,
    quantile_edges: np.ndarray,
) -> torch.Tensor:
    """
    Map values to group ids by precomputed quantile edges.

    Args
        y               tensor [B] or [B, 1]
        quantile_edges  numpy array [G+1]

    Returns
        g               long tensor [B] in [0, G-1]
    """
    y_cpu = y.view(-1).detach().cpu().numpy()
    g = np.digitize(y_cpu, quantile_edges[1:-1], right=False)  # 0..G-1
    return torch.as_tensor(g, device=y.device, dtype=torch.long)


def make_group_id(
    y: torch.Tensor,
    y_min: torch.Tensor | float,
    y_max: torch.Tensor | float,
    G: int,
) -> torch.Tensor:
    """
    Uniformly bucketize by min-max range.

    Args
        y      tensor [B] or [B, 1]
        y_min  scalar or 0-dim tensor
        y_max  scalar or 0-dim tensor
        G      number of groups

    Returns
        g      long tensor [B] in [0, G-1]
    """
    y = y.view(-1)
    y_min = torch.as_tensor(y_min, device=y.device, dtype=y.dtype)
    y_max = torch.as_tensor(y_max, device=y.device, dtype=y.dtype)
    r = (y - y_min) / (y_max - y_min + 1e-8)  # [0, 1]
    g = torch.clamp((r * G).long(), max=G - 1)
    return g


def make_descending_soft_labels(
    g_idx: torch.Tensor,
    G: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Soft label centered at the true group that decays with |Δg|.

    Args
        g_idx  long tensor [B]
        G      number of groups
        beta   larger beta → sharper distribution

    Returns
        q_soft float tensor [B, G] that sums to 1 over G
    """
    B = g_idx.shape[0]
    ar = torch.arange(G, device=g_idx.device).unsqueeze(0).expand(B, -1)  # [B, G]
    center = g_idx.view(-1, 1)                                            # [B, 1]
    scores = G - beta * (ar - center).abs().float()
    return torch.softmax(scores, dim=1)


# -------------------------
# Multi-expert head
# -------------------------
class MultiExpertDIR(nn.Module):
    """
    Group classifier + group-specific regressors
    """

    def __init__(self, d_embed: int, num_groups: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.cls_head = nn.Linear(d_embed, num_groups)
        self.experts = nn.ModuleList([MLP(d_embed, hidden, 1, dropout=dropout) for _ in range(num_groups)])

    def forward(
        self,
        z: torch.Tensor,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
            z           features [B, D]
            return_all  if False return only (logits, p_g). If True also return y_all

        Returns
            logits  [B, G]
            p_g     [B, G]
            y_all   [B, G] only when return_all=True
        """
        logits = self.cls_head(z)               # [B, G]
        p_g = torch.softmax(logits, dim=1)      # [B, G]
        if not return_all:
            return logits, p_g

        y_list = [exp(z) for exp in self.experts]   # G tensors [B, 1]
        y_all = torch.cat(y_list, dim=1)            # [B, G]
        return logits, p_g, y_all


# -------------------------
# Loss helpers
# -------------------------
def _soft_ce_from_logits(logits: torch.Tensor, targets_soft: torch.Tensor) -> torch.Tensor:
    """- Σ q log p with log-softmax for stability"""
    logp = F.log_softmax(logits, dim=1)
    return -(targets_soft * logp).sum(dim=1).mean()


def _pick_expert_outputs(y_hat_all: torch.Tensor, g_true: torch.Tensor) -> torch.Tensor:
    """Gather expert outputs by true group id"""
    idx = g_true.view(-1, 1)  # [B, 1]
    return torch.gather(y_hat_all, dim=1, index=idx)  # [B, 1]


# -------------------------
# Joint DIR objective
# -------------------------
def dir_loss(
    logits: torch.Tensor,
    y_hat_all: torch.Tensor,
    y_true: torch.Tensor,
    g_true: torch.Tensor,
    q_soft: torch.Tensor,
    lambda_soft: float = 0.5,
    lambda_mse: float = 1.0,
    teacher_force: bool = True,
    sample_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined loss
      L = λ_mse * MSE + λ_soft * SoftCE

    Args
        logits         [B, G]
        y_hat_all      [B, G] expert outputs
        y_true         [B, 1]
        g_true         [B]
        q_soft         [B, G]
        sample_weight  [B] optional density or relevance weights

    Returns
        loss, logs
    """
    # 1) soft-label classification
    L_soft = _soft_ce_from_logits(logits, q_soft)

    # 2) regression
    if teacher_force:
        y_hat = _pick_expert_outputs(y_hat_all, g_true)  # [B, 1]
    else:
        p = torch.softmax(logits, dim=1)                 # [B, G]
        y_hat = (p * y_hat_all).sum(dim=1, keepdim=True) # [B, 1]

    se = (y_hat - y_true) ** 2
    if sample_weight is not None:
        sw = sample_weight.view(-1, 1)
        L_mse = torch.sum(sw * se) / (sw.sum() + EPS)
    else:
        L_mse = torch.mean(se)

    L = lambda_mse * L_mse + lambda_soft * L_soft
    logs = {"L_total": float(L.detach()), "L_mse": float(L_mse.detach()), "L_soft": float(L_soft.detach())}
    return L, logs


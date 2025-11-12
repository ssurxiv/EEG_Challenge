# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# 선택 1: EEGNeX 사용
from braindecode.models import EEGNeX
# 선택 2: 본인 모듈 사용
# from NICE_EEG import NICE_EEG

FACTOR_ORDER = ['internalizing', 'attention', 'p-factor', 'age', 'sex']
IDX = {k: i for i, k in enumerate(FACTOR_ORDER)}


# -------------------------
# Small MLP block
# -------------------------
class MLP(nn.Module):
    """2-layer MLP with GELU and dropout"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 dropout: float = 0.1, last_act: nn.Module | None = None) -> None:
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
# Cascaded model
# -------------------------
class CascadedEEGModel(nn.Module):
    """
    EEG → factors[5] → externalizing

    factors order: ['internalizing','attention','p-factor','age','sex']
      - sex는 BCE용 로짓을 내고 필요 시 sigmoid 확률도 제공
      - externalizing head 입력에는 옵션에 따라 sex의 logit 또는 prob를 사용

    Args
      n_channels, n_times     입력 형태
      d_eeg                   encoder 출력 차원
      h_factor, h_ext         factor/externalizing 헤드 은닉 차원
      dropout                 드롭아웃 확률
      n_factor                factor 개수 (기본 5)
      stop_grad_factors       ext 헤드에 들어가는 factor 벡터를 detach할지
      use_residual_fusion     z와 factor 기반 게이트 z*gate를 추가로 결합할지
      use_layernorm_on_z      z에 LayerNorm 적용 여부
      ext_nonneg              externalizing이 음수 불가면 Softplus 사용
      feed_sex_prob_to_ext    ext 입력에 sex 확률(sigmoid) vs sex 로짓
      inference               True면 forward가 externalizing만 반환하도록 단축
      encoder                 외부에서 주입할 인코더 인스턴스(옵션)

    Inputs
      X: [B, C, T]

    Returns
      inference=True  → torch.Tensor [B, 1]  (externalizing)
      inference=False → dict
        {
          'factors_vec': [B,5],
          'internalizing': [B,1], 'attention': [B,1],
          'p-factor': [B,1], 'age': [B,1],
          'sex_logit': [B,1], 'sex_prob': [B,1],
          'externalizing': [B,1],
          'z': [B, d_eeg] or None
        }
    """

    def __init__(self,
                 n_channels: int = 128,
                 n_times: int = 200,
                 d_eeg: int = 64,
                 h_factor: int = 128,
                 h_ext: int = 128,
                 dropout: float = 0.1,
                 n_factor: int = 5,
                 stop_grad_factors: bool = False,
                 use_residual_fusion: bool = True,
                 use_layernorm_on_z: bool = True,
                 ext_nonneg: bool = False,
                 feed_sex_prob_to_ext: bool = True,
                 inference: bool = False,
                 encoder: nn.Module | None = None) -> None:
        super().__init__()

        # ----- Encoder -----
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = EEGNeX(n_chans=n_channels, n_times=n_times, n_outputs=d_eeg)

        self.stop_grad_factors = stop_grad_factors
        self.use_residual_fusion = use_residual_fusion
        self.use_layernorm_on_z = use_layernorm_on_z
        self.feed_sex_prob_to_ext = feed_sex_prob_to_ext
        self.inference = inference

        self.z_norm = nn.LayerNorm(d_eeg) if use_layernorm_on_z else nn.Identity()

        # ----- Factor head -----
        # 연속 4 + 이진 1(sex logit) → 합계 5
        self.factor_head = MLP(d_eeg, h_factor, out_dim=n_factor, dropout=dropout)

        # ----- Residual fusion gate (optional) -----
        self.fdim_for_ext = n_factor
        ext_in_dim = d_eeg + self.fdim_for_ext
        if self.use_residual_fusion:
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.fdim_for_ext, d_eeg),
                nn.GELU(),
                nn.Linear(d_eeg, d_eeg),
                nn.Sigmoid()
            )
            ext_in_dim += d_eeg  # concat z_gate

        # ----- Externalizing head -----
        last_act = nn.Softplus() if ext_nonneg else None
        self.ext_head = MLP(ext_in_dim, h_ext, out_dim=1, dropout=dropout, last_act=last_act)



    def forward(self,
                X: torch.Tensor,
                return_latent: bool = False,
                detach_factors: bool | None = None,
                reverse: bool = False):
        # 1) 인코딩
        z = self.encoder(X)                       # [B, d_eeg]
        z = self.z_norm(z) if self.use_layernorm_on_z else z

        # 2) factor 예측 및 분리
        factors_vec = self.factor_head(z)         # [B, 5]
        intl, attn, pfac, age, sex_logit, sex_prob = self._split_factors(factors_vec)

        # 3) ext 입력 벡터 구성
        if detach_factors is None:
            detach_factors = self.stop_grad_factors
        f_for_ext = self._build_factors_for_ext(
            intl, attn, pfac, age, sex_logit, sex_prob,
            use_prob=self.feed_sex_prob_to_ext,
            detach=detach_factors
        )
        ext_in = self._concat_ext_input(z, f_for_ext)

        # 4) externalizing
        ext = self.ext_head(ext_in)               # [B, 1]

        if self.inference:
            # inference 모드면 외부로는 ext만 반환
            return ext

        out = {
            "factors_vec": factors_vec,
            "internalizing": intl,
            "attention": attn,
            "p-factor": pfac,
            "age": age,
            "sex_logit": sex_logit,
            "sex_prob": sex_prob,
            "externalizing": ext,
            "z": z if return_latent else None
        }

        return out

    def predict_externalizing(self, X: torch.Tensor,
                              detach_factors: bool | None = None,
                              reverse: bool = False) -> torch.Tensor:
        """externalizing만 바로 얻고 싶을 때 사용"""
        old = self.inference
        self.inference = True
        y = self.forward(X, detach_factors=detach_factors, reverse=reverse)
        self.inference = old
        return y

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    @staticmethod
    def _split_factors(factors_vec: torch.Tensor):
        """[B,5] → 각 요소 [B,1]로 분리하고 sex 확률까지 생성"""
        intl = factors_vec[:, IDX['internalizing']:IDX['internalizing'] + 1]
        attn = factors_vec[:, IDX['attention']:IDX['attention'] + 1]
        pfac = factors_vec[:, IDX['p-factor']:IDX['p-factor'] + 1]
        age  = factors_vec[:, IDX['age']:IDX['age'] + 1]
        sex_logit = factors_vec[:, IDX['sex']:IDX['sex'] + 1]
        sex_prob  = torch.sigmoid(sex_logit)
        return intl, attn, pfac, age, sex_logit, sex_prob

    def _build_factors_for_ext(self,
                               intl: torch.Tensor,
                               attn: torch.Tensor,
                               pfac: torch.Tensor,
                               age: torch.Tensor,
                               sex_logit: torch.Tensor,
                               sex_prob: torch.Tensor,
                               use_prob: bool = True,
                               detach: bool = False) -> torch.Tensor:
        """ext 입력에 사용할 factor 벡터 구성"""
        sex_feat = sex_prob if use_prob else sex_logit
        f = torch.cat([intl, attn, pfac, age, sex_feat], dim=-1)  # [B, 5]
        return f.detach() if detach else f

    def _concat_ext_input(self, z: torch.Tensor, f_for_ext: torch.Tensor) -> torch.Tensor:
        """residual fusion 옵션에 따라 ext 입력을 생성"""
        if self.use_residual_fusion:
            g = self.gate_mlp(f_for_ext)  # [B, d_eeg]
            z_gate = z * g
            return torch.cat([z, f_for_ext, z_gate], dim=-1)
        return torch.cat([z, f_for_ext], dim=-1)


"""
OpenMythos Long Context — YaRN Dynamic NTK-aware RoPE Scaling。

YaRN (Yet another RoPE extensioN) は RoPE の周波数をスケーリングし、
学習時より長いシーケンスへの外挿を可能にする手法。
(Peng et al., 2023: https://arxiv.org/abs/2309.00071)

設計:
    yarn_rope_freqs()      -- YaRN スケーリング済み RoPE 周波数を生成
    get_rope_freqs()       -- config に応じて通常 or YaRN RoPE を選択
    RopeScalingConfig      -- スケーリング設定データクラス

使い方::

    from open_mythos.rope_extension import get_rope_freqs, RopeScalingConfig

    # 32K context で YaRN スケーリング
    cfg = RopeScalingConfig(type="yarn", factor=8.0, original_max_len=4096)
    freqs = get_rope_freqs(dim=128, max_len=32768, theta=500000.0, scaling=cfg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from open_mythos.main import OpenMythos

import torch


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------


@dataclass
class RopeScalingConfig:
    """
    RoPE スケーリング設定。

    Args:
        type              -- スケーリング手法: "none" / "linear" / "yarn" / "ntk"
        factor            -- スケーリング係数。4K→32K なら factor=8.0
        original_max_len  -- 学習時の最大シーケンス長 (デフォルト 4096)
        yarn_beta_fast    -- YaRN: high-freq 回転閾値 (デフォルト 32)
        yarn_beta_slow    -- YaRN: low-freq 回転閾値 (デフォルト 1)
        yarn_attn_factor  -- YaRN: attention スケール補正係数 (デフォルト 1.0)
        mscale            -- YaRN: magnitude スケーリング係数 (デフォルト 1.0)
    """

    type: Literal["none", "linear", "yarn", "ntk"] = "none"
    factor: float = 1.0
    original_max_len: int = 4096
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_attn_factor: float = 1.0
    mscale: float = 1.0

    @classmethod
    def for_32k(cls) -> "RopeScalingConfig":
        """4K→32K (factor=8) の推奨 YaRN 設定を返す。"""
        return cls(
            type="yarn",
            factor=8.0,
            original_max_len=4096,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            yarn_attn_factor=0.1 * math.log(8.0) + 1.0,
        )

    @classmethod
    def for_8k(cls) -> "RopeScalingConfig":
        """4K→8K (factor=2) の推奨 YaRN 設定を返す。"""
        return cls(
            type="yarn",
            factor=2.0,
            original_max_len=4096,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            yarn_attn_factor=0.1 * math.log(2.0) + 1.0,
        )


# ---------------------------------------------------------------------------
# RoPE 周波数計算
# ---------------------------------------------------------------------------


def _linear_scaled_freqs(
    dim: int,
    max_len: int,
    theta: float,
    factor: float,
) -> torch.Tensor:
    """
    Linear RoPE スケーリング。

    周波数を factor で割ることで実効的なコンテキスト長を伸ばす。
    シンプルだが品質低下しやすい。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = freqs / factor
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _ntk_scaled_freqs(
    dim: int,
    max_len: int,
    theta: float,
    factor: float,
) -> torch.Tensor:
    """
    NTK-aware RoPE スケーリング。

    base theta を factor^(dim/(dim-2)) でスケールし、
    高周波成分を保護しながら低周波成分を延伸する。
    """
    alpha = factor
    new_theta = theta * (alpha ** (dim / (dim - 2)))
    freqs = 1.0 / (new_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def yarn_rope_freqs(
    dim: int,
    max_len: int,
    theta: float = 500000.0,
    factor: float = 8.0,
    original_max_len: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attn_factor: float = 1.0,
    mscale: float = 1.0,
) -> torch.Tensor:
    """
    YaRN (Yet another RoPE extensioN) スケーリング済み RoPE 周波数を生成する。

    各周波数次元に対して interpolation ratio r を計算し:
        - r=0 (高周波): 元の周波数をそのまま使う (補間なし)
        - r=1 (低周波): Linear スケーリングを適用
        - 0<r<1 (中間周波): 線形混合

    これにより、高周波成分 (位置情報) は保護され、
    低周波成分 (グローバル文脈) のみが延伸される。

    Args:
        dim              -- head dimension (偶数)
        max_len          -- 生成する最大シーケンス長
        theta            -- RoPE base 周波数
        factor           -- 延伸倍率 (4K→32K なら 8.0)
        original_max_len -- 学習時の max_seq_len
        beta_fast        -- high-freq 閾値 (波長がこれより短い次元は補間なし)
        beta_slow        -- low-freq 閾値 (波長がこれより長い次元は完全補間)
        attn_factor      -- attention スコアへの補正スケール
        mscale           -- magnitude スケーリング

    Returns:
        complex64 tensor of shape (max_len, dim//2)
    """
    # 各次元の原周波数
    freq_indices = torch.arange(0, dim, 2, dtype=torch.float32)
    inv_freqs = 1.0 / (theta ** (freq_indices / dim))  # shape: (dim//2,)

    # wavelength (波長): 大きいほど低周波
    wavelengths = 2.0 * math.pi / inv_freqs  # shape: (dim//2,)

    # original_max_len における波長との比較
    low_freq_wavelen = original_max_len / beta_slow   # この波長より長い → 低周波 → 完全補間
    high_freq_wavelen = original_max_len / beta_fast  # この波長より短い → 高周波 → 補間なし

    # interpolation ratio r: 0=高周波(補間なし) / 1=低周波(完全補間)
    r = torch.zeros_like(inv_freqs)

    # 低周波領域 (波長 > low_freq_wavelen)
    low_mask = wavelengths > low_freq_wavelen
    r[low_mask] = 1.0

    # 中間周波数領域: 滑らかに補間
    mid_mask = ~low_mask & (wavelengths > high_freq_wavelen)
    if mid_mask.any():
        # YaRN 論文式: r = (original_max_len / wavelength - beta_fast) / (beta_slow - beta_fast)
        r_mid = (original_max_len / wavelengths[mid_mask] - beta_fast) / (beta_slow - beta_fast)
        r[mid_mask] = r_mid.clamp(0.0, 1.0)

    # 高周波領域 (wavelength <= high_freq_wavelen): r=0 (補間なし、既設定)

    # scaled_inv_freqs: r=0 → 元, r=1 → /factor の線形混合
    scaled_inv_freqs = inv_freqs / factor
    mixed_inv_freqs = (1.0 - r) * inv_freqs + r * scaled_inv_freqs  # shape: (dim//2,)

    # magnitude scaling (attention score correction)
    # attn_factor を適用することで extended context での attention entropy を補正
    if mscale != 1.0 or attn_factor != 1.0:
        scale = mscale * attn_factor
        mixed_inv_freqs = mixed_inv_freqs * scale

    # 位置ベクトルとの outer product
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, mixed_inv_freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def get_rope_freqs(
    dim: int,
    max_len: int,
    theta: float = 500000.0,
    scaling: Optional[RopeScalingConfig] = None,
) -> torch.Tensor:
    """
    設定に応じて適切な RoPE 周波数テンソルを返す。

    Args:
        dim     -- head dimension
        max_len -- 最大シーケンス長
        theta   -- RoPE base 周波数
        scaling -- スケーリング設定 (None または type="none" で通常 RoPE)

    Returns:
        complex64 tensor of shape (max_len, dim//2)
    """
    if scaling is None or scaling.type == "none" or scaling.factor <= 1.0:
        # 標準 RoPE (open_mythos.main.precompute_rope_freqs と同等)
        from open_mythos.main import precompute_rope_freqs
        return precompute_rope_freqs(dim, max_len, theta)

    if scaling.type == "linear":
        return _linear_scaled_freqs(dim, max_len, theta, scaling.factor)

    if scaling.type == "ntk":
        return _ntk_scaled_freqs(dim, max_len, theta, scaling.factor)

    if scaling.type == "yarn":
        return yarn_rope_freqs(
            dim=dim,
            max_len=max_len,
            theta=theta,
            factor=scaling.factor,
            original_max_len=scaling.original_max_len,
            beta_fast=scaling.yarn_beta_fast,
            beta_slow=scaling.yarn_beta_slow,
            attn_factor=scaling.yarn_attn_factor,
            mscale=scaling.mscale,
        )

    raise ValueError(f"Unknown rope scaling type: {scaling.type!r}")


# ---------------------------------------------------------------------------
# ユーティリティ: max_seq_len 拡張ヘルパー
# ---------------------------------------------------------------------------


def extend_model_context(
    model: "OpenMythos",
    new_max_len: int,
    scaling_type: Literal["yarn", "ntk", "linear"] = "yarn",
) -> "OpenMythos":
    """
    学習済みモデルの context 長を動的に拡張する。

    モデルの freqs_cis バッファを YaRN スケーリング済みのものに差し替える。
    重みは変更せず、RoPE 周波数のみを更新する。

    Args:
        model          -- OpenMythos インスタンス
        new_max_len    -- 新しい最大シーケンス長 (例: 32768)
        scaling_type   -- スケーリング手法

    Returns:
        freqs_cis を更新した同じモデルインスタンス (in-place)
    """
    original_max_len = model.cfg.max_seq_len
    factor = new_max_len / original_max_len

    scaling = RopeScalingConfig(
        type=scaling_type,
        factor=factor,
        original_max_len=original_max_len,
        yarn_attn_factor=0.1 * math.log(max(factor, 1.0)) + 1.0 if scaling_type == "yarn" else 1.0,
    )

    device = next(model.parameters()).device

    # GQA 用 freqs_cis 更新
    head_dim = model.cfg.dim // model.cfg.n_heads
    new_freqs_gqa = get_rope_freqs(head_dim, new_max_len, model.cfg.rope_theta, scaling).to(device)
    model.register_buffer("freqs_cis", new_freqs_gqa, persistent=False)

    # MLA 用 freqs_cis 更新
    new_freqs_mla = get_rope_freqs(
        model.cfg.qk_rope_head_dim, new_max_len, model.cfg.rope_theta, scaling
    ).to(device)
    model.register_buffer("freqs_cis_mla", new_freqs_mla, persistent=False)

    # cfg も更新 (フォワードパスが max_seq_len を参照する箇所のため)
    model.cfg.max_seq_len = new_max_len

    return model

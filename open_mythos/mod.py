"""
Mixture-of-Depths (MoD) Transformer
=====================================
Paper: "Mixture-of-Depths: Dynamically allocating compute in transformer
        language models" — Raposo et al., 2024 (arXiv:2404.02258)

Core idea
---------
Every transformer layer processes all tokens in a standard model.  MoD breaks
this by learning *which* tokens deserve full attention + FFN computation at
each layer and letting the rest skip through via a plain residual connection.

Architecture of one MoD layer::

    x ─── TokenRouter ──────→ scores (B, T)
               │
               ├─ top-k indices ─→ gather ─→ [Norm → Attn → Norm → FFN] ─→ scatter
               │                                                                │
               └─ rest indices  ──────────────────────────────────── identity  │
                                                                                │
                                                                     merged x (B, T, dim)

Key properties
--------------
* **FLOPs reduction**: only ``capacity_factor`` fraction of tokens go through
  the heavy block per layer → roughly ``(1 - capacity_factor)`` FLOPs saved.
* **Positional correctness**: selected tokens keep their original sequence
  positions.  After sorting the selected indices, the standard upper-triangular
  causal mask remains valid (earlier positions still precede later ones).
* **Per-batch routing**: different batch items can select different tokens,
  handled by gathering ``freqs_cis`` rows per batch item.
* **Auxiliary loss**: a soft load-balancing loss prevents routing collapse
  where the same tokens are always (or never) routed.

Exports
-------
MoDConfig, TokenRouter, MixtureOfDepthsBlock, MoDTransformer, MoDAnalytics,
precompute_mod_rope_freqs, apply_mod_rope, routing_entropy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse RMSNorm from core module (DRY — no duplicate definition)
from open_mythos.main import RMSNorm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MoDConfig:
    """
    Hyperparameters for a Mixture-of-Depths Transformer.

    Core
    ----
    vocab_size      : vocabulary size
    dim             : hidden dimension (must equal n_heads * head_dim)
    n_heads         : number of query attention heads
    n_kv_heads      : GQA key/value heads (must divide n_heads evenly)
    max_seq_len     : maximum sequence length for RoPE precomputation
    n_layers        : number of MoD layers

    MoD routing
    -----------
    capacity_factor       : fraction of tokens routed per layer (0 < x ≤ 1).
                            E.g. 0.5 → top-50% tokens run the full block.
    router_aux_loss_coef  : weight applied to the load-balancing auxiliary loss.
                            Set to 0.0 to disable.

    FFN
    ---
    ffn_hidden_mult : FFN hidden dim = dim * ffn_hidden_mult (SwiGLU uses 2/3
                      of this internally; default 4 matches standard practice)

    Other
    -----
    dropout   : residual dropout (0.0 disables)
    rope_theta: RoPE base frequency
    """

    # Core
    vocab_size: int = 32000
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2        # GQA: fewer KV heads than Q heads
    max_seq_len: int = 2048
    n_layers: int = 6
    # MoD routing
    capacity_factor: float = 0.5
    router_aux_loss_coef: float = 0.01
    # FFN
    ffn_hidden_mult: int = 4
    # Training
    dropout: float = 0.0
    # RoPE
    rope_theta: float = 10000.0


# ---------------------------------------------------------------------------
# RoPE helpers (positional encoding, batch-aware for MoD routing)
# ---------------------------------------------------------------------------


def precompute_mod_rope_freqs(
    head_dim: int,
    max_len: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Precompute complex RoPE rotation phasors for positions 0..max_len-1.

    Identical formula to the core model but kept in this module so ``mod.py``
    can be used standalone.  The result is a lookup table; each MoD layer
    gathers the rows corresponding to the selected token positions.

    Args:
        head_dim : per-head dimension (must be even)
        max_len  : maximum sequence length
        theta    : RoPE base frequency (higher = slower decay over positions)

    Returns:
        Complex64 tensor of shape ``(max_len, head_dim // 2)``
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_len, head_dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_mod_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE with **per-batch-item** positional frequencies.

    Standard ``apply_rope`` assumes the same positions for every element in the
    batch (shape ``(T, head_dim//2)``).  MoD routing selects different token
    positions per batch item, so this variant accepts a batched frequency
    tensor gathered from the full RoPE table.

    Args:
        x         : ``(B, T, H, head_dim)`` — query or key tensor
        freqs_cis : ``(B, T, head_dim//2)`` complex — already gathered per
                    batch item from the full ``(max_seq_len, head_dim//2)``
                    RoPE table

    Returns:
        Rotated tensor of the same shape and dtype as ``x``
    """
    # View last dim as complex pairs: (B, T, H, head_dim//2) complex
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis: (B, T, head_dim//2) → (B, T, 1, head_dim//2) to broadcast over H
    rotated = xc * freqs_cis.unsqueeze(2)
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Routing-diversity metric
# ---------------------------------------------------------------------------


def routing_entropy(scores: torch.Tensor) -> torch.Tensor:
    """
    Per-token binary routing entropy (in nats).

    Treats each token's routing logit as a Bernoulli parameter and returns
    the binary entropy H = -p*log(p) - (1-p)*log(1-p), where p = sigmoid(scores).

    Higher entropy ≈ more uncertain routing (router treats both "route" and
    "skip" as equally likely).  Values near 0 mean the router is confident.

    Args:
        scores : ``(B, T)`` routing logits (un-normalised)

    Returns:
        ``(B, T)`` per-token entropy in nats (always ≥ 0, ≤ ln 2 ≈ 0.693)

    Example::

        scores = torch.zeros(2, 8)      # 50/50 uncertainty
        h = routing_entropy(scores)
        assert (h - math.log(2)).abs().max() < 1e-5   # maximum entropy
    """
    eps = 1e-8
    p = torch.sigmoid(scores)                    # (B, T)
    h = -(p * (p + eps).log() + (1.0 - p) * (1.0 - p + eps).log())
    return h


# ---------------------------------------------------------------------------
# TokenRouter
# ---------------------------------------------------------------------------


class TokenRouter(nn.Module):
    """
    Lightweight per-token scalar router for Mixture-of-Depths.

    A single linear projection ``dim → 1`` assigns a routing score to each
    token.  Higher scores indicate tokens that benefit most from the full
    attention + FFN computation at this layer.

    The router is differentiable end-to-end; the auxiliary load-balancing loss
    is applied to the soft sigmoid scores so gradients flow back through
    routing decisions even though the hard top-k selection is not differentiable.
    """

    def __init__(self, dim: int) -> None:
        """
        Args:
            dim : model hidden dimension (input feature size)
        """
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token routing scores (logits).

        Args:
            x : ``(B, T, dim)`` hidden states

        Returns:
            ``(B, T)`` routing logits (unbounded; apply sigmoid for probabilities)
        """
        return self.proj(x).squeeze(-1)  # (B, T)

    def select_top_k(
        self,
        scores: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """
        Select the top-``capacity`` token indices per batch item, sorted by
        their **original sequence position** (not by score).

        Sorting is critical for causal attention correctness: after gathering
        the selected tokens, the standard upper-triangular causal mask remains
        valid because the relative ordering of positions is preserved.

        Args:
            scores   : ``(B, T)`` routing logits
            capacity : number of tokens to select (clipped to T if larger)

        Returns:
            ``(B, capacity)`` long tensor of position-sorted selected indices
        """
        B, T = scores.shape
        capacity = max(1, min(capacity, T))

        # Pick top-k positions by score (not yet sorted)
        _, topk_indices = scores.topk(capacity, dim=1)  # (B, capacity)

        # Sort by original position so causal mask stays valid
        selected_sorted, _ = topk_indices.sort(dim=1)  # (B, capacity)
        return selected_sorted


# ---------------------------------------------------------------------------
# MoD-aware attention (batched per-item RoPE)
# ---------------------------------------------------------------------------


class MoDGQAttention(nn.Module):
    """
    Grouped Query Attention adapted for Mixture-of-Depths.

    Identical to the GQA in ``open_mythos.main`` except that ``freqs_cis``
    is expected to have a **batch dimension** ``(B, T, head_dim//2)`` because
    MoD routing selects different token positions per batch item.

    This allows each sequence in a batch to process different token subsets
    while still receiving positionally-correct RoPE embeddings.
    """

    def __init__(self, cfg: MoDConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x         : ``(B, T_sel, dim)`` — selected-token hidden states
            freqs_cis : ``(B, T_sel, head_dim//2)`` complex — per-batch RoPE
            mask      : ``(1, 1, T_sel, T_sel)`` additive causal mask or None

        Returns:
            ``(B, T_sel, dim)`` attended output
        """
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply batched RoPE
        q = apply_mod_rope(q, freqs_cis)
        k = apply_mod_rope(k, freqs_cis)

        # Expand KV heads to match Q head count (GQA)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=2)  # (B, T, n_heads, head_dim)
            v = v.repeat_interleave(self.groups, dim=2)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


class MoDSwiGLU(nn.Module):
    """
    Dense SwiGLU feed-forward network for MoD blocks.

    Uses the standard SwiGLU formulation:
        FFN(x) = (SiLU(x W_gate) ⊙ (x W_up)) W_down

    Hidden dimension is scaled to ``dim * ffn_hidden_mult * 2/3`` (rounded to
    nearest multiple of 64) to keep parameter count equivalent to a standard
    4× FFN while benefiting from the gating expressiveness.
    """

    def __init__(self, dim: int, hidden_mult: int = 4) -> None:
        super().__init__()
        # SwiGLU effective hidden = 2/3 * full_hidden; round to multiple of 64
        full_hidden = dim * hidden_mult
        hidden = int(2 * full_hidden / 3)
        hidden = ((hidden + 63) // 64) * 64  # round up to multiple of 64

        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : ``(B, T, dim)``
        Returns:
            ``(B, T, dim)``
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


class MoDAnalytics:
    """
    Lightweight per-layer routing statistics collector.

    Attach an instance to ``MoDTransformer`` or pass one to individual
    ``MixtureOfDepthsBlock.forward`` calls to record how many tokens each
    layer processes and analyse compute utilisation.

    Typical usage::

        analytics = MoDAnalytics(n_layers=cfg.n_layers)
        logits, aux = model(input_ids, analytics=analytics)
        print(analytics.summary())
        analytics.reset()
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self._routed: List[List[int]] = [[] for _ in range(n_layers)]
        self._total: List[List[int]] = [[] for _ in range(n_layers)]
        self._entropy: List[List[float]] = [[] for _ in range(n_layers)]

    def record(
        self,
        layer_idx: int,
        n_routed: int,
        n_total: int,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        """Record one forward-pass observation for ``layer_idx``.

        Args:
            layer_idx : zero-based layer index
            n_routed  : number of tokens actually routed this step
            n_total   : total sequence length this step
            scores    : optional ``(B, T)`` routing logits; when provided,
                        the mean binary routing entropy for this step is
                        stored and included in :meth:`summary`.
        """
        self._routed[layer_idx].append(n_routed)
        self._total[layer_idx].append(n_total)
        if scores is not None:
            with torch.no_grad():
                mean_h = routing_entropy(scores).mean().item()
            self._entropy[layer_idx].append(mean_h)

    def summary(self) -> Dict[str, float]:
        """
        Return a flat dict with per-layer routing statistics.

        Keys (always present when data exists):
            ``layer_{i}_avg_routed``   — average number of tokens routed
            ``layer_{i}_avg_capacity`` — average fraction of tokens routed (0–1)

        Keys (present only when scores were supplied to :meth:`record`):
            ``layer_{i}_avg_entropy``  — average mean binary routing entropy (nats)
        """
        result: Dict[str, float] = {}
        for i in range(self.n_layers):
            if not self._routed[i]:
                continue
            avg_routed = sum(self._routed[i]) / len(self._routed[i])
            avg_cap = sum(
                r / max(t, 1)
                for r, t in zip(self._routed[i], self._total[i])
            ) / len(self._routed[i])
            result[f"layer_{i}_avg_routed"] = avg_routed
            result[f"layer_{i}_avg_capacity"] = avg_cap
            if self._entropy[i]:
                result[f"layer_{i}_avg_entropy"] = (
                    sum(self._entropy[i]) / len(self._entropy[i])
                )
        return result

    def reset(self) -> None:
        """Clear all recorded observations."""
        self._routed = [[] for _ in range(self.n_layers)]
        self._total = [[] for _ in range(self.n_layers)]
        self._entropy = [[] for _ in range(self.n_layers)]


# ---------------------------------------------------------------------------
# MixtureOfDepthsBlock
# ---------------------------------------------------------------------------


class MixtureOfDepthsBlock(nn.Module):
    """
    One Mixture-of-Depths transformer layer.

    Selects the top ``capacity_factor`` fraction of tokens (by routing score)
    and routes them through a full pre-norm attention + FFN block.  All other
    tokens pass through unchanged via the residual connection.

    The scatter-back step is fully differentiable for the selected tokens and
    simply copies the input for unselected tokens, so gradients propagate
    correctly to all earlier layers.

    Args:
        cfg       : ``MoDConfig`` controlling routing and architecture
        layer_idx : zero-based layer index (used by ``MoDAnalytics``)
    """

    def __init__(self, cfg: MoDConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.capacity_factor = cfg.capacity_factor
        self.layer_idx = layer_idx
        self.dim = cfg.dim

        self.router = TokenRouter(cfg.dim)
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MoDGQAttention(cfg)
        self.ffn = MoDSwiGLU(cfg.dim, cfg.ffn_hidden_mult)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis_table: torch.Tensor,
        analytics: Optional[MoDAnalytics] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route top-k tokens through the transformer block; rest skip via residual.

        Args:
            x               : ``(B, T, dim)`` input hidden states
            freqs_cis_table : ``(max_seq_len, head_dim//2)`` complex — full
                              RoPE lookup table; rows are gathered per batch item
            analytics       : optional analytics recorder

        Returns:
            output        : ``(B, T, dim)`` — same shape as input
            router_scores : ``(B, T)`` routing logits (for auxiliary loss)
        """
        B, T, D = x.shape
        capacity = max(1, int(T * self.capacity_factor))

        # ── 1. Compute routing scores ─────────────────────────────────────
        scores = self.router(x)  # (B, T)

        # ── 2. Select top-k token indices (position-sorted) ──────────────
        sel_idx = self.router.select_top_k(scores, capacity)  # (B, capacity)
        actual_capacity = sel_idx.shape[1]

        # ── 3. Gather selected tokens ─────────────────────────────────────
        idx_exp = sel_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, cap, D)
        x_sel = x.gather(1, idx_exp)  # (B, cap, D)

        # ── 4. Gather per-batch-item RoPE frequencies ─────────────────────
        # freqs_cis_table: (max_seq_len, head_dim//2)
        # sel_idx:         (B, capacity)  — position indices to look up
        # result:          (B, capacity, head_dim//2)
        freqs_sel = freqs_cis_table[sel_idx]

        # ── 5. Build causal mask for the selected token subset ────────────
        # Since sel_idx is sorted by position, the relative ordering is
        # preserved → standard upper-triangular causal mask is valid.
        mask = torch.triu(
            torch.full(
                (1, 1, actual_capacity, actual_capacity),
                float("-inf"),
                device=x.device,
                dtype=x.dtype,
            ),
            diagonal=1,
        )

        # ── 6. Apply pre-norm transformer to selected tokens ──────────────
        h = x_sel + self.resid_drop(
            self.attn(self.attn_norm(x_sel), freqs_sel, mask)
        )
        h = h + self.resid_drop(self.ffn(self.ffn_norm(h)))

        # ── 7. Scatter transformed tokens back; non-selected are unchanged ─
        out = x.clone()
        out.scatter_(1, idx_exp, h)

        # ── 8. Record analytics ───────────────────────────────────────────
        if analytics is not None:
            analytics.record(self.layer_idx, actual_capacity, T, scores=scores)

        return out, scores


# ---------------------------------------------------------------------------
# Full MoD Transformer
# ---------------------------------------------------------------------------


class MoDTransformer(nn.Module):
    """
    Full Mixture-of-Depths Transformer language model.

    Stacks ``cfg.n_layers`` ``MixtureOfDepthsBlock`` layers between an input
    embedding and an output language-model head.  Each layer independently
    routes tokens using its own ``TokenRouter``, so routing patterns can
    diverge across layers as the model learns which tokens need refinement at
    each depth.

    Forward pass returns both logits and an optional auxiliary load-balancing
    loss that prevents routing collapse (all tokens always routed or never
    routed).

    Example::

        cfg = MoDConfig(vocab_size=1000, dim=128, n_heads=4, n_kv_heads=1,
                        max_seq_len=64, n_layers=2, capacity_factor=0.5)
        model = MoDTransformer(cfg)
        ids = torch.randint(0, 1000, (2, 32))
        logits, aux_loss = model(ids)
        # logits: (2, 32, 1000)
        # aux_loss: scalar tensor (None if return_aux_loss=False)
    """

    def __init__(self, cfg: MoDConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Precompute RoPE frequency table shared across all layers
        head_dim = cfg.dim // cfg.n_heads
        freqs = precompute_mod_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis_table", freqs)  # (max_seq_len, head_dim//2)

        self.layers = nn.ModuleList(
            [MixtureOfDepthsBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all linear and embedding weights with N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def compute_aux_loss(
        self, all_router_scores: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Load-balancing auxiliary loss to prevent routing collapse.

        Computes the squared difference between the mean soft routing
        probability for each layer and the target ``capacity_factor``.
        Summing over layers creates a global pressure towards even token
        distribution.

        Args:
            all_router_scores : list of ``(B, T)`` routing logits, one per layer

        Returns:
            Scalar loss tensor (multiplied by ``cfg.router_aux_loss_coef``)
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        target = self.cfg.capacity_factor
        for scores in all_router_scores:
            probs = torch.sigmoid(scores)          # soft routing probability (B, T)
            mean_prob = probs.mean()               # scalar
            total = total + (mean_prob - target).pow(2)
        return total * self.cfg.router_aux_loss_coef

    def forward(
        self,
        input_ids: torch.Tensor,
        return_aux_loss: bool = True,
        analytics: Optional[MoDAnalytics] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Embedding → n MoD layers → Norm → LM Head.

        Args:
            input_ids      : ``(B, T)`` token indices
            return_aux_loss: if True, compute and return the routing aux loss
            analytics      : optional analytics recorder passed to each layer

        Returns:
            logits   : ``(B, T, vocab_size)``
            aux_loss : scalar tensor if ``return_aux_loss`` else None
        """
        x = self.embed(input_ids)  # (B, T, dim)

        all_router_scores: List[torch.Tensor] = []
        for layer in self.layers:
            x, r_scores = layer(x, self.freqs_cis_table, analytics)
            all_router_scores.append(r_scores)

        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)

        aux_loss = self.compute_aux_loss(all_router_scores) if return_aux_loss else None
        return logits, aux_loss

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the training loss: cross-entropy + optional routing aux loss.

        This is a convenience method that combines the language-modelling
        objective with the load-balancing penalty returned by :meth:`forward`.

        Args:
            logits   : ``(B, T, vocab_size)`` raw logits from :meth:`forward`
            targets  : ``(B, T)`` target token IDs; positions with value
                       ``-100`` are masked out of the loss (standard PyTorch
                       convention for padding / teacher-forcing ignoring)
            aux_loss : scalar routing aux loss from :meth:`forward` (or
                       ``None`` to compute CE only)

        Returns:
            Scalar total loss = CE_loss + (aux_loss if provided else 0)

        Example::

            model = MoDTransformer(cfg)
            ids = torch.randint(0, cfg.vocab_size, (B, T))
            logits, aux = model(ids)
            # Shift for next-token prediction
            loss = model.compute_loss(logits[:, :-1], ids[:, 1:], aux)
            loss.backward()
        """
        B, T, V = logits.shape
        ce = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=-100,
        )
        if aux_loss is not None:
            return ce + aux_loss
        return ce

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Greedy / top-k autoregressive generation.

        Each new token is produced by a full forward pass over the growing
        sequence (no KV cache for simplicity; the MoD routing mask changes
        with each new token).

        Args:
            input_ids      : ``(B, T_prompt)`` prompt token indices
            max_new_tokens : number of tokens to generate
            temperature    : logit scaling (1.0 = unchanged, <1.0 = sharper)
            top_k          : if set, restrict sampling to the top-k logits

        Returns:
            ``(B, T_prompt + max_new_tokens)`` token indices
        """
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            # Clip to max_seq_len (no KV cache here; recompute full context)
            ids_ctx = ids[:, -self.cfg.max_seq_len :]
            logits, _ = self.forward(ids_ctx, return_aux_loss=False)
            next_logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-8)
            if top_k is not None:
                topk_vals, _ = next_logits.topk(top_k, dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

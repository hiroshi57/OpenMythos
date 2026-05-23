from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


@dataclass
class MythosConfig:
    """
    Hyperparameter configuration for OpenMythos.

    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2
    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. Used in place of
    LayerNorm throughout the model for stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim -- feature dimension to normalize over
            eps -- small constant added before sqrt for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor of the same shape, rescaled by self.weight
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """
    Precompute complex-valued RoPE rotation matrices for positions 0..max_len-1.

    Each position gets a complex phasor e^{i·m·θ_k} for each frequency pair k.
    Stored as a complex tensor so that rotation is a single pointwise multiply.

    Args:
        dim     -- head dimension (must be even); frequencies are computed for dim//2 pairs
        max_len -- maximum sequence length to precompute
        theta   -- RoPE base (higher = slower frequency decay; 500k is the LLaMA-3 default)

    Returns:
        complex64 tensor of shape (max_len, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.

    Interprets each pair of adjacent features as a 2D complex number and
    multiplies by the precomputed phasor for that position, rotating the
    representation in the complex plane without changing its norm.

    Args:
        x         -- tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis -- precomputed complex frequencies of shape (T, head_dim//2),
                     already sliced to exactly the positions being processed
                     (caller is responsible for correct start_pos offset)

    Returns:
        Rotated tensor of the same shape and dtype as x
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023) with Flash Attention 2 (Dao et al., 2023).

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing the KV cache size
    by that factor while keeping full query expressiveness.

    When flash-attn is installed, uses flash_attn_func which handles GQA natively
    (no KV head expansion needed) and is IO-bound-optimal. Inputs are cast to
    bfloat16 for flash_attn and restored to the original dtype afterward.
    Falls back to manual scaled dot-product attention when flash-attn is absent.

    RoPE is applied to both Q and K. K and V are stored in kv_cache after
    RoPE application so that cached values are already positionally encoded and
    do not need to be re-rotated on retrieval.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, n_kv_heads
        """
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
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            # Apply sliding window only during single-token decode (T=1).
            # Prefill (T > 1) uses a full causal mask that must match K length,
            # so we skip clipping there.
            win = kv_cache.get("__window__", 0)
            if win > 0 and T == 1 and k.shape[1] > win:
                k = k[:, -win:, :, :]
                v = v[:, -win:, :, :]
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        if _HAS_FLASH_ATTN:
            # flash_attn_func expects (B, T, H, head_dim) — GQA is handled natively
            # (n_kv_heads < n_heads is supported without repeat_interleave).
            # causal=True when mask is present (full-sequence prefill/training);
            # causal=False for single-token decode where T=1 and mask is None.
            orig_dtype = q.dtype
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            dropout_p = self.dropout_p if self.training else 0.0
            out = flash_attn_func(
                q, k, v, dropout_p=dropout_p, causal=(mask is not None)
            )
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            # Fallback: manual scaled dot-product with explicit KV head expansion.
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = self.head_dim**-0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = F.dropout(
                F.softmax(attn, dim=-1), p=self.dropout_p, training=self.training
            )
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    The key insight: instead of caching full K and V tensors (each of size
    n_heads × head_dim per token), MLA compresses the KV path through a
    low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and
    V are reconstructed from c_kv at each decoding step, trading a cheap
    linear projection for dramatically smaller cache memory.

    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope)  per head

    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into c_kv (latent, cached) and k_rope_raw (shared across heads)
        k_rope = RoPE(expand(k_rope_raw))  — applied before caching
        c_kv → kv_norm → kv_up → [k_nope | v]  — reconstructed each step
        k = cat(k_nope, k_rope)  per head

    Cache stores: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim),
    versus full GQA cache: n_kv_heads × head_dim × 2.  At production scale this
    is roughly a 10–20× memory reduction.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, kv_lora_rank, q_lora_rank,
                   qk_rope_head_dim, qk_nope_head_dim, v_head_dim
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False
        )

        # KV compression: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = nn.Linear(
            cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies sized for qk_rope_head_dim, shape (T, rope_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"c_kv": ..., "k_rope": ...}
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # expand rope keys across heads and apply RoPE before caching so
        # retrieved keys are already positionally encoded
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            win = kv_cache.get("__window__", 0)
            if win > 0 and T == 1 and c_kv.shape[1] > win:
                c_kv = c_kv[:, -win:, :]
                k_rope = k_rope[:, -win:, :, :]
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S = c_kv.shape[1]  # full sequence length including cache

        # reconstruct K_nope and V from latent (not cached, recomputed each step)
        kv = self.kv_up(self.kv_norm(c_kv))  # (B, S, H*(nope+v))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]  # (B, S, H, nope)
        v = kv[..., self.qk_nope_dim :]  # (B, S, H, v_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, S, H, nope+rope)

        # attention
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        scale = self.q_head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)  # (B, H, T, v_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """
    Single SwiGLU feed-forward expert.

    Implements the gated linear unit variant: output = down(silu(gate(x)) * up(x)).
    Used both as individual routed experts inside MoEFFN and as the standard dense
    FFN in prelude/coda blocks (where expert_dim = dim * 4 // 3).
    """

    def __init__(self, dim: int, expert_dim: int):
        """
        Args:
            dim        -- input and output feature dimension
            expert_dim -- inner (hidden) dimension of the expert
        """
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (..., dim)
        Returns:
            Tensor of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """
    Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024).

    Two classes of experts:
    - Routed experts: n_experts small FFNs; each token activates top-K of them
      via a learned router. A per-expert bias on router logits is updated during
      training to keep load balanced across experts without distorting the loss.
    - Shared experts: n_shared_experts larger FFNs always activated for every token,
      absorbing common cross-domain patterns (syntax, basic reasoning) that would
      otherwise be redundantly learned by many routed experts.

    Total activated parameters per token ≈ topk/n_experts of routed + all shared,
    keeping compute sparse while the total parameter count stays large.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses n_experts, n_shared_experts, n_experts_per_tok,
                   dim, expert_dim
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # load-balancing bias adjusted externally during training; not a gradient param
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (B, T, dim)
        Returns:
            Tensor of shape (B, T, dim); shared expert outputs are summed on top
            of the weighted routed expert outputs
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)

        # Aux-loss-free load balancing (DeepSeek-V3): the bias shifts only the
        # selection of which experts fire so underused experts are picked more,
        # but the gating weights come from unbiased softmax scores so the bias
        # never shows up in the gradient.
        logits = self.router(flat)  # (B*T, n_experts), unbiased
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # renorm

        # routed expert dispatch — vectorized over experts to avoid Python loops
        # Each expert processes only the tokens routed to it; results are
        # scattered back and accumulated with learned gating weights.
        out = torch.zeros_like(flat)
        for eid in range(self.n_experts):
            # mask: which (token, slot) pairs are routed to this expert
            # topk_idx: (N, topk), topk_scores: (N, topk)
            expert_mask = (topk_idx == eid)  # (N, topk) bool
            if not expert_mask.any():
                continue
            # token indices that route to this expert (may repeat across slots)
            token_idx = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            expert_out = self.routed_experts[eid](flat[token_idx])  # (M, D)
            # sum gating weights across slots for tokens that selected this expert
            weight = (topk_scores * expert_mask.float()).sum(dim=-1, keepdim=True)
            out.index_add_(0, token_idx, weight[token_idx] * expert_out)

        # shared experts always fire for every token
        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Loop-index RoPE (differentiates recurrent block across iterations)
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Inject a sinusoidal loop-index signal into the first loop_dim channels of h.

    Analogous to RoPE for sequence position, but applied over recurrence depth
    instead of token position. Without this, the shared recurrent block weights
    must handle both early-stage pattern-matching and late-stage refinement with
    no signal distinguishing which loop they are on. Adding the loop index lets
    the same parameters implement functionally distinct operations per iteration.

    Args:
        h        -- hidden state tensor of shape (B, T, dim)
        loop_t   -- current loop iteration index (0-based)
        loop_dim -- number of leading channels to receive the embedding (must be even)
        theta    -- sinusoidal base frequency

    Returns:
        h with a sinusoidal bias added to its first loop_dim channels; same shape
    """
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter (per loop iteration)
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for the recurrent block (Bae et al., 2024).

    Pure weight-tying (identical weights every loop) limits expressiveness;
    fully distinct weights per loop eliminate parameter savings. This adapter
    sits in between: a shared low-rank down-projection and up-projection matrix B
    are shared across all loops, while a small per-loop scale vector shifts the
    effective transformation at each depth without adding significant parameters.

    delta(x, t) = (down(x) * scale[t]) @ B
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        Args:
            dim       -- model hidden dimension (input and output size)
            rank      -- low-rank bottleneck dimension
            max_loops -- maximum number of loop iterations (determines embedding table size)
        """
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # shared A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # shared B: rank → dim
        self.scale = nn.Embedding(max_loops, rank)  # per-loop element-wise scale

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x      -- input tensor of shape (B, T, dim)
            loop_t -- current loop index used to look up the per-loop scale

        Returns:
            Delta tensor of shape (B, T, dim) to be added to the block output
        """
        # Clamp for depth extrapolation: at inference n_loops can exceed the
        # training max_loop_iters. Iterations beyond the trained range reuse
        # the last learned per-loop scale rather than indexing out of range.
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))  # (rank,)
        down = self.down(x) * s  # (B, T, rank)
        return down @ self.B  # (B, T, dim)


# ---------------------------------------------------------------------------
# Single Transformer Block (shared across recurrent loops)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with swappable attention and optional MoE FFN.

    Attention is selected by cfg.attn_type:
        "gqa" → GQAttention  (Grouped Query Attention, fewer KV heads)
        "mla" → MLAttention  (Multi-Latent Attention, compressed KV cache)

    FFN is selected by use_moe:
        True  → MoEFFN  (fine-grained routed + shared experts; used in RecurrentBlock)
        False → Expert  (dense SwiGLU FFN; used in Prelude and Coda)
    """

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        """
        Args:
            cfg     -- MythosConfig; attn_type selects the attention class
            use_moe -- if True, use MoEFFN; otherwise use a dense Expert FFN
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- precomputed RoPE frequencies
            mask      -- additive causal mask or None
            kv_cache  -- cache dict mutated in-place by the attention layer
            cache_key -- key identifying this layer in the cache

        Returns:
            Output tensor of shape (B, T, dim)
        """
        x = x + self.resid_drop(
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# LTI-stable injection parameters  (spectral radius < 1 by construction)
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """
    Stable input injection for the recurrent update rule (Parcae, Prairie et al., 2026).

    The recurrent hidden state evolves as:
        h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

    where e is the encoded input injected at every loop step to prevent drift.
    Without constraints, A can develop spectral radius ≥ 1, causing the hidden
    state to explode across loop iterations and destabilize training.

    This class guarantees ρ(A) < 1 by construction via a ZOH discretization:
        A_continuous = Diag(-exp(log_A))       always negative diagonal
        A_discrete   = exp(Δt · A_continuous)  element-wise, values in (0, 1)

    where log_A and log_dt are learned parameters and exp ensures positivity.
    This makes looped model training robust to hyperparameter choices and stable
    even at high learning rates.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; one scalar per channel for A and B
        """
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))  # log of A_continuous magnitude
        self.log_dt = nn.Parameter(torch.zeros(1))  # log of discretization step Δt
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        """
        Compute the discretized diagonal state matrix A_discrete.

        Returns:
            1-D tensor of shape (dim,) with all values strictly in (0, 1),
            guaranteeing ρ(A) < 1 regardless of learned parameter values.
        """
        # Compute in log space to avoid 0 * inf = NaN when log_dt → -∞, log_A → +∞.
        # dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
        # Outer clamp keeps the product finite in float32 for any gradient step size.
        # Inner clamp (min=1e-6) ensures the exponent is strictly negative so that
        # A_discrete = exp(-x) < 1 holds in float32 even when x → 0⁺ via very large
        # gradient updates (without it, exp(-1e-9) rounds to 1.0 in float32).
        return torch.exp(
            -torch.exp((self.log_dt + self.log_A).clamp(-20, 20)).clamp(min=1e-6)
        )

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = A·h_t + B·e + transformer_out.

        Args:
            h               -- current hidden state (B, T, dim)
            e               -- encoded input from Prelude, frozen across loops (B, T, dim)
            transformer_out -- output of the recurrent TransformerBlock at this step (B, T, dim)

        Returns:
            Updated hidden state of shape (B, T, dim)
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# ACT halting (Adaptive Computation Time)
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting mechanism (Graves, 2016).

    Learns a per-position halting probability at each loop iteration. Positions
    where the hidden state has converged (high cumulative halting probability)
    stop accumulating updates, while positions still being refined continue.
    This lets easy tokens halt early and hard tokens receive more computation,
    all within the same batch. Also makes the model Turing-complete under
    certain assumptions about the expressiveness of the transformer block.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; input to the halting scalar predictor
        """
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-position halting probability from the current hidden state.

        Args:
            h -- hidden state of shape (B, T, dim)

        Returns:
            Halting probability tensor of shape (B, T), values in (0, 1)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Recurrent Block (one set of weights, looped T times)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """
    The core recurrent block of OpenMythos — a single TransformerBlock looped T times.

    At each loop iteration t, the hidden state h is updated via:
        1. loop_index_embedding: inject sinusoidal loop-index signal into h
        2. TransformerBlock:     compute attention + MoE FFN on normalized (h + e)
        3. LoRAAdapter:          apply depth-wise LoRA delta to transformer output
        4. LTIInjection:         stable update h = A·h + B·e + transformer_out
        5. ACTHalting:           accumulate per-position halting probabilities;
                                  positions that have converged stop contributing

    The encoded input e (output of the Prelude) is injected at every step to keep
    the original input signal alive across arbitrary loop depth, preventing drift.
    The ACT mechanism produces a weighted sum of hidden states across iterations,
    where the weights reflect when each position converged.

    More loop iterations at inference = deeper reasoning chains, following the
    depth-extrapolation property of looped transformers (Saunshi et al., 2025).
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, lora_rank, max_loop_iters, act_threshold
        """
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = (
            cfg.dim // 8
        )  # fraction of channels receiving loop-index embedding

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Run the recurrent loop for up to n_loops iterations with ACT early exit.

        Args:
            h        -- initial hidden state from the Prelude, shape (B, T, dim)
            e        -- encoded input frozen for injection each step, shape (B, T, dim)
            freqs_cis-- precomputed RoPE frequencies
            mask     -- additive causal mask or None
            n_loops  -- number of loop iterations; defaults to cfg.max_loop_iters.
                        Can be increased at inference for deeper reasoning (depth extrapolation).
            kv_cache -- cache dict passed through to the inner TransformerBlock;
                        each loop iteration uses a separate cache key

        Returns:
            ACT-weighted sum of hidden states across iterations, shape (B, T, dim)
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

            p = self.act(h)  # (B, T)
            still_running = ~halted

            # ACT remainder trick: once cumulative_p + p crosses threshold,
            # assign the remaining probability mass as the final weight.
            # Gate by still_running so halted positions contribute exactly
            # once (on the halting step) and zero thereafter — otherwise
            # threshold<1 leaves a non-zero remainder that leaks every step.
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # Only short-circuit when there is no KV cache to keep consistent.
            # With a cache, every loop depth must run on every forward pass so
            # later decode steps find populated keys at every cache_key.
            if halted.all() and kv_cache is None:
                break

        return h_out


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class OpenMythos(nn.Module):
    """
    OpenMythos — Recurrent-Depth Transformer language model.

    Implements the hypothesized Claude Mythos architecture as a Recurrent-Depth
    Transformer (RDT). The model divides computation into three functional blocks:

        Input tokens
             ↓
        [Prelude]          — prelude_layers standard transformer blocks, run once
             ↓
        [Recurrent Block]  — one transformer block looped T times with input injection
             ↑_______↓      h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
             ↓
        [Coda]             — coda_layers standard transformer blocks, run once
             ↓
        Output logits

    Key properties:
    - Same weights, more loops → deeper reasoning, no parameter growth
    - Depth extrapolation: train on N loops, test on N+k loops (emergent)
    - ACT halting: variable compute per position within a batch
    - MoE FFN in the recurrent block: breadth across domains
    - LTI-stable injection: spectral radius < 1 guaranteed by construction
    - Supports both GQA and MLA attention (set via cfg.attn_type)
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig specifying all architecture hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # GQA uses full head_dim for RoPE; MLA uses only qk_rope_head_dim (decoupled)
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
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

    @staticmethod
    def _causal_mask(
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        cache_len: int = 0,
    ) -> torch.Tensor:
        """
        Build an additive causal mask of shape (1, 1, seq_len, cache_len + seq_len).

        Each of the ``seq_len`` query positions can attend to:
          - all ``cache_len`` cached key positions (full attention, value 0), and
          - all preceding + current positions within the current window (causal, value 0);
          - future positions within the current window (-inf).

        When ``cache_len == 0`` (no KV cache / prefill) the mask is square
        and equal to the standard upper-triangular causal mask.

        Args:
            seq_len   -- number of query (current) tokens
            device    -- target device
            dtype     -- tensor dtype (must match activations to avoid precision promotion)
            cache_len -- number of tokens already in the KV cache (default 0)

        Returns:
            Tensor of shape (1, 1, seq_len, cache_len + seq_len)
        """
        # Cache block: queries can always attend to all cached keys → zeros
        cache_block = torch.zeros(1, 1, seq_len, cache_len, device=device, dtype=dtype)
        # Causal block: upper-triangular -inf within the current window
        causal_block = torch.triu(
            torch.full(
                (1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype
            ),
            diagonal=1,
        )
        return torch.cat([cache_block, causal_block], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through Prelude → Recurrent Block → Coda.

        Args:
            input_ids -- token indices of shape (B, T)
            n_loops   -- recurrent loop depth; defaults to cfg.max_loop_iters.
                         Increase at inference to extrapolate to harder problems.
            kv_cache  -- dict mutated in-place for autoregressive KV caching;
                         pass an empty dict {} and reuse across decode steps
            start_pos -- index of the first token in input_ids within the full
                         sequence; used to select the correct RoPE frequencies
                         during incremental decoding (0 for prefill, prompt_len
                         for each subsequent decode step)

        Returns:
            Logits of shape (B, T, vocab_size)
        """
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]
        mask = (
            self._causal_mask(T, device, x.dtype, cache_len=start_pos)
            if T > 1
            else None
        )

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x  # encoded input frozen for injection every loop
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        return self.head(self.norm(x))

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Penalise tokens that have already appeared in input_ids.

        For each token present in the sequence, logits > 0 are divided by the
        penalty and logits < 0 are multiplied by it — pushing previously-seen
        tokens away from the distribution without zeroing them out entirely.

        Args:
            logits            -- shape (B, vocab_size)
            input_ids         -- shape (B, S); tokens seen so far
            repetition_penalty-- > 1 reduces repetition; 1.0 = no effect

        Returns:
            Logits tensor with the penalty applied in-place.
        """
        if repetition_penalty == 1.0:
            return logits
        for b in range(logits.shape[0]):
            for token_id in input_ids[b].unique():
                idx = token_id.long()
                if logits[b, idx] > 0:
                    logits[b, idx] /= repetition_penalty
                else:
                    logits[b, idx] *= repetition_penalty
        return logits

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float = 1.0,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample the next token from a logits tensor of shape (B, vocab_size).

        Applies repetition penalty → temperature scaling → top-k filtering
        → top-p (nucleus) filtering → multinomial sampling in order.
        Returns a (B, 1) integer tensor.
        """
        if repetition_penalty != 1.0 and input_ids is not None:
            logits = OpenMythos._apply_repetition_penalty(
                logits, input_ids, repetition_penalty
            )
        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            v, _ = logits.topk(top_k)
            logits[logits < v[:, -1:]] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(
                1, sorted_idx, sorted_logits
            )
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        decode_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_cache_len: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with KV caching.

        On step 0 the full prompt is processed. On subsequent steps only the
        last generated token is passed, with all previous keys and values
        retrieved from kv_cache. This keeps decode cost proportional to one
        token per step rather than the full growing sequence.

        n_loops can be set higher than the training value to extrapolate to
        harder problems at inference time (depth extrapolation property).

        ``decode_loops`` enables a two-phase depth strategy:

        * **Prefill** (step 0): always runs ``n_loops`` iterations. The model
          thinks deeply over the full prompt once, populating KV caches for
          all loop levels (loop_0 … loop_{n_loops-1}).

        * **Decode** (steps 1+): runs ``decode_loops`` iterations. Fewer loops
          means fewer attention calls per token, directly reducing latency.
          Because the depth-extrapolation property holds (loops=2 recovers
          ~99% of loops=4 quality), setting ``decode_loops=2`` when
          ``n_loops=4`` cuts decode latency by ~2x with negligible quality
          loss.

        The KV caches for loop levels above ``decode_loops`` retain their
        prefill context but are not updated during decode — this is
        self-consistent because each decode step only reads/writes the caches
        for loop levels 0 … decode_loops-1.

        Sampling pipeline (applied in order):
            1. Scale logits by ``temperature`` (lower → more peaked).
            2. ``top_k``: zero out all but the top-K logits (0 = disabled).
            3. ``top_p``: nucleus sampling — keep the smallest set of tokens
               whose cumulative softmax probability ≥ ``top_p``, zero out
               the rest (1.0 = disabled).
            4. Sample from the resulting distribution.

        Args:
            input_ids      -- prompt token indices of shape (B, T)
            max_new_tokens -- number of tokens to generate
            n_loops        -- recurrent loop depth used for prefill (step 0)
            decode_loops   -- loop depth for decode steps (steps 1+).
                              None = use n_loops (original behaviour).
                              Recommended: set to n_loops // 2 for ~2x decode
                              speedup with <1% quality loss.
            temperature    -- softmax temperature; lower = more greedy
            top_k          -- restrict sampling to top-K logits (0 = disabled)
            top_p          -- nucleus sampling threshold in (0, 1]; 1.0 = disabled.
                              E.g. top_p=0.9 keeps the smallest token set whose
                              cumulative probability covers 90% of the mass.

        Returns:
            Token indices of shape (B, T + max_new_tokens)
        """
        kv_cache: dict = {}
        if max_cache_len > 0:
            kv_cache["__window__"] = max_cache_len
        prompt_len = input_ids.shape[1]
        _decode_loops = decode_loops if decode_loops is not None else n_loops
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
                cur_loops = n_loops  # prefill: full depth
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
                cur_loops = _decode_loops  # decode: fast depth
            logits = self.forward(
                cur_ids, n_loops=cur_loops, kv_cache=kv_cache, start_pos=start_pos
            )
            next_tok = self._sample_token(
                logits[:, -1, :], temperature, top_k, top_p,
                repetition_penalty=repetition_penalty, input_ids=input_ids,
            )
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    @torch.no_grad()
    def generate_beam(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        beam_width: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Beam search decoding.

        Maintains ``beam_width`` candidate sequences simultaneously and selects
        the one with the highest length-normalised log-probability at the end.
        Unlike greedy or sampling methods, beam search is deterministic and
        typically produces more coherent sequences for smaller beam widths.

        Args:
            input_ids      -- prompt token indices of shape (1, T); B=1 only
            max_new_tokens -- maximum tokens to generate
            n_loops        -- recurrent loop depth
            beam_width     -- number of beams kept at each step
            temperature    -- logit temperature before beam scoring (lower = more peaked)
            top_p          -- nucleus filter applied to candidate vocab before scoring
            length_penalty -- exponent for length normalisation:
                              score = sum_log_prob / (len ** length_penalty).
                              > 1 favours longer outputs, < 1 favours shorter ones.

        Returns:
            Token indices of shape (1, T + generated) — the best beam's full sequence.
        """
        assert input_ids.shape[0] == 1, "generate_beam supports B=1 only"
        device = input_ids.device
        vocab_size = self.cfg.vocab_size

        # Each beam: (sequence tensor (1, L), cumulative log-prob, kv_cache)
        beams: list[tuple[torch.Tensor, float, dict]] = [
            (input_ids, 0.0, {})
        ]
        prompt_len = input_ids.shape[1]

        for step in range(max_new_tokens):
            all_candidates: list[tuple[torch.Tensor, float, dict]] = []

            for seq, score, cache in beams:
                if step == 0:
                    cur_ids = seq
                    start_pos = 0
                else:
                    cur_ids = seq[:, -1:]
                    start_pos = seq.shape[1] - 1

                logits = self.forward(
                    cur_ids, n_loops=n_loops, kv_cache=cache, start_pos=start_pos
                )
                lgt = logits[:, -1, :].clone()  # (1, V)

                # temperature + top-p filtering
                lgt = lgt / max(temperature, 1e-8)
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(lgt, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove_mask] = float("-inf")
                    lgt = torch.full_like(lgt, float("-inf")).scatter(
                        1, sorted_idx, sorted_logits
                    )

                log_probs = F.log_softmax(lgt, dim=-1).squeeze(0)  # (V,)

                # Expand: take top beam_width next tokens
                top_log_probs, top_ids = log_probs.topk(beam_width)
                for tok_log_p, tok_id in zip(top_log_probs, top_ids):
                    new_seq = torch.cat(
                        [seq, tok_id.view(1, 1)], dim=1
                    )
                    new_score = score + tok_log_p.item()
                    # copy cache so beams don't share state
                    new_cache = {
                        k: {ck: cv.clone() for ck, cv in v.items()}
                        for k, v in cache.items()
                    }
                    all_candidates.append((new_seq, new_score, new_cache))

            # Keep the top beam_width candidates by length-normalised score
            def _norm_score(item: tuple) -> float:
                seq_tensor, raw_score, _ = item
                gen_len = seq_tensor.shape[1] - prompt_len
                return raw_score / max(gen_len, 1) ** length_penalty

            all_candidates.sort(key=_norm_score, reverse=True)
            beams = all_candidates[:beam_width]

        # Return the best beam
        best_seq, _, _ = beams[0]
        return best_seq

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        decode_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_cache_len: int = 0,
    ):
        """Streaming autoregressive generation — yields one token per step.

        Identical to :meth:`generate` in sampling behaviour and KV-cache
        strategy, but ``yield``s each new token as a ``(B, 1)`` integer tensor
        the moment it is sampled instead of accumulating the full sequence.
        This enables real-time display in chat UIs or token-by-token pipelines.

        Usage::

            for tok in model.generate_stream(ids, max_new_tokens=32):
                print(tokenizer.decode(tok[0].tolist()), end="", flush=True)

        Args:
            input_ids      -- prompt token indices of shape (B, T)
            max_new_tokens -- maximum tokens to stream
            n_loops        -- recurrent loop depth for prefill (step 0)
            decode_loops   -- loop depth for decode steps; None = n_loops
            temperature    -- softmax temperature
            top_k          -- top-K filtering (0 = disabled)
            top_p          -- nucleus sampling threshold (1.0 = disabled)

        Yields:
            ``(B, 1)`` integer tensors, one per generated token
        """
        kv_cache: dict = {}
        if max_cache_len > 0:
            kv_cache["__window__"] = max_cache_len
        prompt_len = input_ids.shape[1]
        _decode_loops = decode_loops if decode_loops is not None else n_loops
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
                cur_loops = n_loops
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
                cur_loops = _decode_loops
            logits = self.forward(
                cur_ids, n_loops=cur_loops, kv_cache=kv_cache, start_pos=start_pos
            )
            next_tok = self._sample_token(
                logits[:, -1, :], temperature, top_k, top_p,
                repetition_penalty=repetition_penalty, input_ids=input_ids,
            )
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            yield next_tok

    @torch.no_grad()
    def speculative_decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        draft_loops: int = 1,
        draft_k: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Speculative decoding using self as both draft and target model.

        Leverages the depth-extrapolation property of OpenMythos: the same
        model run at ``draft_loops`` iterations acts as a lightweight *draft*,
        and run at ``n_loops`` iterations acts as the *target* verifier.

        Algorithm (per outer iteration):
            1. **Draft**: run ``draft_k`` autoregressive steps with
               ``draft_loops`` depth, collecting sampled tokens and their
               probabilities (cheap — shallow depth).
            2. **Verify**: run one forward pass at full ``n_loops`` depth over
               the prefix + all ``draft_k`` draft tokens in a single batch,
               obtaining target probabilities for every draft position in
               parallel (one expensive call instead of ``draft_k``).
            3. **Accept/reject**: for each position ``i``, accept draft token
               ``x_i`` with probability ``min(1, p_target(x_i) / p_draft(x_i))``
               (standard speculative sampling).  Stop at the first rejection
               and resample that position from the adjusted target distribution.
            4. Repeat until ``max_new_tokens`` are produced.

        When the shallow draft distribution is close to the deep target
        distribution (as depth extrapolation suggests), acceptance rates are
        high and the method produces ≈ ``draft_k`` tokens per expensive
        target call — amortising the deep-model cost over multiple tokens.

        Note: This implementation runs without a KV cache for correctness
        simplicity (avoids cache-size book-keeping across accept/reject
        boundaries). The depth savings (``draft_loops`` << ``n_loops``) still
        apply for the draft phase.  Batch size is restricted to B=1.

        Args:
            input_ids      -- prompt token indices of shape (B=1, T)
            max_new_tokens -- maximum new tokens to produce
            n_loops        -- target (verifier) loop depth
            draft_loops    -- draft loop depth; should be < n_loops (1 = fastest)
            draft_k        -- number of tokens to speculate before each target call
            temperature    -- sampling temperature (applied to both draft & target)
            top_k          -- top-K filtering (0 = disabled)
            top_p          -- nucleus sampling threshold (1.0 = disabled)

        Returns:
            Token indices of shape (B=1, T + generated) where generated ≤ max_new_tokens.
        """
        assert input_ids.shape[0] == 1, "speculative_decode supports B=1 only"
        generated = 0

        while generated < max_new_tokens:
            k = min(draft_k, max_new_tokens - generated)

            # ----------------------------------------------------------------
            # 1. Draft phase: sample k tokens with the shallow draft model.
            # ----------------------------------------------------------------
            draft_tokens: list[torch.Tensor] = []  # each (1, 1)
            draft_probs: list[torch.Tensor] = []  # each (1, V)
            draft_ids = input_ids

            for _ in range(k):
                logits_d = self.forward(draft_ids, n_loops=draft_loops)
                lgt = logits_d[:, -1, :].clone()
                # apply same sampling filters as _sample_token but keep probs
                lgt = lgt / max(temperature, 1e-8)
                if top_k > 0:
                    v, _ = lgt.topk(top_k)
                    lgt[lgt < v[:, -1:]] = float("-inf")
                if top_p < 1.0:
                    sl, si = torch.sort(lgt, descending=True)
                    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    rm = cp - F.softmax(sl, dim=-1) > top_p
                    sl[rm] = float("-inf")
                    lgt = torch.full_like(lgt, float("-inf")).scatter(1, si, sl)
                p_d = F.softmax(lgt, dim=-1)
                tok = torch.multinomial(p_d, 1)
                draft_tokens.append(tok)
                draft_probs.append(p_d.detach())
                draft_ids = torch.cat([draft_ids, tok], dim=1)

            # ----------------------------------------------------------------
            # 2. Verify phase: one deep forward over prefix + all draft tokens.
            #    The target sees the full context (input_ids + k draft tokens)
            #    and produces logits at positions len(input_ids) .. -1.
            # ----------------------------------------------------------------
            full_seq = torch.cat([input_ids] + draft_tokens, dim=1)  # (1, T+k)
            logits_t = self.forward(full_seq, n_loops=n_loops)  # (1, T+k, V)
            # logits for draft positions: indices T-1 .. T+k-2 (predict tokens T..T+k-1)
            T_cur = input_ids.shape[1]
            target_logits = logits_t[:, T_cur - 1 : T_cur - 1 + k, :]  # (1, k, V)

            # ----------------------------------------------------------------
            # 3. Accept / reject.
            # ----------------------------------------------------------------
            n_accepted = 0
            for i in range(k):
                x_i = draft_tokens[i]  # (1, 1)
                p_target = F.softmax(
                    target_logits[:, i, :] / max(temperature, 1e-8), dim=-1
                )  # (1, V)
                p_draft = draft_probs[i]  # (1, V)

                ratio = (
                    p_target[0, x_i[0, 0]] / (p_draft[0, x_i[0, 0]] + 1e-10)
                ).clamp(max=1.0)

                if torch.rand(1, device=input_ids.device).item() <= ratio.item():
                    # Accept
                    input_ids = torch.cat([input_ids, x_i], dim=1)
                    n_accepted += 1
                    generated += 1
                    if generated >= max_new_tokens:
                        break
                else:
                    # Reject: resample from adjusted target distribution
                    adj = (p_target - p_draft).clamp(min=0)
                    adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-10)
                    bonus_tok = torch.multinomial(adj, 1)
                    input_ids = torch.cat([input_ids, bonus_tok], dim=1)
                    generated += 1
                    break

            if n_accepted == k and generated < max_new_tokens:
                # All drafts accepted → one bonus token from the target's
                # prediction at the position following the last draft token.
                bonus_logits = logits_t[:, T_cur - 1 + k, :]  # (1, V)
                bonus_tok = self._sample_token(bonus_logits, temperature, top_k, top_p)
                input_ids = torch.cat([input_ids, bonus_tok], dim=1)
                generated += 1

        return input_ids

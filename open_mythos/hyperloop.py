"""
HyperloopBlock — nested recurrent-depth architecture.

Reference: "Hyperloop Transformers" (arXiv 2604.21254) and
           "The Recurrent Transformer: Greater Effective Depth and Efficient Decoding"
           (arXiv 2604.21215).

Architecture overview
---------------------
A flat RecurrentBlock runs the same weights T times, giving linear depth scaling.
HyperloopBlock organises those iterations into a two-level hierarchy:

    for outer_t in range(outer_loops):          ← macro loop
        for inner_t in range(inner_loops):      ← micro loop (same as RecurrentBlock)
            h = inner_transformer(h, e)
            h = inner_injection(h, e, trans_out)
        outer_h = ACT-weighted sum of inner iterations
        h = outer_injection(h, e, outer_h)      ← outer update

Total effective depth = outer_loops × inner_loops.
Unique parameters     ≈ single RecurrentBlock  (one shared TransformerBlock,
                        two LTIInjections, two LoRAAdapters, two ACT halters).

Why this is better than a flat loop of the same depth
------------------------------------------------------
At depth d (outer) × k (inner):
- The *outer* injection sees a hidden state that has already been refined k times
  in the inner micro-loop. This gives the outer update a much cleaner signal than
  a flat loop would have at the same position.
- The two separate LTI injection matrices (inner + outer) allow independent spectral
  radius constraints at each level of the hierarchy, making training more stable
  than one large coupled system.
- Per-level LoRA adapters let the model learn behaviourally distinct operations
  at each level (e.g., inner = pattern extraction, outer = reasoning assembly).

Usage
-----
    from open_mythos.hyperloop import HyperloopConfig, HyperloopMythos

    cfg = HyperloopConfig(
        vocab_size=32000, dim=2048,
        outer_loops=4, inner_loops=4,   # effective depth = 16
        ...
    )
    model = HyperloopMythos(cfg)
    logits = model(input_ids)           # same API as OpenMythos
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from open_mythos.main import (
    ACTHalting,
    LoRAAdapter,
    LTIInjection,
    MythosConfig,
    RMSNorm,
    TransformerBlock,
    loop_index_embedding,
    precompute_rope_freqs,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HyperloopConfig(MythosConfig):
    """Extends MythosConfig with two-level loop hyperparameters.

    Fields inherited from MythosConfig are all still respected.
    ``max_loop_iters`` is re-interpreted as ``inner_loops`` when the
    HyperloopBlock is used (the outer loop count is controlled by
    ``outer_loops``).

    Attributes:
        outer_loops  -- number of outer (macro) recurrent iterations.  At each
                        outer step the inner micro-loop runs ``inner_loops``
                        times and its ACT-weighted output is consumed by the
                        outer update rule.
        inner_loops  -- number of inner (micro) recurrent iterations per outer
                        step.  Inherits ``max_loop_iters`` semantics (ACT can
                        halte early, depth extrapolation applies).
        outer_lora_rank -- LoRA rank for the per-outer-iteration depth adapter.
                           Defaults to the same as ``lora_rank`` (inner).
    """

    outer_loops: int = 4
    inner_loops: int = 4  # overrides max_loop_iters inside HyperloopBlock
    outer_lora_rank: int = 0  # 0 = use lora_rank (same as inner)

    def __post_init__(self):
        # Keep max_loop_iters in sync so any code that reads it still works.
        # We set it to the total effective depth so OpenMythos compatibility
        # helpers (e.g. depth-extrapolation checks) see the right number.
        self.max_loop_iters = self.outer_loops * self.inner_loops
        if self.outer_lora_rank == 0:
            self.outer_lora_rank = self.lora_rank


# ---------------------------------------------------------------------------
# Inner micro-loop (reuses RecurrentBlock logic in a standalone function)
# ---------------------------------------------------------------------------


def _run_inner_loop(
    h: torch.Tensor,
    e: torch.Tensor,
    block: TransformerBlock,
    injection: LTIInjection,
    lora: LoRAAdapter,
    act: ACTHalting,
    norm: RMSNorm,
    freqs_cis: torch.Tensor,
    loop_dim: int,
    inner_loops: int,
    act_threshold: float,
    mask: Optional[torch.Tensor],
    kv_cache: Optional[dict],
    loop_offset: int = 0,
) -> torch.Tensor:
    """Run `inner_loops` recurrent iterations and return the ACT-weighted sum.

    This mirrors ``RecurrentBlock.forward`` but is factored out so it can be
    called multiple times (once per outer step) with the same parameters.

    Args:
        h            -- hidden state entering the inner loop, shape (B, T, dim)
        e            -- encoded input injected at every step, shape (B, T, dim)
        block        -- shared TransformerBlock (weights reused across ALL loops)
        injection    -- LTI injection parameters for this level
        lora         -- depth-wise LoRA adapter for this level
        act          -- ACT halting predictor for this level
        norm         -- pre-norm RMSNorm applied to (h + e) before the block
        freqs_cis    -- RoPE frequencies, shape (T, head_dim//2)
        loop_dim     -- number of channels receiving the loop-index embedding
        inner_loops  -- number of micro-iterations to run
        act_threshold-- cumulative halting probability threshold
        mask         -- additive causal mask or None
        kv_cache     -- optional KV cache dict (mutated in-place)
        loop_offset  -- global loop index offset (for unique cache_key strings
                        when this function is called from multiple outer steps)

    Returns:
        ACT-weighted hidden state, shape (B, T, dim).
    """
    B, T, D = h.shape
    halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
    cumulative_p = torch.zeros(B, T, device=h.device)
    h_out = torch.zeros_like(h)

    for t in range(inner_loops):
        global_t = loop_offset + t
        h_loop = loop_index_embedding(h, global_t, loop_dim)
        combined = norm(h_loop + e)
        cache_key = f"hyperloop_inner_{global_t}"
        trans_out = block(combined, freqs_cis, mask, kv_cache, cache_key)
        trans_out = trans_out + lora(trans_out, min(t, lora.scale.num_embeddings - 1))
        h = injection(h, e, trans_out)

        p = act(h)
        still_running = ~halted
        remainder = (1.0 - cumulative_p).clamp(min=0)
        weight = torch.where(cumulative_p + p >= act_threshold, remainder, p)
        weight = weight * still_running.float()
        h_out = h_out + weight.unsqueeze(-1) * h

        cumulative_p = cumulative_p + p * still_running.float()
        halted = halted | (cumulative_p >= act_threshold)

        if halted.all() and kv_cache is None:
            break

    return h_out


# ---------------------------------------------------------------------------
# HyperloopBlock
# ---------------------------------------------------------------------------


class HyperloopBlock(nn.Module):
    """Two-level nested recurrent block.

    Architecture (for each forward call with outer_loops=K, inner_loops=N):

    ::

        h₀ = input from Prelude
        e  = encoded input (frozen across all loops)

        for outer_t = 0 … K-1:
            # ── Inner micro-loop (N iterations) ──────────────────────────
            h_inner = run_inner_loop(h_{outer_t}, e,
                                     inner_block, inner_injection, ...)
            # ── Outer injection (LTI update at macro level) ───────────────
            h_{outer_t+1} = outer_injection(h_{outer_t}, e, h_inner)

        return outer_ACT-weighted sum of {h_1, …, h_K}

    The inner and outer loops each have:
      - Their own ``LTIInjection`` (separate spectral-radius constraints).
      - Their own ``LoRAAdapter`` (per-level depth differentiation).
      - Their own ``ACTHalting`` predictor (independent early-exit per level).

    But they **share the same** ``TransformerBlock`` — all K × N iterations
    run through the same attention + MoE weights.  This keeps the total unique
    parameter count close to a single ``RecurrentBlock``.
    """

    def __init__(self, cfg: HyperloopConfig):
        """Build the HyperloopBlock from a HyperloopConfig.

        Args:
            cfg -- HyperloopConfig; uses dim, outer_loops, inner_loops,
                   lora_rank, outer_lora_rank, act_threshold, max_loop_iters.
        """
        super().__init__()
        self.cfg = cfg
        self.outer_loops = cfg.outer_loops
        self.inner_loops = cfg.inner_loops
        self.act_threshold = cfg.act_threshold
        self.loop_dim = cfg.dim // 8

        # ── Shared transformer block (weights reused for every inner + outer step)
        self.block = TransformerBlock(cfg, use_moe=True)

        # ── Inner micro-loop components
        self.inner_injection = LTIInjection(cfg.dim)
        self.inner_lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.inner_loops)
        self.inner_act = ACTHalting(cfg.dim)
        self.inner_norm = RMSNorm(cfg.dim)

        # ── Outer macro-loop components
        self.outer_injection = LTIInjection(cfg.dim)
        self.outer_lora = LoRAAdapter(cfg.dim, cfg.outer_lora_rank, cfg.outer_loops)
        self.outer_act = ACTHalting(cfg.dim)
        self.outer_norm = RMSNorm(cfg.dim)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        outer_loops: Optional[int] = None,
        inner_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run the two-level nested loop.

        Args:
            h          -- hidden state from Prelude, shape (B, T, dim)
            e          -- encoded input frozen for injection, shape (B, T, dim)
            freqs_cis  -- RoPE frequencies, shape (T, head_dim//2)
            mask       -- causal mask or None
            outer_loops-- override for number of outer iterations at inference
            inner_loops-- override for number of inner iterations at inference
            kv_cache   -- optional KV cache dict (mutated in-place per step)

        Returns:
            ACT-weighted output of shape (B, T, dim).
        """
        K = outer_loops if outer_loops is not None else self.outer_loops
        N = inner_loops if inner_loops is not None else self.inner_loops
        B, T, D = h.shape

        # Outer ACT state
        outer_halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        outer_cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for outer_t in range(K):
            # ── Inner micro-loop ───────────────────────────────────────────
            loop_offset = outer_t * N  # ensures globally unique loop indices
            h_inner = _run_inner_loop(
                h=h,
                e=e,
                block=self.block,
                injection=self.inner_injection,
                lora=self.inner_lora,
                act=self.inner_act,
                norm=self.inner_norm,
                freqs_cis=freqs_cis,
                loop_dim=self.loop_dim,
                inner_loops=N,
                act_threshold=self.act_threshold,
                mask=mask,
                kv_cache=kv_cache,
                loop_offset=loop_offset,
            )

            # ── Outer macro update ─────────────────────────────────────────
            # Apply outer loop-index embedding before the outer injection to
            # let the outer injection weights behave differently per macro step.
            h_outer_emb = loop_index_embedding(h, outer_t, self.loop_dim)
            outer_lora_delta = self.outer_lora(
                h_inner, min(outer_t, self.outer_lora.scale.num_embeddings - 1)
            )
            h = self.outer_injection(h_outer_emb, e, h_inner + outer_lora_delta)

            # Outer ACT accumulation
            p_outer = self.outer_act(h)
            still_running = ~outer_halted
            remainder = (1.0 - outer_cumulative_p).clamp(min=0)
            weight = torch.where(
                outer_cumulative_p + p_outer >= self.act_threshold,
                remainder,
                p_outer,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            outer_cumulative_p = outer_cumulative_p + p_outer * still_running.float()
            outer_halted = outer_halted | (outer_cumulative_p >= self.act_threshold)

            if outer_halted.all() and kv_cache is None:
                break

        return h_out


# ---------------------------------------------------------------------------
# Full HyperloopMythos model
# ---------------------------------------------------------------------------


class HyperloopMythos(nn.Module):
    """Full language model using HyperloopBlock instead of RecurrentBlock.

    Shares the same Prelude → Recurrent → Coda structure as ``OpenMythos``,
    but replaces the flat ``RecurrentBlock`` with a ``HyperloopBlock`` for
    exponentially deeper reasoning at the same unique parameter count.

    The public API is identical to ``OpenMythos``:

    ::

        model = HyperloopMythos(cfg)
        logits = model(input_ids)               # forward pass
        logits = model(input_ids,
                       outer_loops=8,
                       inner_loops=8)           # 64-step reasoning at inference
        out = model.generate(ids, max_new_tokens=32)
    """

    def __init__(self, cfg: HyperloopConfig):
        """Build the full HyperloopMythos language model.

        Args:
            cfg -- HyperloopConfig specifying all hyperparameters.
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Precompute RoPE frequencies for both GQA and MLA paths
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
        self.recurrent = HyperloopBlock(cfg)  # ← two-level nested loop
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build an additive upper-triangular causal mask."""
        mask = torch.full(
            (1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype
        )
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        outer_loops: Optional[int] = None,
        inner_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass through Prelude → HyperloopBlock → Coda.

        Args:
            input_ids  -- token indices, shape (B, T)
            outer_loops-- override outer macro-loop count at inference
            inner_loops-- override inner micro-loop count at inference
            kv_cache   -- optional KV cache for incremental decoding
            start_pos  -- token offset into the sequence (for RoPE during decode)

        Returns:
            Logits of shape (B, T, vocab_size).
        """
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"hl_prelude_{i}")

        e = x  # encoded input frozen for injection
        x = self.recurrent(
            x,
            e,
            freqs_cis,
            mask,
            outer_loops=outer_loops,
            inner_loops=inner_loops,
            kv_cache=kv_cache,
        )

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"hl_coda_{i}")

        return self.head(self.norm(x))

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float = 1.0,
        generated_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample the next token from a logits tensor of shape (B, vocab_size).

        Applies repetition penalty → temperature → top-k → top-p → multinomial.
        Returns a (B, 1) integer tensor.

        Args:
            logits            -- raw logits, shape (B, vocab_size)
            temperature       -- softmax temperature
            top_k             -- keep only the top-K logits (0 = disabled)
            top_p             -- nucleus threshold (1.0 = disabled)
            repetition_penalty-- divide logits of already-generated tokens by
                                 this value (>1.0 = more penalty, 1.0 = off).
            generated_ids     -- previously generated token ids, shape (B, T).
        """
        import torch.nn.functional as F

        logits = logits.clone()  # avoid in-place modification of caller's tensor
        if repetition_penalty != 1.0 and generated_ids is not None:
            for b in range(logits.shape[0]):
                unique_ids = generated_ids[b].unique()
                scores = logits[b, unique_ids]
                scores = torch.where(
                    scores > 0,
                    scores / repetition_penalty,
                    scores * repetition_penalty,
                )
                logits[b, unique_ids] = scores
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
        outer_loops: int = 4,
        inner_loops: int = 4,
        decode_outer_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache.

        Supports the same two-phase depth strategy as ``OpenMythos.generate``:

        * **Prefill** (step 0): runs ``outer_loops`` macro iterations over
          the full prompt, populating KV caches at all outer-loop levels.
        * **Decode** (steps 1+): runs ``decode_outer_loops`` outer iterations.
          Setting ``decode_outer_loops=1`` when ``outer_loops=2`` roughly
          halves decode latency. The depth extrapolation result shows that
          outer=1 degrades quality by only +0.0085 eval-loss, so the
          quality/speed tradeoff is favourable.

        ``inner_loops`` is kept constant across both phases — only the outer
        macro-loop count changes, which gives a coarser but architecturally
        cleaner knob than adjusting inner loops.

        Sampling pipeline (applied in order):
            1. Scale logits by ``temperature``.
            2. ``top_k``: keep only the top-K logits (0 = disabled).
            3. ``top_p``: nucleus sampling — zero out tokens whose cumulative
               probability exceeds ``top_p`` (1.0 = disabled).
            4. Sample from the resulting distribution.

        Args:
            input_ids         -- prompt token indices, shape (B, T)
            max_new_tokens    -- tokens to generate
            outer_loops       -- outer macro-loop depth for prefill (step 0)
            inner_loops       -- inner micro-loop depth (constant, all steps)
            decode_outer_loops-- outer loops for decode steps (steps 1+).
                                 None = use outer_loops (original behaviour).
            temperature       -- softmax temperature
            top_k             -- restrict sampling to top-K logits (0 = off)
            top_p             -- nucleus sampling threshold in (0, 1]; 1.0 = disabled.
            repetition_penalty-- penalise already-generated tokens (1.0 = off).

        Returns:
            Token indices of shape (B, T + max_new_tokens).
        """

        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        _decode_outer = (
            decode_outer_loops if decode_outer_loops is not None else outer_loops
        )

        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
                cur_outer = outer_loops  # prefill: full depth
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
                cur_outer = _decode_outer  # decode: fast depth
            logits = self.forward(
                cur_ids,
                outer_loops=cur_outer,
                inner_loops=inner_loops,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )
            next_tok = self._sample_token(
                logits[:, -1, :],
                temperature,
                top_k,
                top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=input_ids,
            )
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        outer_loops: int = 4,
        inner_loops: int = 4,
        decode_outer_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        """Streaming autoregressive generation — yields one token per step.

        Identical to :meth:`generate` in sampling behaviour, but ``yield``s
        each new ``(B, 1)`` token tensor as soon as it is sampled.

        Usage::

            for tok in model.generate_stream(ids, outer_loops=2, inner_loops=2):
                print(tokenizer.decode(tok[0].tolist()), end="", flush=True)

        Args:
            input_ids         -- prompt token indices, shape (B, T)
            max_new_tokens    -- maximum tokens to stream
            outer_loops       -- outer macro-loop depth for prefill
            inner_loops       -- inner micro-loop depth (constant)
            decode_outer_loops-- outer loops for decode steps; None = outer_loops
            temperature       -- softmax temperature
            top_k             -- top-K filtering (0 = disabled)
            top_p             -- nucleus threshold (1.0 = disabled)
            repetition_penalty-- penalise already-generated tokens (1.0 = off)

        Yields:
            ``(B, 1)`` integer tensors, one per generated token
        """
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        _decode_outer = (
            decode_outer_loops if decode_outer_loops is not None else outer_loops
        )
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
                cur_outer = outer_loops
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
                cur_outer = _decode_outer
            logits = self.forward(
                cur_ids,
                outer_loops=cur_outer,
                inner_loops=inner_loops,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )
            next_tok = self._sample_token(
                logits[:, -1, :],
                temperature,
                top_k,
                top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=input_ids,
            )
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            yield next_tok

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        outer_loops: int = 4,
        inner_loops: int = 4,
        beam_width: int = 4,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Beam search decoding for HyperloopMythos.

        Maintains ``beam_width`` candidate sequences and selects the highest
        cumulative log-probability beam.  Each step runs a full forward pass
        per active beam (no shared KV cache across diverged beams).

        ``beam_width=1`` is equivalent to greedy decoding.

        Args:
            input_ids         -- prompt token indices, shape **(1, T)**
            max_new_tokens    -- tokens to generate
            outer_loops       -- outer macro-loop depth
            inner_loops       -- inner micro-loop depth
            beam_width        -- number of beams (1 = greedy)
            repetition_penalty-- penalise already-generated tokens (1.0 = off)

        Returns:
            Best beam as shape (1, T + max_new_tokens).
        """
        import torch.nn.functional as F

        assert input_ids.shape[0] == 1, "beam_search requires batch_size=1"
        device = input_ids.device
        V = self.cfg.vocab_size

        beams = input_ids.repeat(beam_width, 1)
        scores = torch.zeros(beam_width, device=device)

        for step in range(max_new_tokens):
            n_active = 1 if step == 0 else beam_width
            log_probs_list: list[torch.Tensor] = []
            for b in range(n_active):
                logits = self.forward(
                    beams[b : b + 1],
                    outer_loops=outer_loops,
                    inner_loops=inner_loops,
                )[
                    :, -1, :
                ]  # (1, V)
                if repetition_penalty != 1.0:
                    logits = logits.clone()
                    unique_ids = beams[b].unique()
                    tok_scores = logits[0, unique_ids]
                    tok_scores = torch.where(
                        tok_scores > 0,
                        tok_scores / repetition_penalty,
                        tok_scores * repetition_penalty,
                    )
                    logits[0, unique_ids] = tok_scores
                log_probs_list.append(F.log_softmax(logits[0], dim=-1))

            lp = torch.stack(log_probs_list, dim=0)  # (n_active, V)
            candidate = scores[:n_active].unsqueeze(1) + lp
            flat = candidate.view(-1)
            k = min(beam_width, flat.numel())
            top_scores, top_flat = flat.topk(k)
            parent = top_flat // V
            token = top_flat % V
            beams = torch.cat([beams[parent], token.unsqueeze(1)], dim=1)
            scores = top_scores

        best = scores.argmax()
        return beams[best : best + 1]

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[torch.Tensor],
        max_new_tokens: int = 64,
        outer_loops: int = 4,
        inner_loops: int = 4,
        decode_outer_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> list[torch.Tensor]:
        """Independent autoregressive generation for a list of prompts.

        Args:
            prompts           -- list of prompt tensors, each shape (1, T_i)
            max_new_tokens    -- tokens to generate per prompt
            outer_loops       -- outer macro-loop depth for prefill
            inner_loops       -- inner micro-loop depth
            decode_outer_loops-- outer loops for decode steps (None = outer_loops)
            temperature       -- softmax temperature
            top_k             -- top-K filtering (0 = disabled)
            top_p             -- nucleus sampling threshold (1.0 = disabled)
            repetition_penalty-- repetition penalty (1.0 = disabled)

        Returns:
            List of tensors, each shape (1, T_i + max_new_tokens).
        """
        return [
            self.generate(
                ids,
                max_new_tokens=max_new_tokens,
                outer_loops=outer_loops,
                inner_loops=inner_loops,
                decode_outer_loops=decode_outer_loops,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            for ids in prompts
        ]

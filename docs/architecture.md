# OpenMythos — Architecture Reference

> Version: 0.6.0 | Updated: 2026-05-23

---

## Overview

OpenMythos implements a **Recurrent-Depth Transformer (RDT)** — a looped transformer that separates computation into three functional stages:

```
Input tokens
     ↓
[Prelude]           prelude_layers × TransformerBlock  (run once)
     ↓
[Recurrent Block]   one TransformerBlock looped T times
     ↑_________↓    h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
     ↓
[Coda]              coda_layers × TransformerBlock  (run once)
     ↓
Output logits
```

The same recurrent block weights are reused at every loop iteration. Increasing loop depth at inference (depth extrapolation) extends the model's effective reasoning chain without adding parameters.

---

## Component Map

```
OpenMythos
├── embed                 nn.Embedding(vocab_size, dim)
├── freqs_cis             buffer — precomputed RoPE complex frequencies (GQA)
├── freqs_cis_mla         buffer — precomputed RoPE for MLA rope head dim
├── prelude               ModuleList of TransformerBlock (use_moe=False)
├── recurrent             RecurrentBlock
│   ├── block             TransformerBlock (use_moe=True)
│   │   ├── attn_norm     RMSNorm
│   │   ├── attn          MLAttention | GQAttention  (selected by cfg.attn_type)
│   │   ├── ffn_norm      RMSNorm
│   │   └── ffn           MoEFFN
│   ├── injection         LTIInjection
│   ├── act               ACTHalting
│   ├── lora              LoRAAdapter
│   └── norm              RMSNorm
├── coda                  ModuleList of TransformerBlock (use_moe=False)
├── norm                  RMSNorm  (final pre-head norm)
└── head                  nn.Linear(dim, vocab_size, bias=False)  [weight-tied to embed]
```

---

## Attention

### GQA — Grouped Query Attention (`attn_type="gqa"`)

| Parameter | Role |
|---|---|
| `n_heads` | Number of query heads |
| `n_kv_heads` | Number of key/value heads (`< n_heads`); each KV head is shared across `n_heads // n_kv_heads` query heads |
| `dim // n_heads` | Per-head dimension |

RoPE is applied to Q and K before writing into the KV cache, so cached values are already positionally encoded and need not be re-rotated.

Flash Attention 2 is used automatically when `flash-attn>=2.8.3` is installed. Falls back to manual scaled dot-product attention otherwise.

### MLA — Multi-Latent Attention (`attn_type="mla"`)

Compresses the KV cache from `n_kv_heads × head_dim × 2` to `kv_lora_rank + n_heads × qk_rope_head_dim` per token — roughly 10–20× smaller at production scale.

| Parameter | Role |
|---|---|
| `kv_lora_rank` | Compressed KV latent dimension stored in cache |
| `q_lora_rank` | Compressed Q latent dimension |
| `qk_rope_head_dim` | Per-head dimensions receiving RoPE |
| `qk_nope_head_dim` | Per-head dimensions without RoPE |
| `v_head_dim` | Per-head value dimension |

K_nope and V are reconstructed from the latent at each decode step; only `c_kv` and `k_rope` are cached.

---

## Feed-Forward

### Dense Expert (`use_moe=False`)

Standard SwiGLU FFN used in Prelude and Coda blocks:

```
output = down(silu(gate(x)) * up(x))
inner_dim = dim * 4 // 3
```

### MoEFFN (`use_moe=True`)

Fine-grained Mixture-of-Experts used inside the Recurrent Block:

| Parameter | Role |
|---|---|
| `n_experts` | Total routed experts |
| `n_shared_experts` | Always-active shared experts |
| `n_experts_per_tok` | Top-K experts selected per token |
| `expert_dim` | Hidden dim inside each fine-grained expert |

Router selects top-K experts by unbiased softmax scores; a per-expert bias (not a gradient parameter) is updated externally to keep load balanced without distorting the loss (DeepSeek-V3 aux-loss-free balancing).

---

## Recurrent Block

### Loop update rule

At each iteration `t` from `0` to `n_loops - 1`:

```
h_loop  = loop_index_embedding(h, t)      # inject sinusoidal loop-depth signal
combined = RMSNorm(h_loop + e)            # fuse with frozen input encoding
trans   = TransformerBlock(combined)      # attention + MoE FFN
trans   = trans + LoRAAdapter(trans, t)   # depth-wise LoRA delta
h       = LTIInjection(h, e, trans)       # stable state update
p       = ACTHalting(h)                   # per-position halting probability
```

### LTI Injection — stability guarantee

```
A_continuous = Diag(-exp(log_A))          # always negative diagonal
A_discrete   = exp(Δt · A_continuous)     # element-wise exp → values in (0, 1)
h_{t+1}      = A_discrete · h_t + B · e + trans_out
```

`ρ(A) < 1` is guaranteed by construction for any value of the learned parameters `log_A` and `log_dt`. Training is stable even at high learning rates.

### ACT Halting

Each position accumulates a halting probability `p_t` at every loop step. When the cumulative sum exceeds `act_threshold` (default 0.99), the position stops contributing updates. The final hidden state is a weighted sum of `h_t` across iterations:

```
h_out = Σ_t weight_t · h_t
```

Easy tokens halt early; hard tokens receive more computation — all within the same batch.

### Loop Index Embedding

A sinusoidal embedding of the loop index `t` is injected into the first `dim // 8` channels of `h` at each step. This lets the same shared weights implement functionally distinct operations at different loop depths.

### Depth-wise LoRA Adapter

A shared down-projection `A` and up-projection `B` are reused across all loops, while a per-loop scale vector `s[t]` shifts behavior per depth:

```
delta(x, t) = (A(x) * s[t]) @ B
```

Parameter overhead: `2 × dim × rank + max_loop_iters × rank` — negligible vs. the main block.

---

## Inference Engine

### KV Cache

All attention layers support an incremental KV cache passed as a `dict`. Cache keys follow the pattern `"prelude_i"`, `"recurrent_loop_t"`, `"coda_i"`.

**Sliding window** (`max_cache_len > 0`): during single-token decode steps, cache entries are clipped to the last `max_cache_len` positions, bounding memory at long sequence lengths.

### HyperloopMythos — Two-Phase Depth Strategy

```
Prefill  (step 0):   n_loops iterations   — deep reasoning over the full prompt
Decode   (steps 1+): decode_loops iters   — fast per-token generation
```

Benchmark result: `decode_loops = n_loops // 2` delivers **2.54× decode speedup** with < 1% quality degradation (Sprint 1, 227 tests PASS).

### Generation Methods

| Method | Algorithm | B |
|---|---|---|
| `generate()` | Autoregressive sampling with KV cache | ≥1 |
| `generate_beam()` | Beam search with length penalty | 1 |
| `generate_stream()` | Streaming autoregressive (yields per token) | ≥1 |
| `speculative_decode()` | Self-speculative: shallow draft + deep verify | 1 |
| `generate_batch()` | Left-padded batch, variable-length prompts | ≥1 |

### Sampling Pipeline

Applied in order inside `_sample_token`:

1. `repetition_penalty` — divide/multiply logits of seen tokens
2. Temperature scaling — `logits / temperature`
3. Top-K filtering — zero out all but top-K logits
4. Top-P (nucleus) filtering — keep smallest set covering probability mass `≥ top_p`
5. Multinomial sampling

### Quantization

| Mode | API | Effect |
|---|---|---|
| FP16 | `model.quantize("fp16")` | `half()` — halves memory vs. float32 |
| INT8 | `model.quantize("int8")` | `quantize_dynamic` on all `nn.Linear` layers — ~4× smaller |

---

## Configuration Reference

`MythosConfig` fields:

| Field | Default | Description |
|---|---|---|
| `vocab_size` | 32000 | Token vocabulary size |
| `dim` | 2048 | Model hidden dimension |
| `n_heads` | 16 | Query attention heads |
| `n_kv_heads` | 4 | KV heads (GQA); ignored by MLA |
| `max_seq_len` | 4096 | Maximum sequence length for RoPE precomputation |
| `max_loop_iters` | 16 | Default recurrent loop depth at inference |
| `prelude_layers` | 2 | Standard transformer layers before the loop |
| `coda_layers` | 2 | Standard transformer layers after the loop |
| `attn_type` | `"mla"` | `"gqa"` or `"mla"` |
| `kv_lora_rank` | 512 | MLA compressed KV latent dim |
| `q_lora_rank` | 1536 | MLA compressed Q latent dim |
| `qk_rope_head_dim` | 64 | MLA per-head RoPE dim |
| `qk_nope_head_dim` | 128 | MLA per-head non-RoPE dim |
| `v_head_dim` | 128 | MLA per-head value dim |
| `n_experts` | 64 | Total routed MoE experts |
| `n_shared_experts` | 2 | Always-active shared experts |
| `n_experts_per_tok` | 4 | Top-K experts per token |
| `expert_dim` | 512 | Hidden dim inside each expert |
| `act_threshold` | 0.99 | ACT cumulative halting threshold |
| `rope_theta` | 500000.0 | RoPE base frequency |
| `lora_rank` | 16 | LoRA adapter rank |
| `max_output_tokens` | 4096 | Max tokens per forward pass |
| `dropout` | 0.0 | Dropout probability (0 = disabled) |

---

## File Layout

```
open_mythos/
├── main.py        Core model: all nn.Module classes + generation methods
├── hyperloop.py   HyperloopMythos two-phase inference engine (Sprint 1)
├── moda.py        MoDa (Mixture-of-Depths Attention) variant
├── tokenizer.py   MythosTokenizer wrapping openai/gpt-oss-20b
├── variants.py    Pre-configured model scales mythos_1b … mythos_1t
└── __init__.py    Public exports

tests/
├── test_main.py       257 unit tests covering all model components and generation methods
├── test_hyperloop.py  HyperloopMythos inference engine tests
├── test_moda.py       MoDa variant tests
├── test_integration.py End-to-end integration tests
└── small_benchmark.py Speed benchmark: HyperloopMythos vs OpenMythos vs Baseline

docs/
├── architecture.md    This file — component map, algorithms, config reference
├── open_mythos.md     Full API reference
├── datasets.md        Training dataset recommendations
└── operations.md      Deployment and operations guide
```

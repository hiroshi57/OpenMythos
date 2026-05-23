# Changelog

All notable changes to OpenMythos are documented here.

---

## [0.6.0] — 2026-05-23

### Sprint 2: Inference 高度化

**New generation methods**

- `generate_beam()` — beam search with configurable `beam_width` and `length_penalty` (B=1)
- `generate_batch()` — parallel generation for variable-length prompt lists via left-padding
- `repetition_penalty` parameter added to `generate()` and `generate_stream()`
- `max_cache_len` (sliding window KV cache) parameter added to `generate()` and `generate_stream()`

**Quantization**

- `model.quantize("fp16")` — in-place FP16 cast; dtype-safe through ACT, LTI, and MoEFFN
- `model.quantize("int8")` — dynamic INT8 quantization of all `nn.Linear` layers

**Internal fixes**

- `MoEFFN.forward`: `weight` and `expert_out` now cast to `flat.dtype` for FP16/BF16 safety
- `RecurrentBlock`: `cumulative_p` and boolean masks cast to `h.dtype`
- `LTIInjection.forward`: `A` and `B` cast to `h.dtype`
- `ACTHalting.forward`: output cast to `h.dtype`

**Tests**: 257 PASS (up from 227)

---

## [0.5.0] — 2026-05-21

### Sprint 1: HyperloopMythos & Inference Engine

- `HyperloopMythos` two-phase inference engine (prefill `n_loops`, decode `decode_loops`)
- `generate_stream()` — streaming autoregressive generation
- `speculative_decode()` — self-speculative decoding with depth-based draft/target split
- `top_p` nucleus sampling in `_sample_token`
- `decode_loops` two-phase depth strategy: 2.54× decode speedup
- `_causal_mask` cache_len fix for incremental decoding
- LTI `get_A()` `.clamp(min=1e-6)` fix for float32 saturation
- `freqs_cis[:T]` slice fix for apply_rope broadcast errors
- Lint: ruff/black pass, `.gitattributes` CRLF/LF unification

**Tests**: 227 PASS

---

## [0.1.0] — 2026-05-01

Initial release: OpenMythos RDT Phase 0–4

- `OpenMythos` model with GQA / MLA attention
- `MoEFFN` fine-grained mixture of experts
- `LTIInjection` with spectral radius < 1 guarantee
- `ACTHalting` adaptive computation time
- `LoRAAdapter` depth-wise LoRA
- `RecurrentBlock` with loop index embedding
- `HyperloopMythos` v0.1 (basic)
- Pre-configured variants `mythos_1b` … `mythos_1t`
- `MythosTokenizer`
- Training script `training/3b_fine_web_edu.py`

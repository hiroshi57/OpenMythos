#!/usr/bin/env python3
"""
OpenMythos Throughput & Latency Benchmark.

Measures:
  - Time-to-first-token (TTFT): prefill latency for prompt of length P
  - Time-per-output-token (TPOT): average decode latency per generated token
  - Throughput: tokens/sec (prompt+generated) for given batch sizes
  - Peak memory: GPU VRAM or CPU RSS at batch inference time

Usage
-----
# nano variant, smoke test (CPU)
python benchmark/throughput.py --variant nano --max-new-tokens 32

# 1b on CUDA, sweep batch sizes
python benchmark/throughput.py --variant 1b --device cuda --batch-sizes 1,2,4,8

# From checkpoint
python benchmark/throughput.py --checkpoint path/to/model.pt --prompt "Once upon"
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.variants import (
    mythos_nano,
    mythos_1b,
    mythos_3b,
    mythos_7b,
    mythos_10b,
)

_VARIANTS = {
    "nano": mythos_nano,
    "1b": mythos_1b,
    "3b": mythos_3b,
    "7b": mythos_7b,
    "10b": mythos_10b,
}

_DEFAULT_PROMPT = (
    "The history of artificial intelligence is a long and winding road "
    "that stretches back to the earliest days of computing. "
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LatencyResult:
    batch_size: int
    prompt_len: int
    max_new_tokens: int
    ttft_ms: float  # time-to-first-token (ms)
    tpot_ms: float  # time-per-output-token (ms)
    total_ms: float  # total generation time (ms)
    generated_tokens: int
    throughput_tok_s: float  # (prompt + generated) tokens / sec
    peak_mem_mb: float  # peak GPU VRAM or CPU RSS (MB)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(args) -> tuple[OpenMythos, str]:
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg: MythosConfig = ckpt["cfg"]
        model = OpenMythos(cfg)
        model.load_state_dict(ckpt["model"])
    else:
        fn = _VARIANTS.get(args.variant or "nano")
        if fn is None:
            raise ValueError(f"Unknown variant '{args.variant}'")
        cfg = fn()
        model = OpenMythos(cfg)
    return model.to(dev).eval(), dev


def _tokenize(prompt: str, vocab_size: int) -> list[int]:
    try:
        from open_mythos.tokenizer import MythosTokenizer

        enc = MythosTokenizer()
        ids = enc.encode(prompt)
    except Exception:
        ids = [ord(c) for c in prompt]
    return [min(i, vocab_size - 1) for i in ids]


def _peak_mem_mb(device: str) -> float:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    try:
        import psutil
        import os

        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except ImportError:
        return float("nan")


def _reset_mem_stats(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------


def measure_latency(
    model: OpenMythos,
    prompt_ids: list[int],
    device: str,
    max_new_tokens: int = 32,
    batch_size: int = 1,
    n_loops: Optional[int] = None,
    warmup_iters: int = 1,
) -> LatencyResult:
    """Measure TTFT / TPOT / throughput for one (prompt, batch_size) combination.

    The prompt is replicated across the batch dimension.  All timing uses
    ``time.perf_counter`` and, for CUDA, surrounding ``torch.cuda.synchronize``
    calls so that async kernels are correctly attributed.
    """
    ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    ids_t = ids_t.unsqueeze(0).expand(batch_size, -1)  # (B, P)

    def _sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    gen_kwargs: dict = {}
    if n_loops is not None:
        gen_kwargs["n_loops"] = n_loops

    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            model(ids_t, **gen_kwargs)
        _sync()

    _reset_mem_stats(device)
    gc.collect()

    # --- TTFT: single forward pass (prefill) ---
    _sync()
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        _ = model(ids_t, **gen_kwargs)
    _sync()
    ttft_ms = (time.perf_counter() - t_prefill_start) * 1000.0

    # --- Autoregressive decode ---
    _sync()
    t_decode_start = time.perf_counter()
    cur_ids = ids_t  # (B, P)
    generated = 0
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(cur_ids, **gen_kwargs)  # (B, T, V)
        _sync()
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        cur_ids = torch.cat([cur_ids, next_id], dim=1)
        generated += 1
    t_decode_end = time.perf_counter()
    total_decode_ms = (t_decode_end - t_decode_start) * 1000.0
    tpot_ms = total_decode_ms / max(generated, 1)
    total_ms = ttft_ms + total_decode_ms

    peak_mb = _peak_mem_mb(device)
    total_tokens = (len(prompt_ids) + generated) * batch_size
    throughput = total_tokens / (total_ms / 1000.0 + 1e-9)

    return LatencyResult(
        batch_size=batch_size,
        prompt_len=len(prompt_ids),
        max_new_tokens=max_new_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_ms=total_ms,
        generated_tokens=generated,
        throughput_tok_s=throughput,
        peak_mem_mb=peak_mb,
    )


def sweep_batch_sizes(
    model: OpenMythos,
    prompt_ids: list[int],
    device: str,
    batch_sizes: list[int],
    max_new_tokens: int = 32,
    n_loops: Optional[int] = None,
) -> list[LatencyResult]:
    results = []
    for bs in batch_sizes:
        try:
            r = measure_latency(
                model,
                prompt_ids,
                device,
                max_new_tokens=max_new_tokens,
                batch_size=bs,
                n_loops=n_loops,
            )
            results.append(r)
        except RuntimeError as exc:
            print(f"  [SKIP] batch_size={bs}: {exc}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenMythos throughput & latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", default="nano", choices=list(_VARIANTS))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt", default=_DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--batch-sizes", default="1", help="Comma-separated list, e.g. 1,2,4"
    )
    p.add_argument("--n-loops", type=int, default=None)
    p.add_argument(
        "--warmup", type=int, default=1, help="Warmup forward passes before timing"
    )
    p.add_argument("--output-json", default=None, help="Write results to JSON file")
    return p.parse_args()


def _fmt_row(r: LatencyResult) -> str:
    mem_str = f"{r.peak_mem_mb:.1f}" if not math.isnan(r.peak_mem_mb) else "N/A"
    return (
        f"  bs={r.batch_size:<3}  "
        f"TTFT={r.ttft_ms:7.1f}ms  "
        f"TPOT={r.tpot_ms:6.1f}ms  "
        f"total={r.total_ms:8.1f}ms  "
        f"thr={r.throughput_tok_s:8.0f} tok/s  "
        f"mem={mem_str} MB"
    )


def main() -> None:
    args = _parse_args()
    model, device = _load_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model      : {args.variant or 'checkpoint'}  ({n_params:,} params)")
    print(f"Device     : {device}")
    print(f"Max tokens : {args.max_new_tokens}")

    prompt_ids = _tokenize(args.prompt, model.cfg.vocab_size)
    if not prompt_ids:
        prompt_ids = [0]
    print(f"Prompt len : {len(prompt_ids)} tokens")

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    print(f"\nSweeping batch sizes: {batch_sizes}")
    print("─" * 72)

    results = sweep_batch_sizes(
        model,
        prompt_ids,
        device,
        batch_sizes=batch_sizes,
        max_new_tokens=args.max_new_tokens,
        n_loops=args.n_loops,
    )

    for r in results:
        print(_fmt_row(r))

    print("─" * 72)

    if args.output_json:
        data = [asdict(r) for r in results]
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

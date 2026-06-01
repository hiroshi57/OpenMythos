#!/usr/bin/env python3
"""
OpenMythos Perplexity Evaluation — WikiText-2 / WikiText-103.

Measures token-level perplexity (PPL) on a text corpus using a sliding
context window so that every token is predicted conditioned on as much
prior context as the model's max_seq_len allows.

Usage
-----
# nano variant on WikiText-2 test split (CPU)
python benchmark/perplexity.py --variant nano --dataset wikitext-2-raw-v1

# From a checkpoint
python benchmark/perplexity.py --checkpoint path/to/model.pt --stride 512

# Full WikiText-103
python benchmark/perplexity.py --variant 1b --dataset wikitext-103-raw-v1 --device cuda

Algorithm
---------
The corpus is tokenised and treated as one long sequence of token IDs.
A sliding window of size ``seq_len`` advances by ``stride`` tokens at a time.
Loss is accumulated only on the ``stride`` right-most tokens of each window
(i.e. tokens that have ``seq_len - stride`` tokens of prior context) to
avoid double-counting tokens near window boundaries.

This matches the standard WikiText-2 evaluation protocol used by GPT-2.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional

import torch
import torch.nn.functional as F

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.variants import (
    mythos_nano, mythos_1b, mythos_3b, mythos_7b, mythos_10b,
)

_VARIANTS = {
    "nano": mythos_nano,
    "1b": mythos_1b,
    "3b": mythos_3b,
    "7b": mythos_7b,
    "10b": mythos_10b,
}


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


def _load_corpus(dataset_name: str, split: str = "test") -> str:
    """Download and return the raw text of the requested WikiText split."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets to run perplexity benchmark")

    ds = load_dataset("wikitext", dataset_name, split=split)
    return "\n\n".join(ds["text"])


def _tokenize_corpus(text: str, vocab_size: int) -> list[int]:
    """Tokenize corpus; use MythosTokenizer with byte-level fallback."""
    try:
        from open_mythos.tokenizer import MythosTokenizer
        enc = MythosTokenizer()
        ids = enc.encode(text)
    except Exception:
        ids = [ord(c) for c in text]
    return [min(i, vocab_size - 1) for i in ids]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_perplexity(
    model: OpenMythos,
    token_ids: list[int],
    device: str,
    seq_len: int = 512,
    stride: int = 256,
    batch_size: int = 1,
    n_loops: Optional[int] = None,
) -> dict:
    """Compute perplexity over token_ids using a sliding context window.

    Args:
        model      -- eval-mode OpenMythos
        token_ids  -- full tokenised corpus as a flat int list
        device     -- torch device string
        seq_len    -- context window size (≤ model.cfg.max_seq_len)
        stride     -- step between windows; smaller = more context per token
        batch_size -- sequences per forward pass (1 is safe on any HW)
        n_loops    -- recurrent depth override; None = model default

    Returns:
        dict with keys: ``ppl``, ``nll``, ``n_tokens``, ``elapsed_sec``,
        ``tokens_per_sec``.
    """
    ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    n_total = len(ids_tensor)

    total_nll = 0.0
    n_predicted = 0
    t0 = time.perf_counter()

    for begin in range(0, n_total - 1, stride):
        end = min(begin + seq_len, n_total - 1)
        input_chunk = ids_tensor[begin:end].unsqueeze(0).to(device)
        target_chunk = ids_tensor[begin + 1:end + 1].unsqueeze(0).to(device)

        # Tokens that benefit from full left context (exclude first overlap)
        context_len = end - begin
        predict_start = max(0, seq_len - stride)
        if context_len <= predict_start:
            continue

        kwargs = {}
        if n_loops is not None:
            kwargs["n_loops"] = n_loops

        with torch.no_grad():
            logits = model(input_chunk, **kwargs)  # (1, T, V)

        # Accumulate loss only on the stride-right portion
        log_probs = F.log_softmax(logits[0, predict_start:], dim=-1)
        targets = target_chunk[0, predict_start:]
        nll = F.nll_loss(log_probs, targets, reduction="sum").item()
        total_nll += nll
        n_predicted += targets.shape[0]

        if end >= n_total - 1:
            break

    elapsed = time.perf_counter() - t0
    avg_nll = total_nll / max(n_predicted, 1)
    ppl = math.exp(min(avg_nll, 88.0))

    return {
        "ppl": ppl,
        "nll": avg_nll,
        "n_tokens": n_predicted,
        "elapsed_sec": elapsed,
        "tokens_per_sec": n_predicted / max(elapsed, 1e-9),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenMythos perplexity evaluation on WikiText",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", default="nano", choices=list(_VARIANTS))
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0")
    p.add_argument("--dataset", default="wikitext-2-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--n-loops", type=int, default=None, help="Override recurrent depth")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Truncate corpus to first N tokens (faster for smoke-test)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model, device = _load_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model      : {args.variant or 'checkpoint'}  ({n_params:,} params)")
    print(f"Device     : {device}")
    print(f"Dataset    : {args.dataset} / {args.split}")
    print(f"Seq len    : {args.seq_len}   Stride: {args.stride}")

    print("Loading corpus...", flush=True)
    text = _load_corpus(args.dataset, args.split)
    ids = _tokenize_corpus(text, model.cfg.vocab_size)
    if args.max_tokens:
        ids = ids[:args.max_tokens]
    print(f"Corpus     : {len(ids):,} tokens")

    print("Evaluating...", flush=True)
    result = evaluate_perplexity(
        model, ids, device,
        seq_len=args.seq_len,
        stride=args.stride,
        n_loops=args.n_loops,
    )

    print(f"\n{'─' * 40}")
    print(f"  PPL      : {result['ppl']:.2f}")
    print(f"  NLL      : {result['nll']:.4f}")
    print(f"  Tokens   : {result['n_tokens']:,}")
    print(f"  Elapsed  : {result['elapsed_sec']:.1f}s")
    print(f"  Speed    : {result['tokens_per_sec']:.0f} tok/s")
    print(f"{'─' * 40}")


if __name__ == "__main__":
    main()

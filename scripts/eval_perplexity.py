#!/usr/bin/env python3
"""
Perplexity evaluation for OpenMythos on WikiText-2.

Measures how perplexity changes as n_loops increases, producing both
a CSV of results and a plot — the key evidence that more loops = better reasoning.

Usage (CPU, quick smoke-test):
    python scripts/eval_perplexity.py --max-batches 20

Usage (full eval):
    python scripts/eval_perplexity.py

Usage (GPU):
    python scripts/eval_perplexity.py --device cuda
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from open_mythos.main import MythosConfig, OpenMythos

# ---------------------------------------------------------------------------
# Small model preset for quick CPU evaluation
# ---------------------------------------------------------------------------


def small_eval_config() -> MythosConfig:
    """256-dim model that fits comfortably in CPU RAM for eval."""
    return MythosConfig(
        vocab_size=50257,  # GPT-2 tokenizer vocab
        dim=256,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=512,
        max_loop_iters=16,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        act_threshold=0.99,
        lora_rank=8,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_perplexity(
    model: OpenMythos,
    token_ids: torch.Tensor,
    n_loops: int,
    seq_len: int,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """
    Compute perplexity over a flat token sequence with a sliding window.

    Returns (perplexity, elapsed_seconds).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_chunks = (len(token_ids) - 1) // seq_len

    t0 = time.time()
    for i, start in enumerate(range(0, n_chunks * seq_len, seq_len)):
        if max_batches is not None and i >= max_batches:
            break
        chunk = token_ids[start : start + seq_len + 1].to(device)
        input_ids = chunk[:-1].unsqueeze(0)  # (1, seq_len)
        targets = chunk[1:].unsqueeze(0)  # (1, seq_len)

        logits = model(input_ids, n_loops=n_loops)  # (1, seq_len, vocab)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += targets.numel()

    elapsed = time.time() - t0
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    return ppl, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per loop setting (use 20-50 for quick smoke-test)",
    )
    parser.add_argument(
        "--loops",
        default="1,2,4,8,16",
        help="Comma-separated n_loops values to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Fine-tuned checkpoint path (.pt). Empty = random weights (architectural benchmark).",
    )
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device = torch.device(args.device)
    loop_values = [int(x) for x in args.loops.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # --- tokenizer (GPT-2, no special tokens needed for LM eval) ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # --- dataset ---
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(ds["text"])
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"  {len(token_ids):,} tokens in test set")

    # --- model ---
    print("Building model...")
    cfg = small_eval_config()
    model = OpenMythos(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print("  Using random weights (architectural benchmark mode)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters | dim={cfg.dim} | attn={cfg.attn_type}")

    # --- evaluate across loop counts ---
    results = []
    print(f"\n{'loops':>6}  {'perplexity':>12}  {'time(s)':>8}")
    print("-" * 32)
    for n_loops in loop_values:
        ppl, elapsed = compute_perplexity(
            model,
            token_ids,
            n_loops,
            seq_len=args.seq_len,
            device=device,
            max_batches=args.max_batches,
        )
        results.append({"loops": n_loops, "perplexity": ppl, "time_s": elapsed})
        print(f"{n_loops:>6}  {ppl:>12.2f}  {elapsed:>8.1f}s")

    # --- save CSV ---
    csv_path = out_dir / "perplexity_vs_loops.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["loops", "perplexity", "time_s"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved → {csv_path}")

    # --- plot (optional, skips gracefully if matplotlib unavailable) ---
    try:
        import matplotlib.pyplot as plt

        loops_x = [r["loops"] for r in results]
        ppls_y = [r["perplexity"] for r in results]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(loops_x, ppls_y, marker="o", linewidth=2, color="#3670A0")
        ax.set_xlabel("n_loops (recurrent depth)")
        ax.set_ylabel("Perplexity ↓ better")
        ax.set_title("OpenMythos — Perplexity vs. Loop Depth (WikiText-2)")
        ax.set_xticks(loops_x)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = out_dir / "perplexity_vs_loops.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved  → {plot_path}")
    except ImportError:
        print("matplotlib not found - skipping plot (pip install matplotlib to enable)")


if __name__ == "__main__":
    main()

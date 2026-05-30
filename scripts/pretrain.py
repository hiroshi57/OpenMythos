#!/usr/bin/env python3
"""
OpenMythos pre-training script — FineWeb-Edu streaming.

Trains from scratch on FineWeb-Edu (sample-10BT subset by default) with a
3-phase warmup-stable-decay LR schedule, bf16 autocast on CUDA, gradient
checkpointing, and periodic perplexity evaluation.

Checkpoint format (compatible with perplexity.py / lm_eval_harness.py):
    {"model": state_dict, "cfg": MythosConfig}

Usage
-----
# Quick smoke-test (CPU, nano, 1 000 tokens)
python scripts/pretrain.py --variant nano --max-tokens 1000 --save-every 100 --eval-every 50

# GCP T4 (run via scripts/pretrain_gcp.sh)
python scripts/pretrain.py --variant 1b --device cuda --batch 8 --grad-accum 4 \
    --max-tokens 300000000 --seq-len 1024
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

from open_mythos.main import OpenMythos, TransformerBlock
from open_mythos.variants import mythos_nano, mythos_1b, mythos_3b, mythos_7b, mythos_10b
from open_mythos.logger_utils import TrainLogger

_VARIANTS = {
    "nano": mythos_nano,
    "1b": mythos_1b,
    "3b": mythos_3b,
    "7b": mythos_7b,
    "10b": mythos_10b,
}

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def warmup_stable_decay_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """3-phase schedule: linear warmup → stable LR → cosine decay."""

    total = warmup_steps + stable_steps + decay_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            decay_step = step - warmup_steps - stable_steps
            progress = min(decay_step / max(decay_steps, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

def _simple_tokenize(text: str, vocab_size: int) -> list[int]:
    """Byte-level tokenizer clipped to vocab_size — no external deps."""
    return [min(b, vocab_size - 1) for b in text.encode("utf-8", errors="replace")]


class StreamingTokenDataset(IterableDataset):
    """
    Wraps FineWeb-Edu (or any HuggingFace streaming dataset) and yields
    fixed-length token chunks of size ``seq_len + 1`` (input + target).

    Parameters
    ----------
    hf_dataset_name : str
        HuggingFace dataset name.
    hf_subset : str
        Dataset subset / config name.
    split : str
        Dataset split.
    seq_len : int
        Sequence length for each training chunk.
    max_tokens : int
        Stop after consuming this many tokens (approximate).
    vocab_size : int
        Clip token IDs to this range.
    text_field : str
        Field name containing the text in each example.
    """

    def __init__(
        self,
        hf_dataset_name: str = "HuggingFaceFW/fineweb-edu",
        hf_subset: str = "sample-10BT",
        split: str = "train",
        seq_len: int = 1024,
        max_tokens: int = 300_000_000,
        vocab_size: int = 1024,
        text_field: str = "text",
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.hf_subset = hf_subset
        self.split = split
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.text_field = text_field

    def __iter__(self) -> Iterator[torch.Tensor]:
        from datasets import load_dataset

        ds = load_dataset(
            self.hf_dataset_name,
            name=self.hf_subset,
            streaming=True,
            split=self.split,
        )

        buffer: list[int] = []
        consumed = 0
        chunk_size = self.seq_len + 1  # +1 for the target shift

        for example in ds:
            if consumed >= self.max_tokens:
                break
            text = example.get(self.text_field, "")
            if not text:
                continue
            ids = _simple_tokenize(text, self.vocab_size)
            buffer.extend(ids)

            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                consumed += self.seq_len
                yield torch.tensor(chunk, dtype=torch.long)

                if consumed >= self.max_tokens:
                    return


# ---------------------------------------------------------------------------
# Gradient checkpointing helpers
# ---------------------------------------------------------------------------

def _enable_gradient_checkpointing(model: OpenMythos) -> None:
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            module.gradient_checkpointing = True


def _disable_gradient_checkpointing(model: OpenMythos) -> None:
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            module.gradient_checkpointing = False


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_checkpoint(model: OpenMythos, step: int, ckpt_dir: Path) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_step{step}.pt"
    torch.save({"model": model.state_dict(), "cfg": model.cfg}, path)
    return path


def _load_checkpoint(path: str, device: str) -> tuple[dict, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Derive step from filename if not stored
    step = ckpt.get("step", 0)
    if step == 0:
        stem = Path(path).stem  # e.g. "ckpt_step1000"
        parts = stem.split("step")
        if len(parts) == 2 and parts[1].isdigit():
            step = int(parts[1])
    return ckpt, step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pretrain(args: argparse.Namespace) -> None:
    # ── Device ──────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # ── Model ────────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume:
        ckpt, global_step = _load_checkpoint(args.resume, str(device))
        from open_mythos.main import OpenMythos as _OM
        model = _OM(ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"[pretrain] resumed from {args.resume} at step {global_step}")
    else:
        variant_fn = _VARIANTS.get(args.variant)
        if variant_fn is None:
            raise ValueError(f"Unknown variant: {args.variant}")
        cfg = variant_fn()
        model = OpenMythos(cfg).to(device)

    model.train()
    _enable_gradient_checkpointing(model)

    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = model.cfg.vocab_size
    # Clip seq_len to model's max_seq_len to avoid RoPE index-out-of-bounds
    seq_len = min(args.seq_len, model.cfg.max_seq_len)
    if seq_len < args.seq_len:
        print(f"[pretrain] seq_len clipped {args.seq_len} → {seq_len} (model max_seq_len)")
    args.seq_len = seq_len
    print(f"[pretrain] variant={args.variant}  params={n_params:,}  device={device}")
    print(f"[pretrain] max_tokens={args.max_tokens:,}  seq_len={args.seq_len}  batch={args.batch}")

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    tokens_per_step = args.batch * args.grad_accum * args.seq_len
    total_steps = max(args.max_tokens // tokens_per_step, 1)
    warmup_steps = max(int(total_steps * 0.02), 10)
    stable_steps = int(total_steps * 0.68)
    decay_steps = total_steps - warmup_steps - stable_steps

    scheduler = warmup_stable_decay_schedule(optimizer, warmup_steps, stable_steps, decay_steps)
    # fast-forward scheduler to resume step
    for _ in range(global_step):
        scheduler.step()

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = TrainLogger(
        backend=args.logger,
        run_name=f"pretrain_{args.variant}",
        config=vars(args),
    )

    # ── Autocast ─────────────────────────────────────────────────────────────
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda
        else torch.amp.autocast(device_type="cpu", dtype=torch.float32)
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = StreamingTokenDataset(
        seq_len=args.seq_len,
        max_tokens=args.max_tokens,
        vocab_size=vocab_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=0,          # Windows + streaming: must be 0
        persistent_workers=False,
    )

    ckpt_dir = Path(args.ckpt_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    consumed_tokens = global_step * tokens_per_step
    accum_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in loader:
        if consumed_tokens >= args.max_tokens:
            break

        # batch: (B, seq_len+1)
        batch = batch.to(device)
        x = batch[:, :-1]   # (B, seq_len)
        y = batch[:, 1:]    # (B, seq_len)

        with autocast_ctx:
            logits = model(x)           # (B, T, V)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )
            loss = loss / args.grad_accum

        loss.backward()
        accum_loss += loss.item()
        consumed_tokens += x.numel()

        # ── Optimizer step (gradient accumulation) ────────────────────────
        if (global_step + 1) % args.grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            elapsed = time.time() - t0
            tok_per_sec = consumed_tokens / max(elapsed, 1e-6)
            lr_now = scheduler.get_last_lr()[0] * args.lr

            print(
                f"step={global_step + 1:>6}  loss={accum_loss:.4f}"
                f"  lr={lr_now:.2e}  tok/s={tok_per_sec:.0f}"
                f"  consumed={consumed_tokens / 1e6:.1f}M"
            )
            logger.log({"train/loss": accum_loss, "train/lr": lr_now}, step=global_step + 1)
            accum_loss = 0.0

        global_step += 1

        # ── Checkpoint ────────────────────────────────────────────────────
        if global_step % args.save_every == 0:
            path = _save_checkpoint(model, global_step, ckpt_dir)
            print(f"[pretrain] saved checkpoint: {path}")

        # ── Perplexity eval ───────────────────────────────────────────────
        if global_step % args.eval_every == 0:
            _run_eval(model, device, global_step, logger, args)

    # ── Final checkpoint ─────────────────────────────────────────────────────
    path = _save_checkpoint(model, global_step, ckpt_dir)
    print(f"[pretrain] training complete. final checkpoint: {path}")
    logger.finish()


def _run_eval(
    model: OpenMythos,
    device: torch.device,
    step: int,
    logger: TrainLogger,
    args: argparse.Namespace,
) -> None:
    """Run WikiText-2 perplexity eval using the existing benchmark module."""
    try:
        from benchmark.perplexity import evaluate_perplexity, _load_corpus, _tokenize_corpus

        _disable_gradient_checkpointing(model)
        model.eval()
        with torch.no_grad():
            text = _load_corpus("wikitext-2-raw-v1", split="validation")
            token_ids = _tokenize_corpus(text, model.cfg.vocab_size)
            result = evaluate_perplexity(
                model, token_ids, str(device),
                seq_len=min(512, args.seq_len),
                stride=256,
                batch_size=1,
            )
        ppl = result["ppl"]
        print(f"[eval] step={step}  PPL={ppl:.2f}")
        logger.log({"eval/ppl": ppl}, step=step)
        model.train()
        _enable_gradient_checkpointing(model)
    except Exception as e:
        print(f"[eval] skipped ({e})")
        model.train()
        _enable_gradient_checkpointing(model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenMythos pre-training on FineWeb-Edu")
    p.add_argument("--variant", default="nano", choices=list(_VARIANTS), help="Model variant")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument("--batch", type=int, default=8, help="Per-device batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    p.add_argument("--max-tokens", type=int, default=300_000_000, dest="max_tokens",
                   help="Stop after consuming this many tokens")
    p.add_argument("--ckpt-dir", default="checkpoints/pretrain", dest="ckpt_dir",
                   help="Directory to save checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seq-len", type=int, default=1024, dest="seq_len",
                   help="Sequence length")
    p.add_argument("--grad-accum", type=int, default=4, dest="grad_accum",
                   help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    p.add_argument("--save-every", type=int, default=1000, dest="save_every",
                   help="Save checkpoint every N steps")
    p.add_argument("--eval-every", type=int, default=500, dest="eval_every",
                   help="Run perplexity eval every N steps")
    p.add_argument("--logger", default="none", choices=["none", "wandb", "mlflow", "tensorboard"],
                   help="Experiment logger backend")
    return p.parse_args()


if __name__ == "__main__":
    pretrain(_parse_args())

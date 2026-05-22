#!/usr/bin/env python3
"""
Side-by-side training + benchmark of OpenMythos, HyperloopMythos, and a vanilla
transformer on a small HuggingFace dataset (TinyStories by default, streamed).

All three models share the same tiny MLA config and see the exact same batches in
the same order, so per-step train loss and throughput are directly comparable.

Models
------
1. Baseline      -- dense stack of TransformerBlock (use_moe=False),
                    depth = prelude + 1 + coda (parameter-matched)
2. OpenMythos    -- flat recurrent-depth transformer, max_loop_iters iterations
3. HyperloopMythos -- 2-level nested loop (outer x inner), same effective depth

What the script measures
------------------------
1. Per-step training loss + tokens/sec for all three models, fed identical batches.
2. Periodic held-out eval loss on a separate dataset split (--eval-every).
3. Depth-extrapolation sweep at the end:
   - OpenMythos: vary n_loops from --depth-sweep
   - HyperloopMythos: vary outer_loops from --hl-depth-sweep (inner fixed at training)
4. Summary table with initial/final/avg train loss, wall-clock, avg tok/s,
   and sec/step for all three models.

Defaults are tuned for a laptop CPU run in reasonable time; pass --device cuda
and bump --steps / --batch-size / --seq-len for a real comparison.

    # Default CPU smoke run (TinyStories, 100 steps, batch 8, seq 64)
    python tests/small_benchmark.py

    # Heavier GPU run
    python tests/small_benchmark.py --steps 5000 --batch-size 64 --seq-len 512 --device cuda

    # Save results to file
    python tests/small_benchmark.py 2>&1 | tee benchmark_results/hyperloop_vs_flat_cpu_$(date +%Y-%m-%d).txt

    # Wikitext instead of TinyStories
    python tests/small_benchmark.py --dataset wikitext --dataset-config wikitext-2-raw-v1

    # Aggressive depth extrapolation sweep
    python tests/small_benchmark.py --depth-sweep 1,2,4,8,16 --hl-depth-sweep 1,2,4,8
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from open_mythos import MythosConfig, OpenMythos
from open_mythos.hyperloop import HyperloopConfig, HyperloopMythos
from open_mythos.main import (
    RMSNorm,
    TransformerBlock,
    precompute_rope_freqs,
)

# ---------------------------------------------------------------------------
# Baseline: dense GQA + SwiGLU transformer
# ---------------------------------------------------------------------------


class BaselineTransformer(nn.Module):
    """Vanilla decoder-only transformer with dense SwiGLU FFNs.

    Reuses OpenMythos's TransformerBlock (attention + FFN kernels are identical)
    so any measured delta reflects the looped recurrent-depth architecture, not
    kernel differences. Supports both attn_type="gqa" and "mla".
    """

    def __init__(self, cfg: MythosConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        # MLA applies RoPE to qk_rope_head_dim only; GQA rotates the full head_dim.
        rope_dim = (
            cfg.qk_rope_head_dim if cfg.attn_type == "mla" else cfg.dim // cfg.n_heads
        )
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(rope_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, T, T), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[1]
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[:T]
        mask = self._causal_mask(T, x.device) if T > 1 else None
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, mask, cache_key=f"layer_{i}")
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# Dataset: tokenize once, pack into fixed-length next-token pairs
# ---------------------------------------------------------------------------


class PackedLMDataset(Dataset):
    """Flatten an HF text dataset into one token buffer, slice fixed-length pairs.

    Accepts either map-style or streaming (`IterableDataset`) HF datasets --
    iteration stops once `max_tokens` are collected, so large corpora like
    TinyStories can be streamed without downloading the whole thing.
    """

    def __init__(
        self,
        hf_ds,
        tokenizer,
        seq_len: int,
        max_tokens: int,
        text_field: str = "text",
    ):
        buf: list[int] = []
        for sample in hf_ds:
            text = sample[text_field]
            if not text or not text.strip():
                continue
            buf.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(buf) >= max_tokens:
                break
        self.seq_len = seq_len
        n_pairs = max(1, (len(buf) - 1) // seq_len)
        buf = buf[: n_pairs * seq_len + 1]
        self.data = torch.tensor(buf, dtype=torch.long)

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        s = idx * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    total_loss: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0
    steps: int = 0
    first_losses: list[float] = field(default_factory=list)
    last_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def update(self, loss: float, tokens: int, seconds: float) -> None:
        self.total_loss += loss
        self.total_tokens += tokens
        self.total_time += seconds
        self.steps += 1
        if len(self.first_losses) < 10:
            self.first_losses.append(loss)
        self.last_losses.append(loss)

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(1, self.steps)

    @property
    def tok_per_sec(self) -> float:
        return self.total_tokens / max(1e-9, self.total_time)

    @property
    def initial_loss(self) -> float:
        return sum(self.first_losses) / max(1, len(self.first_losses))

    @property
    def final_loss(self) -> float:
        return sum(self.last_losses) / max(1, len(self.last_losses))


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    vocab_size: int,
) -> tuple[float, float]:
    """Run one optimizer step; return (loss, wall-clock seconds)."""
    t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return loss.item(), time.perf_counter() - t0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    max_batches: int | None = None,
    n_loops: int | None = None,
) -> float:
    """Mean cross-entropy over (up to `max_batches`) of the loader.

    ``n_loops`` semantics per model type:
    - OpenMythos      -> passed as ``n_loops`` (flat recurrent depth)
    - HyperloopMythos -> passed as ``outer_loops`` (inner_loops kept at training value)
    - BaselineTransformer -> ignored

    This lets the same function benchmark all three models uniformly while
    supporting depth-extrapolation sweeps for both looped architectures.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if isinstance(model, OpenMythos):
            logits = model(x, n_loops=n_loops)
        elif isinstance(model, HyperloopMythos):
            # n_loops is re-interpreted as outer_loops for the hyperloop sweep
            logits = model(x, outer_loops=n_loops)
        else:
            logits = model(x)
        # sum-reduction so we weight by token count, not batch count
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / max(1, total_tokens)


# ---------------------------------------------------------------------------
# Config + utilities
# ---------------------------------------------------------------------------


def build_tiny_cfg(vocab_size: int, seq_len: int) -> MythosConfig:
    """Tiny shared config with MLA attention -- runs in reasonable time on CPU.

    MLA LoRA ranks and head dims scale with ``dim=128`` instead of the
    2048-dim-sized defaults (q_lora_rank=1536, qk_nope_head_dim=128, ...),
    which would otherwise dominate the parameter count at this scale.
    """
    return MythosConfig(
        vocab_size=vocab_size,
        dim=128,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=seq_len,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        v_head_dim=32,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        lora_rank=4,
        rope_theta=10000.0,
        dropout=0.0,
    )


def build_tiny_hyperloop_cfg(
    vocab_size: int,
    seq_len: int,
    outer_loops: int = 2,
    inner_loops: int = 2,
) -> HyperloopConfig:
    """HyperloopConfig for the 3-way comparison.

    Default ``outer_loops=2, inner_loops=2`` gives effective depth 4,
    matching OpenMythos ``max_loop_iters=4``.  Both models therefore see the
    same total number of recurrent iterations while using different structures
    (flat vs. nested), making the comparison fair.

    Parameters inheriting from ``build_tiny_cfg`` are copied verbatim so that
    the attention kernel, MoE layout, and embedding sizes are identical.
    """
    base = build_tiny_cfg(vocab_size, seq_len)
    return HyperloopConfig(
        vocab_size=base.vocab_size,
        dim=base.dim,
        n_heads=base.n_heads,
        n_kv_heads=base.n_kv_heads,
        max_seq_len=base.max_seq_len,
        max_loop_iters=base.max_loop_iters,  # will be overwritten by __post_init__
        prelude_layers=base.prelude_layers,
        coda_layers=base.coda_layers,
        attn_type=base.attn_type,
        kv_lora_rank=base.kv_lora_rank,
        q_lora_rank=base.q_lora_rank,
        qk_rope_head_dim=base.qk_rope_head_dim,
        qk_nope_head_dim=base.qk_nope_head_dim,
        v_head_dim=base.v_head_dim,
        n_experts=base.n_experts,
        n_shared_experts=base.n_shared_experts,
        n_experts_per_tok=base.n_experts_per_tok,
        expert_dim=base.expert_dim,
        lora_rank=base.lora_rank,
        rope_theta=base.rope_theta,
        dropout=base.dropout,
        outer_loops=outer_loops,
        inner_loops=inner_loops,
        outer_lora_rank=0,  # 0 = use lora_rank (same as inner)
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fmt_count(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}T"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    # Defaults point at TinyStories -- simpler vocabulary + shorter documents
    # lets a dim=128 model actually reach a meaningful loss in modest time.
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument(
        "--dataset-config",
        default="",
        help="pass '' for datasets with no config (e.g. TinyStories)",
    )
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument(
        "--train-tokens",
        type=int,
        default=5_000_000,
        help="max tokens to materialize for the training buffer",
    )
    p.add_argument(
        "--eval-tokens",
        type=int,
        default=200_000,
        help="max tokens to materialize for the held-out eval buffer",
    )
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="run held-out eval every N steps (0 disables)",
    )
    p.add_argument("--eval-batches", type=int, default=5)
    p.add_argument(
        "--depth-sweep",
        default="1,2,4",
        help="comma-separated n_loops values for OpenMythos depth-extrapolation eval",
    )
    # HyperloopMythos specific
    p.add_argument(
        "--hl-outer-loops",
        type=int,
        default=2,
        help="outer_loops for HyperloopMythos during training",
    )
    p.add_argument(
        "--hl-inner-loops",
        type=int,
        default=2,
        help="inner_loops for HyperloopMythos during training",
    )
    p.add_argument(
        "--hl-depth-sweep",
        default="1,2,4",
        help="comma-separated outer_loops values for HyperloopMythos depth-extrapolation eval"
        " (inner_loops is kept at the training value)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def load_text_ds(name: str, config: str, split: str):
    """Streaming ``load_dataset`` with optional config (empty string == no config)."""
    if config:
        return load_dataset(name, config, split=split, streaming=True)
    return load_dataset(name, split=split, streaming=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(
        f"[setup] device={device}  batch={args.batch_size}  "
        f"seq_len={args.seq_len}  steps={args.steps}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # AutoTokenizer.vocab_size can be smaller than the head size for BPE
    # tokenizers with added tokens; use len(tokenizer) to be safe.
    vocab_size = len(tokenizer)
    print(f"[setup] tokenizer={args.tokenizer}  vocab_size={vocab_size:,}")

    # ------------------------------------------------------------------
    # Data: streamed train + held-out eval splits
    # ------------------------------------------------------------------
    print(f"[setup] dataset={args.dataset}  config={args.dataset_config or '(none)'}")
    raw_train = load_text_ds(args.dataset, args.dataset_config, args.train_split)
    train_ds = PackedLMDataset(
        raw_train, tokenizer, args.seq_len, args.train_tokens, args.text_field
    )
    raw_eval = load_text_ds(args.dataset, args.dataset_config, args.eval_split)
    eval_ds = PackedLMDataset(
        raw_eval, tokenizer, args.seq_len, args.eval_tokens, args.text_field
    )
    print(
        f"[setup] train tokens={train_ds.data.numel():,}  pairs={len(train_ds)}  |  "
        f"eval tokens={eval_ds.data.numel():,}  pairs={len(eval_ds)}"
    )

    torch.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # ------------------------------------------------------------------
    # Models -- same init seed so all three start from the same embedding
    # ------------------------------------------------------------------
    cfg = build_tiny_cfg(vocab_size, args.seq_len)
    hl_cfg = build_tiny_hyperloop_cfg(
        vocab_size,
        args.seq_len,
        outer_loops=args.hl_outer_loops,
        inner_loops=args.hl_inner_loops,
    )

    torch.manual_seed(args.seed)
    mythos = OpenMythos(cfg).to(device)

    torch.manual_seed(args.seed)
    hyperloop = HyperloopMythos(hl_cfg).to(device)

    # Parameter-matched depth: prelude + one unique recurrent block + coda.
    baseline_layers = cfg.prelude_layers + 1 + cfg.coda_layers
    torch.manual_seed(args.seed)
    baseline = BaselineTransformer(cfg, n_layers=baseline_layers).to(device)

    n_m, n_hl, n_b = (
        count_params(mythos),
        count_params(hyperloop),
        count_params(baseline),
    )
    hl_total_depth = (
        cfg.prelude_layers + hl_cfg.outer_loops * hl_cfg.inner_loops + cfg.coda_layers
    )
    print(
        f"[setup] OpenMythos   params = {fmt_count(n_m)}  ({n_m:,})\n"
        f"[setup] HyperloopMythos params = {fmt_count(n_hl)}  ({n_hl:,})  "
        f"[outer={hl_cfg.outer_loops} x inner={hl_cfg.inner_loops}]\n"
        f"[setup] Baseline     params = {fmt_count(n_b)}  ({n_b:,})  "
        f"[{baseline_layers} layers]"
    )
    print(
        f"[setup] OpenMythos runtime depth   = prelude({cfg.prelude_layers}) + "
        f"loops({cfg.max_loop_iters}) + coda({cfg.coda_layers}) = "
        f"{cfg.prelude_layers + cfg.max_loop_iters + cfg.coda_layers}"
    )
    print(
        f"[setup] HyperloopMythos runtime depth = prelude({cfg.prelude_layers}) + "
        f"outer({hl_cfg.outer_loops})xinner({hl_cfg.inner_loops}) + coda({cfg.coda_layers}) = "
        f"{hl_total_depth}"
    )

    opt_m = torch.optim.AdamW(
        mythos.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    opt_hl = torch.optim.AdamW(
        hyperloop.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    opt_b = torch.optim.AdamW(
        baseline.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    mm, hlm, bm = Metrics(), Metrics(), Metrics()
    # (step, mythos_eval, hyperloop_eval, base_eval)
    eval_history: list[tuple[int, float, float, float]] = []

    header = (
        f"\n{'step':>6} | {'mythos':>10} | {'hyperloop':>10} | {'baseline':>10} | "
        f"{'m tok/s':>9} | {'hl tok/s':>9} | {'b tok/s':>9}"
    )
    print(header)
    print("-" * len(header))

    # ------------------------------------------------------------------
    # Training loop with periodic held-out eval
    # ------------------------------------------------------------------
    data_iter = iter(train_loader)
    t_total = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tokens = x.numel()

        loss_m, dt_m = train_step(mythos, x, y, opt_m, device, vocab_size)
        loss_hl, dt_hl = train_step(hyperloop, x, y, opt_hl, device, vocab_size)
        loss_b, dt_b = train_step(baseline, x, y, opt_b, device, vocab_size)

        mm.update(loss_m, tokens, dt_m)
        hlm.update(loss_hl, tokens, dt_hl)
        bm.update(loss_b, tokens, dt_b)

        if step == 1 or step % args.log_every == 0:
            print(
                f"{step:>6} | {loss_m:>10.4f} | {loss_hl:>10.4f} | {loss_b:>10.4f} | "
                f"{tokens / dt_m:>9,.0f} | {tokens / dt_hl:>9,.0f} | {tokens / dt_b:>9,.0f}"
            )

        if args.eval_every and step % args.eval_every == 0:
            eval_m = evaluate(
                mythos, eval_loader, device, vocab_size, args.eval_batches
            )
            eval_hl = evaluate(
                hyperloop, eval_loader, device, vocab_size, args.eval_batches
            )
            eval_b = evaluate(
                baseline, eval_loader, device, vocab_size, args.eval_batches
            )
            eval_history.append((step, eval_m, eval_hl, eval_b))
            best = min(eval_m, eval_hl, eval_b)
            winner = (
                "mythos"
                if best == eval_m
                else "hyperloop" if best == eval_hl else "baseline"
            )
            print(
                f"  [eval @ step {step}]  "
                f"mythos {eval_m:.4f}  "
                f"hyperloop {eval_hl:.4f}  "
                f"baseline {eval_b:.4f}  "
                f"<- {winner} wins"
            )

    total_wall = time.perf_counter() - t_total

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    bar = "=" * 80
    print(f"\n{bar}")
    print(f"Summary ({args.steps} steps, wall clock {total_wall:.1f}s)")
    print(bar)
    print(f"  {'':<28} {'OpenMythos':>14}   {'HyperloopMythos':>16}   {'Baseline':>12}")
    print(
        f"  {'params':<28} {fmt_count(n_m):>14}   {fmt_count(n_hl):>16}   {fmt_count(n_b):>12}"
    )
    print(
        f"  {'initial train (first 10)':<28} "
        f"{mm.initial_loss:>14.4f}   {hlm.initial_loss:>16.4f}   {bm.initial_loss:>12.4f}"
    )
    print(
        f"  {'final train (last 10)':<28} "
        f"{mm.final_loss:>14.4f}   {hlm.final_loss:>16.4f}   {bm.final_loss:>12.4f}"
    )
    print(
        f"  {'avg train (all steps)':<28} "
        f"{mm.avg_loss:>14.4f}   {hlm.avg_loss:>16.4f}   {bm.avg_loss:>12.4f}"
    )
    print(
        f"  {'train time (sec)':<28} "
        f"{mm.total_time:>14.2f}   {hlm.total_time:>16.2f}   {bm.total_time:>12.2f}"
    )
    print(
        f"  {'avg tok/s':<28} "
        f"{mm.tok_per_sec:>14,.0f}   {hlm.tok_per_sec:>16,.0f}   {bm.tok_per_sec:>12,.0f}"
    )
    print(
        f"  {'sec/step':<28} "
        f"{mm.total_time / max(1, mm.steps):>14.4f}   "
        f"{hlm.total_time / max(1, hlm.steps):>16.4f}   "
        f"{bm.total_time / max(1, bm.steps):>12.4f}"
    )

    # Winner per metric
    print("\n  Quality ranking (lower = better):")
    models_final = [
        ("OpenMythos", mm.final_loss),
        ("HyperloopMythos", hlm.final_loss),
        ("Baseline", bm.final_loss),
    ]
    for rank, (name, loss) in enumerate(sorted(models_final, key=lambda x: x[1]), 1):
        print(f"    #{rank}  {name:<20}  final_train_loss = {loss:.4f}")

    # ------------------------------------------------------------------
    # Depth extrapolation: OpenMythos -- vary n_loops
    # ------------------------------------------------------------------
    loops_sweep = sorted({int(s) for s in args.depth_sweep.split(",") if s.strip()})
    print(f"\n{bar}")
    print("Depth extrapolation -- OpenMythos (vary n_loops, full eval set)")
    print(bar)
    baseline_eval = evaluate(baseline, eval_loader, device, vocab_size)
    print(f"  Baseline (fixed depth)              : eval loss = {baseline_eval:.4f}")
    sweep_m: list[tuple[int, float]] = []
    for nl in loops_sweep:
        sweep_m.append(
            (nl, evaluate(mythos, eval_loader, device, vocab_size, n_loops=nl))
        )
    trained_loss_m = next(
        (loss for nl, loss in sweep_m if nl == cfg.max_loop_iters), None
    )
    print(f"  OpenMythos (trained at n_loops={cfg.max_loop_iters}):")
    print(f"    {'n_loops':>8}  {'eval loss':>10}  {'delta vs trained':>16}")
    for nl, loss in sweep_m:
        if trained_loss_m is None or nl == cfg.max_loop_iters:
            delta_str = ""
        else:
            delta_str = f"{loss - trained_loss_m:+.4f}"
        marker = "  [trained]" if nl == cfg.max_loop_iters else ""
        print(f"    {nl:>8}  {loss:>10.4f}  {delta_str:>16}{marker}")

    # ------------------------------------------------------------------
    # Depth extrapolation: HyperloopMythos -- vary outer_loops
    # ------------------------------------------------------------------
    hl_outer_sweep = sorted(
        {int(s) for s in args.hl_depth_sweep.split(",") if s.strip()}
    )
    print(f"\n{bar}")
    print(
        f"Depth extrapolation -- HyperloopMythos "
        f"(vary outer_loops, inner_loops fixed={hl_cfg.inner_loops}, full eval set)"
    )
    print(bar)
    sweep_hl: list[tuple[int, float]] = []
    for ol in hl_outer_sweep:
        sweep_hl.append(
            (ol, evaluate(hyperloop, eval_loader, device, vocab_size, n_loops=ol))
        )
    trained_loss_hl = next(
        (loss for ol, loss in sweep_hl if ol == hl_cfg.outer_loops), None
    )
    print(
        f"  HyperloopMythos (trained at outer={hl_cfg.outer_loops}xinner={hl_cfg.inner_loops}):"
    )
    print(f"    {'outer_loops':>11}  {'eval loss':>10}  {'delta vs trained':>16}")
    for ol, loss in sweep_hl:
        if trained_loss_hl is None or ol == hl_cfg.outer_loops:
            delta_str = ""
        else:
            delta_str = f"{loss - trained_loss_hl:+.4f}"
        marker = "  [trained]" if ol == hl_cfg.outer_loops else ""
        print(f"    {ol:>11}  {loss:>10.4f}  {delta_str:>16}{marker}")

    # ------------------------------------------------------------------
    # Cross-model comparison at equal effective depth
    # ------------------------------------------------------------------
    print(f"\n{bar}")
    print(
        f"Cross-model comparison at equal effective depth "
        f"(flat n_loops={cfg.max_loop_iters} vs nested {hl_cfg.outer_loops}x{hl_cfg.inner_loops})"
    )
    print(bar)
    eval_m_trained = next(
        (loss for nl, loss in sweep_m if nl == cfg.max_loop_iters),
        evaluate(mythos, eval_loader, device, vocab_size, n_loops=cfg.max_loop_iters),
    )
    eval_hl_trained = next(
        (loss for ol, loss in sweep_hl if ol == hl_cfg.outer_loops),
        evaluate(
            hyperloop, eval_loader, device, vocab_size, n_loops=hl_cfg.outer_loops
        ),
    )
    delta = eval_hl_trained - eval_m_trained
    winner_str = (
        f"HyperloopMythos wins by {-delta:.4f}"
        if delta < 0
        else f"OpenMythos wins by {delta:.4f}" if delta > 0 else "tie"
    )
    print(
        f"  OpenMythos flat (depth {cfg.max_loop_iters})          : {eval_m_trained:.4f}"
    )
    print(
        f"  HyperloopMythos nested ({hl_cfg.outer_loops}x{hl_cfg.inner_loops}=depth {hl_cfg.outer_loops * hl_cfg.inner_loops}): {eval_hl_trained:.4f}"
    )
    print(f"  -> {winner_str}")
    print(
        f"  Baseline (fixed depth {baseline_layers})             : {baseline_eval:.4f}"
    )


if __name__ == "__main__":
    main()

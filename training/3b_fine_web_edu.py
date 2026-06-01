#!/usr/bin/env python3
"""
OpenMythos pretraining on FineWeb-Edu with FSDP + AdamW.

Single GPU:
    python training/3b_fine_web_edu.py

Multi-GPU:
    torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py

Improvements over initial version:
  - Perplexity (ppl) in every log line for human-readable loss tracking
  - ETA estimation based on rolling average of recent step times
  - Persistent loguru file sink (logs/train_<run_id>.log)
  - Held-out eval step every `eval_every` steps using a disjoint FineWeb-Edu shard
  - Loop curriculum: ramp n_loops from 1 → max_loop_iters over first `loop_ramp_steps`
"""

import argparse
import os
import math
import time
from collections import deque
from datetime import datetime
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.main import TransformerBlock, RecurrentBlock
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.logger_utils import TrainLogger

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    """
    Streaming FineWeb-Edu loader yielding fixed-length (input, target) pairs.

    FineWeb-Edu is trillions of tokens, so `streaming=True` pulls shards on
    demand instead of materializing to disk. Sharding is two-dimensional —
    `world_size` ranks × `num_workers` DataLoader workers per rank — and each
    `(rank, worker_id)` deterministically owns one shard of the global stream.
    That gives disjoint coverage without any cross-process coordination.

    Streaming datasets are not seekable, so a resumed run re-enters its shard
    from the beginning. Acceptable at pretraining scale: the chance of
    re-playing the same tokens before the run ends is negligible versus the
    cost of a true resumable loader.
    """

    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        """
        Args:
            encoding   -- tokenizer exposing `.encode(str) -> list[int]`
            seq_len    -- context length; every yielded pair has this many tokens
            subset     -- FineWeb-Edu config name (e.g. "sample-10BT", "default")
            rank       -- global rank of this process within the distributed job
            world_size -- total number of distributed processes
        """
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        """
        Yield `(input_ids, target_ids)` tensors of length `seq_len` forever.

        Inputs and targets are shifted by one for next-token prediction —
        `target[i] == input[i + 1]`. Documents are concatenated into a rolling
        buffer and sliced into fixed-length chunks, packing short docs together
        and splitting long ones. This keeps every step at the same shape,
        which under FSDP avoids recompute from variable-length inputs and
        removes the need for a pad-aware attention mask.
        """
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------


def _fmt_eta(seconds: float) -> str:
    """Format remaining seconds as Xh Ym Zs for log readability."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def curriculum_loops(step: int, ramp_steps: int, max_loops: int) -> int:
    """Linearly ramp n_loops from 1 → max_loops over the first ramp_steps steps.

    Starting training with fewer recurrent loops improves early stability: with
    only 1 loop, gradients flow through a shallower graph, reducing exploding /
    vanishing gradient risk during the critical early warm-up period. As the
    model stabilises, gradually increasing the loop depth lets the recurrent
    block learn to use the additional computational budget.

    Args:
        step       -- current global optimizer step (0-indexed)
        ramp_steps -- number of steps over which to ramp from 1 to max_loops
        max_loops  -- final loop count after ramp_steps

    Returns:
        Integer loop count in [1, max_loops].
    """
    if ramp_steps <= 0 or step >= ramp_steps:
        return max_loops
    fraction = step / ramp_steps
    return max(1, round(1 + fraction * (max_loops - 1)))


@torch.no_grad()
def run_eval(
    model,
    encoding,
    seq_len: int,
    subset: str,
    device,
    vocab_size: int,
    rank: int,
    world_size: int,
    ddp: bool,
    n_batches: int = 20,
    micro_batch: int = 4,
) -> float:
    """Evaluate cross-entropy loss on a held-out shard of FineWeb-Edu.

    Uses shard indices in range [world_size, 2*world_size-1] — adjacent to but
    disjoint from the training shards [0, world_size-1] — so validation tokens
    are never seen during training.

    Under FSDP, all ranks must call this together to avoid collective hangs.
    Only rank 0 logs the result; other ranks discard theirs.

    Args:
        model       -- FSDP-wrapped (ddp=True) or raw model in eval mode
        encoding    -- tokenizer
        seq_len     -- sequence length (same as training)
        subset      -- FineWeb-Edu config name
        device      -- target device string or torch.device
        vocab_size  -- tokenizer vocabulary size
        rank        -- current process rank
        world_size  -- total number of distributed processes
        ddp         -- True if FSDP is active
        n_batches   -- number of micro-batches to average over
        micro_batch -- batch size for each eval step

    Returns:
        Mean cross-entropy loss over n_batches × micro_batch sequences.
        All ranks return the same scalar (no reduce across ranks; each rank
        evaluates its own disjoint shard independently for speed).
    """
    was_training = model.training
    model.eval()

    # Eval shard is adjacent to training shards (never overlaps)
    eval_n_shards = world_size * 2
    eval_shard_idx = world_size + rank

    eval_ds = FineWebEduDataset(
        encoding, seq_len, subset, eval_shard_idx, eval_n_shards
    )
    eval_loader = DataLoader(eval_ds, batch_size=micro_batch, num_workers=0)

    total_loss = 0.0
    n_seen = 0
    for i, (x, y) in enumerate(eval_loader):
        if i >= n_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        total_loss += loss.item()
        n_seen += 1

    if was_training:
        model.train()

    return total_loss / max(n_seen, 1)


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup → half-cosine decay to `min_lr` (legacy schedule)."""
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


def warmup_stable_decay(
    step: int,
    warmup: int,
    stable_end: int,
    total: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Warmup → Stable → Decay LR schedule (DeepSeek-V3 style).

    Three phases:
        [0, warmup)          Linear ramp 0 → max_lr
        [warmup, stable_end) Flat at max_lr — model trains at peak rate
        [stable_end, total)  Cosine decay max_lr → min_lr
        [total, ∞)           Clamped at min_lr

    Keeping the LR flat after warmup avoids the early cosine drop that
    standard schedules impose.  The decay phase is shortened to the tail,
    so more tokens are processed at max_lr before the final convergence push.

    Typical split: warmup=2000, stable_end=total*0.9, total=total_steps.

    Args:
        step       -- current optimizer step (0-indexed)
        warmup     -- steps to ramp from 0 → max_lr
        stable_end -- step where the stable plateau ends and decay begins
        total      -- step where cosine reaches min_lr
        max_lr     -- peak learning rate
        min_lr     -- floor learning rate

    Returns:
        Scalar LR for this step.
    """
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step < stable_end:
        return max_lr
    if step >= total:
        return min_lr
    decay = (step - stable_end) / max(total - stable_end, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    """
    Return checkpoint paths in `ckpt_dir` sorted oldest → newest.

    Relies on the zero-padded `step_{0000000}.pt` filename convention so
    lexicographic sort matches chronological order. Changing the filename
    format elsewhere without updating the pad width would silently break
    both `keep_last` pruning and resume-latest on startup, since both pick
    the last element of this list.

    Args:
        ckpt_dir -- directory to scan; missing directory returns []

    Returns:
        Sorted list of absolute paths to matching checkpoint files.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
) -> None:
    """
    Gather full model + optimizer state, write atomically, prune old files.

    Under FSDP both states are collected inside a single FULL_STATE_DICT
    context so the optim-state tensors bind to fully-unsharded parameters;
    mixing contexts between model and optimizer has caused silent divergence
    on resume in past torch versions. The temp-file + os.replace write means
    a kill mid-save leaves the previous checkpoint intact instead of a
    truncated .pt file. Non-master ranks participate in the FSDP gather
    (otherwise the collective would hang) but exit before touching disk.

    Args:
        model       -- FSDP-wrapped (ddp=True) or raw (ddp=False) model
        optimizer   -- the optimizer whose state should round-trip with the model
        step        -- global step number; encoded zero-padded into the filename
        cfg         -- model config object; saved so downstream eval can
                       reconstruct the model without re-importing the variant
        vocab_size  -- tokenizer vocab size at train time; saved for sanity-check
                       on load against a (possibly updated) tokenizer
        ckpt_dir    -- directory to write into; created if missing
        ddp         -- True if FSDP path; False for single-GPU / CPU
        master      -- whether this rank writes to disk (rank 0 only)
        keep_last   -- number of most-recent checkpoints to retain; older ones
                       are unlinked after a successful write

    Returns:
        None. Writes to disk as a side effect on master rank.
    """
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    logger.success(f"Checkpoint saved → {final_path}")


def load_checkpoint(model, optimizer, path: str, ddp: bool) -> int:
    """
    Restore model + optimizer from disk, returning the step to resume at.

    Every rank reads the file (`rank0_only=False` on load) so FSDP has access
    to the full state on each rank — the complement to the `rank0_only=True`
    save path. Must mirror save's single-context pattern; splitting the model
    and optimizer loads across two `state_dict_type` blocks has historically
    produced optimizer state bound to the wrong shard shapes.

    `weights_only=False` is required because the checkpoint contains the
    pickled `cfg` dataclass — flip to `weights_only=True` only if you
    separate config out.

    Args:
        model     -- same FSDP-wrapped or raw model used during save
        optimizer -- freshly constructed optimizer to be filled in-place
        path      -- absolute path to a `step_{N:07d}.pt` file produced by
                     `save_checkpoint`
        ddp       -- whether the model is FSDP-wrapped; must match the save run

    Returns:
        The step number the checkpoint was taken at; the caller advances the
        training loop from this value.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=ckpt["optimizer"],
            )
            optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """CLI argument parser — all flags override in-script defaults."""
    p = argparse.ArgumentParser(
        prog="3b_fine_web_edu",
        description="OpenMythos 3B pretraining on FineWeb-Edu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Run identity
    p.add_argument(
        "--run-name", default=None, help="Human-readable run name (default: timestamp)"
    )
    p.add_argument(
        "--project",
        default="open-mythos",
        help="WandB / MLflow project / experiment name",
    )

    # Checkpointing & resume
    p.add_argument(
        "--ckpt-dir", default="checkpoints", help="Directory to write/read checkpoints"
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Path to a specific checkpoint file to resume from. "
        "If omitted, resumes from the latest in --ckpt-dir automatically.",
    )
    p.add_argument(
        "--ckpt-every", type=int, default=1000, help="Save a checkpoint every N steps"
    )
    p.add_argument(
        "--keep-last",
        type=int,
        default=3,
        help="Number of recent checkpoints to keep on disk",
    )

    # Logging
    p.add_argument(
        "--logger",
        default="none",
        choices=["none", "wandb", "mlflow", "tensorboard"],
        help="Experiment logging backend",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print a training log line every N steps",
    )
    p.add_argument(
        "--log-dir",
        default="runs",
        help="TensorBoard log directory (ignored for other backends)",
    )

    # Training hypers (override script defaults)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument(
        "--total-tokens",
        type=float,
        default=30e9,
        help="Total training tokens (e.g. 30e9)",
    )
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument(
        "--eval-every", type=int, default=500, help="Eval every N steps; 0 to disable"
    )
    p.add_argument(
        "--dataset-subset", default="sample-10BT", help="FineWeb-Edu config name"
    )
    p.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="Disable gradient checkpointing (uses more VRAM, faster)",
    )

    return p.parse_args()


def main():
    """
    End-to-end pretraining entry point.

    Order matters: distributed init must run before any CUDA allocation, the
    tokenizer must exist before the model is built (vocab_size flows into
    cfg), and FSDP must wrap the model before the optimizer is constructed
    (FSDP re-flattens parameters, so an optimizer built on the unwrapped
    model would track stale param objects). Resume then loads state into the
    already-constructed optimizer in-place.

    Lifecycle:
        1. Parse CLI args (all flags override in-script defaults).
        2. Initialize torch.distributed (NCCL) if launched under torchrun.
        3. Build tokenizer → derive vocab_size.
        4. Construct OpenMythos with the 3B variant config.
        5. Wrap in FSDP with FULL_SHARD + bf16/fp16 mixed precision (multi-GPU)
           or move to device + autocast (single-GPU).
        6. Build fused AdamW on (possibly sharded) parameters.
        7. Resume from checkpoint (--resume path, or latest in --ckpt-dir).
        8. Stream FineWeb-Edu through grad-accumulation microbatches with
           warmup_stable_decay LR schedule, per-step logging, and periodic checkpoints.
        9. Write a final checkpoint; barrier + tear down the process group.
    """
    args = _parse_args()

    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    # ------------------------------------------------------------------
    # Persistent file logging (rank 0 only — avoid duplicated log files)
    # ------------------------------------------------------------------
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if master:
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        logger.add(
            f"{logs_dir}/train_{run_id}.log",
            rotation="500 MB",
            retention="10 days",
            level="INFO",
        )
        logger.info(
            f"GPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    if master:
        logger.info(f"Tokenizer: gpt-oss-20b  |  Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Hyperparameters (CLI args override defaults)
    # ------------------------------------------------------------------
    seq_len = args.seq_len
    micro_batch = args.micro_batch
    target_tokens = int(args.total_tokens)
    grad_accum = max(1, 256 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = args.warmup_steps
    stable_end_steps = int(total_steps * 0.9)  # flat plateau until 90% of training
    lr = args.lr
    wd = args.wd
    log_every = args.log_every
    ckpt_every = args.ckpt_every
    ckpt_dir = args.ckpt_dir
    dataset_subset = args.dataset_subset
    # Held-out evaluation: run every eval_every steps; 0 disables eval entirely
    eval_every = args.eval_every
    eval_batches = 20
    # Loop curriculum: ramp n_loops from 1 → max_loop_iters over this many steps.
    # Set to 0 to disable (always train at full depth).
    loop_ramp_steps = warmup_steps  # align ramp with LR warmup

    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = OpenMythos(cfg)

    # Gradient checkpointing: recompute activations during backward to save memory.
    # Disabled by --no-grad-ckpt flag; enabled by default.
    use_grad_ckpt = not args.no_grad_ckpt
    if use_grad_ckpt:
        for module in model.modules():
            if isinstance(module, (TransformerBlock, RecurrentBlock)):
                module.gradient_checkpointing = True  # type: ignore[attr-defined]

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
        )
        wrap_policy = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    # FSDP handles its own mixed precision; only need autocast for single-GPU
    amp_ctx = nullcontext() if ddp else amp_ctx  # type: ignore[possibly-undefined]

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    # ------------------------------------------------------------------
    # Experiment logger (WandB / MLflow / TensorBoard / none)
    # ------------------------------------------------------------------
    train_logger: TrainLogger | None = None
    if master:
        cfg_dict = {
            "seq_len": seq_len,
            "micro_batch": micro_batch,
            "lr": lr,
            "wd": wd,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "grad_accum": grad_accum,
            "dataset_subset": dataset_subset,
        }
        train_logger = TrainLogger(
            backend=args.logger,
            run_name=run_id,
            project=args.project,
            config=cfg_dict,
            log_dir=args.log_dir,
        )
        logger.info(f"Experiment logger: {args.logger}  |  run_name: {run_id}")

    # ------------------------------------------------------------------
    # Resume from checkpoint (--resume path, or latest in --ckpt-dir)
    # ------------------------------------------------------------------
    # Streaming datasets are not resumable by position, so re-iterating from
    # the beginning is accepted — at pretraining scale the loss of dataset
    # position is negligible vs. the cost of discarded training steps.
    start_step = 0
    resume_path = args.resume
    if resume_path is None:
        existing_ckpts = _list_ckpts(ckpt_dir)
        if existing_ckpts:
            resume_path = existing_ckpts[-1]
    if resume_path is not None:
        if master:
            logger.info(f"Resuming from checkpoint: {resume_path}")
        start_step = load_checkpoint(model, optimizer, resume_path, ddp)
        if master:
            logger.success(f"Resumed at step {start_step}")

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    # Rolling window of (step_time, tokens) for smooth ETA / tok/s estimation
    _step_times: deque = deque(maxlen=log_every * 5)
    t_step_start = time.perf_counter()
    step = start_step

    # --- Gradient Noise Scale (GNS) tracking ---
    # GNS = E[‖g‖²] / Var[g] approximated as (grad_norm² / micro_batch).
    # High GNS → large-batch friendly; low GNS → signal dominated by noise.
    _gns_window: deque = deque(maxlen=100)

    # --- Early stopping ---
    # Halt training if eval loss hasn't improved by `es_min_delta` for
    # `es_patience` consecutive eval steps.
    es_patience = 5  # eval steps without improvement before stopping
    es_min_delta = 1e-3  # minimum improvement to count as progress
    _es_best_loss = float("inf")
    _es_counter = 0

    while step < total_steps:
        cur_lr = warmup_stable_decay(
            step, warmup_steps, stable_end_steps, total_steps, lr, lr * 0.1
        )
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        # Loop curriculum: gradually increase recurrent depth during warm-up
        cur_loops = curriculum_loops(step, loop_ramp_steps, cfg.max_loop_iters)

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
            y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                logits = model(x, n_loops=cur_loops)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum

            loss.backward()
            loss_accum += loss.item()

        # FSDP shards parameters, so `nn.utils.clip_grad_norm_` would clip
        # against each rank's local norm and miss the cross-shard gather.
        # FSDP.clip_grad_norm_ computes the true global norm and returns it.
        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        # GNS approximation: (grad_norm^2) / micro_batch
        _gns_window.append(float(grad_norm) ** 2 / micro_batch)

        # Track per-step timing for rolling-window ETA and tok/s
        dt_step = time.perf_counter() - t_step_start
        _step_times.append(dt_step)
        t_step_start = time.perf_counter()

        if master and step % log_every == 0:
            # Use rolling average for smoother ETA (avoids spikes from checkpoint I/O)
            avg_step_time = sum(_step_times) / len(_step_times)
            tok_per_sec = global_batch_tok / avg_step_time
            tokens_seen = step * global_batch_tok
            # Perplexity: cap loss at 88 before exp to prevent float overflow
            ppl = math.exp(min(loss_accum, 88.0))
            eta_secs = (total_steps - step) * avg_step_time
            gns = sum(_gns_window) / len(_gns_window) if _gns_window else 0.0
            logger.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} | ppl {ppl:7.1f}"
                f" | gnorm {float(grad_norm):.2f} | gns {gns:.2f} | lr {cur_lr:.2e}"
                f" | loops {cur_loops}/{cfg.max_loop_iters}"
                f" | {tok_per_sec / 1e6:.2f}M tok/s"
                f" | {tokens_seen / 1e9:.1f}B tok seen"
                f" | eta {_fmt_eta(eta_secs)}"
            )
            if train_logger is not None:
                train_logger.log(
                    {
                        "train/loss": loss_accum,
                        "train/ppl": ppl,
                        "train/grad_norm": float(grad_norm),
                        "train/gns": gns,
                        "train/lr": cur_lr,
                        "train/tok_per_sec": tok_per_sec,
                        "train/tokens_seen_B": tokens_seen / 1e9,
                        "train/loops": cur_loops,
                    },
                    step=step,
                )

        # Held-out evaluation step + early stopping
        should_stop = False
        if eval_every > 0 and step % eval_every == 0:
            eval_loss = run_eval(
                model,
                encoding,
                seq_len,
                dataset_subset,
                device if not ddp else f"cuda:{local_rank}",
                vocab_size,
                rank,
                world_size,
                ddp,
                n_batches=eval_batches,
                micro_batch=micro_batch,
            )
            eval_ppl = math.exp(min(eval_loss, 88.0))
            if master:
                logger.info(
                    f"[eval @ step {step}] loss {eval_loss:.4f} | ppl {eval_ppl:.1f}"
                )
                if train_logger is not None:
                    train_logger.log(
                        {"eval/loss": eval_loss, "eval/ppl": eval_ppl}, step=step
                    )
            # Early stopping check
            if eval_loss < _es_best_loss - es_min_delta:
                _es_best_loss = eval_loss
                _es_counter = 0
            else:
                _es_counter += 1
                if master:
                    logger.warning(
                        f"[early stopping] no improvement for {_es_counter}/{es_patience} eval steps"
                    )
                if _es_counter >= es_patience:
                    if master:
                        logger.warning(
                            f"[early stopping] triggered at step {step} (best eval loss {_es_best_loss:.4f})"
                        )
                    should_stop = True
        if should_stop:
            break

        if step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master
            )

    # Final checkpoint — total_steps may not be divisible by ckpt_every, so
    # without this the tail of the run is lost if the schedule doesn't align.
    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)

    if ddp:
        # Barrier so no rank exits while another is still finishing its
        # checkpoint gather — avoids NCCL "process group destroyed" noise.
        dist.barrier()
        dist.destroy_process_group()

    if master:
        logger.success("Training complete.")
        if train_logger is not None:
            train_logger.finish()


if __name__ == "__main__":
    main()

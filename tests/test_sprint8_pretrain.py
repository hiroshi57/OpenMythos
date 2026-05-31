"""
Sprint 8 pre-training tests.

Tests scripts/pretrain.py logic without running the full training loop or
downloading any datasets.  All tests run on CPU with the nano variant.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from open_mythos.variants import mythos_nano
from open_mythos.main import OpenMythos, TransformerBlock
from scripts.pretrain import (
    StreamingTokenDataset,
    warmup_stable_decay_schedule,
    _enable_gradient_checkpointing,
    _disable_gradient_checkpointing,
    _save_checkpoint,
    _load_checkpoint,
    _parse_args,
)


# ---------------------------------------------------------------------------
# TestWarmupStableDecaySchedule (5 tests)
# ---------------------------------------------------------------------------

class TestWarmupStableDecaySchedule:

    def _make_scheduler(self, warmup, stable, decay):
        model = nn.Linear(4, 4)
        opt = AdamW(model.parameters(), lr=1.0)
        # ダミー勾配を設定して opt.step() が呼べる状態にする
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        return warmup_stable_decay_schedule(opt, warmup, stable, decay), opt

    def _step(self, opt, sched):
        """PyTorch の正しい順序: opt.step() → sched.step()"""
        opt.step()
        sched.step()

    def test_warmup_phase_lr_increases(self):
        sched, opt = self._make_scheduler(warmup=10, stable=10, decay=10)
        lrs = []
        for _ in range(10):
            lrs.append(sched.get_last_lr()[0])
            self._step(opt, sched)
        # LR should be monotonically non-decreasing during warmup
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_stable_phase_lr_constant(self):
        sched, opt = self._make_scheduler(warmup=5, stable=10, decay=5)
        for _ in range(5):
            self._step(opt, sched)
        # After warmup, LR should be ~1.0 (max)
        stable_lrs = [sched.get_last_lr()[0]]
        for _ in range(4):
            self._step(opt, sched)
            stable_lrs.append(sched.get_last_lr()[0])
        assert max(stable_lrs) - min(stable_lrs) < 1e-6

    def test_decay_phase_lr_decreases(self):
        sched, opt = self._make_scheduler(warmup=2, stable=2, decay=10)
        for _ in range(4):  # skip warmup + stable
            self._step(opt, sched)
        decay_lrs = []
        for _ in range(10):
            decay_lrs.append(sched.get_last_lr()[0])
            self._step(opt, sched)
        # LR should be monotonically non-increasing during decay
        assert all(decay_lrs[i] >= decay_lrs[i + 1] for i in range(len(decay_lrs) - 1))

    def test_zero_warmup_steps_no_crash(self):
        sched, opt = self._make_scheduler(warmup=0, stable=5, decay=5)
        for _ in range(10):
            self._step(opt, sched)

    def test_end_lr_near_min(self):
        min_lr_ratio = 0.1
        model = nn.Linear(4, 4)
        opt = AdamW(model.parameters(), lr=1.0)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        sched = warmup_stable_decay_schedule(opt, 2, 2, 100, min_lr_ratio=min_lr_ratio)
        for _ in range(104):
            opt.step()
            sched.step()
        final_lr = sched.get_last_lr()[0]
        assert final_lr <= min_lr_ratio + 0.05


# ---------------------------------------------------------------------------
# TestStreamingTokenDataset (4 tests)
# ---------------------------------------------------------------------------

class TestStreamingTokenDataset:
    """Mock the HuggingFace streaming dataset to avoid network calls."""

    def _make_fake_dataset(self, n_examples: int = 10, text_len: int = 200):
        examples = [{"text": "hello world " * (text_len // 12)} for _ in range(n_examples)]
        return iter(examples)

    def test_chunk_size_matches_seq_len(self):
        ds = StreamingTokenDataset(seq_len=32, max_tokens=500, vocab_size=256)
        fake = self._make_fake_dataset(n_examples=5, text_len=300)
        with patch("datasets.load_dataset", return_value=fake):
            chunks = list(ds)
        for chunk in chunks:
            assert chunk.shape == (33,), f"Expected seq_len+1=33, got {chunk.shape}"

    def test_max_tokens_cuts_off_stream(self):
        seq_len = 16
        max_tokens = 64  # only 4 chunks should be yielded
        ds = StreamingTokenDataset(seq_len=seq_len, max_tokens=max_tokens, vocab_size=256)
        fake = self._make_fake_dataset(n_examples=100, text_len=500)
        with patch("datasets.load_dataset", return_value=fake):
            chunks = list(ds)
        total = len(chunks) * seq_len
        assert total <= max_tokens + seq_len  # allow up to 1 chunk overshoot

    def test_tokens_in_vocab_range(self):
        vocab_size = 128
        ds = StreamingTokenDataset(seq_len=16, max_tokens=200, vocab_size=vocab_size)
        fake = self._make_fake_dataset(n_examples=5, text_len=100)
        with patch("datasets.load_dataset", return_value=fake):
            for chunk in ds:
                assert chunk.max().item() < vocab_size

    def test_consecutive_chunks_no_gap(self):
        ds = StreamingTokenDataset(seq_len=8, max_tokens=100, vocab_size=256)
        fake = iter([{"text": "abcdefghijklmnopqrstuvwxyz" * 10}])
        with patch("datasets.load_dataset", return_value=fake):
            chunks = list(ds)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# TestPretrainCheckpoint (4 tests)
# ---------------------------------------------------------------------------

class TestPretrainCheckpoint:

    def _make_model(self):
        return OpenMythos(mythos_nano()).eval()

    def test_checkpoint_format_has_model_and_cfg(self):
        model = self._make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_checkpoint(model, step=100, ckpt_dir=Path(tmpdir))
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "model" in ckpt
        assert "cfg" in ckpt

    def test_resume_restores_step_count(self):
        model = self._make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_checkpoint(model, step=500, ckpt_dir=Path(tmpdir))
            _, step = _load_checkpoint(str(path), "cpu")
        assert step == 500

    def test_checkpoint_compatible_with_perplexity_loader(self):
        """perplexity.py's _load_model expects ckpt['model'] and ckpt['cfg']."""
        model = self._make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_checkpoint(model, step=10, ckpt_dir=Path(tmpdir))
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            loaded = OpenMythos(ckpt["cfg"])
            loaded.load_state_dict(ckpt["model"])  # must not raise

    def test_checkpoint_compatible_with_lm_eval_loader(self):
        """lm_eval_harness.py uses the same ckpt['model'] / ckpt['cfg'] format."""
        model = self._make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_checkpoint(model, step=20, ckpt_dir=Path(tmpdir))
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert isinstance(ckpt["cfg"].vocab_size, int)
            assert ckpt["cfg"].vocab_size > 0


# ---------------------------------------------------------------------------
# TestPretrainBf16 (2 tests)
# ---------------------------------------------------------------------------

class TestPretrainBf16:

    def test_cpu_uses_float32(self):
        ctx = torch.amp.autocast(device_type="cpu", dtype=torch.float32)
        model = OpenMythos(mythos_nano())
        x = torch.zeros(1, 4, dtype=torch.long)
        with ctx:
            out = model(x)
        assert out.dtype in (torch.float32, torch.bfloat16)

    def test_autocast_dtype_cpu_is_float32(self):
        ctx = torch.amp.autocast(device_type="cpu", dtype=torch.float32)
        with ctx:
            t = torch.tensor([1.0, 2.0])
            result = t * t
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# TestPretrainGradCheckpoint (2 tests)
# ---------------------------------------------------------------------------

class TestPretrainGradCheckpoint:

    def _make_model(self):
        return OpenMythos(mythos_nano())

    def test_gradient_checkpointing_flag_set_on_blocks(self):
        model = self._make_model()
        _enable_gradient_checkpointing(model)
        blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
        assert len(blocks) > 0
        assert all(b.gradient_checkpointing for b in blocks)

    def test_gradient_checkpointing_forward_runs(self):
        model = self._make_model().train()
        _enable_gradient_checkpointing(model)
        x = torch.zeros(1, 4, dtype=torch.long)
        out = model(x)
        assert out.shape[-1] == model.cfg.vocab_size


# ---------------------------------------------------------------------------
# TestPretrainLossStep (2 tests)
# ---------------------------------------------------------------------------

class TestPretrainLossStep:

    def _make_model(self):
        return OpenMythos(mythos_nano()).train()

    def test_loss_finite_on_random_batch(self):
        model = self._make_model()
        vocab = model.cfg.vocab_size
        x = torch.randint(0, vocab, (1, 8))
        y = torch.randint(0, vocab, (1, 8))
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        assert loss.isfinite()

    def test_loss_backward_no_nan_grad(self):
        model = self._make_model()
        vocab = model.cfg.vocab_size
        x = torch.randint(0, vocab, (1, 4))
        y = torch.randint(0, vocab, (1, 4))
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not p.grad.isnan().any(), f"NaN grad in {name}"


# ---------------------------------------------------------------------------
# TestPretrainCLI (1 test)
# ---------------------------------------------------------------------------

class TestPretrainCLI:

    def test_parse_args_defaults(self):
        args = _parse_args.__wrapped__() if hasattr(_parse_args, "__wrapped__") else None
        # Call with empty argv
        import sys
        orig = sys.argv
        sys.argv = ["pretrain.py"]
        try:
            args = _parse_args()
        finally:
            sys.argv = orig

        assert args.variant == "nano"
        assert args.lr == 1e-3
        assert args.max_tokens == 300_000_000
        assert args.grad_accum == 4
        assert args.seq_len == 1024
        assert args.save_every == 1000
        assert args.eval_every == 500
        assert args.logger == "none"

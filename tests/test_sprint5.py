"""
Sprint 5 feature tests.

Covers:
  5.1.1  warmup_stable_decay LR schedule
  5.1.2  MuP weight initialisation (init_mup)
  5.1.3  enable_lora_finetuning / trainable_parameters
  5.2.1  CLI — mythos generate / mythos info (subprocess smoke tests)
  5.2.2  push_to_hub / from_pretrained — local round-trip (no network)
  5.3.1  enable_lora_finetuning parameter counts
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types

import pytest
import torch
import torch.nn as nn

from open_mythos import OpenMythos
from open_mythos.variants import mythos_nano


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nano() -> OpenMythos:
    return OpenMythos(mythos_nano())


def _batch(cfg, b: int = 2, s: int = 8):
    x = torch.randint(0, cfg.vocab_size, (b, s))
    y = torch.randint(0, cfg.vocab_size, (b, s))
    return x, y


# ---------------------------------------------------------------------------
# 5.1.1  warmup_stable_decay
# ---------------------------------------------------------------------------

def _load_warmup_stable_decay():
    """Extract warmup_stable_decay from the training script via ast+compile.

    The training script has top-level FSDP/dataset imports that break on CPU;
    we extract only the target function's source without executing the rest.
    """
    import ast, inspect, math, pathlib, textwrap, types

    src = (pathlib.Path(__file__).parents[1] / "training" / "3b_fine_web_edu.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "warmup_stable_decay":
            fn_src = ast.get_source_segment(src, node)
            ns: dict = {"math": math}
            exec(compile(textwrap.dedent(fn_src), "<warmup_stable_decay>", "exec"), ns)
            return ns["warmup_stable_decay"]
    raise RuntimeError("warmup_stable_decay not found in training script")


class TestWarmupStableDecay:
    """Validate warmup_stable_decay schedule shape."""

    @pytest.fixture(autouse=True)
    def _import(self):
        self.fn = _load_warmup_stable_decay()

    def test_zero_at_step_zero(self):
        lr = self.fn(0, warmup=100, stable_end=900, total=1000, max_lr=1e-3, min_lr=1e-4)
        assert lr == pytest.approx(0.0, abs=1e-9)

    def test_max_at_end_of_warmup(self):
        lr = self.fn(100, warmup=100, stable_end=900, total=1000, max_lr=1e-3, min_lr=1e-4)
        assert lr == pytest.approx(1e-3, rel=1e-5)

    def test_flat_plateau(self):
        lr_mid = self.fn(500, warmup=100, stable_end=900, total=1000, max_lr=1e-3, min_lr=1e-4)
        assert lr_mid == pytest.approx(1e-3, rel=1e-5)

    def test_min_at_end_of_decay(self):
        lr = self.fn(1000, warmup=100, stable_end=900, total=1000, max_lr=1e-3, min_lr=1e-4)
        assert lr == pytest.approx(1e-4, abs=1e-7)

    def test_monotone_during_decay(self):
        steps = range(900, 1001)
        lrs = [self.fn(s, warmup=100, stable_end=900, total=1000, max_lr=1e-3, min_lr=1e-4) for s in steps]
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


# ---------------------------------------------------------------------------
# 5.1.2  init_mup
# ---------------------------------------------------------------------------

class TestInitMup:
    def test_returns_self(self):
        m = _nano()
        assert m.init_mup(base_dim=128) is m

    def test_embed_std_near_one(self):
        m = _nano().init_mup(base_dim=128)
        # Token embedding should be initialised with std≈1.0
        emb_w = m.embed.weight
        assert 0.5 < emb_w.std().item() < 2.0

    def test_head_weight_tied_to_embed(self):
        m = _nano().init_mup(base_dim=128)
        # Weight tying: head and embed share the same tensor
        assert m.head.weight.data_ptr() == m.embed.weight.data_ptr()

    def test_forward_still_works_after_mup(self):
        cfg = mythos_nano()
        m = OpenMythos(cfg).init_mup(base_dim=128).eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        logits = m(x)
        assert logits.shape == (1, 8, cfg.vocab_size)
        assert not torch.isnan(logits).any()


# ---------------------------------------------------------------------------
# 5.3.1  enable_lora_finetuning / trainable_parameters
# ---------------------------------------------------------------------------

class TestLoraFinetuning:
    def test_returns_self(self):
        m = _nano()
        assert m.enable_lora_finetuning() is m

    def test_only_lora_params_trainable(self):
        m = _nano().enable_lora_finetuning()
        for name, p in m.named_parameters():
            if ".lora." in name:
                assert p.requires_grad, f"LoRA param not trainable: {name}"
            else:
                assert not p.requires_grad, f"Non-LoRA param still trainable: {name}"

    def test_trainable_parameters_yields_lora_only(self):
        m = _nano().enable_lora_finetuning()
        tp = list(m.trainable_parameters())
        assert len(tp) > 0, "No trainable parameters after enable_lora_finetuning"
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in tp)
        assert trainable < total, "LoRA params should be a subset of all params"

    def test_lora_grad_flows_in_backward(self):
        cfg = mythos_nano()
        m = OpenMythos(cfg).enable_lora_finetuning().train()
        opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-3)
        x, y = _batch(cfg)
        logits = m(x)
        loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        for name, p in m.named_parameters():
            if ".lora." in name and p.requires_grad:
                assert p.grad is not None and not torch.isnan(p.grad).any(), \
                    f"Bad grad for LoRA param {name}"

    def test_non_lora_params_have_no_grad(self):
        cfg = mythos_nano()
        m = OpenMythos(cfg).enable_lora_finetuning().train()
        x, y = _batch(cfg)
        logits = m(x)
        loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        for name, p in m.named_parameters():
            if ".lora." not in name:
                assert p.grad is None, f"Frozen param has grad: {name}"

    def test_lora_param_count_small_fraction(self):
        m = _nano().enable_lora_finetuning()
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.trainable_parameters())
        ratio = trainable / total
        assert ratio < 0.10, f"LoRA fraction unexpectedly large: {ratio:.2%}"


# ---------------------------------------------------------------------------
# 5.2.2  from_pretrained round-trip (local, no network)
# ---------------------------------------------------------------------------

class TestFromPretrainedLocal:
    def test_load_from_pt_file(self):
        cfg = mythos_nano()
        m = OpenMythos(cfg).eval()
        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "model.pt")
            torch.save({"model": m.state_dict(), "cfg": cfg}, ckpt_path)
            m2 = OpenMythos.from_pretrained(ckpt_path)
        assert isinstance(m2, OpenMythos)

    def test_load_from_directory(self):
        cfg = mythos_nano()
        m = OpenMythos(cfg).eval()
        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "model.pt")
            torch.save({"model": m.state_dict(), "cfg": cfg}, ckpt_path)
            m2 = OpenMythos.from_pretrained(d)
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1 = m(x)
            out2 = m2(x)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_outputs_match_original(self):
        cfg = mythos_nano()
        m = OpenMythos(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "model.pt")
            torch.save({"model": m.state_dict(), "cfg": cfg}, ckpt_path)
            m2 = OpenMythos.from_pretrained(ckpt_path)
        with torch.no_grad():
            assert torch.allclose(m(x), m2(x), atol=1e-5)


# ---------------------------------------------------------------------------
# 5.2.1  CLI smoke tests (subprocess)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_info_nano(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "open_mythos.cli", "info", "--variant", "nano"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "Parameters" in result.stdout
        assert "mythos_nano" in result.stdout or "nano" in result.stdout

    def test_generate_nano_no_crash(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "open_mythos.cli", "generate",
             "--variant", "nano", "--prompt", "Hello", "--max-tokens", "5"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=120,
        )
        assert result.returncode == 0

    def test_generate_stream_no_crash(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "open_mythos.cli", "generate",
             "--variant", "nano", "--prompt", "test", "--max-tokens", "3", "--stream"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=120,
        )
        assert result.returncode == 0

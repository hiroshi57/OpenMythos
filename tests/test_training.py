"""
Training smoke tests — verify 1-step training on CPU using mythos_nano.

These tests do NOT require a GPU or the FineWeb-Edu dataset.
They construct a minimal synthetic batch and run a single optimizer step
to confirm gradient flow, loss decrease, and checkpoint round-trip.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from open_mythos import OpenMythos
from open_mythos.variants import mythos_nano


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nano_model() -> OpenMythos:
    cfg = mythos_nano()
    return OpenMythos(cfg).train()


def _random_batch(cfg, batch: int = 2, seq: int = 16):
    """Return (input_ids, target_ids) on CPU."""
    x = torch.randint(0, cfg.vocab_size, (batch, seq))
    y = torch.randint(0, cfg.vocab_size, (batch, seq))
    return x, y


# ---------------------------------------------------------------------------
# 4.1.3 — Training smoke tests
# ---------------------------------------------------------------------------


class TestTrainingSmokeNano:
    def setup_method(self):
        self.cfg = mythos_nano()
        self.model = OpenMythos(self.cfg).train()
        self.x, self.y = _random_batch(self.cfg)

    def test_forward_returns_logits(self):
        logits = self.model(self.x)
        assert logits.shape == (2, 16, self.cfg.vocab_size)

    def test_loss_is_finite(self):
        logits = self.model(self.x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.cfg.vocab_size), self.y.view(-1)
        )
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_backward_no_nan_gradients(self):
        logits = self.model(self.x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.cfg.vocab_size), self.y.view(-1)
        )
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"

    def test_one_optimizer_step_decreases_loss(self):
        """After one AdamW step the loss on the same batch must drop."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        logits0 = self.model(self.x)
        loss0 = nn.functional.cross_entropy(
            logits0.view(-1, self.cfg.vocab_size), self.y.view(-1)
        )
        loss0.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            logits1 = self.model(self.x)
            loss1 = nn.functional.cross_entropy(
                logits1.view(-1, self.cfg.vocab_size), self.y.view(-1)
            )
        assert loss1.item() < loss0.item()

    def test_gradient_checkpointing_forward(self):
        """Gradient checkpointing must not break forward/backward."""
        from open_mythos.main import TransformerBlock, RecurrentBlock
        for module in self.model.modules():
            if isinstance(module, (TransformerBlock, RecurrentBlock)):
                module.gradient_checkpointing = True

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        logits = self.model(self.x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.cfg.vocab_size), self.y.view(-1)
        )
        loss.backward()
        optimizer.step()
        assert not torch.isnan(loss)

    def test_checkpoint_save_load_round_trip(self):
        """Save model state, reload into a fresh model, verify outputs match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nano.pt")
            torch.save(self.model.state_dict(), path)

            model2 = OpenMythos(self.cfg).eval()
            model2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

            self.model.eval()
            with torch.no_grad():
                out1 = self.model(self.x)
                out2 = model2(self.x)
            assert torch.allclose(out1, out2, atol=1e-5)

    def test_n_loops_override(self):
        """Passing n_loops=1 must complete without error (curriculum loop entry)."""
        logits = self.model(self.x, n_loops=1)
        assert logits.shape == (2, 16, self.cfg.vocab_size)

    def test_mythos_nano_param_count(self):
        """Nano config must be small enough to instantiate comfortably on CPU."""
        n = sum(p.numel() for p in self.model.parameters())
        assert n < 50_000_000, f"mythos_nano is unexpectedly large: {n:,} params"

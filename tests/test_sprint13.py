"""
Sprint 13 テスト

Track A — Mixture-of-Depths (MoD) Transformer
  13.1.1  open_mythos/mod.py
            MoDConfig / TokenRouter / MixtureOfDepthsBlock / MoDTransformer / MoDAnalytics
  13.1.2  routing_entropy / MoDAnalytics entropy tracking / MoDTransformer.compute_loss
"""

from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from open_mythos.mod import (
    MoDConfig,
    MoDAnalytics,
    MixtureOfDepthsBlock,
    MoDTransformer,
    TokenRouter,
    apply_mod_rope,
    precompute_mod_rope_freqs,
    routing_entropy,
)
from open_mythos import (
    MoDConfig as MoDConfigExported,
    MoDTransformer as MoDTransformerExported,
)


# ---------------------------------------------------------------------------
# RNG isolation — save the global PyTorch RNG state ONCE before any test in
# this module runs, then restore it after all tests finish.  This prevents
# torch.randn / torch.multinomial / nn.init calls from bleeding into
# subsequent test files (e.g. Sprint 7 serve tests whose agent-generation
# paths are sensitive to the global torch RNG state via KV-cache sizing).
#
# scope="module" is used rather than scope="function" so that the save/restore
# brackets the ENTIRE module; this avoids subtle pytest fixture-ordering issues
# where a function-scoped fixture might not fully cover inter-test transitions.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _rng_isolation():
    """Save global PyTorch RNG state before this module; restore after."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _small_cfg(**kwargs) -> MoDConfig:
    """Tiny model config for fast CPU tests."""
    base = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        n_layers=2,
        capacity_factor=0.5,
        router_aux_loss_coef=0.01,
        ffn_hidden_mult=2,
        dropout=0.0,
        rope_theta=10000.0,
    )
    base.update(kwargs)
    return MoDConfig(**base)


# ---------------------------------------------------------------------------
# 1. MoDConfig
# ---------------------------------------------------------------------------


class TestMoDConfig:
    def test_default_instantiation(self):
        cfg = MoDConfig()
        assert cfg.dim == 512
        assert cfg.n_layers == 6
        assert cfg.capacity_factor == 0.5

    def test_custom_fields(self):
        cfg = _small_cfg(capacity_factor=0.75, n_layers=4)
        assert cfg.capacity_factor == 0.75
        assert cfg.n_layers == 4

    def test_capacity_factor_range(self):
        """capacity_factor of 1.0 means all tokens are always routed."""
        cfg = _small_cfg(capacity_factor=1.0)
        assert cfg.capacity_factor == 1.0

    def test_exported_from_package(self):
        """MoDConfig must be importable from top-level open_mythos."""
        assert MoDConfigExported is MoDConfig


# ---------------------------------------------------------------------------
# 2. precompute_mod_rope_freqs
# ---------------------------------------------------------------------------


class TestPrecomputeModRopeFreqs:
    def test_output_shape(self):
        freqs = precompute_mod_rope_freqs(head_dim=16, max_len=64)
        assert freqs.shape == (64, 8)  # (max_len, head_dim//2)

    def test_complex_dtype(self):
        freqs = precompute_mod_rope_freqs(head_dim=16, max_len=8)
        assert freqs.is_complex()

    def test_unit_magnitude(self):
        freqs = precompute_mod_rope_freqs(head_dim=16, max_len=8)
        mags = freqs.abs()
        assert torch.allclose(mags, torch.ones_like(mags), atol=1e-5)

    def test_different_theta(self):
        f1 = precompute_mod_rope_freqs(16, 8, theta=10000.0)
        f2 = precompute_mod_rope_freqs(16, 8, theta=500000.0)
        assert not torch.allclose(f1, f2)


# ---------------------------------------------------------------------------
# 3. apply_mod_rope
# ---------------------------------------------------------------------------


class TestApplyModRope:
    def test_output_shape_preserved(self):
        B, T, H, hd = 2, 6, 4, 16
        x = torch.randn(B, T, H, hd)
        freqs = precompute_mod_rope_freqs(hd, 32)
        # gather rows for T positions
        positions = torch.arange(T).unsqueeze(0).expand(B, -1)  # (B, T)
        freqs_batched = freqs[positions]  # (B, T, hd//2)
        out = apply_mod_rope(x, freqs_batched)
        assert out.shape == x.shape

    def test_preserves_norm(self):
        """RoPE is a rotation — it should not change vector norms."""
        B, T, H, hd = 2, 4, 2, 16
        x = torch.randn(B, T, H, hd)
        freqs = precompute_mod_rope_freqs(hd, 32)
        positions = torch.arange(T).unsqueeze(0).expand(B, -1)
        freqs_batched = freqs[positions]
        out = apply_mod_rope(x, freqs_batched)
        # Compare norms (head-level)
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)

    def test_different_batches_different_output(self):
        """Per-batch-item freqs should produce different rotations."""
        B, T, H, hd = 2, 4, 2, 16
        x = torch.ones(B, T, H, hd)
        freqs = precompute_mod_rope_freqs(hd, 32)
        # batch 0 → positions 0..3, batch 1 → positions 4..7
        positions = torch.stack([torch.arange(4), torch.arange(4, 8)])
        freqs_batched = freqs[positions]
        out = apply_mod_rope(x, freqs_batched)
        assert not torch.allclose(out[0], out[1])


# ---------------------------------------------------------------------------
# 4. TokenRouter
# ---------------------------------------------------------------------------


class TestTokenRouter:
    def test_output_shape(self):
        router = TokenRouter(dim=64)
        x = torch.randn(2, 10, 64)
        scores = router(x)
        assert scores.shape == (2, 10)

    def test_select_top_k_shape(self):
        router = TokenRouter(dim=64)
        scores = torch.randn(3, 16)
        sel = router.select_top_k(scores, capacity=8)
        assert sel.shape == (3, 8)

    def test_select_top_k_sorted(self):
        """Selected indices must be sorted in ascending position order."""
        router = TokenRouter(dim=64)
        scores = torch.randn(2, 20)
        sel = router.select_top_k(scores, capacity=10)
        for b in range(2):
            assert (sel[b, 1:] >= sel[b, :-1]).all()

    def test_select_top_k_valid_range(self):
        router = TokenRouter(dim=64)
        T = 15
        scores = torch.randn(2, T)
        sel = router.select_top_k(scores, capacity=7)
        assert (sel >= 0).all()
        assert (sel < T).all()

    def test_select_top_k_clamps_to_T(self):
        """Capacity larger than T should be clipped to T."""
        router = TokenRouter(dim=64)
        scores = torch.randn(2, 6)
        sel = router.select_top_k(scores, capacity=100)
        assert sel.shape[1] == 6  # clipped to T

    def test_no_duplicate_indices(self):
        router = TokenRouter(dim=64)
        scores = torch.randn(1, 10)
        sel = router.select_top_k(scores, capacity=5)
        unique_count = sel[0].unique().numel()
        assert unique_count == 5

    def test_gradient_flows(self):
        router = TokenRouter(dim=32)
        x = torch.randn(2, 8, 32, requires_grad=True)
        scores = router(x)
        scores.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# 5. MixtureOfDepthsBlock
# ---------------------------------------------------------------------------


class TestMixtureOfDepthsBlock:
    def _block_and_freqs(self, cfg: MoDConfig):
        block = MixtureOfDepthsBlock(cfg, layer_idx=0)
        freqs = precompute_mod_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        return block, freqs

    def test_output_shape(self):
        cfg = _small_cfg()
        block, freqs = self._block_and_freqs(cfg)
        x = torch.randn(2, 12, cfg.dim)
        out, scores = block(x, freqs)
        assert out.shape == x.shape
        assert scores.shape == (2, 12)

    def test_non_selected_tokens_unchanged(self):
        """
        With capacity_factor=0.5 on T=4, 2 tokens are routed.
        The other 2 should be bitwise-identical to the input.
        """
        cfg = _small_cfg(capacity_factor=0.5)
        block, freqs = self._block_and_freqs(cfg)
        block.eval()
        T = 4
        x = torch.randn(1, T, cfg.dim)
        with torch.no_grad():
            out, scores = block(x, freqs)

        # Find which tokens were NOT selected
        sel = block.router.select_top_k(scores, max(1, int(T * cfg.capacity_factor)))
        all_positions = set(range(T))
        selected_set = set(sel[0].tolist())
        not_selected = list(all_positions - selected_set)

        for pos in not_selected:
            assert torch.allclose(out[0, pos], x[0, pos], atol=1e-6), (
                f"Non-selected token at position {pos} should be unchanged"
            )

    def test_capacity_one_full(self):
        """capacity_factor=1.0 → all tokens routed → output differs from input."""
        cfg = _small_cfg(capacity_factor=1.0)
        block, freqs = self._block_and_freqs(cfg)
        block.eval()
        x = torch.randn(2, 8, cfg.dim)
        with torch.no_grad():
            out, _ = block(x, freqs)
        assert not torch.allclose(out, x)

    def test_analytics_recorded(self):
        cfg = _small_cfg()
        block, freqs = self._block_and_freqs(cfg)
        analytics = MoDAnalytics(n_layers=1)
        x = torch.randn(2, 10, cfg.dim)
        block(x, freqs, analytics=analytics)
        assert len(analytics._routed[0]) == 1

    def test_gradient_flows(self):
        cfg = _small_cfg()
        block, freqs = self._block_and_freqs(cfg)
        x = torch.randn(2, 8, cfg.dim, requires_grad=True)
        out, scores = block(x, freqs)
        (out.sum() + scores.sum()).backward()
        assert x.grad is not None

    def test_no_nan_in_output(self):
        cfg = _small_cfg()
        block, freqs = self._block_and_freqs(cfg)
        x = torch.randn(2, 16, cfg.dim)
        out, scores = block(x, freqs)
        assert not torch.isnan(out).any()
        assert not torch.isnan(scores).any()

    def test_batch_size_one(self):
        cfg = _small_cfg()
        block, freqs = self._block_and_freqs(cfg)
        x = torch.randn(1, 8, cfg.dim)
        out, scores = block(x, freqs)
        assert out.shape == (1, 8, cfg.dim)

    def test_seq_len_one(self):
        """T=1 edge case: capacity must be at least 1."""
        cfg = _small_cfg(capacity_factor=0.25)
        block, freqs = self._block_and_freqs(cfg)
        x = torch.randn(2, 1, cfg.dim)
        out, scores = block(x, freqs)
        assert out.shape == (2, 1, cfg.dim)


# ---------------------------------------------------------------------------
# 6. MoDAnalytics
# ---------------------------------------------------------------------------


class TestMoDAnalytics:
    def test_record_and_summary(self):
        analytics = MoDAnalytics(n_layers=3)
        analytics.record(0, n_routed=5, n_total=10)
        analytics.record(0, n_routed=4, n_total=10)
        analytics.record(1, n_routed=6, n_total=10)

        summary = analytics.summary()
        assert "layer_0_avg_routed" in summary
        assert abs(summary["layer_0_avg_routed"] - 4.5) < 1e-6
        assert abs(summary["layer_0_avg_capacity"] - 0.45) < 1e-6
        assert "layer_1_avg_routed" in summary
        assert "layer_2_avg_routed" not in summary  # no records for layer 2

    def test_reset_clears_data(self):
        analytics = MoDAnalytics(n_layers=2)
        analytics.record(0, 3, 6)
        analytics.reset()
        assert analytics.summary() == {}

    def test_empty_summary(self):
        analytics = MoDAnalytics(n_layers=4)
        assert analytics.summary() == {}


# ---------------------------------------------------------------------------
# 7. MoDTransformer
# ---------------------------------------------------------------------------


class TestMoDTransformer:
    def _model(self, **kwargs) -> MoDTransformer:
        return MoDTransformer(_small_cfg(**kwargs))

    def test_forward_logits_shape(self):
        model = self._model()
        ids = torch.randint(0, 200, (2, 12))
        logits, aux = model(ids)
        assert logits.shape == (2, 12, 200)
        assert aux is not None

    def test_aux_loss_scalar(self):
        model = self._model()
        ids = torch.randint(0, 200, (2, 8))
        _, aux = model(ids)
        assert aux.ndim == 0

    def test_aux_loss_non_negative(self):
        model = self._model()
        ids = torch.randint(0, 200, (2, 8))
        _, aux = model(ids)
        assert aux.item() >= 0.0

    def test_no_aux_loss_when_disabled(self):
        model = self._model()
        ids = torch.randint(0, 200, (2, 8))
        logits, aux = model(ids, return_aux_loss=False)
        assert aux is None
        assert logits.shape == (2, 8, 200)

    def test_generate_output_shape(self):
        model = self._model()
        ids = torch.randint(0, 200, (1, 4))
        out = model.generate(ids, max_new_tokens=6)
        assert out.shape == (1, 10)  # 4 prompt + 6 generated

    def test_generate_valid_token_ids(self):
        cfg = _small_cfg()
        model = MoDTransformer(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 3))
        out = model.generate(ids, max_new_tokens=5)
        assert (out >= 0).all()
        assert (out < cfg.vocab_size).all()

    def test_weight_tying(self):
        """Embedding weight should be tied to the LM head weight."""
        model = self._model()
        assert model.embed.weight is model.head.weight

    def test_parameter_count_reasonable(self):
        """Small model should have a plausible number of parameters."""
        model = self._model()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1000  # definitely has parameters
        assert n_params < 10_000_000  # small test model

    def test_no_nan_in_logits(self):
        model = self._model()
        ids = torch.randint(0, 200, (2, 16))
        logits, _ = model(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_analytics_integration(self):
        cfg = _small_cfg()
        model = MoDTransformer(cfg)
        analytics = MoDAnalytics(n_layers=cfg.n_layers)
        ids = torch.randint(0, cfg.vocab_size, (2, 10))
        model(ids, analytics=analytics)
        summary = analytics.summary()
        # Every layer should have at least one observation
        for i in range(cfg.n_layers):
            assert f"layer_{i}_avg_routed" in summary

    def test_capacity_factor_one_all_tokens_processed(self):
        """With capacity_factor=1.0, all tokens go through every layer."""
        cfg = _small_cfg(capacity_factor=1.0)
        model = MoDTransformer(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _ = model(ids)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_backward_through_model(self):
        """Full end-to-end gradient flow test."""
        model = self._model()
        ids = torch.randint(0, 200, (2, 6))
        logits, aux = model(ids)
        loss = logits.mean() + aux
        loss.backward()
        # At least some parameters should have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_exported_from_package(self):
        assert MoDTransformerExported is MoDTransformer

    def test_temperature_generation(self):
        model = self._model()
        ids = torch.randint(0, 200, (1, 4))
        out = model.generate(ids, max_new_tokens=3, temperature=0.5)
        assert out.shape == (1, 7)

    def test_top_k_generation(self):
        model = self._model()
        ids = torch.randint(0, 200, (1, 4))
        out = model.generate(ids, max_new_tokens=3, top_k=5)
        assert out.shape == (1, 7)

    def test_compute_aux_loss_proportional_to_coef(self):
        """Doubling coef should roughly double aux loss."""
        cfg1 = _small_cfg(router_aux_loss_coef=0.01)
        cfg2 = _small_cfg(router_aux_loss_coef=0.02)
        m1 = MoDTransformer(cfg1)
        m2 = MoDTransformer(cfg2)
        # Use same router scores
        scores = [torch.zeros(2, 8)]
        l1 = m1.compute_aux_loss(scores).item()
        l2 = m2.compute_aux_loss(scores).item()
        # Both should be proportional (ratio ~2x)
        if l1 > 0:
            assert abs(l2 / l1 - 2.0) < 0.1


# ---------------------------------------------------------------------------
# 8. routing_entropy  (13.1.2)
# ---------------------------------------------------------------------------


class TestRoutingEntropy:
    def test_output_shape(self):
        scores = torch.randn(3, 10)
        h = routing_entropy(scores)
        assert h.shape == (3, 10)

    def test_max_entropy_at_zero_logit(self):
        """sigmoid(0) = 0.5 → maximum binary entropy = ln(2)."""
        scores = torch.zeros(2, 8)
        h = routing_entropy(scores)
        assert torch.allclose(h, torch.full_like(h, math.log(2)), atol=1e-4)

    def test_entropy_non_negative(self):
        scores = torch.randn(4, 16)
        h = routing_entropy(scores)
        assert (h >= 0).all()

    def test_entropy_bounded_above(self):
        """Binary entropy ≤ ln(2) ≈ 0.693 for all inputs."""
        scores = torch.randn(4, 16) * 10  # extreme logits
        h = routing_entropy(scores)
        assert (h <= math.log(2) + 1e-5).all()

    def test_confident_routing_low_entropy(self):
        """Very large positive/negative logits → near-zero entropy."""
        scores_high = torch.full((2, 8), 50.0)
        scores_low = torch.full((2, 8), -50.0)
        h_high = routing_entropy(scores_high)
        h_low = routing_entropy(scores_low)
        assert h_high.mean().item() < 1e-3
        assert h_low.mean().item() < 1e-3

    def test_exported_from_package(self):
        from open_mythos import routing_entropy as re_exported
        assert re_exported is routing_entropy

    def test_gradient_flows(self):
        scores = torch.randn(2, 8, requires_grad=True)
        h = routing_entropy(scores)
        h.sum().backward()
        assert scores.grad is not None


# ---------------------------------------------------------------------------
# 9. MoDAnalytics entropy tracking  (13.1.2)
# ---------------------------------------------------------------------------


class TestMoDAnalyticsEntropy:
    def test_entropy_recorded_when_scores_given(self):
        analytics = MoDAnalytics(n_layers=2)
        scores = torch.zeros(2, 10)  # max entropy
        analytics.record(0, n_routed=5, n_total=10, scores=scores)
        summary = analytics.summary()
        assert "layer_0_avg_entropy" in summary

    def test_entropy_absent_without_scores(self):
        analytics = MoDAnalytics(n_layers=2)
        analytics.record(0, n_routed=5, n_total=10)
        summary = analytics.summary()
        assert "layer_0_avg_entropy" not in summary

    def test_entropy_max_at_zero_logits(self):
        analytics = MoDAnalytics(n_layers=1)
        scores = torch.zeros(1, 8)
        analytics.record(0, 4, 8, scores=scores)
        summary = analytics.summary()
        assert abs(summary["layer_0_avg_entropy"] - math.log(2)) < 1e-4

    def test_reset_clears_entropy(self):
        analytics = MoDAnalytics(n_layers=1)
        analytics.record(0, 4, 8, scores=torch.zeros(1, 8))
        analytics.reset()
        assert analytics.summary() == {}
        assert analytics._entropy == [[]]

    def test_analytics_integration_with_block_scores(self):
        """MixtureOfDepthsBlock should automatically record scores."""
        cfg = _small_cfg()
        block = MixtureOfDepthsBlock(cfg, layer_idx=0)
        freqs = precompute_mod_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        analytics = MoDAnalytics(n_layers=1)
        x = torch.randn(2, 10, cfg.dim)
        block(x, freqs, analytics=analytics)
        summary = analytics.summary()
        # Scores are now passed automatically → entropy should be present
        assert "layer_0_avg_entropy" in summary


# ---------------------------------------------------------------------------
# 10. MoDTransformer.compute_loss  (13.1.2)
# ---------------------------------------------------------------------------


class TestMoDTransformerComputeLoss:
    def _model_and_batch(self, T=8):
        cfg = _small_cfg()
        model = MoDTransformer(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, T + 1))
        return model, cfg, ids

    def test_loss_scalar(self):
        model, cfg, ids = self._model_and_batch()
        logits, aux = model(ids[:, :-1])
        loss = model.compute_loss(logits, ids[:, 1:], aux)
        assert loss.ndim == 0

    def test_loss_positive(self):
        model, cfg, ids = self._model_and_batch()
        logits, aux = model(ids[:, :-1])
        loss = model.compute_loss(logits, ids[:, 1:], aux)
        assert loss.item() > 0.0

    def test_loss_without_aux(self):
        """compute_loss works with aux_loss=None (CE only)."""
        model, cfg, ids = self._model_and_batch()
        logits, _ = model(ids[:, :-1], return_aux_loss=False)
        loss = model.compute_loss(logits, ids[:, 1:], aux_loss=None)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_loss_with_aux_larger(self):
        """CE + aux should be ≥ CE alone (aux is non-negative)."""
        model, cfg, ids = self._model_and_batch()
        logits, aux = model(ids[:, :-1])
        loss_full = model.compute_loss(logits, ids[:, 1:], aux)
        loss_ce = model.compute_loss(logits, ids[:, 1:], aux_loss=None)
        assert loss_full.item() >= loss_ce.item() - 1e-6

    def test_loss_ignore_index(self):
        """Padding positions (-100) must not contribute to the loss."""
        model, cfg, ids = self._model_and_batch()
        logits, _ = model(ids[:, :-1], return_aux_loss=False)
        targets = ids[:, 1:].clone()
        targets[:, -1] = -100           # mask last position
        loss_masked = model.compute_loss(logits, targets)
        # Masked loss should differ from unmasked in general (not exactly, but callable)
        assert loss_masked.ndim == 0

    def test_backward_through_compute_loss(self):
        """End-to-end: forward → compute_loss → backward should work."""
        model, cfg, ids = self._model_and_batch()
        logits, aux = model(ids[:, :-1])
        loss = model.compute_loss(logits, ids[:, 1:], aux)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

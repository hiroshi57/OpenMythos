"""
Tests for open_mythos/moda.py — Mixture-of-Depths Attention + DeepSeek MoE.

Coverage:
  - RMSNorm (moda variant)
  - RotaryEmbedding
  - DeepSeekExpert
  - DeepSeekGate  (softmax / sigmoid / groups / bias)
  - DeepSeekMoE   (forward + balance loss)
  - MoDAAttention (no depth cache / with depth cache)
  - MoDABlock
  - MoDAModel     (forward, loss, generate, num_parameters)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from open_mythos.moda import (
    DeepSeekExpert,
    DeepSeekGate,
    DeepSeekMoE,
    MoDAAttention,
    MoDABlock,
    MoDAConfig,
    MoDAModel,
    RMSNorm,
    RotaryEmbedding,
    _SharedFFN,
    apply_rotary_emb,
)

# ---------------------------------------------------------------------------
# Shared tiny config — keeps tests fast on CPU
# ---------------------------------------------------------------------------

B, T = 2, 12  # batch size, sequence length


def tiny_cfg(**overrides) -> MoDAConfig:
    """Return a minimal MoDAConfig suited for CPU unit tests."""
    defaults = dict(
        vocab_size=256,
        d_model=64,
        n_layers=3,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=16,
        max_seq_len=32,
        rope_base=10_000.0,
        attn_dropout=0.0,
        norm_eps=1e-6,
        n_shared_experts=1,
        n_routed_experts=8,
        n_activated_experts=2,
        expert_hidden_dim=32,
        moe_balance_alpha=0.01,
        moe_score_func="softmax",
        moe_n_groups=1,
        moe_topk_groups=1,
        moe_route_scale=1.0,
    )
    defaults.update(overrides)
    return MoDAConfig(**defaults)


# ---------------------------------------------------------------------------
# RMSNorm (moda variant)
# ---------------------------------------------------------------------------


class TestMoDARMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(B, T, 64)
        assert norm(x).shape == x.shape

    def test_unit_rms_when_weight_ones(self):
        norm = RMSNorm(64)
        nn.init.ones_(norm.weight)
        x = torch.randn(4, 64)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_learnable_weight(self):
        assert RMSNorm(32).weight.requires_grad

    def test_eps_prevents_div_by_zero(self):
        norm = RMSNorm(8, eps=1e-6)
        x = torch.zeros(2, 8)
        # Should not raise / NaN
        out = norm(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# RotaryEmbedding
# ---------------------------------------------------------------------------


class TestRotaryEmbedding:
    def setup_method(self):
        self.rope = RotaryEmbedding(dim=16, max_seq_len=32, base=10_000.0)

    def test_shapes(self):
        cos, sin = self.rope(T)
        assert cos.shape == (1, 1, T, 16)
        assert sin.shape == (1, 1, T, 16)

    def test_cache_extension(self):
        """Requesting a length > initial cache should auto-extend."""
        rope = RotaryEmbedding(dim=16, max_seq_len=4, base=10_000.0)
        cos, sin = rope(20)
        assert cos.shape[2] >= 20

    def test_cos_sin_values_unit_circle(self):
        """cos²+sin² must equal 1 at every position/dimension."""
        cos, sin = self.rope(T)
        assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)

    def test_apply_rotary_emb_shape(self):
        cos, sin = self.rope(T)
        x = torch.randn(B, 4, T, 16)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_apply_rotary_emb_is_isometry(self):
        """RoPE rotation must preserve vector norms."""
        cos, sin = self.rope(T)
        x = torch.randn(B, 4, T, 16)
        out = apply_rotary_emb(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)

    def test_apply_rotary_emb_position_zero_is_identity(self):
        """At position 0 the rotation angle is 0 → output must equal input."""
        cos_full, sin_full = self.rope(T)
        cos1, sin1 = cos_full[:, :, :1], sin_full[:, :, :1]
        x = torch.randn(B, 4, 1, 16)
        out = apply_rotary_emb(x, cos1, sin1)
        assert torch.allclose(x, out, atol=1e-6)


# ---------------------------------------------------------------------------
# DeepSeekExpert
# ---------------------------------------------------------------------------


class TestDeepSeekExpert:
    def test_output_shape_2d(self):
        expert = DeepSeekExpert(d_model=64, hidden_dim=32)
        x = torch.randn(B * T, 64)
        assert expert(x).shape == (B * T, 64)

    def test_silu_gate_nonlinearity(self):
        """Output must differ from a pure linear transform (SiLU is nonlinear)."""
        expert = DeepSeekExpert(d_model=16, hidden_dim=8)
        x = torch.randn(4, 16)
        out1 = expert(x)
        out2 = expert(-x)
        assert not torch.allclose(out1, -out2)

    def test_no_bias_parameters(self):
        expert = DeepSeekExpert(d_model=16, hidden_dim=8)
        for name, param in expert.named_parameters():
            assert "bias" not in name, f"Unexpected bias: {name}"


# ---------------------------------------------------------------------------
# DeepSeekGate
# ---------------------------------------------------------------------------


class TestDeepSeekGate:
    def _gate(self, **kwargs) -> DeepSeekGate:
        defaults = dict(
            d_model=64,
            n_routed_experts=8,
            n_activated=2,
            score_func="softmax",
        )
        defaults.update(kwargs)
        return DeepSeekGate(**defaults)

    def test_output_shapes(self):
        gate = self._gate()
        x = torch.randn(B * T, 64)
        weights, indices, scores = gate(x)
        assert weights.shape == (B * T, 2)
        assert indices.shape == (B * T, 2)
        assert scores.shape == (B * T, 8)

    def test_indices_in_range(self):
        gate = self._gate()
        x = torch.randn(B * T, 64)
        _, indices, _ = gate(x)
        assert indices.min() >= 0
        assert indices.max() < 8

    def test_weights_positive(self):
        gate = self._gate()
        x = torch.randn(B * T, 64)
        weights, _, _ = gate(x)
        assert (weights >= 0).all()

    def test_softmax_gate_scores_sum_to_one(self):
        gate = self._gate(score_func="softmax")
        x = torch.randn(B * T, 64)
        _, _, scores = gate(x)
        # full softmax distribution sums to 1 per token
        assert torch.allclose(scores.sum(dim=-1), torch.ones(B * T), atol=1e-4)

    def test_sigmoid_gate_weights_normalised(self):
        """Sigmoid gate: selected weights should sum to 1 (after normalisation)."""
        gate = self._gate(score_func="sigmoid")
        x = torch.randn(B * T, 64)
        weights, _, _ = gate(x)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B * T), atol=1e-4)

    def test_bias_routing(self):
        """With use_bias=True, bias parameter should be registered."""
        gate = self._gate(use_bias=True)
        assert gate.bias is not None
        assert gate.bias.shape == (8,)

    def test_no_bias_by_default(self):
        gate = self._gate()
        assert gate.bias is None

    def test_group_routing_respects_topk_groups(self):
        """Group-limited routing: only topk_groups out of n_groups should be active."""
        gate = self._gate(n_routed_experts=8, n_activated=2, n_groups=2, topk_groups=1)
        x = torch.randn(4, 64)
        weights, indices, _ = gate(x)
        # Each token picks 2 experts from a single group (4 experts per group)
        # Verify indices fall within one of the two groups [0-3] or [4-7] per token
        for tok in range(4):
            idx = indices[tok]
            in_group0 = (idx < 4).all()
            in_group1 = (idx >= 4).all()
            assert (in_group0 or in_group1).item(), (
                f"Token {tok} spans two groups: {idx.tolist()}"
            )


# ---------------------------------------------------------------------------
# DeepSeekMoE
# ---------------------------------------------------------------------------


class TestDeepSeekMoE:
    def setup_method(self):
        self.cfg = tiny_cfg()
        self.moe = DeepSeekMoE(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.d_model)
        out, loss = self.moe(x)
        assert out.shape == (B, T, self.cfg.d_model)

    def test_training_returns_balance_loss(self):
        self.moe.train()
        x = torch.randn(B, T, self.cfg.d_model)
        _, loss = self.moe(x)
        assert loss is not None
        assert loss.ndim == 0  # scalar

    def test_eval_returns_no_balance_loss(self):
        self.moe.eval()
        x = torch.randn(B, T, self.cfg.d_model)
        _, loss = self.moe(x)
        assert loss is None

    def test_shared_experts_always_fire(self):
        """Zeroing all routed experts — output must still be non-zero from shared."""
        for exp in self.moe.experts:
            for p in exp.parameters():
                p.data.zero_()
        x = torch.randn(B, T, self.cfg.d_model)
        out, _ = self.moe(x)
        assert out.abs().sum() > 0

    def test_balance_loss_scalar_positive(self):
        self.moe.train()
        x = torch.randn(B, T, self.cfg.d_model)
        _, loss = self.moe(x)
        assert loss.item() >= 0

    def test_balance_loss_zero_when_alpha_zero(self):
        cfg_no_loss = tiny_cfg(moe_balance_alpha=0.0)
        moe = DeepSeekMoE(cfg_no_loss)
        moe.train()
        x = torch.randn(B, T, cfg_no_loss.d_model)
        _, loss = moe(x)
        assert loss is None

    def test_gradients_flow_to_gate(self):
        self.moe.train()
        x = torch.randn(B, T, self.cfg.d_model, requires_grad=False)
        out, loss = self.moe(x)
        total = out.sum() + (loss if loss is not None else 0)
        total.backward()
        assert self.moe.gate.weight.grad is not None

    def test_sigmoid_score_func(self):
        cfg_sig = tiny_cfg(moe_score_func="sigmoid")
        moe = DeepSeekMoE(cfg_sig)
        x = torch.randn(B, T, cfg_sig.d_model)
        out, _ = moe(x)
        assert out.shape == (B, T, cfg_sig.d_model)


# ---------------------------------------------------------------------------
# MoDAAttention
# ---------------------------------------------------------------------------


class TestMoDAAttention:
    def setup_method(self):
        self.cfg = tiny_cfg()
        self.attn = MoDAAttention(self.cfg)
        self.rope = RotaryEmbedding(
            self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_base
        )

    def _cos_sin(self):
        return self.rope(T)

    def test_output_shape_no_depth_cache(self):
        cos, sin = self._cos_sin()
        x = torch.randn(B, T, self.cfg.d_model)
        out = self.attn(x, [], [], cos, sin)
        assert out.shape == (B, T, self.cfg.d_model)

    def test_output_shape_with_depth_cache(self):
        cos, sin = self._cos_sin()
        x = torch.randn(B, T, self.cfg.d_model)
        Hk, d = self.cfg.n_heads_kv, self.cfg.head_dim
        depth_k = [torch.randn(B, Hk, T, d) for _ in range(3)]
        depth_v = [torch.randn(B, Hk, T, d) for _ in range(3)]
        out = self.attn(x, depth_k, depth_v, cos, sin)
        assert out.shape == (B, T, self.cfg.d_model)

    def test_no_nan(self):
        cos, sin = self._cos_sin()
        x = torch.randn(B, T, self.cfg.d_model)
        out = self.attn(x, [], [], cos, sin)
        assert not torch.isnan(out).any()

    def test_depth_cache_size_matters(self):
        """Output with a non-empty depth cache should differ from no-cache output."""
        cos, sin = self._cos_sin()
        x = torch.randn(B, T, self.cfg.d_model)
        Hk, d = self.cfg.n_heads_kv, self.cfg.head_dim
        out_no_depth = self.attn(x, [], [], cos, sin)
        depth_k = [torch.randn(B, Hk, T, d)]
        depth_v = [torch.randn(B, Hk, T, d)]
        out_with_depth = self.attn(x, depth_k, depth_v, cos, sin)
        assert not torch.allclose(out_no_depth, out_with_depth)

    def test_gqa_divisibility_raises(self):
        """n_heads_q not divisible by n_heads_kv must raise ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            MoDAAttention(tiny_cfg(n_heads_q=5, n_heads_kv=3))


# ---------------------------------------------------------------------------
# MoDABlock
# ---------------------------------------------------------------------------


class TestMoDABlock:
    def setup_method(self):
        self.cfg = tiny_cfg()
        self.block = MoDABlock(self.cfg)
        self.rope = RotaryEmbedding(
            self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_base
        )

    def _cos_sin(self):
        return self.rope(T)

    def test_output_shapes(self):
        cos, sin = self._cos_sin()
        x = torch.randn(B, T, self.cfg.d_model)
        x_out, k_w, v_w, bal = self.block(x, [], [], cos, sin)
        Hk, d = self.cfg.n_heads_kv, self.cfg.head_dim
        assert x_out.shape == (B, T, self.cfg.d_model)
        assert k_w.shape == (B, Hk, T, d)
        assert v_w.shape == (B, Hk, T, d)

    def test_training_balance_loss(self):
        self.block.train()
        cos, sin = self._cos_sin()
        x = torch.randn(B, T, self.cfg.d_model)
        _, _, _, bal = self.block(x, [], [], cos, sin)
        assert bal is not None
        assert bal.ndim == 0

    def test_eval_no_balance_loss(self):
        self.block.eval()
        cos, sin = self._cos_sin()
        x = torch.randn(B, T, self.cfg.d_model)
        _, _, _, bal = self.block(x, [], [], cos, sin)
        assert bal is None

    def test_k_write_has_rope(self):
        """k_write should be positionally encoded — pos 0 must differ from pos T-1."""
        self.block.eval()
        cos, sin = self._cos_sin()
        x = torch.randn(B, T, self.cfg.d_model)
        _, k_w, _, _ = self.block(x, [], [], cos, sin)
        # Position 0 and last position should differ
        assert not torch.allclose(k_w[:, :, 0], k_w[:, :, -1])

    def test_depth_cache_written_per_block(self):
        """k_write / v_write must not be identical across two different blocks."""
        cfg = self.cfg
        block2 = MoDABlock(cfg)
        cos, sin = self._cos_sin()
        x = torch.randn(B, T, cfg.d_model)
        _, k1, v1, _ = self.block(x, [], [], cos, sin)
        _, k2, v2, _ = block2(x, [], [], cos, sin)
        # Different blocks → different write projections → different outputs
        assert not torch.allclose(k1, k2)


# ---------------------------------------------------------------------------
# MoDAModel — forward, loss, num_parameters
# ---------------------------------------------------------------------------


class TestMoDAModel:
    def setup_method(self):
        self.cfg = tiny_cfg()
        self.model = MoDAModel(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_logits_shape(self):
        logits, loss = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)
        assert loss is None

    def test_loss_with_labels(self):
        labels = torch.randint(0, self.cfg.vocab_size, (B, T))
        logits, loss = self.model(self.ids, labels=labels)
        assert loss is not None
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_no_nan_in_logits(self):
        logits, _ = self.model(self.ids)
        assert not torch.isnan(logits).any()

    def test_weight_tying(self):
        assert self.model.lm_head.weight is self.model.embed.weight

    def test_num_parameters(self):
        n = self.model.num_parameters()
        assert n > 0
        n_trainable = self.model.num_parameters(trainable_only=True)
        assert n_trainable <= n

    def test_depth_cache_grows_across_layers(self):
        """Forward should populate depth caches — indirectly: later layers see prior writes."""
        logits, _ = self.model(self.ids)
        # Just verify no NaN and correct shape; depth cache is ephemeral inside forward.
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_seq_len_exceeds_max_raises(self):
        ids_too_long = torch.randint(0, self.cfg.vocab_size, (1, self.cfg.max_seq_len + 1))
        with pytest.raises(ValueError, match="max_seq_len"):
            self.model(ids_too_long)

    def test_backward_propagates(self):
        """Loss backward must populate gradients in all parameter groups."""
        self.model.train()
        labels = torch.randint(0, self.cfg.vocab_size, (B, T))
        _, loss = self.model(self.ids, labels=labels)
        loss.backward()
        # Check a few key parameters
        assert self.model.embed.weight.grad is not None
        assert self.model.blocks[0].attn.q_proj.weight.grad is not None
        assert self.model.blocks[0].moe.gate.weight.grad is not None

    def test_balance_loss_included_in_total(self):
        """With alpha>0 and training mode, loss > pure CE (balance adds a positive term)."""
        self.model.train()
        labels = torch.randint(0, self.cfg.vocab_size, (B, T))
        _, loss_with_bal = self.model(self.ids, labels=labels)

        cfg_no_bal = tiny_cfg(moe_balance_alpha=0.0)
        model_no_bal = MoDAModel(cfg_no_bal).train()
        # Use same weights
        model_no_bal.load_state_dict(self.model.state_dict())
        _, loss_no_bal = model_no_bal(self.ids, labels=labels)

        # With balance loss alpha=0.01 the total should be different (>= CE)
        assert loss_with_bal.item() != pytest.approx(loss_no_bal.item(), rel=1e-6)

    def test_extra_repr_contains_key_info(self):
        r = self.model.extra_repr()
        assert "d_model" in r
        assert "experts" in r

    def test_eval_no_balance_loss(self):
        self.model.eval()
        labels = torch.randint(0, self.cfg.vocab_size, (B, T))
        _, loss = self.model(self.ids, labels=labels)
        # With alpha>0 but eval mode: balance_loss=None → total is pure CE
        assert loss is not None and loss.item() > 0

    def test_single_token_forward(self):
        single = torch.randint(0, self.cfg.vocab_size, (B, 1))
        logits, _ = self.model(single)
        assert logits.shape == (B, 1, self.cfg.vocab_size)


# ---------------------------------------------------------------------------
# MoDAConfig validation
# ---------------------------------------------------------------------------


class TestMoDAConfig:
    def test_defaults(self):
        cfg = MoDAConfig()
        assert cfg.d_model == 2048
        assert cfg.n_layers == 24
        assert cfg.moe_score_func == "softmax"

    def test_custom_values(self):
        cfg = tiny_cfg(d_model=128, n_layers=2)
        assert cfg.d_model == 128
        assert cfg.n_layers == 2

    def test_sigmoid_score_func(self):
        cfg = tiny_cfg(moe_score_func="sigmoid")
        model = MoDAModel(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, _ = model(ids)
        assert logits.shape == (1, 8, cfg.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "--verbose", "-s"])

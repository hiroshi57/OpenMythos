"""
Tests for open_mythos/hyperloop.py — HyperloopBlock and HyperloopMythos.

Coverage:
  - HyperloopConfig defaults and post_init behaviour
  - HyperloopBlock forward shape, no-NaN, depth scaling
  - HyperloopBlock spectral radius guarantees on both injection levels
  - HyperloopBlock outer / inner loop override at inference
  - HyperloopMythos forward shape, generate, weight tying, KV-cache first pass
  - Comparison: HyperloopMythos vs OpenMythos at equal total depth
"""

from __future__ import annotations

import pytest
import torch

from open_mythos.hyperloop import (
    HyperloopBlock,
    HyperloopConfig,
    HyperloopMythos,
)
from open_mythos.main import (
    OpenMythos,
    MythosConfig,
    precompute_rope_freqs,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

B, T = 2, 8
VOCAB = 256


def tiny_cfg(**overrides) -> HyperloopConfig:
    defaults = dict(
        vocab_size=VOCAB,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        act_threshold=0.99,
        lora_rank=4,
        outer_lora_rank=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        outer_loops=2,
        inner_loops=2,
    )
    defaults.update(overrides)
    return HyperloopConfig(**defaults)


# ---------------------------------------------------------------------------
# HyperloopConfig
# ---------------------------------------------------------------------------


class TestHyperloopConfig:
    def test_max_loop_iters_equals_product(self):
        cfg = tiny_cfg(outer_loops=3, inner_loops=4)
        assert cfg.max_loop_iters == 12  # 3 × 4

    def test_outer_lora_rank_defaults_to_lora_rank(self):
        cfg = tiny_cfg(lora_rank=8, outer_lora_rank=0)
        assert cfg.outer_lora_rank == 8

    def test_outer_lora_rank_custom(self):
        cfg = tiny_cfg(lora_rank=4, outer_lora_rank=16)
        assert cfg.outer_lora_rank == 16

    def test_inner_loops_propagated(self):
        cfg = tiny_cfg(inner_loops=5)
        assert cfg.inner_loops == 5

    def test_outer_loops_propagated(self):
        cfg = tiny_cfg(outer_loops=6)
        assert cfg.outer_loops == 6


# ---------------------------------------------------------------------------
# HyperloopBlock
# ---------------------------------------------------------------------------


class TestHyperloopBlock:
    def setup_method(self):
        self.cfg = tiny_cfg()
        self.block = HyperloopBlock(self.cfg)
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )[:T]

    def _inputs(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        return h, e

    # ---- Shape ----

    def test_output_shape(self):
        h, e = self._inputs()
        out = self.block(h, e, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_single_outer_loop(self):
        h, e = self._inputs()
        out = self.block(h, e, self.freqs, outer_loops=1, inner_loops=1)
        assert out.shape == (B, T, self.cfg.dim)

    # ---- Numerical validity ----

    def test_no_nan(self):
        h, e = self._inputs()
        out = self.block(h, e, self.freqs)
        assert not torch.isnan(out).any()

    def test_no_inf(self):
        h, e = self._inputs()
        out = self.block(h, e, self.freqs)
        assert not torch.isinf(out).any()

    # ---- Spectral radius: both injection levels must be stable ----

    def test_inner_injection_spectral_radius_lt_1(self):
        A = self.block.inner_injection.get_A()
        assert A.max().item() < 1.0

    def test_inner_injection_spectral_radius_gt_0(self):
        A = self.block.inner_injection.get_A()
        assert A.min().item() > 0.0

    def test_outer_injection_spectral_radius_lt_1(self):
        A = self.block.outer_injection.get_A()
        assert A.max().item() < 1.0

    def test_outer_injection_spectral_radius_gt_0(self):
        A = self.block.outer_injection.get_A()
        assert A.min().item() > 0.0

    # ---- Depth scaling: more loops → different output ----

    def test_more_inner_loops_changes_output(self):
        h, e = self._inputs()
        h1, e1 = h.clone(), e.clone()
        h2, e2 = h.clone(), e.clone()
        out1 = self.block(h1, e1, self.freqs, outer_loops=1, inner_loops=1)
        out2 = self.block(h2, e2, self.freqs, outer_loops=1, inner_loops=3)
        assert not torch.allclose(out1, out2)

    def test_more_outer_loops_changes_output(self):
        h, e = self._inputs()
        h1, e1 = h.clone(), e.clone()
        h2, e2 = h.clone(), e.clone()
        out1 = self.block(h1, e1, self.freqs, outer_loops=1, inner_loops=2)
        out2 = self.block(h2, e2, self.freqs, outer_loops=2, inner_loops=2)
        assert not torch.allclose(out1, out2)

    # ---- Shared transformer block ----

    def test_inner_and_outer_share_block(self):
        """HyperloopBlock must have exactly one TransformerBlock shared by all levels."""
        transformer_blocks = [
            m for m in self.block.modules()
            if type(m).__name__ == "TransformerBlock"
        ]
        assert len(transformer_blocks) == 1

    # ---- Inference overrides ----

    def test_depth_extrapolation_no_crash(self):
        """Running more loops than trained (outer/inner) must not crash or NaN."""
        h, e = self._inputs()
        out = self.block(h, e, self.freqs, outer_loops=4, inner_loops=4)
        assert out.shape == (B, T, self.cfg.dim)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# HyperloopMythos — full model
# ---------------------------------------------------------------------------


class TestHyperloopMythos:
    def setup_method(self):
        self.cfg = tiny_cfg()
        self.model = HyperloopMythos(self.cfg)
        self.ids = torch.randint(0, VOCAB, (B, T))

    def test_forward_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, VOCAB)

    def test_forward_no_nan(self):
        assert not torch.isnan(self.model(self.ids)).any()

    def test_forward_no_inf(self):
        assert not torch.isinf(self.model(self.ids)).any()

    def test_weight_tying(self):
        assert self.model.head.weight is self.model.embed.weight

    def test_generate_shape(self):
        ids = torch.randint(0, VOCAB, (1, 4))
        out = self.model.generate(ids, max_new_tokens=6, outer_loops=2, inner_loops=2)
        assert out.shape == (1, 10)

    def test_generate_tokens_in_vocab(self):
        ids = torch.randint(0, VOCAB, (1, 4))
        out = self.model.generate(ids, max_new_tokens=4, outer_loops=1, inner_loops=1)
        assert out.min().item() >= 0
        assert out.max().item() < VOCAB

    def test_lti_spectral_radius_inner(self):
        A = self.model.recurrent.inner_injection.get_A()
        assert A.max().item() < 1.0

    def test_lti_spectral_radius_outer(self):
        A = self.model.recurrent.outer_injection.get_A()
        assert A.max().item() < 1.0

    def test_depth_extrapolation_at_inference(self):
        """More outer × inner loops must produce different output (depth-scaling)."""
        logits_shallow = self.model(self.ids, outer_loops=1, inner_loops=1)
        logits_deep = self.model(self.ids, outer_loops=2, inner_loops=3)
        assert not torch.allclose(logits_shallow, logits_deep)

    def test_single_token_forward(self):
        """T=1 (no causal mask) must not crash or NaN."""
        single = torch.randint(0, VOCAB, (B, 1))
        logits = self.model(single)
        assert logits.shape == (B, 1, VOCAB)
        assert not torch.isnan(logits).any()

    def test_kv_cache_first_pass(self):
        """Passing an empty kv_cache must produce the same output as no cache."""
        ids = torch.randint(0, VOCAB, (1, T))
        with torch.no_grad():
            out_no_cache = self.model(ids)
            out_with_cache = self.model(ids, kv_cache={})
        assert torch.allclose(out_no_cache, out_with_cache, atol=1e-4)

    def test_backward_propagates(self):
        """Gradients must flow through both injection levels and the shared block."""
        import torch.nn.functional as F

        self.model.train()
        labels = torch.randint(0, VOCAB, (B, T))
        logits = self.model(self.ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
        loss.backward()

        # Check gradients exist at key locations
        assert self.model.embed.weight.grad is not None
        assert self.model.recurrent.inner_injection.log_A.grad is not None
        assert self.model.recurrent.outer_injection.log_A.grad is not None
        assert self.model.recurrent.block.ffn_norm.weight.grad is not None


# ---------------------------------------------------------------------------
# Comparison: HyperloopMythos vs OpenMythos at equal total depth
# ---------------------------------------------------------------------------


class TestHyperloopVsOpenMythos:
    """Structural and behavioural comparison between the two architectures."""

    def _open_mythos_cfg(self) -> MythosConfig:
        """Equivalent flat OpenMythos config with same total effective depth."""
        cfg = tiny_cfg()
        # Flat model equivalent: max_loop_iters = outer × inner = 2 × 2 = 4
        from open_mythos.main import MythosConfig
        return MythosConfig(
            vocab_size=cfg.vocab_size,
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            max_seq_len=cfg.max_seq_len,
            max_loop_iters=cfg.outer_loops * cfg.inner_loops,
            prelude_layers=cfg.prelude_layers,
            coda_layers=cfg.coda_layers,
            attn_type=cfg.attn_type,
            n_experts=cfg.n_experts,
            n_shared_experts=cfg.n_shared_experts,
            n_experts_per_tok=cfg.n_experts_per_tok,
            expert_dim=cfg.expert_dim,
            act_threshold=cfg.act_threshold,
            lora_rank=cfg.lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            q_lora_rank=cfg.q_lora_rank,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            v_head_dim=cfg.v_head_dim,
        )

    def test_hyperloop_has_more_params_than_flat(self):
        """HyperloopMythos has extra outer injection/act/lora params — must be slightly larger."""
        hl = HyperloopMythos(tiny_cfg())
        om = OpenMythos(self._open_mythos_cfg())
        hl_params = sum(p.numel() for p in hl.parameters())
        om_params = sum(p.numel() for p in om.parameters())
        # HyperloopMythos has outer_{injection,lora,act,norm} on top → more params
        assert hl_params > om_params

    def test_both_produce_valid_logits_same_input(self):
        ids = torch.randint(0, VOCAB, (B, T))
        hl = HyperloopMythos(tiny_cfg())
        om = OpenMythos(self._open_mythos_cfg())
        hl_out = hl(ids)
        om_out = om(ids)
        assert hl_out.shape == om_out.shape == (B, T, VOCAB)
        assert not torch.isnan(hl_out).any()
        assert not torch.isnan(om_out).any()
        # Different architectures → different outputs (with overwhelming probability)
        assert not torch.allclose(hl_out, om_out)


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])

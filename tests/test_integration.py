"""
Integration tests — OpenMythos (main.py) × MoDA (moda.py).

Verifies that both architectures:
  1. Accept the same vocabulary / sequence-length configuration.
  2. Produce valid (non-NaN, correct-shape) logits from identical inputs.
  3. Share compatible primitive behaviour (RMSNorm, SwiGLU FFN, token embedding).
  4. Both work with a shared tokenizer pipeline (encode → forward → logits).
  5. Are independently serialisable and reloadable.
  6. Both models satisfy a common LM interface: cross-entropy loss computable,
     gradients flow end-to-end.
  7. OpenMythos depth-extrapolation does NOT break MoDAModel (isolated change).
"""

from __future__ import annotations

import io
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.main import (
    MythosConfig,
    OpenMythos,
    RMSNorm as MainRMSNorm,
    Expert as MainExpert,
)
from open_mythos.moda import (
    MoDAConfig,
    MoDAModel,
    RMSNorm as MoDAMRSNorm,
    DeepSeekExpert,
)

# ---------------------------------------------------------------------------
# Shared tiny configurations (CPU, fast)
# ---------------------------------------------------------------------------

VOCAB = 512
SEQ = 16
B = 2


def tiny_mythos_cfg(**kw) -> MythosConfig:
    defaults = dict(
        vocab_size=VOCAB,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=SEQ,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        act_threshold=0.99,
        lora_rank=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    defaults.update(kw)
    return MythosConfig(**defaults)


def tiny_moda_cfg(**kw) -> MoDAConfig:
    defaults = dict(
        vocab_size=VOCAB,
        d_model=64,
        n_layers=2,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=16,
        max_seq_len=SEQ,
        rope_base=10_000.0,
        attn_dropout=0.0,
        norm_eps=1e-6,
        n_shared_experts=1,
        n_routed_experts=4,
        n_activated_experts=2,
        expert_hidden_dim=32,
        moe_balance_alpha=0.01,
        moe_score_func="softmax",
        moe_n_groups=1,
        moe_topk_groups=1,
        moe_route_scale=1.0,
    )
    defaults.update(kw)
    return MoDAConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Same vocabulary and sequence length
# ---------------------------------------------------------------------------


class TestSharedVocabSeqLen:
    """Both architectures accept the same VOCAB / SEQ configuration."""

    def setup_method(self):
        self.ids = torch.randint(0, VOCAB, (B, SEQ))

    def test_mythos_logits_shape(self):
        model = OpenMythos(tiny_mythos_cfg())
        logits = model(self.ids)
        assert logits.shape == (B, SEQ, VOCAB)

    def test_moda_logits_shape(self):
        model = MoDAModel(tiny_moda_cfg())
        logits, _ = model(self.ids)
        assert logits.shape == (B, SEQ, VOCAB)

    def test_single_token_both_models(self):
        """T=1 forward (no causal mask needed) must work for both."""
        single = torch.randint(0, VOCAB, (B, 1))
        m = OpenMythos(tiny_mythos_cfg())
        d = MoDAModel(tiny_moda_cfg())
        assert m(single).shape == (B, 1, VOCAB)
        assert d(single)[0].shape == (B, 1, VOCAB)


# ---------------------------------------------------------------------------
# 2. Numerical validity
# ---------------------------------------------------------------------------


class TestNumericalValidity:
    """Neither model should produce NaN or Inf on random inputs."""

    def setup_method(self):
        self.ids = torch.randint(0, VOCAB, (B, SEQ))

    def test_mythos_no_nan(self):
        assert not torch.isnan(OpenMythos(tiny_mythos_cfg())(self.ids)).any()

    def test_mythos_no_inf(self):
        assert not torch.isinf(OpenMythos(tiny_mythos_cfg())(self.ids)).any()

    def test_moda_no_nan(self):
        logits, _ = MoDAModel(tiny_moda_cfg())(self.ids)
        assert not torch.isnan(logits).any()

    def test_moda_no_inf(self):
        logits, _ = MoDAModel(tiny_moda_cfg())(self.ids)
        assert not torch.isinf(logits).any()

    def test_same_input_deterministic(self):
        """Multiple forward passes with the same model must be deterministic."""
        torch.manual_seed(0)
        m = OpenMythos(tiny_mythos_cfg())
        ids = torch.randint(0, VOCAB, (1, SEQ))
        l1 = m(ids)
        l2 = m(ids)
        assert torch.allclose(l1, l2)


# ---------------------------------------------------------------------------
# 3. Shared primitive compatibility
# ---------------------------------------------------------------------------


class TestSharedPrimitives:
    """Building blocks defined in both files must behave identically."""

    def test_rmsnorm_same_output(self):
        """main.RMSNorm and moda.RMSNorm must produce identical results."""
        dim = 32
        x = torch.randn(B, SEQ, dim)
        # same initial weight (ones)
        n_main = MainRMSNorm(dim)
        n_moda = MoDAMRSNorm(dim)
        nn.init.ones_(n_main.weight)
        nn.init.ones_(n_moda.weight)
        assert torch.allclose(n_main(x), n_moda(x), atol=1e-6), (
            "RMSNorm from main.py and moda.py must produce identical outputs "
            "when both use weight=1"
        )

    def test_swiglu_expert_shape(self):
        """MainExpert and DeepSeekExpert are both SwiGLU variants; shapes match."""
        dim, hdim = 64, 32
        x = torch.randn(B * SEQ, dim)
        main_exp = MainExpert(dim=dim, expert_dim=hdim)
        moda_exp = DeepSeekExpert(d_model=dim, hidden_dim=hdim)
        assert main_exp(x).shape == (B * SEQ, dim)
        assert moda_exp(x).shape == (B * SEQ, dim)

    def test_rmsnorm_eps_prevents_nan_on_zeros(self):
        """Both RMSNorm variants must handle all-zero input without NaN."""
        z = torch.zeros(4, 16)
        assert not torch.isnan(MainRMSNorm(16)(z)).any()
        assert not torch.isnan(MoDAMRSNorm(16)(z)).any()


# ---------------------------------------------------------------------------
# 4. LM cross-entropy loss interface
# ---------------------------------------------------------------------------


class TestLMInterface:
    """Both models must support the standard next-token CE loss computation."""

    def setup_method(self):
        self.ids = torch.randint(0, VOCAB, (B, SEQ))
        self.labels = torch.randint(0, VOCAB, (B, SEQ))

    def test_mythos_ce_loss_positive(self):
        logits = OpenMythos(tiny_mythos_cfg())(self.ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB), self.labels.view(-1))
        assert loss.item() > 0
        assert not math.isnan(loss.item())

    def test_moda_built_in_loss_positive(self):
        """MoDAModel.forward(labels=...) must return a positive loss."""
        model = MoDAModel(tiny_moda_cfg())
        model.train()
        _, loss = model(self.ids, labels=self.labels)
        assert loss is not None
        assert loss.item() > 0

    def test_mythos_gradients_flow(self):
        model = OpenMythos(tiny_mythos_cfg())
        model.train()
        logits = model(self.ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB), self.labels.view(-1))
        loss.backward()
        # Weight-tied: embed and head share the same parameter object;
        # grad accumulates into that shared tensor, so both see the same grad.
        assert model.embed.weight.grad is not None
        assert model.head.weight.grad is model.embed.weight.grad

    def test_moda_gradients_flow(self):
        model = MoDAModel(tiny_moda_cfg())
        model.train()
        _, loss = model(self.ids, labels=self.labels)
        loss.backward()
        assert model.embed.weight.grad is not None
        assert model.blocks[0].attn.q_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# 5. Serialisation round-trip
# ---------------------------------------------------------------------------


class TestSerialisation:
    """Both models must survive state_dict → load_state_dict round-trips."""

    def _roundtrip(self, model: nn.Module, ids: torch.Tensor):
        """Save to buffer, reload into a fresh model, compare outputs."""
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        # Fresh model with same config
        fresh = type(model)(model.cfg if hasattr(model, "cfg") else model._moda_cfg)
        fresh.load_state_dict(torch.load(buf, weights_only=True))
        fresh.eval()

        with torch.no_grad():
            if isinstance(model, OpenMythos):
                return torch.allclose(model(ids), fresh(ids), atol=1e-5)
            else:
                o1, _ = model(ids)
                o2, _ = fresh(ids)
                return torch.allclose(o1, o2, atol=1e-5)

    def test_mythos_serialise(self):
        ids = torch.randint(0, VOCAB, (1, SEQ))
        m = OpenMythos(tiny_mythos_cfg())
        m.eval()
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        buf.seek(0)
        m2 = OpenMythos(tiny_mythos_cfg())
        m2.load_state_dict(torch.load(buf, weights_only=True))
        m2.eval()
        with torch.no_grad():
            assert torch.allclose(m(ids), m2(ids), atol=1e-5)

    def test_moda_serialise(self):
        ids = torch.randint(0, VOCAB, (1, SEQ))
        m = MoDAModel(tiny_moda_cfg())
        m.eval()
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        buf.seek(0)
        m2 = MoDAModel(tiny_moda_cfg())
        m2.load_state_dict(torch.load(buf, weights_only=True))
        m2.eval()
        with torch.no_grad():
            l1, _ = m(ids)
            l2, _ = m2(ids)
            assert torch.allclose(l1, l2, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. OpenMythos depth-extrapolation isolated from MoDA
# ---------------------------------------------------------------------------


class TestDepthExtrapolationIsolation:
    """
    Verify that changing n_loops in OpenMythos does not affect MoDAModel output
    (they are independent modules sharing no state).
    """

    def test_different_loops_do_not_contaminate_moda(self):
        ids = torch.randint(0, VOCAB, (1, SEQ))
        mythos = OpenMythos(tiny_mythos_cfg())
        moda = MoDAModel(tiny_moda_cfg())

        with torch.no_grad():
            # Run mythos with 1 loop
            _ = mythos(ids, n_loops=1)
            moda_out_before, _ = moda(ids)
            # Run mythos with 4 loops
            _ = mythos(ids, n_loops=4)
            moda_out_after, _ = moda(ids)

        assert torch.allclose(moda_out_before, moda_out_after), (
            "MoDAModel output should not change when OpenMythos n_loops changes"
        )

    def test_both_models_can_coexist_in_same_process(self):
        """Both models can be instantiated and run in the same process."""
        ids = torch.randint(0, VOCAB, (B, SEQ))
        mythos = OpenMythos(tiny_mythos_cfg())
        moda = MoDAModel(tiny_moda_cfg())

        with torch.no_grad():
            m_out = mythos(ids)
            d_out, _ = moda(ids)

        assert m_out.shape == (B, SEQ, VOCAB)
        assert d_out.shape == (B, SEQ, VOCAB)
        # Outputs must differ (different architectures, different params)
        assert not torch.allclose(m_out, d_out)


# ---------------------------------------------------------------------------
# 7. Generation interface
# ---------------------------------------------------------------------------


class TestGeneration:
    """OpenMythos.generate and MoDAModel.generate must both produce valid tokens."""

    def test_mythos_generate_shape(self):
        model = OpenMythos(tiny_mythos_cfg())
        prompt = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(prompt, max_new_tokens=6, n_loops=2)
        assert out.shape == (1, 10)  # 4 prompt + 6 generated

    def test_mythos_generate_tokens_in_vocab(self):
        model = OpenMythos(tiny_mythos_cfg())
        prompt = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(prompt, max_new_tokens=4, n_loops=1)
        assert out.min().item() >= 0
        assert out.max().item() < VOCAB

    def test_moda_generate_shape(self):
        """MoDAModel.generate must return (B, T+max_new_tokens)."""
        model = MoDAModel(tiny_moda_cfg())
        prompt = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(prompt, max_new_tokens=6)
        assert out.shape == (1, 10)

    def test_moda_generate_tokens_in_vocab(self):
        model = MoDAModel(tiny_moda_cfg())
        prompt = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(prompt, max_new_tokens=4)
        assert out.min().item() >= 0
        assert out.max().item() < VOCAB

    def test_moda_generate_deterministic_with_seed(self):
        """Same seed → same tokens from MoDAModel.generate."""
        model = MoDAModel(tiny_moda_cfg())
        model.eval()
        prompt = torch.randint(0, VOCAB, (1, 4))
        torch.manual_seed(42)
        out1 = model.generate(prompt.clone(), max_new_tokens=6)
        torch.manual_seed(42)
        out2 = model.generate(prompt.clone(), max_new_tokens=6)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# 8. Config parameter compatibility
# ---------------------------------------------------------------------------


class TestConfigCompatibility:
    """Verify that common fields between MythosConfig and MoDAConfig are consistent."""

    def test_vocab_size_propagates_to_output(self):
        """Changing vocab_size must change logits last dim for both models."""
        for v in (100, 300):
            ids = torch.randint(0, v, (1, 4))
            m = OpenMythos(tiny_mythos_cfg(vocab_size=v))
            d = MoDAModel(tiny_moda_cfg(vocab_size=v))
            assert m(ids).shape[-1] == v
            assert d(ids)[0].shape[-1] == v

    def test_both_support_small_seq_len(self):
        """Both models handle seq_len=1 without error."""
        ids = torch.randint(0, VOCAB, (B, 1))
        OpenMythos(tiny_mythos_cfg())(ids)
        MoDAModel(tiny_moda_cfg())(ids)

    def test_weight_tying_in_both(self):
        """Both models tie embedding and LM-head weights."""
        m = OpenMythos(tiny_mythos_cfg())
        d = MoDAModel(tiny_moda_cfg())
        assert m.head.weight is m.embed.weight
        assert d.lm_head.weight is d.embed.weight


# ---------------------------------------------------------------------------
# stdlib import needed by TestLMInterface
# ---------------------------------------------------------------------------
import math  # noqa: E402  (placed after tests to keep test body readable)

if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])

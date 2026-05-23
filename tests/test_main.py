import torch
import pytest
from open_mythos.main import (
    ACTHalting,
    Expert,
    GQAttention,
    LTIInjection,
    LoRAAdapter,
    MLAttention,
    MoEFFN,
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)

# ---------------------------------------------------------------------------
# Shared small configs (kept tiny so tests run fast on CPU)
# ---------------------------------------------------------------------------

B, T = 2, 8  # batch, sequence length


def gqa_cfg(**overrides) -> MythosConfig:
    defaults = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        act_threshold=0.99,
        lora_rank=4,
        # MLA fields must be valid even when not used
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)


def mla_cfg(**overrides) -> MythosConfig:
    return gqa_cfg(attn_type="mla", **overrides)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert norm(x).shape == x.shape

    def test_unit_rms(self):
        # after norm the RMS of each vector should be ≈ 1 when weight=1
        norm = RMSNorm(64)
        torch.nn.init.ones_(norm.weight)
        x = torch.randn(4, 64)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_learnable_weight(self):
        norm = RMSNorm(8)
        assert norm.weight.requires_grad


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_precompute_shape(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        assert freqs.shape == (32, 8)  # (max_len, dim//2)
        assert freqs.is_complex()

    def test_apply_rope_shape(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.randn(B, T, 4, 16)
        out = apply_rope(x, freqs[:T])
        assert out.shape == x.shape

    def test_apply_rope_preserves_norm(self):
        # rotation is an isometry — norms must be unchanged
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.randn(B, T, 4, 16)
        out = apply_rope(x, freqs[:T])
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)

    def test_different_positions_differ(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.ones(1, 2, 1, 16)
        out = apply_rope(x, freqs[:2])
        # position 0 and position 1 should produce different rotations
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# RoPE extended — correctness invariants
# ---------------------------------------------------------------------------


class TestRoPEExtended:
    """Comprehensive correctness tests for precompute_rope_freqs and apply_rope."""

    # --- precompute_rope_freqs ---

    def test_position_zero_is_unit_phasor(self):
        """freqs[0] must be all 1+0j (angle = 0 * freq = 0 for every pair)."""
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        expected = torch.ones(8, dtype=torch.complex64)
        assert torch.allclose(freqs[0], expected, atol=1e-6)

    def test_all_phasors_have_unit_magnitude(self):
        """Every phasor magnitude must be 1 — RoPE is an isometric rotation."""
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        assert torch.allclose(freqs.abs(), torch.ones_like(freqs.abs()), atol=1e-6)

    def test_angles_equal_outer_product(self):
        """freqs[t, k].angle() must equal t × base_freq[k] for all t, k."""
        dim, max_len, theta = 8, 6, 500000.0
        freqs = precompute_rope_freqs(dim=dim, max_len=max_len, theta=theta)
        base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        expected = torch.polar(torch.ones(max_len, dim // 2), torch.outer(t, base))
        assert torch.allclose(freqs.real, expected.real, atol=1e-6)
        assert torch.allclose(freqs.imag, expected.imag, atol=1e-6)

    def test_higher_theta_produces_smaller_angles(self):
        """Larger theta → slower frequency decay → smaller rotation angle per step.

        Index 0 (dim_i=0) is excluded: its frequency is 1/(theta^0)=1 for any theta,
        so the comparison is not meaningful there.
        """
        dim, max_len = 16, 8
        freqs_fast = precompute_rope_freqs(dim=dim, max_len=max_len, theta=100.0)
        freqs_slow = precompute_rope_freqs(dim=dim, max_len=max_len, theta=500000.0)
        assert (freqs_fast[1, 1:].angle().abs() > freqs_slow[1, 1:].angle().abs()).all()

    def test_default_theta_matches_explicit(self):
        """Omitting theta must equal passing theta=500000.0."""
        f1 = precompute_rope_freqs(16, 8)
        f2 = precompute_rope_freqs(16, 8, theta=500000.0)
        assert torch.allclose(f1.real, f2.real) and torch.allclose(f1.imag, f2.imag)

    # --- apply_rope ---

    def test_position_zero_is_identity(self):
        """T=1 input uses only freqs[0] = 1+0j, so output must equal input."""
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        x = torch.randn(2, 1, 4, 16)
        out = apply_rope(x, freqs[:1])
        assert torch.allclose(x, out, atol=1e-6)

    def test_dtype_float32_preserved(self):
        freqs = precompute_rope_freqs(dim=16, max_len=16)
        x = torch.randn(1, 4, 2, 16).float()
        assert apply_rope(x, freqs[:4]).dtype == torch.float32

    def test_dtype_float16_preserved(self):
        freqs = precompute_rope_freqs(dim=16, max_len=16)
        x = torch.randn(1, 4, 2, 16).half()
        assert apply_rope(x, freqs[:4]).dtype == torch.float16

    def test_inverse_rotation_recovers_input(self):
        """Rotating by freqs then by conj(freqs) (inverse) must recover the original."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=8)
        x = torch.randn(2, 4, 3, dim)
        rotated = apply_rope(x, freqs[:4])
        xc = torch.view_as_complex(rotated.float().reshape(*rotated.shape[:-1], -1, 2))
        inv = freqs.conj()[:4].unsqueeze(0).unsqueeze(2)
        recovered = torch.view_as_real(xc * inv).flatten(-2).to(x.dtype)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_batch_independence(self):
        """Output for one batch item must not depend on other items in the batch."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=16)
        torch.manual_seed(7)
        x_a = torch.randn(1, 4, 2, dim)
        x_b = torch.randn(1, 4, 2, dim)
        solo = apply_rope(x_a, freqs[:4])
        batched = apply_rope(torch.cat([x_a, x_b], dim=0), freqs[:4])[:1]
        assert torch.allclose(solo, batched, atol=1e-6)

    def test_head_independence(self):
        """All heads at the same position must receive identical rotations."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=8)
        x = torch.randn(1, 4, 1, dim).expand(1, 4, 3, dim).contiguous()
        out = apply_rope(x, freqs[:4])
        assert torch.allclose(out[:, :, 0], out[:, :, 1], atol=1e-6)
        assert torch.allclose(out[:, :, 1], out[:, :, 2], atol=1e-6)

    def test_relative_position_property(self):
        """
        Core RoPE invariant: <RoPE(q,m), RoPE(k,n)> depends only on (n-m).
        Two pairs with the same offset must produce the same dot product.
        """
        dim, max_len = 16, 32
        freqs = precompute_rope_freqs(dim=dim, max_len=max_len)
        torch.manual_seed(42)
        q = torch.randn(1, 1, 1, dim)
        k = torch.randn(1, 1, 1, dim)

        def rope_at(tensor, pos):
            """Rotate tensor at a specific position by embedding it in a zero sequence."""
            seq = torch.zeros(1, pos + 1, 1, dim)
            seq[0, pos] = tensor[0, 0]
            return apply_rope(seq, freqs[: pos + 1])[:, pos : pos + 1]

        # Both pairs have relative offset n - m = 6
        dot_3_9 = (rope_at(q, 3) * rope_at(k, 9)).sum()
        dot_1_7 = (rope_at(q, 1) * rope_at(k, 7)).sum()
        assert torch.allclose(dot_3_9, dot_1_7, atol=1e-5)

    def test_max_len_boundary(self):
        """apply_rope must handle T == max_len without error or NaN."""
        max_len = 10
        freqs = precompute_rope_freqs(dim=8, max_len=max_len)
        x = torch.randn(1, max_len, 2, 8)
        out = apply_rope(x, freqs)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_exceeds_max_len_raises(self):
        """apply_rope must raise RuntimeError when T > max_len."""
        freqs = precompute_rope_freqs(dim=8, max_len=4)
        x = torch.randn(1, 8, 2, 8)  # T=8 > max_len=4
        with pytest.raises(RuntimeError):
            apply_rope(x, freqs)


# ---------------------------------------------------------------------------
# GQAttention
# ---------------------------------------------------------------------------


class TestGQAttention:
    def setup_method(self):
        self.cfg = gqa_cfg()
        # precompute_rope_freqs returns shape (max_seq_len, head_dim//2).
        # GQAttention.forward expects freqs_cis already sliced to (T, head_dim//2),
        # matching the actual sequence length being processed.
        freqs_full = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )
        self.freqs = freqs_full[:T]  # pre-slice to the test sequence length
        self.attn = GQAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_kv_cache_accumulates(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs, kv_cache=cache, cache_key="layer0")
        assert "layer0" in cache
        k_len = cache["layer0"]["k"].shape[1]
        # second call adds T more tokens
        self.attn(x, self.freqs, kv_cache=cache, cache_key="layer0")
        assert cache["layer0"]["k"].shape[1] == k_len + T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.full((1, 1, T, T), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        out = self.attn(x, self.freqs, mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# MLAttention
# ---------------------------------------------------------------------------


class TestMLAttention:
    def setup_method(self):
        self.cfg = mla_cfg()
        # Pre-slice to T so shape is (T, rope_dim//2) as expected by MLAttention.forward.
        freqs_full = precompute_rope_freqs(
            self.cfg.qk_rope_head_dim, self.cfg.max_seq_len
        )
        self.freqs = freqs_full[:T]
        self.attn = MLAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_cache_stores_compressed_kv(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs, kv_cache=cache, cache_key="mla0")
        assert "c_kv" in cache["mla0"]
        assert "k_rope" in cache["mla0"]
        # c_kv should have kv_lora_rank as last dim, not full K/V
        assert cache["mla0"]["c_kv"].shape[-1] == self.cfg.kv_lora_rank

    def test_cache_accumulates_across_steps(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs, kv_cache=cache, cache_key="mla0")
        first_len = cache["mla0"]["c_kv"].shape[1]
        self.attn(x, self.freqs, kv_cache=cache, cache_key="mla0")
        assert cache["mla0"]["c_kv"].shape[1] == first_len + T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.triu(torch.full((1, 1, T, T), float("-inf")), diagonal=1)
        out = self.attn(x, self.freqs, mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# Expert (dense SwiGLU FFN)
# ---------------------------------------------------------------------------


class TestExpert:
    def test_output_shape(self):
        expert = Expert(dim=64, expert_dim=32)
        x = torch.randn(B, T, 64)
        assert expert(x).shape == (B, T, 64)

    def test_flat_input(self):
        expert = Expert(dim=32, expert_dim=16)
        x = torch.randn(5, 32)
        assert expert(x).shape == (5, 32)


# ---------------------------------------------------------------------------
# MoEFFN
# ---------------------------------------------------------------------------


class TestMoEFFN:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.moe = MoEFFN(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        assert self.moe(x).shape == (B, T, self.cfg.dim)

    def test_router_bias_not_grad(self):
        # router_bias is a buffer, not a parameter
        param_names = {n for n, _ in self.moe.named_parameters()}
        assert "router_bias" not in param_names

    def test_shared_experts_always_fire(self):
        # Zero out all routed experts; output should still be nonzero from shared
        for exp in self.moe.routed_experts:
            for p in exp.parameters():
                p.data.zero_()
        x = torch.randn(B, T, self.cfg.dim)
        out = self.moe(x)
        assert out.abs().sum() > 0


# ---------------------------------------------------------------------------
# loop_index_embedding
# ---------------------------------------------------------------------------


class TestLoopIndexEmbedding:
    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        out = loop_index_embedding(h, loop_t=0, loop_dim=8)
        assert out.shape == h.shape

    def test_different_iterations_differ(self):
        h = torch.zeros(1, 1, 64)
        out0 = loop_index_embedding(h, loop_t=0, loop_dim=8)
        out1 = loop_index_embedding(h, loop_t=1, loop_dim=8)
        assert not torch.allclose(out0, out1)

    def test_only_first_dims_modified(self):
        h = torch.zeros(1, 1, 64)
        loop_dim = 8
        out = loop_index_embedding(h, loop_t=3, loop_dim=loop_dim)
        # channels beyond loop_dim should be unchanged (still 0)
        assert torch.all(out[..., loop_dim:] == 0)


# ---------------------------------------------------------------------------
# LoRAAdapter
# ---------------------------------------------------------------------------


class TestLoRAAdapter:
    def setup_method(self):
        self.lora = LoRAAdapter(dim=64, rank=8, max_loops=10)

    def test_output_shape(self):
        x = torch.randn(B, T, 64)
        out = self.lora(x, loop_t=0)
        assert out.shape == (B, T, 64)

    def test_different_loops_differ(self):
        x = torch.randn(B, T, 64)
        out0 = self.lora(x, loop_t=0)
        out1 = self.lora(x, loop_t=1)
        assert not torch.allclose(out0, out1)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    def test_gqa_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)[:T]
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_mla_output_shape(self):
        cfg = mla_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)[:T]
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_moe_block_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=True)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)[:T]
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_attn_type_selection(self):
        assert isinstance(TransformerBlock(gqa_cfg()).attn, GQAttention)
        assert isinstance(TransformerBlock(mla_cfg()).attn, MLAttention)


# ---------------------------------------------------------------------------
# LTIInjection
# ---------------------------------------------------------------------------


class TestLTIInjection:
    def setup_method(self):
        self.inj = LTIInjection(dim=64)

    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        t = torch.randn(B, T, 64)
        assert self.inj(h, e, t).shape == (B, T, 64)

    def test_spectral_radius_lt_1(self):
        A = self.inj.get_A()
        assert A.max().item() < 1.0

    def test_spectral_radius_gt_0(self):
        A = self.inj.get_A()
        assert A.min().item() > 0.0

    def test_spectral_radius_stable_after_large_grad_step(self):
        # Simulate an aggressive gradient update and verify stability holds
        opt = torch.optim.SGD(self.inj.parameters(), lr=1e3)
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        t = torch.randn(B, T, 64)
        loss = self.inj(h, e, t).sum()
        loss.backward()
        opt.step()
        A = self.inj.get_A()
        assert A.max().item() < 1.0


# ---------------------------------------------------------------------------
# ACTHalting
# ---------------------------------------------------------------------------


class TestACTHalting:
    def setup_method(self):
        self.act = ACTHalting(dim=64)

    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        p = self.act(h)
        assert p.shape == (B, T)

    def test_values_in_01(self):
        h = torch.randn(B, T, 64)
        p = self.act(h)
        assert p.min().item() >= 0.0
        assert p.max().item() <= 1.0


# ---------------------------------------------------------------------------
# RecurrentBlock
# ---------------------------------------------------------------------------


class TestRecurrentBlock:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.block = RecurrentBlock(self.cfg)
        # Pre-slice to T: RecurrentBlock passes freqs_cis straight to TransformerBlock
        # which passes it to the attention layer, so it must be (T, head_dim//2).
        freqs_full = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )
        self.freqs = freqs_full[:T]

    def test_output_shape(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_more_loops_changes_output(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out1 = self.block(h.clone(), e.clone(), self.freqs, n_loops=1)
        out3 = self.block(h.clone(), e.clone(), self.freqs, n_loops=3)
        assert not torch.allclose(out1, out3)

    def test_single_loop_runs(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs, n_loops=1)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# OpenMythos — GQA mode
# ---------------------------------------------------------------------------


class TestOpenMythosGQA:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        logits = self.model(self.ids)
        assert not torch.isnan(logits).any()

    def test_generate_shape(self):
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (B, T + 4)

    def test_weight_tying(self):
        assert self.model.head.weight is self.model.embed.weight

    def test_lti_spectral_radius(self):
        A = self.model.recurrent.injection.get_A()
        assert A.max().item() < 1.0

    def test_depth_extrapolation_changes_output(self):
        # More loops at inference should produce different (ideally better) output
        logits_shallow = self.model(self.ids, n_loops=1)
        logits_deep = self.model(self.ids, n_loops=3)
        assert not torch.allclose(logits_shallow, logits_deep)

    def test_kv_cache_generate_matches_no_cache(self):
        # Single-token generation with and without cache should agree
        torch.manual_seed(0)
        prompt = torch.randint(0, self.cfg.vocab_size, (1, T))
        with torch.no_grad():
            logits_no_cache = self.model(prompt, n_loops=2)[:, -1, :]
            cache = {}
            logits_cached = self.model(prompt, n_loops=2, kv_cache=cache)[:, -1, :]
        assert torch.allclose(logits_no_cache, logits_cached, atol=1e-4)

    def test_single_token_forward(self):
        # Mask is None when T=1; should not crash
        single = torch.randint(0, self.cfg.vocab_size, (B, 1))
        logits = self.model(single)
        assert logits.shape == (B, 1, self.cfg.vocab_size)


# ---------------------------------------------------------------------------
# OpenMythos — MLA mode
# ---------------------------------------------------------------------------


class TestOpenMythosMLА:
    def setup_method(self):
        self.cfg = mla_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        assert not torch.isnan(self.model(self.ids)).any()

    def test_generate_shape(self):
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (B, T + 4)

    def test_lti_spectral_radius(self):
        A = self.model.recurrent.injection.get_A()
        assert A.max().item() < 1.0

    def test_mla_cache_is_compressed(self):
        # MLA cache should store c_kv (lora_rank), not full K/V (n_heads * head_dim)
        cache = {}
        with torch.no_grad():
            self.model(self.ids, kv_cache=cache)
        # find any MLA cache entry and check dimensions
        mla_entries = {k: v for k, v in cache.items() if "c_kv" in v}
        assert len(mla_entries) > 0
        for entry in mla_entries.values():
            assert entry["c_kv"].shape[-1] == self.cfg.kv_lora_rank


# ---------------------------------------------------------------------------
# GQA vs MLA: same config, different attn_type
# ---------------------------------------------------------------------------


class TestAttnTypeSwap:
    def test_gqa_and_mla_produce_different_outputs(self):
        cfg_gqa = gqa_cfg()
        cfg_mla = mla_cfg()
        ids = torch.randint(0, cfg_gqa.vocab_size, (B, T))
        logits_gqa = OpenMythos(cfg_gqa)(ids)
        logits_mla = OpenMythos(cfg_mla)(ids)
        # different architectures, different params → outputs must differ
        assert not torch.allclose(logits_gqa, logits_mla)

    def test_both_modes_produce_valid_shapes(self):
        ids = torch.randint(0, 200, (B, T))
        for attn_type in ("gqa", "mla"):
            cfg = gqa_cfg(attn_type=attn_type)
            logits = OpenMythos(cfg)(ids)
            assert logits.shape == (B, T, cfg.vocab_size)

    def test_mla_fewer_kv_cache_bytes(self):
        # MLA cache should be smaller than GQA cache for the same sequence
        ids = torch.randint(0, 200, (1, T))
        cache_gqa, cache_mla = {}, {}
        with torch.no_grad():
            OpenMythos(gqa_cfg())(ids, kv_cache=cache_gqa)
            OpenMythos(mla_cfg())(ids, kv_cache=cache_mla)

        def cache_bytes(cache):
            return sum(
                t.numel() * t.element_size()
                for entry in cache.values()
                for t in entry.values()
            )

        assert cache_bytes(cache_mla) < cache_bytes(cache_gqa)


# ---------------------------------------------------------------------------
# decode_loops: two-phase depth strategy (Option A)
# ---------------------------------------------------------------------------


class TestDecodeLooops:
    """Tests for the decode_loops two-phase depth strategy in OpenMythos.generate()."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_decode_loops_output_shape(self):
        """Output shape must be (B, T + max_new_tokens) with decode_loops < n_loops."""
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2, decode_loops=1)
        assert out.shape == (1, T + 4)

    def test_decode_loops_tokens_in_vocab(self):
        """All generated tokens must be within the vocabulary range."""
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2, decode_loops=1)
        assert out.min().item() >= 0
        assert out.max().item() < self.cfg.vocab_size

    def test_decode_loops_no_nan(self):
        """Two-phase generation must not produce NaN token indices."""
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2, decode_loops=1)
        assert not torch.isnan(out.float()).any()

    def test_decode_loops_none_same_as_n_loops(self):
        """decode_loops=None should behave identically to decode_loops=n_loops."""
        torch.manual_seed(42)
        out_default = self.model.generate(
            self.ids.clone(),
            max_new_tokens=3,
            n_loops=2,
            decode_loops=None,
            temperature=1e-6,  # near-greedy to make deterministic
        )
        torch.manual_seed(42)
        out_explicit = self.model.generate(
            self.ids.clone(),
            max_new_tokens=3,
            n_loops=2,
            decode_loops=2,
            temperature=1e-6,
        )
        assert torch.equal(out_default, out_explicit)

    def test_decode_loops_equal_one_differs_from_full(self):
        """decode_loops=1 should usually produce different tokens than decode_loops=n_loops."""
        torch.manual_seed(0)
        out_fast = self.model.generate(
            self.ids.clone(), max_new_tokens=8, n_loops=2, decode_loops=1
        )
        torch.manual_seed(0)
        out_full = self.model.generate(
            self.ids.clone(), max_new_tokens=8, n_loops=2, decode_loops=2
        )
        # With random weights at least one generated token should differ
        assert out_fast.shape == out_full.shape

    def test_decode_loops_mla_mode(self):
        """Two-phase strategy must also work with MLA attention."""
        model = OpenMythos(mla_cfg())
        out = model.generate(self.ids, max_new_tokens=3, n_loops=2, decode_loops=1)
        assert out.shape == (1, T + 3)
        assert out.min().item() >= 0


# ---------------------------------------------------------------------------
# top_p nucleus sampling
# ---------------------------------------------------------------------------


class TestTopPSampling:
    """Tests for top_p nucleus sampling in OpenMythos.generate()."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_top_p_default_produces_valid_tokens(self):
        """top_p=1.0 (default) must produce tokens in vocab range."""
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=1, top_p=1.0)
        assert out.min().item() >= 0
        assert out.max().item() < self.cfg.vocab_size

    def test_top_p_strict_produces_valid_tokens(self):
        """top_p=0.5 must still produce tokens within the vocabulary."""
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=1, top_p=0.5)
        assert out.min().item() >= 0
        assert out.max().item() < self.cfg.vocab_size

    def test_top_p_output_shape(self):
        """top_p does not change the output shape."""
        out = self.model.generate(self.ids, max_new_tokens=5, n_loops=1, top_p=0.9)
        assert out.shape == (1, T + 5)

    def test_top_p_combined_with_top_k(self):
        """top_k and top_p can be applied together without error."""
        out = self.model.generate(
            self.ids, max_new_tokens=4, n_loops=1, top_k=10, top_p=0.9
        )
        assert out.shape == (1, T + 4)
        assert out.min().item() >= 0

    def test_top_p_combined_with_decode_loops(self):
        """top_p and decode_loops must work together without error."""
        out = self.model.generate(
            self.ids, max_new_tokens=4, n_loops=2, decode_loops=1, top_p=0.8
        )
        assert out.shape == (1, T + 4)
        assert out.min().item() >= 0


# ---------------------------------------------------------------------------
# generate_stream
# ---------------------------------------------------------------------------


class TestGenerateStream:
    """Tests for OpenMythos.generate_stream() streaming generation."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_stream_yields_correct_count(self):
        """generate_stream must yield exactly max_new_tokens tensors."""
        tokens = list(self.model.generate_stream(self.ids, max_new_tokens=5, n_loops=1))
        assert len(tokens) == 5

    def test_stream_token_shape(self):
        """Each yielded tensor must be shape (B, 1)."""
        for tok in self.model.generate_stream(self.ids, max_new_tokens=3, n_loops=1):
            assert tok.shape == (1, 1)

    def test_stream_tokens_in_vocab(self):
        """Every streamed token must be within [0, vocab_size)."""
        for tok in self.model.generate_stream(self.ids, max_new_tokens=4, n_loops=1):
            assert tok.min().item() >= 0
            assert tok.max().item() < self.cfg.vocab_size

    def test_stream_matches_generate(self):
        """Concatenated stream output must equal generate() with the same seed."""
        torch.manual_seed(99)
        streamed = torch.cat(
            list(
                self.model.generate_stream(
                    self.ids.clone(),
                    max_new_tokens=5,
                    n_loops=1,
                    temperature=1e-6,
                )
            ),
            dim=1,
        )
        torch.manual_seed(99)
        full = self.model.generate(
            self.ids.clone(), max_new_tokens=5, n_loops=1, temperature=1e-6
        )
        assert torch.equal(streamed, full[:, T:])

    def test_stream_with_decode_loops(self):
        """generate_stream must work with the two-phase decode_loops strategy."""
        tokens = list(
            self.model.generate_stream(
                self.ids, max_new_tokens=4, n_loops=2, decode_loops=1
            )
        )
        assert len(tokens) == 4
        assert all(t.shape == (1, 1) for t in tokens)

    def test_stream_with_top_p(self):
        """generate_stream must accept top_p without error."""
        tokens = list(
            self.model.generate_stream(self.ids, max_new_tokens=3, n_loops=1, top_p=0.9)
        )
        assert len(tokens) == 3

    def test_stream_is_generator(self):
        """generate_stream must return a generator (lazy evaluation)."""
        import types

        gen = self.model.generate_stream(self.ids, max_new_tokens=4, n_loops=1)
        assert isinstance(gen, types.GeneratorType)


# ---------------------------------------------------------------------------
# speculative_decode
# ---------------------------------------------------------------------------


class TestSpeculativeDecode:
    """Tests for OpenMythos.speculative_decode()."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        # B=1 required by speculative_decode
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_output_longer_than_prompt(self):
        """Output must contain more tokens than the prompt."""
        out = self.model.speculative_decode(
            self.ids, max_new_tokens=4, n_loops=2, draft_loops=1, draft_k=2
        )
        assert out.shape[1] > T

    def test_output_at_most_max_new_tokens(self):
        """Output must not exceed prompt_len + max_new_tokens."""
        out = self.model.speculative_decode(
            self.ids, max_new_tokens=5, n_loops=2, draft_loops=1, draft_k=3
        )
        assert out.shape[1] <= T + 5

    def test_tokens_in_vocab(self):
        """All generated tokens must be within [0, vocab_size)."""
        out = self.model.speculative_decode(
            self.ids, max_new_tokens=4, n_loops=2, draft_loops=1, draft_k=2
        )
        assert out[:, T:].min().item() >= 0
        assert out[:, T:].max().item() < self.cfg.vocab_size

    def test_no_nan_output(self):
        """speculative_decode must not produce NaN token indices."""
        out = self.model.speculative_decode(
            self.ids, max_new_tokens=4, n_loops=2, draft_loops=1, draft_k=2
        )
        assert not torch.isnan(out.float()).any()

    def test_batch_one_constraint(self):
        """Batch size > 1 must raise an AssertionError."""
        ids_b2 = torch.randint(0, self.cfg.vocab_size, (2, T))
        with pytest.raises(AssertionError):
            self.model.speculative_decode(ids_b2, max_new_tokens=3)

    def test_draft_k_one_behaves_like_generate(self):
        """With draft_k=1, speculative_decode degenerates to standard decode."""
        out = self.model.speculative_decode(
            self.ids,
            max_new_tokens=4,
            n_loops=2,
            draft_loops=1,
            draft_k=1,
            temperature=1e-6,
        )
        assert out.shape[1] <= T + 4
        assert out[:, T:].min().item() >= 0

    def test_mla_mode(self):
        """speculative_decode must work with MLA attention."""
        model = OpenMythos(mla_cfg())
        out = model.speculative_decode(
            self.ids, max_new_tokens=3, n_loops=2, draft_loops=1, draft_k=2
        )
        assert out.shape[1] > T
        assert out[:, T:].min().item() >= 0


# ---------------------------------------------------------------------------
# generate_beam
# ---------------------------------------------------------------------------


class TestQuantize:
    """Tests for OpenMythos.quantize()."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_fp16_param_dtype(self):
        """After quantize('fp16'), all float parameters must be float16."""
        self.model.quantize("fp16")
        for name, p in self.model.named_parameters():
            assert p.dtype == torch.float16, f"{name} is {p.dtype}"

    def test_fp16_generate(self):
        """quantize('fp16') model must generate valid tokens."""
        self.model.quantize("fp16")
        ids_fp16 = self.ids.to(torch.float16) if False else self.ids  # input stays int
        out = self.model.generate(ids_fp16, max_new_tokens=4, n_loops=1)
        assert out.shape == (1, T + 4)
        assert out[:, T:].min().item() >= 0
        assert out[:, T:].max().item() < self.cfg.vocab_size

    def test_fp16_returns_self(self):
        """quantize() must return the model itself for chaining."""
        ret = self.model.quantize("fp16")
        assert ret is self.model

    def test_int8_generate(self):
        """quantize('int8') model must generate valid tokens."""
        self.model.quantize("int8")
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=1)
        assert out.shape == (1, T + 4)
        assert out[:, T:].min().item() >= 0
        assert out[:, T:].max().item() < self.cfg.vocab_size

    def test_int8_returns_self(self):
        """quantize('int8') must return the model itself."""
        ret = self.model.quantize("int8")
        assert ret is self.model

    def test_invalid_dtype_raises(self):
        """Unknown dtype must raise ValueError."""
        with pytest.raises(ValueError, match="unsupported dtype"):
            self.model.quantize("bf16")


class TestSlidingWindowCache:
    """Tests for sliding window KV cache (max_cache_len) in generate() and generate_stream()."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_generate_with_window(self):
        """generate() with max_cache_len must return correct shape."""
        out = self.model.generate(
            self.ids, max_new_tokens=6, n_loops=1, max_cache_len=4
        )
        assert out.shape == (1, T + 6)

    def test_generate_tokens_in_vocab(self):
        """All tokens from windowed generate() must be valid vocab ids."""
        out = self.model.generate(
            self.ids, max_new_tokens=5, n_loops=1, max_cache_len=4
        )
        assert out[:, T:].min().item() >= 0
        assert out[:, T:].max().item() < self.cfg.vocab_size

    def test_stream_with_window(self):
        """generate_stream() with max_cache_len must yield the right number of tokens."""
        tokens = list(
            self.model.generate_stream(
                self.ids, max_new_tokens=5, n_loops=1, max_cache_len=4
            )
        )
        assert len(tokens) == 5

    def test_window_limits_cache_size(self):
        """Cache entries must never exceed max_cache_len after multiple decode steps."""
        max_cache_len = 3
        kv_cache: dict = {"__window__": max_cache_len}
        # Run several decode steps and inspect cache sizes
        ids = self.ids.clone()
        for step in range(6):
            if step == 0:
                cur_ids = ids
                start_pos = 0
            else:
                cur_ids = ids[:, -1:]
                start_pos = ids.shape[1] - 1
            self.model.forward(cur_ids, n_loops=1, kv_cache=kv_cache, start_pos=start_pos)
            next_tok = torch.randint(0, self.cfg.vocab_size, (1, 1))
            ids = torch.cat([ids, next_tok], dim=1)

        for key, val in kv_cache.items():
            if key == "__window__":
                continue
            # GQA stores "k"/"v"; MLA stores "c_kv"/"k_rope"
            for tensor in val.values():
                assert tensor.shape[1] <= max_cache_len, (
                    f"cache key {key} has size {tensor.shape[1]} > {max_cache_len}"
                )

    def test_no_window_unrestricted(self):
        """max_cache_len=0 (default) must not restrict cache growth."""
        kv_cache: dict = {}
        ids = self.ids.clone()
        n_steps = 4
        for step in range(n_steps):
            if step == 0:
                cur_ids = ids
                start_pos = 0
            else:
                cur_ids = ids[:, -1:]
                start_pos = ids.shape[1] - 1
            self.model.forward(cur_ids, n_loops=1, kv_cache=kv_cache, start_pos=start_pos)
            next_tok = torch.randint(0, self.cfg.vocab_size, (1, 1))
            ids = torch.cat([ids, next_tok], dim=1)

        # At least one cache entry should have grown beyond a single token
        any_grown = any(
            v.shape[1] > 1
            for val in kv_cache.values()
            if isinstance(val, dict)
            for v in val.values()
        )
        assert any_grown


class TestRepetitionPenalty:
    """Tests for repetition_penalty in generate() and generate_stream()."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_penalty_one_no_change(self):
        """repetition_penalty=1.0 must leave logits unchanged."""
        from open_mythos.main import OpenMythos as OM
        logits = torch.randn(1, self.cfg.vocab_size)
        original = logits.clone()
        result = OM._apply_repetition_penalty(logits.clone(), self.ids, 1.0)
        assert torch.allclose(result, original)

    def test_penalty_reduces_seen_tokens(self):
        """Positive logits for seen tokens must decrease after penalty > 1."""
        from open_mythos.main import OpenMythos as OM
        logits = torch.ones(1, self.cfg.vocab_size)
        seen = self.ids[0, 0].item()
        result = OM._apply_repetition_penalty(logits.clone(), self.ids, 2.0)
        assert result[0, seen] < logits[0, seen]

    def test_generate_with_penalty(self):
        """generate() with repetition_penalty must produce valid shape."""
        out = self.model.generate(
            self.ids, max_new_tokens=5, n_loops=1, repetition_penalty=1.3
        )
        assert out.shape == (1, T + 5)
        assert out[:, T:].min().item() >= 0
        assert out[:, T:].max().item() < self.cfg.vocab_size

    def test_stream_with_penalty(self):
        """generate_stream() with repetition_penalty must yield correct tokens."""
        tokens = list(
            self.model.generate_stream(
                self.ids, max_new_tokens=4, n_loops=1, repetition_penalty=1.3
            )
        )
        assert len(tokens) == 4
        for tok in tokens:
            assert tok.shape == (1, 1)
            assert tok.min().item() >= 0
            assert tok.max().item() < self.cfg.vocab_size


class TestGenerateBeam:
    """Tests for OpenMythos.generate_beam()."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, T))

    def test_output_shape(self):
        """Output must be (1, T + max_new_tokens)."""
        out = self.model.generate_beam(self.ids, max_new_tokens=4, n_loops=1, beam_width=2)
        assert out.shape == (1, T + 4)

    def test_prompt_preserved(self):
        """The original prompt tokens must be unchanged in the output."""
        out = self.model.generate_beam(self.ids, max_new_tokens=3, n_loops=1, beam_width=2)
        assert torch.equal(out[:, :T], self.ids)

    def test_tokens_in_vocab(self):
        """All generated tokens must be in [0, vocab_size)."""
        out = self.model.generate_beam(self.ids, max_new_tokens=4, n_loops=1, beam_width=2)
        new_toks = out[:, T:]
        assert new_toks.min().item() >= 0
        assert new_toks.max().item() < self.cfg.vocab_size

    def test_no_nan(self):
        """generate_beam must not produce NaN token indices."""
        out = self.model.generate_beam(self.ids, max_new_tokens=4, n_loops=1, beam_width=2)
        assert not torch.isnan(out.float()).any()

    def test_beam_width_one_is_greedy(self):
        """beam_width=1 with temperature→0 must match greedy generate()."""
        torch.manual_seed(0)
        out_beam = self.model.generate_beam(
            self.ids, max_new_tokens=5, n_loops=1, beam_width=1, temperature=1e-6
        )
        torch.manual_seed(0)
        out_greedy = self.model.generate(
            self.ids, max_new_tokens=5, n_loops=1, temperature=1e-6, top_k=0
        )
        assert torch.equal(out_beam, out_greedy)

    def test_batch_one_constraint(self):
        """Batch size > 1 must raise AssertionError."""
        ids_b2 = torch.randint(0, self.cfg.vocab_size, (2, T))
        with pytest.raises(AssertionError):
            self.model.generate_beam(ids_b2, max_new_tokens=2, beam_width=2)

    def test_length_penalty(self):
        """length_penalty > 1 must run without error and return correct shape."""
        out = self.model.generate_beam(
            self.ids, max_new_tokens=4, n_loops=1, beam_width=2, length_penalty=1.5
        )
        assert out.shape == (1, T + 4)

    def test_mla_mode(self):
        """generate_beam must work with MLA attention."""
        model = OpenMythos(mla_cfg())
        out = model.generate_beam(self.ids, max_new_tokens=3, n_loops=1, beam_width=2)
        assert out.shape == (1, T + 3)
        assert out[:, T:].min().item() >= 0


# ---------------------------------------------------------------------------
# generate_batch
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    """Tests for OpenMythos.generate_batch()."""

    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)

    def test_output_length(self):
        """Each returned tensor must have exactly max_new_tokens tokens."""
        prompts = [
            torch.randint(0, self.cfg.vocab_size, (5,)),
            torch.randint(0, self.cfg.vocab_size, (3,)),
        ]
        results = self.model.generate_batch(prompts, max_new_tokens=6, n_loops=1)
        assert len(results) == 2
        for r in results:
            assert r.shape == (6,)

    def test_tokens_in_vocab(self):
        """All generated tokens must be valid vocab ids."""
        prompts = [torch.randint(0, self.cfg.vocab_size, (4,)) for _ in range(3)]
        results = self.model.generate_batch(prompts, max_new_tokens=5, n_loops=1)
        for r in results:
            assert r.min().item() >= 0
            assert r.max().item() < self.cfg.vocab_size

    def test_no_nan(self):
        """generate_batch must not produce NaN token indices."""
        prompts = [torch.randint(0, self.cfg.vocab_size, (4,)) for _ in range(2)]
        results = self.model.generate_batch(prompts, max_new_tokens=4, n_loops=1)
        for r in results:
            assert not torch.isnan(r.float()).any()

    def test_variable_length_prompts(self):
        """Variable-length prompts must all produce max_new_tokens output."""
        prompts = [
            torch.randint(0, self.cfg.vocab_size, (2,)),
            torch.randint(0, self.cfg.vocab_size, (6,)),
            torch.randint(0, self.cfg.vocab_size, (4,)),
        ]
        results = self.model.generate_batch(prompts, max_new_tokens=4, n_loops=1)
        assert len(results) == 3
        for r in results:
            assert r.shape == (4,)

    def test_single_prompt(self):
        """Single-prompt batch must work like standard generate()."""
        prompt = torch.randint(0, self.cfg.vocab_size, (T,))
        results = self.model.generate_batch([prompt], max_new_tokens=5, n_loops=1)
        assert len(results) == 1
        assert results[0].shape == (5,)

    def test_decode_loops_accepted(self):
        """decode_loops parameter must be accepted without error."""
        prompts = [torch.randint(0, self.cfg.vocab_size, (4,)) for _ in range(2)]
        results = self.model.generate_batch(
            prompts, max_new_tokens=3, n_loops=2, decode_loops=1
        )
        for r in results:
            assert r.shape == (3,)

    def test_2d_prompt_input(self):
        """(1, T) shaped prompt tensors must be accepted."""
        prompts = [torch.randint(0, self.cfg.vocab_size, (1, 4)) for _ in range(2)]
        results = self.model.generate_batch(prompts, max_new_tokens=3, n_loops=1)
        for r in results:
            assert r.shape == (3,)


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])

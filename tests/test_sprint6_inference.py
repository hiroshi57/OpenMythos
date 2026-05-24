"""
Sprint 6.1 — 推論最適化テスト

カバー範囲:
  6.1.1  torch.compile() ラッパー (compile_model)
  6.1.2  SDPA fallback (GQA / MLA 両方) — forward が正常に動くことを確認
  6.1.3  KV cache ページング (allocate_kv_cache / free_kv_cache)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from open_mythos import OpenMythos
from open_mythos.variants import mythos_nano
from open_mythos.main import MythosConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nano_gqa() -> OpenMythos:
    cfg = mythos_nano()  # attn_type="gqa"
    return OpenMythos(cfg).eval()


def _nano_mla() -> OpenMythos:
    cfg = mythos_nano()
    cfg.attn_type = "mla"
    # MLA 用に最小パラメータを調整
    cfg.kv_lora_rank = 16
    cfg.q_lora_rank = 32
    cfg.qk_rope_head_dim = 16
    cfg.qk_nope_head_dim = 16
    cfg.v_head_dim = 16
    return OpenMythos(cfg).eval()


def _rand_ids(cfg: MythosConfig, b: int = 1, s: int = 8) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (b, s))


# ---------------------------------------------------------------------------
# 6.1.1  compile_model
# ---------------------------------------------------------------------------

class TestCompileModel:
    def test_returns_self(self):
        m = _nano_gqa()
        assert m.compile_model() is m

    def test_forward_after_compile_gqa(self):
        # backend="eager" はすべての環境（Windows CPU含む）で動作する
        m = _nano_gqa().compile_model(mode="default", backend="eager")
        x = _rand_ids(m.cfg)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, 8, m.cfg.vocab_size)
        assert not torch.isnan(out).any()

    def test_forward_after_compile_fullgraph_false(self):
        """fullgraph=False はグラフ break を許可するので必ず動く。"""
        m = _nano_gqa().compile_model(mode="default", fullgraph=False, backend="eager")
        x = _rand_ids(m.cfg)
        with torch.no_grad():
            out = m(x)
        assert out.shape[-1] == m.cfg.vocab_size

    def test_compile_idempotent(self):
        """2回呼んでもクラッシュしないこと。"""
        m = _nano_gqa()
        m.compile_model(backend="eager").compile_model(backend="eager")
        x = _rand_ids(m.cfg)
        with torch.no_grad():
            out = m(x)
        assert not torch.isnan(out).any()

    def test_compile_pytorch_lt2_no_crash(self, monkeypatch):
        """torch.compile が存在しない環境でも compile_model() は self を返す。"""
        m = _nano_gqa()
        monkeypatch.delattr(torch, "compile", raising=False)
        result = m.compile_model()
        assert result is m


# ---------------------------------------------------------------------------
# 6.1.2  SDPA fallback (GQA と MLA の forward)
# ---------------------------------------------------------------------------

class TestSDPAFallback:
    def test_gqa_forward_no_flash(self):
        """flash_attn 未インストール環境でも SDPA fallback で forward が通る。"""
        m = _nano_gqa()
        x = _rand_ids(m.cfg)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, 8, m.cfg.vocab_size)
        assert not torch.isnan(out).any()

    def test_mla_forward_no_flash(self):
        m = _nano_mla()
        x = _rand_ids(m.cfg)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, 8, m.cfg.vocab_size)
        assert not torch.isnan(out).any()

    def test_gqa_generate_sdpa(self):
        """SDPA fallback でもトークン生成が完走する。"""
        m = _nano_gqa()
        x = _rand_ids(m.cfg)
        with torch.no_grad():
            gen = m.generate(input_ids=x, max_new_tokens=4, temperature=1.0)
        assert gen.shape[1] == x.shape[1] + 4

    def test_gqa_output_dtype_preserved(self):
        """入力と同じ dtype で出力されること (float32 → float32)。"""
        m = _nano_gqa()
        x = _rand_ids(m.cfg)
        with torch.no_grad():
            out = m(x)
        assert out.dtype == torch.float32

    def test_gqa_with_kv_cache_sdpa(self):
        """kv_cache 付き decode ステップが SDPA でも動く。"""
        m = _nano_gqa()
        cache = {}
        x = _rand_ids(m.cfg, s=4)
        with torch.no_grad():
            # prefill
            _ = m(x, kv_cache=cache)
            # single-token decode
            next_tok = _rand_ids(m.cfg, s=1)
            out = m(next_tok, kv_cache=cache, start_pos=4)
        assert out.shape == (1, 1, m.cfg.vocab_size)

    def test_mla_with_kv_cache_sdpa(self):
        """MLA の kv_cache decode が SDPA で動く。"""
        m = _nano_mla()
        cache = {}
        x = _rand_ids(m.cfg, s=4)
        with torch.no_grad():
            _ = m(x, kv_cache=cache)
            next_tok = _rand_ids(m.cfg, s=1)
            out = m(next_tok, kv_cache=cache, start_pos=4)
        assert out.shape == (1, 1, m.cfg.vocab_size)


# ---------------------------------------------------------------------------
# 6.1.3  KV cache ページング
# ---------------------------------------------------------------------------

class TestKVCachePaging:
    def test_allocate_returns_dict(self):
        m = _nano_gqa()
        cache = m.allocate_kv_cache()
        assert isinstance(cache, dict)
        assert "__window__" in cache

    def test_allocate_window_matches_config(self):
        cfg = mythos_nano()
        cfg.kv_page_size = 32
        cfg.kv_max_pages = 4
        m = OpenMythos(cfg).eval()
        cache = m.allocate_kv_cache()
        assert cache["__window__"] == 32 * 4  # 128

    def test_allocate_no_paging_uses_max_seq_len(self):
        cfg = mythos_nano()
        cfg.kv_max_pages = 0
        m = OpenMythos(cfg).eval()
        cache = m.allocate_kv_cache()
        assert cache["__window__"] == cfg.max_seq_len

    def test_allocate_custom_max_seq_len(self):
        m = _nano_gqa()
        cache = m.allocate_kv_cache(max_seq_len=256)
        assert cache["__window__"] == 256

    def test_forward_with_allocated_cache(self):
        m = _nano_gqa()
        cache = m.allocate_kv_cache()
        x = _rand_ids(m.cfg, s=4)
        with torch.no_grad():
            out = m(x, kv_cache=cache)
        assert out.shape == (1, 4, m.cfg.vocab_size)

    def test_free_kv_cache_clears_layers(self):
        m = _nano_gqa()
        cache = m.allocate_kv_cache()
        x = _rand_ids(m.cfg, s=4)
        with torch.no_grad():
            m(x, kv_cache=cache)
        # キャッシュにレイヤーエントリが追加されているはず
        layer_keys = [k for k in cache if not k.startswith("__")]
        assert len(layer_keys) > 0
        # free 後はメタキーのみ残る
        m.free_kv_cache(cache)
        layer_keys_after = [k for k in cache if not k.startswith("__")]
        assert len(layer_keys_after) == 0
        assert "__window__" in cache  # メタキーは保持

    def test_free_kv_cache_reuse(self):
        """free 後に再利用しても forward が通ること。"""
        m = _nano_gqa()
        cache = m.allocate_kv_cache()
        x = _rand_ids(m.cfg, s=4)
        with torch.no_grad():
            m(x, kv_cache=cache)
        m.free_kv_cache(cache)
        with torch.no_grad():
            out = m(x, kv_cache=cache)
        assert out.shape == (1, 4, m.cfg.vocab_size)

    def test_page_eviction_on_decode_steps(self):
        """single-token decode ステップで window 上限を超えたとき evict される。

        Eviction は decode (T=1) 時のみ発動する設計（prefill は全シーケンスを
        一括処理するため causal mask が必要でスキップされる）。
        """
        cfg = mythos_nano()
        cfg.kv_page_size = 4
        cfg.kv_max_pages = 2  # 最大 8 トークン保持
        m = OpenMythos(cfg).eval()
        # window=8 のキャッシュを作成
        cache = m.allocate_kv_cache()
        assert cache["__window__"] == 8

        # 短い prefill（8 トークン以内）でキャッシュを初期化
        x_prefill = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            m(x_prefill, kv_cache=cache)

        # decode ステップを window*2 回繰り返して eviction を発動させる
        with torch.no_grad():
            for _ in range(16):
                x_tok = torch.randint(0, cfg.vocab_size, (1, 1))
                m(x_tok, kv_cache=cache)

        # decode 後のキャッシュは window (8) を超えないこと
        for key, val in cache.items():
            if key.startswith("__"):
                continue
            if "k" in val:
                assert val["k"].shape[1] <= cfg.kv_page_size * cfg.kv_max_pages

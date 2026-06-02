"""
Sprint 33 — LongTermMemory FAISS ANN インデックステストスイート (40 tests)

FAISS がインストール済みの場合は faiss backend テストも実行する。
インストールされていない場合は linear backend テストのみ実行する。
"""

import math

import numpy as np
import pytest

from open_mythos.long_term_memory import (
    ANN_DIM,
    ANNIndex,
    EpisodicStore,
    LongTermMemoryAgent,
    MemoryEntry,
    _text_to_vector,
)

# FAISS が利用可能かどうかを判定
try:
    import faiss as _faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")


# =========================================================================
# 1. _text_to_vector (5 tests)
# =========================================================================

def test_text_to_vector_returns_ndarray():
    vec = _text_to_vector("hello world")
    assert isinstance(vec, np.ndarray)

def test_text_to_vector_shape():
    vec = _text_to_vector("テスト", dim=256)
    assert vec.shape == (256,)

def test_text_to_vector_l2_normalized():
    vec = _text_to_vector("SEO最適化コンテンツ戦略")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5 or norm == 0.0  # ゼロベクトルか正規化済み

def test_text_to_vector_custom_dim():
    vec = _text_to_vector("test", dim=128)
    assert vec.shape == (128,)

def test_text_to_vector_empty_string():
    vec = _text_to_vector("")
    assert vec.shape == (ANN_DIM,)
    assert float(np.linalg.norm(vec)) == pytest.approx(0.0, abs=1e-6)


# =========================================================================
# 2. ANNIndex — 初期化・基本プロパティ (5 tests)
# =========================================================================

def test_ann_index_default_linear():
    """FAISS が入っていれば faiss、なければ linear — auto の挙動を確認"""
    idx = ANNIndex(backend="linear")
    assert idx.backend == "linear"
    assert idx.is_faiss is False

def test_ann_index_size_starts_zero():
    idx = ANNIndex(backend="linear")
    assert idx.size == 0

def test_ann_index_faiss_available_returns_bool():
    assert isinstance(ANNIndex.faiss_available(), bool)

def test_ann_index_faiss_available_matches_import():
    assert ANNIndex.faiss_available() == HAS_FAISS

def test_ann_index_auto_resolves():
    idx = ANNIndex(backend="auto")
    expected = "faiss" if HAS_FAISS else "linear"
    assert idx.backend == expected


# =========================================================================
# 3. ANNIndex — linear backend add/search (8 tests)
# =========================================================================

def test_ann_linear_add_increments_size():
    idx = ANNIndex(backend="linear")
    vec = _text_to_vector("hello")
    idx.add("id1", vec)
    assert idx.size == 1

def test_ann_linear_add_multiple():
    idx = ANNIndex(backend="linear")
    for i in range(5):
        idx.add(f"id{i}", _text_to_vector(f"text {i}"))
    assert idx.size == 5

def test_ann_linear_search_empty():
    idx = ANNIndex(backend="linear")
    result = idx.search(_text_to_vector("query"), top_k=3)
    assert result == []

def test_ann_linear_search_returns_all_ids():
    idx = ANNIndex(backend="linear")
    idx.add("a", _text_to_vector("apple"))
    idx.add("b", _text_to_vector("banana"))
    result = idx.search(_text_to_vector("fruit"), top_k=10)
    ids = [r[0] for r in result]
    assert "a" in ids and "b" in ids

def test_ann_linear_search_top_k_limit():
    idx = ANNIndex(backend="linear")
    for i in range(10):
        idx.add(f"id{i}", _text_to_vector(f"text {i}"))
    result = idx.search(_text_to_vector("query"), top_k=3)
    assert len(result) <= 3

def test_ann_linear_search_result_format():
    idx = ANNIndex(backend="linear")
    idx.add("eid1", _text_to_vector("sample text"))
    result = idx.search(_text_to_vector("sample"), top_k=1)
    assert len(result) == 1
    eid, score = result[0]
    assert isinstance(eid, str)
    assert isinstance(score, float)

def test_ann_linear_rebuild():
    idx = ANNIndex(backend="linear")
    idx.add("old", _text_to_vector("old text"))
    ids = ["n1", "n2"]
    vecs = np.stack([_text_to_vector(f"new {i}") for i in range(2)])
    idx.rebuild(ids, vecs)
    assert idx.size == 2

def test_ann_linear_clear():
    idx = ANNIndex(backend="linear")
    idx.add("x", _text_to_vector("text"))
    idx.clear()
    assert idx.size == 0
    assert idx.search(_text_to_vector("text"), top_k=1) == []


# =========================================================================
# 4. ANNIndex — FAISS backend (6 tests, skip if not installed)
# =========================================================================

@requires_faiss
def test_ann_faiss_is_faiss_true():
    idx = ANNIndex(backend="faiss")
    assert idx.is_faiss is True

@requires_faiss
def test_ann_faiss_add_search():
    idx = ANNIndex(backend="faiss")
    idx.add("id1", _text_to_vector("SEO最適化"))
    idx.add("id2", _text_to_vector("コンテンツ品質向上"))
    result = idx.search(_text_to_vector("SEO"), top_k=2)
    assert len(result) >= 1
    ids = [r[0] for r in result]
    assert "id1" in ids

@requires_faiss
def test_ann_faiss_scores_in_range():
    idx = ANNIndex(backend="faiss")
    idx.add("v1", _text_to_vector("LoRA fine-tuning"))
    result = idx.search(_text_to_vector("LoRA training"), top_k=1)
    assert len(result) == 1
    _, score = result[0]
    assert -1.1 <= score <= 1.1  # 内積 (コサイン) の範囲

@requires_faiss
def test_ann_faiss_clear():
    idx = ANNIndex(backend="faiss")
    idx.add("z", _text_to_vector("data"))
    idx.clear()
    assert idx.size == 0
    assert idx.search(_text_to_vector("data"), top_k=1) == []

@requires_faiss
def test_ann_faiss_rebuild():
    idx = ANNIndex(backend="faiss")
    idx.add("old", _text_to_vector("old"))
    ids  = ["a", "b", "c"]
    vecs = np.stack([_text_to_vector(f"item {i}") for i in range(3)])
    idx.rebuild(ids, vecs)
    assert idx.size == 3

@requires_faiss
def test_ann_faiss_explicit_request_raises_if_missing(monkeypatch):
    """FAISS が import できない状況をモックして ImportError を確認"""
    import sys, builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("mocked faiss missing")
        return original_import(name, *args, **kwargs)

    # faiss モジュールをキャッシュから除外
    saved = sys.modules.pop("faiss", None)
    monkeypatch.setattr(builtins, "__import__", mock_import)
    try:
        with pytest.raises(ImportError):
            ANNIndex._resolve_backend("faiss")
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)
        if saved is not None:
            sys.modules["faiss"] = saved


# =========================================================================
# 5. EpisodicStore with ANN (8 tests)
# =========================================================================

def test_episodic_ann_property():
    store = EpisodicStore(ann_backend="linear")
    assert isinstance(store.ann, ANNIndex)

def test_episodic_ann_backend_param():
    store = EpisodicStore(ann_backend="linear")
    assert store.ann.backend == "linear"

def test_episodic_append_adds_to_ann():
    store = EpisodicStore(ann_backend="linear", score_threshold=0.0)
    store.append("query", "response", score=0.8)
    assert store.ann.size == 1

def test_episodic_append_multiple_ann_size():
    store = EpisodicStore(ann_backend="linear", score_threshold=0.0)
    for i in range(5):
        store.append(f"context {i}", f"response {i}", score=0.8)
    assert store.ann.size == 5

def test_episodic_search_returns_results():
    store = EpisodicStore(ann_backend="linear", score_threshold=0.0)
    store.append("SEO記事の書き方", "H1→H2→FAQ構成が効果的", score=0.9)
    store.append("プロンプト最適化", "明確な指示と例を使う", score=0.85)
    result = store.search("SEO 最適化", top_k=2)
    assert len(result) >= 1

def test_episodic_search_category_filter():
    store = EpisodicStore(ann_backend="linear", score_threshold=0.0)
    store.append("ctx", "ep text", score=0.8)
    # category_filter でフィルタしても動作する
    result = store.search("ep text", top_k=5, category_filter="episode")
    assert all(e.category == "episode" for e, _ in result)

def test_episodic_stats_includes_ann_info():
    store = EpisodicStore(ann_backend="linear", score_threshold=0.0)
    store.append("ctx", "text", score=0.8)
    s = store.stats()
    assert "ann_backend" in s
    assert "ann_size" in s

def test_episodic_evict_rebuilds_ann():
    store = EpisodicStore(max_size=3, ann_backend="linear", score_threshold=0.0)
    for i in range(5):
        store.append(f"ctx{i}", f"unique response {i} with distinct content", score=0.8)
    # max_size=3 なので evict が発生し、ANN も再構築される
    assert store.ann.size == len(store._entries)


# =========================================================================
# 6. LongTermMemoryAgent with ann_backend (6 tests)
# =========================================================================

def test_ltm_agent_ann_backend_param():
    agent = LongTermMemoryAgent(ann_backend="linear")
    assert agent.episodes.ann.backend == "linear"

def test_ltm_agent_store_and_retrieve():
    agent = LongTermMemoryAgent(ann_backend="linear", score_threshold=0.0)
    agent.store_episode("SEO記事", "効果的なH1設計が重要", score=0.9)
    result = agent.retrieve("SEO 記事 構成", top_k=1)
    assert result.total_searched >= 1

def test_ltm_agent_auto_backend():
    agent = LongTermMemoryAgent(ann_backend="auto")
    expected = "faiss" if HAS_FAISS else "linear"
    assert agent.episodes.ann.backend == expected

def test_ltm_agent_consolidate_rebuilds_ann():
    agent = LongTermMemoryAgent(ann_backend="linear", score_threshold=0.0)
    for i in range(10):
        agent.store_episode(f"context {i}", f"response text {i}", score=0.8)
    result = agent.consolidate()
    assert isinstance(result, dict)
    # consolidate 後も ANN サイズとエントリ数が一致
    assert agent.episodes.ann.size == len(agent.episodes._entries)

def test_ltm_agent_stats_with_ann():
    agent = LongTermMemoryAgent(ann_backend="linear", score_threshold=0.0)
    agent.store_episode("ctx", "response", score=0.9)
    s = agent.stats()
    assert s["episode_count"] >= 1
    assert "ann_backend" not in s  # stats() は既存キーのみ (ann_info は episodes.stats() で)

def test_ltm_agent_retrieve_best_entry():
    agent = LongTermMemoryAgent(ann_backend="linear", score_threshold=0.0)
    agent.store_episode("LLMO最適化", "クエリ対応コンテンツが効果的", score=0.95)
    result = agent.retrieve("LLMO最適化とは", top_k=1)
    assert result.best_entry is not None


# =========================================================================
# 7. 後方互換 — 既存テストとの整合 (2 tests)
# =========================================================================

def test_backward_compat_no_ann_param():
    """ann_backend パラメータなしで従来どおり動作する"""
    agent = LongTermMemoryAgent()
    agent.store_episode("test", "response", score=0.9)
    result = agent.retrieve("test", top_k=1)
    assert isinstance(result.best_entry, MemoryEntry)

def test_backward_compat_episodic_store_defaults():
    """EpisodicStore のデフォルトパラメータが変わっていない"""
    store = EpisodicStore()
    assert store.max_size == 500
    assert store.score_threshold == 0.5
    assert store.dedup_threshold == 0.85

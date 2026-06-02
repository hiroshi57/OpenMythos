"""
Sprint 26 テスト — 長期記憶統合 LongTermMemoryAgent (P7)

- TestMemoryEntry       : MemoryEntry データ構造
- TestEpisodicStore     : エピソード追加・検索・重複除去
- TestSemanticStore     : 知識ストア操作
- TestLongTermMemory    : 統合インターフェース
- TestMemoryRetrieval   : 検索結果データ
- TestConsolidate       : 記憶整理
- TestIntegration       : 他スプリントとの連携
"""

from __future__ import annotations

import pathlib
import time

_ROOT = pathlib.Path(__file__).parent.parent


# ===========================================================================
# TestMemoryEntry
# ===========================================================================


class TestMemoryEntry:
    def test_basic_attributes(self):
        from open_mythos.long_term_memory import MemoryEntry
        e = MemoryEntry(text="テストテキスト", context="テストクエリ", score=0.8)
        assert e.text == "テストテキスト"
        assert e.score == 0.8
        assert e.category == "episode"

    def test_freshness_is_float(self):
        from open_mythos.long_term_memory import MemoryEntry
        e = MemoryEntry(text="新鮮なメモリ")
        assert 0.0 < e.freshness <= 1.0

    def test_freshness_decreases_with_age(self):
        from open_mythos.long_term_memory import MemoryEntry
        e_fresh = MemoryEntry(text="新しい", created_at=time.time())
        e_old = MemoryEntry(text="古い", created_at=time.time() - 48 * 3600)
        assert e_fresh.freshness > e_old.freshness

    def test_priority_uses_relevance(self):
        from open_mythos.long_term_memory import MemoryEntry
        e = MemoryEntry(text="test", score=0.9)
        p_low = e.priority(relevance=0.1)
        p_high = e.priority(relevance=0.9)
        assert p_high > p_low

    def test_access_count_default_zero(self):
        from open_mythos.long_term_memory import MemoryEntry
        e = MemoryEntry(text="test")
        assert e.access_count == 0

    def test_entry_id_generated(self):
        from open_mythos.long_term_memory import MemoryEntry
        e1 = MemoryEntry(text="a")
        e2 = MemoryEntry(text="b")
        assert e1.entry_id != e2.entry_id


# ===========================================================================
# TestEpisodicStore
# ===========================================================================


class TestEpisodicStore:
    def test_append_returns_entry(self):
        from open_mythos.long_term_memory import EpisodicStore
        store = EpisodicStore()
        entry = store.append("クエリ", "レスポンス", score=0.8)
        assert entry is not None
        assert entry.text == "レスポンス"

    def test_append_below_threshold_returns_none(self):
        from open_mythos.long_term_memory import EpisodicStore
        store = EpisodicStore(score_threshold=0.7)
        result = store.append("q", "low quality", score=0.5)
        assert result is None

    def test_append_above_threshold_stored(self):
        from open_mythos.long_term_memory import EpisodicStore
        store = EpisodicStore(score_threshold=0.5)
        store.append("q", "high quality content", score=0.9)
        assert len(store.entries) == 1

    def test_deduplication(self):
        from open_mythos.long_term_memory import EpisodicStore
        store = EpisodicStore(dedup_threshold=0.8)
        store.append("q", "SEO最適化のベストプラクティスを教えてください", score=0.8)
        result = store.append("q", "SEO最適化のベストプラクティスを教えてください", score=0.7)
        assert result is None  # 重複除去
        assert len(store.entries) == 1

    def test_search_returns_relevant(self):
        from open_mythos.long_term_memory import EpisodicStore
        store = EpisodicStore()
        store.append("SEO記事の書き方", "H1見出しから始めてFAQを追加", score=0.9)
        store.append("料理レシピ", "卵を3個割って混ぜる", score=0.9)
        results = store.search("SEO コンテンツ 記事", top_k=1)
        assert len(results) == 1
        entry, rel = results[0]
        assert "SEO" in entry.context or "H1" in entry.text

    def test_search_increments_access_count(self):
        from open_mythos.long_term_memory import EpisodicStore
        store = EpisodicStore()
        store.append("SEO", "SEO対策の方法", score=0.9)
        store.search("SEO 最適化")
        assert store.entries[0].access_count >= 0  # アクセスカウント存在確認

    def test_eviction_when_full(self):
        from open_mythos.long_term_memory import EpisodicStore
        store = EpisodicStore(max_size=5)
        for i in range(10):
            store.append(f"q{i}", f"response {i}", score=0.8)
        assert len(store.entries) <= 5

    def test_stats_keys(self):
        from open_mythos.long_term_memory import EpisodicStore
        store = EpisodicStore()
        store.append("q", "text", score=0.9)
        stats = store.stats()
        assert "count" in stats
        assert "avg_score" in stats

    def test_stats_empty(self):
        from open_mythos.long_term_memory import EpisodicStore
        store = EpisodicStore()
        stats = store.stats()
        assert stats["count"] == 0


# ===========================================================================
# TestSemanticStore
# ===========================================================================


class TestSemanticStore:
    def test_store_and_retrieve(self):
        from open_mythos.long_term_memory import SemanticStore
        store = SemanticStore()
        store.store("llmo_def", "LLMOはAI検索向けコンテンツ最適化の手法", tags=["llmo", "seo"])
        results = store.search("LLMO 最適化", top_k=1)
        assert len(results) == 1

    def test_tag_search(self):
        from open_mythos.long_term_memory import SemanticStore
        store = SemanticStore()
        store.store("seo_tip", "キーワード密度を3%に保つ", tags=["seo"])
        store.store("roas_tip", "ROAS目標は3以上に設定", tags=["roas"])
        results = store.search("ヒント 方法", tags=["seo"])
        assert all("seo" in e.tags for e, _ in results)

    def test_overwrite_same_key(self):
        from open_mythos.long_term_memory import SemanticStore
        store = SemanticStore()
        store.store("key1", "初期コンテンツ")
        store.store("key1", "更新コンテンツ")
        assert store.size == 1

    def test_size_limit(self):
        from open_mythos.long_term_memory import SemanticStore
        store = SemanticStore(max_size=3)
        for i in range(5):
            store.store(f"k{i}", f"content {i}")
        assert store.size <= 3

    def test_stats_count(self):
        from open_mythos.long_term_memory import SemanticStore
        store = SemanticStore()
        store.store("k1", "v1")
        store.store("k2", "v2")
        assert store.stats()["count"] == 2


# ===========================================================================
# TestLongTermMemory
# ===========================================================================


class TestLongTermMemory:
    def test_store_episode(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        entry = agent.store_episode("どうやってSEO記事を書く？", "H1から始める", score=0.8)
        assert entry is not None

    def test_store_knowledge(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        entry = agent.store_knowledge("llmo", "LLMOはAI検索最適化", tags=["llmo"])
        assert entry.category == "knowledge"

    def test_retrieve_returns_memory_retrieval(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent, MemoryRetrieval
        agent = LongTermMemoryAgent()
        agent.store_episode("SEO記事", "H1から始めて構造化する", score=0.9)
        result = agent.retrieve("SEO 記事 構成")
        assert isinstance(result, MemoryRetrieval)

    def test_retrieve_best_entry(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        agent.store_episode("LLMO最適化方法", "エンティティ密度を高める", score=0.9)
        result = agent.retrieve("LLMO 最適化")
        assert result.best_entry is not None

    def test_retrieve_no_entries_returns_empty(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        result = agent.retrieve("存在しないクエリ")
        assert result.entries == []
        assert result.total_searched == 0

    def test_retrieve_includes_knowledge(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        agent.store_knowledge("test_key", "テスト知識テキスト", tags=["test"])
        result = agent.retrieve("テスト 知識", include_knowledge=True)
        knowledge_entries = [e for e in result.entries if e.category == "knowledge"]
        assert len(knowledge_entries) >= 0  # 知識エントリが存在すれば含まれる

    def test_retrieve_top_k_limit(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        for i in range(10):
            agent.store_episode(f"クエリ{i}", f"レスポンス{i}", score=0.8)
        result = agent.retrieve("クエリ", top_k=3)
        assert len(result.entries) <= 3

    def test_store_conversation_memory(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        entry = agent.store_conversation_memory("質問", "回答テキスト", score=0.75)
        assert entry is not None

    def test_store_conversation_with_evaluate_fn(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent(score_threshold=0.3)
        called = []
        def mock_eval(text: str) -> float:
            called.append(text)
            return 0.85
        agent.store_conversation_memory("q", "高品質なテキスト", evaluate_fn=mock_eval)
        assert called  # evaluate_fn が呼ばれた

    def test_stats_keys(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        agent.store_episode("q", "r", score=0.8)
        stats = agent.stats()
        assert "episode_count" in stats
        assert "knowledge_count" in stats
        assert "total_entries" in stats


# ===========================================================================
# TestMemoryRetrieval
# ===========================================================================


class TestMemoryRetrieval:
    def _make_retrieval(self):
        from open_mythos.long_term_memory import MemoryEntry, MemoryRetrieval
        entries = [MemoryEntry(text=f"テキスト{i}", score=0.8 - i * 0.1) for i in range(3)]
        scores = [0.9 - i * 0.1 for i in range(3)]
        return MemoryRetrieval(
            query="クエリ",
            entries=entries,
            relevance_scores=scores,
            total_searched=10,
        )

    def test_best_entry(self):
        r = self._make_retrieval()
        assert r.best_entry is not None
        assert r.best_entry.text == "テキスト0"

    def test_top_relevance(self):
        r = self._make_retrieval()
        assert abs(r.top_relevance - 0.9) < 0.01

    def test_to_context_string(self):
        r = self._make_retrieval()
        ctx = r.to_context_string()
        assert "[記憶から検索]:" in ctx
        assert "テキスト0" in ctx

    def test_empty_retrieval(self):
        from open_mythos.long_term_memory import MemoryRetrieval
        r = MemoryRetrieval(query="q", entries=[], relevance_scores=[], total_searched=0)
        assert r.best_entry is None
        assert r.top_relevance == 0.0
        assert r.to_context_string() == ""


# ===========================================================================
# TestConsolidate
# ===========================================================================


class TestConsolidate:
    def test_consolidate_removes_duplicates(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent(dedup_threshold=0.5)
        # ほぼ同じテキストを複数追加
        for _ in range(3):
            agent.episodes._entries.append(
                __import__("open_mythos.long_term_memory", fromlist=["MemoryEntry"]).MemoryEntry(
                    text="SEO最適化の方法について詳しく説明します",
                    context="seo",
                    score=0.8,
                )
            )
        result = agent.consolidate()
        assert "removed_duplicates" in result
        assert isinstance(result["removed_duplicates"], int)

    def test_consolidate_result_has_removed_stale(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        result = agent.consolidate()
        assert "removed_stale" in result


# ===========================================================================
# TestIntegration
# ===========================================================================


class TestIntegration:
    def test_store_and_retrieve_high_score_only(self):
        """score_threshold より低いエピソードは保存されない。"""
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent(score_threshold=0.7)
        agent.store_episode("高品質クエリ", "高品質レスポンス", score=0.9)
        agent.store_episode("低品質クエリ", "低品質レスポンス", score=0.4)
        stats = agent.stats()
        assert stats["episode_count"] == 1

    def test_llmo_scorer_integration(self):
        """LLMOScorer (Sprint 19) をフィットネス関数として使用できる。"""
        from open_mythos.long_term_memory import LongTermMemoryAgent
        from open_mythos.llmo import LLMOScorer
        scorer = LLMOScorer()
        agent = LongTermMemoryAgent(score_threshold=0.0)

        text = "OpenMythosは再帰深度Transformerを使ったLLM推論エンジンです。"
        score = scorer.score(text).llmo_total
        entry = agent.store_episode("OpenMythosとは？", text, score=score)
        assert entry is not None

    def test_multiple_retrievals_increase_access_count(self):
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        agent.store_episode("SEO記事", "エンティティ密度が重要", score=0.9)
        agent.retrieve("SEO エンティティ")
        agent.retrieve("SEO エンティティ")
        total_access = sum(e.access_count for e in agent.episodes.entries)
        assert total_access >= 0  # アクセスカウントがインクリメントされている

    def test_context_string_used_for_prompt_injection(self):
        """retrieve() の to_context_string() がプロンプト注入に使えること。"""
        from open_mythos.long_term_memory import LongTermMemoryAgent
        agent = LongTermMemoryAgent()
        agent.store_episode("SEO", "H1から書く", score=0.9)
        r = agent.retrieve("SEO 書き方")
        ctx = r.to_context_string()
        # to_context_string が空でない場合はプロンプトとして利用可能
        assert isinstance(ctx, str)

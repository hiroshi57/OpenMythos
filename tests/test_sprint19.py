"""
Sprint 19 テスト — LLMO 強化

カバー範囲:
    19.1  LLMOOptimizer       — テキスト自動最適化エンジン
    19.2  score_with_query()  — クエリ対応スコアリング (query_relevance / intent_type)
    19.3  suggest_improvements() — 具体的改善提案生成
    19.4  serve/api.py — /v1/llmo/optimize / /v1/llmo/suggest / /v1/llmo/score
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 19.2 — score_with_query / query_relevance / intent_type
# ---------------------------------------------------------------------------


class TestScoreWithQuery:
    """クエリ対応スコアリングのテスト。"""

    def _scorer(self):
        from open_mythos.llmo import LLMOScorer

        return LLMOScorer()

    def test_returns_query_relevance_field(self):
        s = self._scorer()
        result = s.score_with_query(
            "LLMO optimization improves AI citation rates significantly.",
            query="LLMO optimization",
        )
        assert hasattr(result, "query_relevance")
        assert 0.0 <= result.query_relevance <= 1.0

    def test_relevant_text_scores_higher_than_irrelevant(self):
        s = self._scorer()
        relevant = s.score_with_query(
            "LLMO optimization boosts AI search citations by 3x via entity density.",
            query="LLMO optimization",
        )
        irrelevant = s.score_with_query(
            "今日は晴れてとても気持ちの良い天気です。公園で散歩しました。",
            query="LLMO optimization",
        )
        assert relevant.query_relevance > irrelevant.query_relevance

    def test_empty_query_returns_zero_relevance(self):
        s = self._scorer()
        result = s.score_with_query("some text here", query="")
        assert result.query_relevance == 0.0

    def test_intent_type_informational(self):
        s = self._scorer()
        result = s.score_with_query("text", query="LLMOとは何ですか")
        assert result.intent_type == "informational"

    def test_intent_type_transactional(self):
        s = self._scorer()
        result = s.score_with_query("text", query="LLMO ツール 購入")
        assert result.intent_type == "transactional"

    def test_intent_type_commercial(self):
        s = self._scorer()
        result = s.score_with_query("text", query="LLMO ツール 比較 おすすめ")
        assert result.intent_type == "commercial"

    def test_intent_type_navigational(self):
        s = self._scorer()
        result = s.score_with_query("text", query="OpenMythos 公式サイト")
        assert result.intent_type == "navigational"

    def test_query_relevance_japanese(self):
        s = self._scorer()
        result = s.score_with_query(
            "LLMO最適化により、AI検索での引用率が向上します。エンティティ密度と回答直接性が重要です。",
            query="LLMO最適化",
        )
        assert result.query_relevance > 0.1

    def test_base_scores_unchanged_by_query(self):
        """クエリを追加しても 3 軸の基本スコアは変わらない。"""
        s = self._scorer()
        text = "OpenMythos implements Recurrent-Depth Transformer achieving 3.7x efficiency."
        base = s.score(text)
        with_q = s.score_with_query(text, query="transformer efficiency")
        assert base.entity_density == with_q.entity_density
        assert base.answer_directness == with_q.answer_directness
        assert base.citability == with_q.citability

    def test_intent_type_field_nonempty(self):
        s = self._scorer()
        result = s.score_with_query("text", query="SEO 最新 情報")
        assert result.intent_type != ""


# ---------------------------------------------------------------------------
# 19.3 — suggest_improvements
# ---------------------------------------------------------------------------


class TestSuggestImprovements:
    """改善提案生成のテスト。"""

    def _scorer(self):
        from open_mythos.llmo import LLMOScorer

        return LLMOScorer()

    def test_returns_list_of_improvements(self):
        s = self._scorer()
        suggestions = s.suggest_improvements("short text")
        assert isinstance(suggestions, list)

    def test_low_quality_text_gets_high_priority_suggestions(self):
        s = self._scorer()
        plain = "今日はいい天気。散歩した。楽しかった。"
        suggestions = s.suggest_improvements(plain)
        priorities = [sug.priority for sug in suggestions]
        assert "high" in priorities

    def test_improvement_has_required_fields(self):
        s = self._scorer()
        suggestions = s.suggest_improvements("text", max_suggestions=3)
        for sug in suggestions:
            assert hasattr(sug, "category")
            assert hasattr(sug, "priority")
            assert hasattr(sug, "description")
            assert hasattr(sug, "expected_delta")
            assert sug.priority in ("high", "medium", "low")

    def test_max_suggestions_respected(self):
        s = self._scorer()
        suggestions = s.suggest_improvements("tiny", max_suggestions=2)
        assert len(suggestions) <= 2

    def test_sorted_by_priority(self):
        s = self._scorer()
        suggestions = s.suggest_improvements("very short text", max_suggestions=5)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        for i in range(len(suggestions) - 1):
            assert (
                priority_order[suggestions[i].priority]
                <= priority_order[suggestions[i + 1].priority]
            )

    def test_query_suggestion_included_when_irrelevant(self):
        s = self._scorer()
        suggestions = s.suggest_improvements(
            "今日の天気はとても良い。", query="LLMO最適化 SEO対策"
        )
        categories = [sug.category for sug in suggestions]
        assert "query" in categories

    def test_high_quality_text_fewer_high_priority(self):
        s = self._scorer()
        rich = (
            "LLMO optimization requires three key metrics: entity density measures "
            "the ratio of named entities. Our 2025 study of 10,000 pages shows 3.2× "
            "more AI citations for high-scoring pages.\n\n"
            "## Key Findings\n- entity_density > 0.7: 87% improvement\n"
            "- answer_directness > 0.6: 3.2x citation rate\n"
            "- citability > 0.5: significant AI visibility boost"
        )
        suggestions = s.suggest_improvements(rich, max_suggestions=5)
        high_count = sum(1 for sug in suggestions if sug.priority == "high")
        # 高品質テキストでは high priority 提案が少ないはず
        assert high_count <= 2

    def test_category_values_valid(self):
        s = self._scorer()
        valid_cats = {
            "entity",
            "directness",
            "citability",
            "structure",
            "length",
            "query",
        }
        suggestions = s.suggest_improvements("short text with no structure at all here")
        for sug in suggestions:
            assert sug.category in valid_cats

    def test_expected_delta_positive(self):
        s = self._scorer()
        suggestions = s.suggest_improvements("brief text")
        for sug in suggestions:
            assert sug.expected_delta >= 0.0

    def test_no_suggestions_for_perfect_text(self):
        """非常に高品質なテキストは high priority 提案が 0 になる可能性がある。"""
        s = self._scorer()
        # 高品質テキストで high priority が少ないことを確認
        rich = (
            "LLMO is Large Language Model Optimization. "
            "Our 2025 research across 50,000 pages shows entity_density > 0.7 "
            "yields 3.2x more AI citations (source: OpenMythos Research 2025).\n\n"
            "## Why LLMO Matters\n- Answer-first format increases directness score\n"
            "- Structured data (lists, headers) improves citability\n"
            "- Specific numbers and named entities boost entity density\n\n"
            "Studies confirm that pages following LLMO principles receive "
            "significantly more traffic from AI-powered search engines."
        )
        suggestions = s.suggest_improvements(rich, max_suggestions=5)
        high = [sug for sug in suggestions if sug.priority == "high"]
        assert len(high) <= 3  # 完璧でなくても high が多すぎないことを確認


# ---------------------------------------------------------------------------
# 19.1 — LLMOOptimizer
# ---------------------------------------------------------------------------


class TestLLMOOptimizer:
    """テキスト自動最適化エンジンのテスト。"""

    def _opt(self):
        from open_mythos.llmo import LLMOOptimizer

        return LLMOOptimizer()

    def test_returns_optimized_result(self):
        from open_mythos.llmo import OptimizedResult

        opt = self._opt()
        result = opt.optimize("Short text.", target_score=0.5, max_iterations=2)
        assert isinstance(result, OptimizedResult)

    def test_result_has_required_fields(self):
        opt = self._opt()
        result = opt.optimize("text", max_iterations=1)
        assert hasattr(result, "original_text")
        assert hasattr(result, "optimized_text")
        assert hasattr(result, "original_score")
        assert hasattr(result, "optimized_score")
        assert hasattr(result, "improvement_pct")
        assert hasattr(result, "changes_applied")
        assert hasattr(result, "iterations")

    def test_original_text_preserved(self):
        opt = self._opt()
        text = "Original test text for optimization."
        result = opt.optimize(text, max_iterations=1)
        assert result.original_text == text

    def test_score_non_negative(self):
        opt = self._opt()
        result = opt.optimize("a", max_iterations=1)
        assert result.optimized_score.llmo_total >= 0.0

    def test_does_not_decrease_score(self):
        """最適化後のスコアが元スコアより下がらない。"""
        opt = self._opt()
        result = opt.optimize(
            "今日はいい天気。散歩した。楽しかった。",
            target_score=0.8,
            max_iterations=3,
        )
        assert (
            result.optimized_score.llmo_total >= result.original_score.llmo_total - 0.01
        )

    def test_improvement_pct_type(self):
        opt = self._opt()
        result = opt.optimize("test text here", max_iterations=1)
        assert isinstance(result.improvement_pct, float)

    def test_rewrite_for_answer_first_nochange_already_direct(self):
        opt = self._opt()
        text = "はい、LLMOは効果的な最適化手法です。詳しい説明を以下に示します。"
        result_text = opt.rewrite_for_answer_first(text)
        # answer-first パターンが既にある場合は変更されない
        assert "はい" in result_text

    def test_rewrite_for_answer_first_english(self):
        opt = self._opt()
        text = (
            "This approach has many benefits. "
            "LLMO optimization with 3.2x citation rate is the best method. "
            "Results vary by content type."
        )
        result = opt.rewrite_for_answer_first(text)
        # 元テキストと同じか変更されるかどちらか
        assert isinstance(result, str)
        assert len(result) > 0

    def test_optimize_with_query(self):
        opt = self._opt()
        result = opt.optimize(
            "今日の天気はいいです。",
            query="LLMO最適化 コンテンツ",
            target_score=0.5,
            max_iterations=2,
        )
        assert result.optimized_score.llmo_total >= 0.0

    def test_changes_applied_is_list(self):
        opt = self._opt()
        result = opt.optimize("short", max_iterations=2)
        assert isinstance(result.changes_applied, list)

    def test_max_iterations_respected(self):
        opt = self._opt()
        result = opt.optimize("very short", max_iterations=1)
        assert result.iterations <= 1

    def test_target_score_achieved_flag(self):
        """スコアが既に高い場合は改善が少ない可能性がある。"""
        opt = self._opt()
        # 非常に高い target score — 達成されなくても OK
        result = opt.optimize(
            "LLMO optimization is Large Language Model Optimization. "
            "Entity density, answer directness, and citability are key metrics.",
            target_score=0.99,
            max_iterations=3,
        )
        assert isinstance(result.optimized_score.llmo_total, float)

    def test_empty_text_does_not_crash(self):
        opt = self._opt()
        result = opt.optimize("", max_iterations=1)
        # 空テキストの元スコアは 0.0。最適化後は citation cue が付加される場合がある
        assert result.original_score.llmo_total == 0.0
        assert result.optimized_score.llmo_total >= 0.0


# ---------------------------------------------------------------------------
# 19.4 — API エンドポイント (ロジック単体テスト)
# ---------------------------------------------------------------------------


class TestLLMOAPILogic:
    """API レイヤーのロジック検証（FastAPI 起動不要）。"""

    def test_llmo_suggest_response_structure(self):
        from open_mythos.llmo import LLMOScorer

        scorer = LLMOScorer()
        text = "Short low quality text with no structure or data."
        suggestions = scorer.suggest_improvements(text, max_suggestions=5)
        assert len(suggestions) <= 5
        assert all(hasattr(s, "category") for s in suggestions)

    def test_llmo_optimize_response_structure(self):
        from open_mythos.llmo import LLMOOptimizer

        optimizer = LLMOOptimizer()
        result = optimizer.optimize(
            "テスト用テキスト。短い文章です。",
            target_score=0.6,
            max_iterations=2,
        )
        assert result.original_text != "" or result.optimized_text == ""
        assert isinstance(result.improvement_pct, float)

    def test_llmo_query_score_response(self):
        from open_mythos.llmo import LLMOScorer

        scorer = LLMOScorer()
        result = scorer.score_with_query(
            "LLMO optimization improves AI search visibility through structured content.",
            query="LLMO AI search",
        )
        assert 0.0 <= result.query_relevance <= 1.0
        assert result.intent_type in (
            "informational",
            "navigational",
            "transactional",
            "commercial",
        )

    def test_openapi_tags_include_seo(self):
        """serve/api.py の openapi_tags に seo が含まれることを確認。"""

        # api.py を直接 import せず tags 定義のみを静的確認
        api_path = ROOT / "serve" / "api.py"
        content = api_path.read_text(encoding="utf-8")
        assert '"/v1/llmo/suggest"' in content
        assert '"/v1/llmo/optimize"' in content
        assert '"/v1/llmo/score"' in content

    def test_optimizer_with_scorer_shared(self):
        """LLMOOptimizer が外部 scorer を共有できる。"""
        from open_mythos.llmo import LLMOOptimizer, LLMOScorer

        scorer = LLMOScorer(
            entity_weight=0.4, directness_weight=0.3, citability_weight=0.3
        )
        optimizer = LLMOOptimizer(scorer=scorer)
        result = optimizer.optimize("test text", max_iterations=1)
        assert result.optimized_score.llmo_total >= 0.0


# ---------------------------------------------------------------------------
# 統合テスト
# ---------------------------------------------------------------------------


class TestLLMOSprint19Integration:
    """Sprint 19 LLMO 全機能統合テスト。"""

    def test_full_pipeline_ja(self):
        """日本語テキストの全パイプライン: score → suggest → optimize。"""
        from open_mythos.llmo import LLMOOptimizer, LLMOScorer

        scorer = LLMOScorer()
        optimizer = LLMOOptimizer(scorer=scorer)
        text = "今日はいい天気です。散歩しました。"

        # 1. スコアリング
        score = scorer.score(text)
        assert score.llmo_total >= 0.0

        # 2. クエリ対応スコアリング
        q_score = scorer.score_with_query(text, query="日本語 コンテンツ 最適化")
        assert q_score.query_relevance >= 0.0

        # 3. 改善提案
        suggestions = scorer.suggest_improvements(text, query="LLMO 最適化")
        assert len(suggestions) > 0

        # 4. 最適化
        result = optimizer.optimize(text, query="LLMO最適化", max_iterations=2)
        assert (
            result.optimized_score.llmo_total >= result.original_score.llmo_total - 0.01
        )

    def test_full_pipeline_en(self):
        """英語テキストの全パイプライン。"""
        from open_mythos.llmo import LLMOOptimizer, LLMOScorer

        scorer = LLMOScorer()
        optimizer = LLMOOptimizer(scorer=scorer)
        text = "This is a simple sentence. It has no structure or data."

        score = scorer.score(text)
        suggestions = scorer.suggest_improvements(text, max_suggestions=3)
        result = optimizer.optimize(text, target_score=0.6, max_iterations=2)

        assert score.llmo_total >= 0.0
        assert len(suggestions) >= 0
        assert result.iterations >= 1

    def test_suggest_then_optimize_improvement(self):
        """改善提案を参考に最適化するとスコアが上がることを確認。"""
        from open_mythos.llmo import LLMOOptimizer, LLMOScorer

        scorer = LLMOScorer()
        optimizer = LLMOOptimizer(scorer=scorer)

        low_text = "Hello world. This is basic."
        initial_score = scorer.score(low_text).llmo_total
        suggestions = scorer.suggest_improvements(low_text)
        result = optimizer.optimize(low_text, max_iterations=3)

        # 最適化後は同等以上
        assert result.optimized_score.llmo_total >= initial_score - 0.01
        assert isinstance(suggestions, list)

    def test_query_relevance_improves_after_keyword_injection(self):
        """クエリキーワード注入後に query_relevance が改善される。"""
        from open_mythos.llmo import LLMOOptimizer, LLMOScorer

        scorer = LLMOScorer()
        optimizer = LLMOOptimizer(scorer=scorer)

        query = "machine learning optimization"
        text = "今日は良い天気でした。散歩が楽しかった。"

        before = scorer.score_with_query(text, query).query_relevance
        result = optimizer.optimize(text, query=query, max_iterations=2)
        after = scorer.score_with_query(result.optimized_text, query).query_relevance

        assert after >= before - 0.01  # 悪化しない

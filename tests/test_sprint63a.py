"""
Sprint 63A — A/B テスト基盤 テスト

対象:
  open_mythos/skills/ab_test.py:
    VariantStatus / ABTestStatus
    VariantStats / Variant
    ABTest / ABTestStore
    TrafficAllocator
    SignificanceResult / ABTestAnalyzer
    ABTestReportEngine
"""
from __future__ import annotations

import pytest

from open_mythos.skills.ab_test import (
    VariantStatus, ABTestStatus,
    VariantStats, Variant,
    ABTest, ABTestStore,
    TrafficAllocator,
    SignificanceResult, ABTestAnalyzer,
    ABTestReportEngine,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEnums:
    def test_variant_status(self):
        assert VariantStatus.DRAFT.value   == "draft"
        assert VariantStatus.RUNNING.value == "running"
        assert VariantStatus.WINNER.value  == "winner"
        assert VariantStatus.LOSER.value   == "loser"
        assert VariantStatus.STOPPED.value == "stopped"

    def test_ab_test_status(self):
        assert ABTestStatus.DRAFT.value     == "draft"
        assert ABTestStatus.RUNNING.value   == "running"
        assert ABTestStatus.COMPLETED.value == "completed"
        assert ABTestStatus.STOPPED.value   == "stopped"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VariantStats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVariantStats:
    def test_initial_zero(self):
        s = VariantStats()
        assert s.impressions == 0
        assert s.ctr == 0.0
        assert s.cvr == 0.0

    def test_record_impression(self):
        s = VariantStats()
        s.record_impression(100)
        assert s.impressions == 100

    def test_record_click(self):
        s = VariantStats()
        s.record_click(10)
        assert s.clicks == 10

    def test_record_conversion(self):
        s = VariantStats()
        s.record_conversion(5)
        assert s.conversions == 5

    def test_ctr(self):
        s = VariantStats(impressions=1000, clicks=50)
        assert s.ctr == 0.05

    def test_ctr_zero_impressions(self):
        s = VariantStats(impressions=0, clicks=10)
        assert s.ctr == 0.0

    def test_cvr(self):
        s = VariantStats(clicks=100, conversions=20)
        assert s.cvr == 0.2

    def test_cvr_zero_clicks(self):
        s = VariantStats(clicks=0, conversions=5)
        assert s.cvr == 0.0

    def test_cvr_per_impression(self):
        s = VariantStats(impressions=1000, conversions=10)
        assert s.cvr_per_impression == 0.01

    def test_to_dict(self):
        s = VariantStats(impressions=100, clicks=10, conversions=2)
        d = s.to_dict()
        assert d["impressions"] == 100
        assert d["ctr"] == 0.1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Variant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_variant(vid="v1", name="案A", weight=1.0) -> Variant:
    return Variant(id=vid, name=name, content=f"{name}のコピー", weight=weight)


class TestVariant:
    def test_default_status_draft(self):
        v = make_variant()
        assert v.status == VariantStatus.DRAFT

    def test_default_weight(self):
        v = make_variant()
        assert v.weight == 1.0

    def test_stats_attached(self):
        v = make_variant()
        v.stats.record_impression(10)
        assert v.stats.impressions == 10

    def test_to_dict(self):
        v = make_variant()
        d = v.to_dict()
        assert d["id"] == "v1"
        assert d["status"] == "draft"
        assert "stats" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABTest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_test(test_id="t1") -> ABTest:
    t = ABTest(id=test_id, name="テストA")
    t.add_variant(make_variant("v1", "案A"))
    t.add_variant(make_variant("v2", "案B"))
    return t


class TestABTest:
    def test_add_variant(self):
        t = ABTest(id="t1", name="test")
        t.add_variant(make_variant())
        assert len(t.variants) == 1

    def test_get_variant(self):
        t = make_test()
        assert t.get_variant("v1") is not None
        assert t.get_variant("nope") is None

    def test_start(self):
        t = make_test()
        t.start()
        assert t.status == ABTestStatus.RUNNING
        assert all(v.status == VariantStatus.RUNNING for v in t.variants)

    def test_start_requires_2_variants(self):
        t = ABTest(id="t1", name="test")
        t.add_variant(make_variant())
        with pytest.raises(ValueError):
            t.start()

    def test_start_from_running_raises(self):
        t = make_test()
        t.start()
        with pytest.raises(ValueError):
            t.start()

    def test_stop(self):
        t = make_test()
        t.start()
        t.stop()
        assert t.status == ABTestStatus.STOPPED

    def test_stop_from_draft_raises(self):
        t = make_test()
        with pytest.raises(ValueError):
            t.stop()

    def test_complete_with_winner(self):
        t = make_test()
        t.start()
        t.complete(winner_id="v1")
        assert t.status == ABTestStatus.COMPLETED
        assert t.get_variant("v1").status == VariantStatus.WINNER
        assert t.get_variant("v2").status == VariantStatus.LOSER

    def test_complete_from_draft_raises(self):
        t = make_test()
        with pytest.raises(ValueError):
            t.complete()

    def test_total_impressions(self):
        t = make_test()
        t.get_variant("v1").stats.record_impression(100)
        t.get_variant("v2").stats.record_impression(50)
        assert t.total_impressions == 150

    def test_to_dict(self):
        t = make_test()
        d = t.to_dict()
        assert d["id"] == "t1"
        assert len(d["variants"]) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABTestStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestABTestStore:
    def test_add_and_get(self):
        store = ABTestStore()
        t = make_test()
        store.add(t)
        assert store.get("t1") is t

    def test_get_missing(self):
        store = ABTestStore()
        assert store.get("nope") is None

    def test_list_all(self):
        store = ABTestStore()
        store.add(make_test("a"))
        store.add(make_test("b"))
        assert len(store.list_all()) == 2

    def test_list_by_status(self):
        store = ABTestStore()
        t = make_test("a")
        t.start()
        store.add(t)
        store.add(make_test("b"))
        assert len(store.list_by_status(ABTestStatus.RUNNING)) == 1
        assert len(store.list_by_status(ABTestStatus.DRAFT)) == 1

    def test_find_by_campaign(self):
        store = ABTestStore()
        t = make_test("a")
        t.campaign_id = "camp-1"
        store.add(t)
        store.add(make_test("b"))
        assert len(store.find_by_campaign("camp-1")) == 1

    def test_delete(self):
        store = ABTestStore()
        store.add(make_test("a"))
        assert store.delete("a") is True
        assert store.get("a") is None

    def test_delete_missing(self):
        store = ABTestStore()
        assert store.delete("x") is False

    def test_count(self):
        store = ABTestStore()
        assert store.count() == 0
        store.add(make_test("a"))
        assert store.count() == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TrafficAllocator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTrafficAllocator:
    def test_allocate_returns_variant(self):
        t = make_test()
        alloc = TrafficAllocator(t)
        v = alloc.allocate("user-1")
        assert v is not None
        assert v.id in ("v1", "v2")

    def test_allocate_deterministic(self):
        t = make_test()
        alloc = TrafficAllocator(t)
        v1 = alloc.allocate("user-42")
        v2 = alloc.allocate("user-42")
        assert v1.id == v2.id

    def test_allocate_empty_test(self):
        t = ABTest(id="t1", name="empty")
        alloc = TrafficAllocator(t)
        assert alloc.allocate("user-1") is None

    def test_distribution_roughly_even(self):
        t = make_test()  # 重み 1:1
        alloc = TrafficAllocator(t)
        users = [f"user-{i}" for i in range(2000)]
        dist = alloc.allocation_distribution(users)
        # 2000 ユーザーがおおよそ半々（±15%）
        assert 800 < dist["v1"] < 1200
        assert 800 < dist["v2"] < 1200

    def test_weighted_distribution(self):
        t = ABTest(id="t1", name="weighted")
        t.add_variant(make_variant("v1", "案A", weight=3.0))
        t.add_variant(make_variant("v2", "案B", weight=1.0))
        alloc = TrafficAllocator(t)
        users = [f"user-{i}" for i in range(2000)]
        dist = alloc.allocation_distribution(users)
        # v1 が約 3 倍
        assert dist["v1"] > dist["v2"]

    def test_zero_weight_fallback(self):
        t = ABTest(id="t1", name="zero")
        t.add_variant(make_variant("v1", "案A", weight=0.0))
        t.add_variant(make_variant("v2", "案B", weight=0.0))
        alloc = TrafficAllocator(t)
        v = alloc.allocate("user-1")
        assert v is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABTestAnalyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestABTestAnalyzer:
    def setup_method(self):
        self.analyzer = ABTestAnalyzer(alpha=0.05)

    def test_compare_ctr_returns_result(self):
        a = make_variant("v1")
        b = make_variant("v2")
        a.stats = VariantStats(impressions=1000, clicks=100)
        b.stats = VariantStats(impressions=1000, clicks=50)
        result = self.analyzer.compare_ctr(a, b)
        assert isinstance(result, SignificanceResult)

    def test_compare_ctr_rates(self):
        a = make_variant("v1")
        b = make_variant("v2")
        a.stats = VariantStats(impressions=1000, clicks=100)
        b.stats = VariantStats(impressions=1000, clicks=50)
        result = self.analyzer.compare_ctr(a, b)
        assert result.rate_a == 0.1
        assert result.rate_b == 0.05

    def test_large_difference_is_significant(self):
        a = make_variant("v1")
        b = make_variant("v2")
        # 大きな差 + 大標本 → 有意
        a.stats = VariantStats(impressions=10000, clicks=1000)  # CTR 10%
        b.stats = VariantStats(impressions=10000, clicks=500)   # CTR 5%
        result = self.analyzer.compare_ctr(a, b)
        assert result.significant is True
        assert result.better_variant_id == "v1"

    def test_small_difference_not_significant(self):
        a = make_variant("v1")
        b = make_variant("v2")
        # 小標本・僅差 → 非有意
        a.stats = VariantStats(impressions=100, clicks=11)
        b.stats = VariantStats(impressions=100, clicks=10)
        result = self.analyzer.compare_ctr(a, b)
        assert result.significant is False

    def test_zero_data_not_significant(self):
        a = make_variant("v1")
        b = make_variant("v2")
        result = self.analyzer.compare_ctr(a, b)
        assert result.significant is False
        assert result.p_value == 1.0

    def test_compare_cvr(self):
        a = make_variant("v1")
        b = make_variant("v2")
        a.stats = VariantStats(clicks=1000, conversions=200)
        b.stats = VariantStats(clicks=1000, conversions=100)
        result = self.analyzer.compare_cvr(a, b)
        assert result.rate_a == 0.2
        assert result.rate_b == 0.1

    def test_determine_winner_ctr(self):
        t = make_test()
        t.get_variant("v1").stats = VariantStats(impressions=1000, clicks=100)
        t.get_variant("v2").stats = VariantStats(impressions=1000, clicks=50)
        winner = self.analyzer.determine_winner(t, metric="ctr")
        assert winner.id == "v1"

    def test_determine_winner_no_data(self):
        t = make_test()
        winner = self.analyzer.determine_winner(t)
        assert winner is None

    def test_significance_result_to_dict(self):
        a = make_variant("v1")
        b = make_variant("v2")
        a.stats = VariantStats(impressions=1000, clicks=100)
        b.stats = VariantStats(impressions=1000, clicks=50)
        result = self.analyzer.compare_ctr(a, b)
        d = result.to_dict()
        assert "z_score" in d
        assert "p_value" in d
        assert "significant" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABTestReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestABTestReportEngine:
    def setup_method(self):
        self.engine = ABTestReportEngine()

    def _populated_test(self) -> ABTest:
        t = make_test()
        t.get_variant("v1").stats = VariantStats(impressions=1000, clicks=100, conversions=20)
        t.get_variant("v2").stats = VariantStats(impressions=1000, clicks=50, conversions=10)
        return t

    def test_summary_json(self):
        t = self._populated_test()
        d = self.engine.summary_json(t)
        assert d["variant_count"] == 2
        assert d["winner_id"] == "v1"

    def test_summary_json_no_data(self):
        t = make_test()
        d = self.engine.summary_json(t)
        assert d["winner_id"] is None

    def test_markdown_contains_name(self):
        t = self._populated_test()
        md = self.engine.markdown(t)
        assert "テストA" in md

    def test_markdown_contains_variants(self):
        t = self._populated_test()
        md = self.engine.markdown(t)
        assert "案A" in md
        assert "案B" in md

    def test_markdown_contains_winner(self):
        t = self._populated_test()
        md = self.engine.markdown(t)
        assert "勝者" in md

    def test_markdown_contains_significance(self):
        t = self._populated_test()
        md = self.engine.markdown(t)
        assert "p 値" in md

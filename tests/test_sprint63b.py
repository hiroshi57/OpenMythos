"""
Sprint 63B — キャンペーン分析ダッシュボード テスト

対象:
  open_mythos/skills/campaign_analytics.py:
    MetricType / MetricPoint / CampaignMetrics
    KpiSet / KpiCalculator
    TrendResult / TrendAnalyzer
    CampaignAnalyticsStore
    AnalyticsSnapshot / CampaignAnalyticsDashboard
    AnalyticsReportEngine
"""
from __future__ import annotations

import pytest

from open_mythos.skills.campaign_analytics import (
    MetricType, MetricPoint, CampaignMetrics,
    KpiSet, KpiCalculator,
    TrendResult, TrendAnalyzer,
    CampaignAnalyticsStore,
    AnalyticsSnapshot, CampaignAnalyticsDashboard,
    AnalyticsReportEngine,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MetricType / MetricPoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMetricType:
    def test_values(self):
        assert MetricType.IMPRESSIONS.value == "impressions"
        assert MetricType.CLICKS.value      == "clicks"
        assert MetricType.CONVERSIONS.value == "conversions"
        assert MetricType.SPEND.value       == "spend"
        assert MetricType.REVENUE.value     == "revenue"


class TestMetricPoint:
    def test_defaults(self):
        p = MetricPoint()
        assert p.impressions == 0
        assert p.spend == 0.0

    def test_to_dict(self):
        p = MetricPoint(impressions=100, clicks=10, spend=50.0)
        d = p.to_dict()
        assert d["impressions"] == 100
        assert d["spend"] == 50.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignMetrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignMetrics:
    def test_record_adds_point(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=100)
        assert len(m.points) == 1

    def test_total_impressions(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=100)
        m.record(impressions=200)
        assert m.total_impressions == 300

    def test_total_clicks(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(clicks=10)
        m.record(clicks=20)
        assert m.total_clicks == 30

    def test_total_conversions(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(conversions=5)
        assert m.total_conversions == 5

    def test_total_spend(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(spend=100.0)
        m.record(spend=50.0)
        assert m.total_spend == 150.0

    def test_total_revenue(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(revenue=500.0)
        assert m.total_revenue == 500.0

    def test_add_point(self):
        m = CampaignMetrics(campaign_id="c1")
        m.add_point(MetricPoint(impressions=50))
        assert m.total_impressions == 50

    def test_to_dict(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=100, clicks=10)
        d = m.to_dict()
        assert d["campaign_id"] == "c1"
        assert d["total_impressions"] == 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KpiCalculator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestKpiCalculator:
    def setup_method(self):
        self.calc = KpiCalculator()

    def _metrics(self) -> CampaignMetrics:
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=10000, clicks=500, conversions=50, spend=1000.0, revenue=5000.0)
        return m

    def test_ctr(self):
        kpis = self.calc.compute(self._metrics())
        assert kpis.ctr == 0.05  # 500 / 10000

    def test_cvr(self):
        kpis = self.calc.compute(self._metrics())
        assert kpis.cvr == 0.1  # 50 / 500

    def test_cpc(self):
        kpis = self.calc.compute(self._metrics())
        assert kpis.cpc == 2.0  # 1000 / 500

    def test_cpa(self):
        kpis = self.calc.compute(self._metrics())
        assert kpis.cpa == 20.0  # 1000 / 50

    def test_cpm(self):
        kpis = self.calc.compute(self._metrics())
        assert kpis.cpm == 100.0  # 1000 / 10000 * 1000

    def test_roas(self):
        kpis = self.calc.compute(self._metrics())
        assert kpis.roas == 5.0  # 5000 / 1000

    def test_roi(self):
        kpis = self.calc.compute(self._metrics())
        assert kpis.roi == 4.0  # (5000 - 1000) / 1000

    def test_zero_safe(self):
        m = CampaignMetrics(campaign_id="empty")
        kpis = self.calc.compute(m)
        assert kpis.ctr == 0.0
        assert kpis.roas == 0.0

    def test_compute_from_point(self):
        p = MetricPoint(impressions=1000, clicks=100, spend=200.0, revenue=600.0)
        kpis = self.calc.compute_from_point(p)
        assert kpis.ctr == 0.1
        assert kpis.roas == 3.0

    def test_kpi_to_dict(self):
        kpis = self.calc.compute(self._metrics())
        d = kpis.to_dict()
        assert "ctr" in d
        assert "roas" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TrendAnalyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTrendAnalyzer:
    def setup_method(self):
        self.analyzer = TrendAnalyzer()

    def test_period_over_period_up(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=100)
        m.record(impressions=150)
        trend = self.analyzer.period_over_period(m, "impressions")
        assert trend.direction == "up"
        assert trend.delta == 50

    def test_period_over_period_down(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=200)
        m.record(impressions=100)
        trend = self.analyzer.period_over_period(m, "impressions")
        assert trend.direction == "down"

    def test_period_over_period_flat(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=100)
        m.record(impressions=100)
        trend = self.analyzer.period_over_period(m, "impressions")
        assert trend.direction == "flat"

    def test_period_over_period_insufficient(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=100)
        assert self.analyzer.period_over_period(m) is None

    def test_delta_pct(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=100)
        m.record(impressions=150)
        trend = self.analyzer.period_over_period(m, "impressions")
        assert trend.delta_pct == 0.5

    def test_moving_average(self):
        m = CampaignMetrics(campaign_id="c1")
        for v in [10, 20, 30, 40]:
            m.record(impressions=v)
        ma = self.analyzer.moving_average(m, "impressions", window=2)
        assert ma[0] == 10       # [10]
        assert ma[1] == 15       # (10+20)/2
        assert ma[3] == 35       # (30+40)/2

    def test_moving_average_invalid_window(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=10)
        with pytest.raises(ValueError):
            self.analyzer.moving_average(m, window=0)

    def test_total_series(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(clicks=5)
        m.record(clicks=10)
        series = self.analyzer.total_series(m, "clicks")
        assert series == [5.0, 10.0]

    def test_trend_to_dict(self):
        m = CampaignMetrics(campaign_id="c1")
        m.record(impressions=100)
        m.record(impressions=150)
        trend = self.analyzer.period_over_period(m, "impressions")
        d = trend.to_dict()
        assert d["direction"] == "up"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignAnalyticsStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignAnalyticsStore:
    def test_get_or_create(self):
        store = CampaignAnalyticsStore()
        m = store.get_or_create("c1")
        assert m.campaign_id == "c1"

    def test_get_or_create_idempotent(self):
        store = CampaignAnalyticsStore()
        m1 = store.get_or_create("c1")
        m2 = store.get_or_create("c1")
        assert m1 is m2

    def test_record(self):
        store = CampaignAnalyticsStore()
        store.record("c1", impressions=100)
        assert store.get("c1").total_impressions == 100

    def test_get_missing(self):
        store = CampaignAnalyticsStore()
        assert store.get("nope") is None

    def test_list_campaigns(self):
        store = CampaignAnalyticsStore()
        store.record("c1", impressions=10)
        store.record("c2", impressions=20)
        assert set(store.list_campaigns()) == {"c1", "c2"}

    def test_count(self):
        store = CampaignAnalyticsStore()
        assert store.count() == 0
        store.record("c1", impressions=10)
        assert store.count() == 1

    def test_delete(self):
        store = CampaignAnalyticsStore()
        store.record("c1", impressions=10)
        assert store.delete("c1") is True
        assert store.get("c1") is None

    def test_delete_missing(self):
        store = CampaignAnalyticsStore()
        assert store.delete("x") is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignAnalyticsDashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignAnalyticsDashboard:
    def setup_method(self):
        self.store = CampaignAnalyticsStore()
        self.dash = CampaignAnalyticsDashboard(store=self.store)

    def test_campaign_kpis(self):
        self.store.record("c1", impressions=1000, clicks=100)
        kpis = self.dash.campaign_kpis("c1")
        assert kpis.ctr == 0.1

    def test_campaign_kpis_missing(self):
        assert self.dash.campaign_kpis("nope") is None

    def test_snapshot_empty(self):
        snap = self.dash.snapshot()
        assert snap.campaign_count == 0

    def test_snapshot_aggregates(self):
        self.store.record("c1", impressions=1000, clicks=100, spend=200.0, revenue=600.0)
        self.store.record("c2", impressions=2000, clicks=100, spend=300.0, revenue=900.0)
        snap = self.dash.snapshot()
        assert snap.campaign_count == 2
        assert snap.total_impressions == 3000
        assert snap.total_clicks == 200

    def test_snapshot_to_dict(self):
        self.store.record("c1", impressions=1000)
        snap = self.dash.snapshot()
        d = snap.to_dict()
        assert "aggregate_kpis" in d

    def test_rank_by_roas(self):
        self.store.record("c1", spend=100.0, revenue=500.0)  # roas 5
        self.store.record("c2", spend=100.0, revenue=200.0)  # roas 2
        ranking = self.dash.rank_by_kpi("roas")
        assert ranking[0]["campaign_id"] == "c1"
        assert ranking[1]["campaign_id"] == "c2"

    def test_rank_by_ctr(self):
        self.store.record("c1", impressions=1000, clicks=200)  # ctr 0.2
        self.store.record("c2", impressions=1000, clicks=50)   # ctr 0.05
        ranking = self.dash.rank_by_kpi("ctr")
        assert ranking[0]["campaign_id"] == "c1"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AnalyticsReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAnalyticsReportEngine:
    def setup_method(self):
        self.store = CampaignAnalyticsStore()
        self.dash = CampaignAnalyticsDashboard(store=self.store)
        self.engine = AnalyticsReportEngine(self.dash)

    def test_summary_json(self):
        self.store.record("c1", impressions=1000, clicks=100, spend=200.0, revenue=600.0)
        d = self.engine.summary_json()
        assert "snapshot" in d
        assert "ranking_by_roas" in d

    def test_campaign_markdown_not_found(self):
        md = self.engine.campaign_markdown("nope")
        assert "エラー" in md

    def test_campaign_markdown_contains_kpi(self):
        self.store.record("c1", impressions=1000, clicks=100, conversions=10, spend=200.0, revenue=600.0)
        md = self.engine.campaign_markdown("c1")
        assert "KPI" in md
        assert "ROAS" in md

    def test_campaign_markdown_contains_id(self):
        self.store.record("c1", impressions=100)
        md = self.engine.campaign_markdown("c1")
        assert "c1" in md

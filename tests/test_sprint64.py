"""
Sprint 64 — A/Bテスト API公開 + 形態素連携 + 予算最適化 テスト

対象:
  64A: serve/api.py — /v1/abtest/* /v1/analytics/* エンドポイント
  64B: open_mythos/skills/campaign_manager.py — CopyGenerator(use_morphology=True)
  64C: open_mythos/skills/budget_optimizer.py — BudgetOptimizer
"""
from __future__ import annotations

import pytest

from open_mythos.skills.campaign_manager import CopyGenerator, AdObjective
from open_mythos.skills.campaign_analytics import CampaignAnalyticsStore
from open_mythos.skills.budget_optimizer import (
    AllocationStrategy, BudgetConstraint, BudgetAllocation,
    OptimizationResult, BudgetOptimizer,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 64C: BudgetOptimizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAllocationStrategy:
    def test_values(self):
        assert AllocationStrategy.EQUAL.value         == "equal"
        assert AllocationStrategy.ROAS_WEIGHTED.value  == "roas_weighted"
        assert AllocationStrategy.PERFORMANCE.value   == "performance"
        assert AllocationStrategy.PROPORTIONAL.value  == "proportional"


class TestBudgetConstraint:
    def test_clamp_min(self):
        c = BudgetConstraint(min_budget=100)
        assert c.clamp(50) == 100

    def test_clamp_max(self):
        c = BudgetConstraint(max_budget=1000)
        assert c.clamp(2000) == 1000

    def test_clamp_within(self):
        c = BudgetConstraint(min_budget=100, max_budget=1000)
        assert c.clamp(500) == 500

    def test_no_max(self):
        c = BudgetConstraint(min_budget=0)
        assert c.clamp(99999) == 99999


class TestBudgetAllocation:
    def test_to_dict(self):
        a = BudgetAllocation(campaign_id="c1", amount=5000.0, share=0.5, roas=3.0)
        d = a.to_dict()
        assert d["campaign_id"] == "c1"
        assert d["amount"] == 5000.0
        assert d["share"] == 0.5


class TestBudgetOptimizer:
    def setup_method(self):
        self.store = CampaignAnalyticsStore()
        # c1: ROAS 5, c2: ROAS 2
        self.store.record("c1", impressions=10000, clicks=500, conversions=50, spend=1000.0, revenue=5000.0)
        self.store.record("c2", impressions=10000, clicks=300, conversions=20, spend=1000.0, revenue=2000.0)
        self.opt = BudgetOptimizer(store=self.store)

    def test_optimize_returns_result(self):
        result = self.opt.optimize(10000, ["c1", "c2"])
        assert isinstance(result, OptimizationResult)

    def test_roas_weighted_favors_high_roas(self):
        result = self.opt.optimize(
            10000, ["c1", "c2"], strategy=AllocationStrategy.ROAS_WEIGHTED
        )
        a1 = result.get("c1")
        a2 = result.get("c2")
        # c1 (ROAS 5) > c2 (ROAS 2)
        assert a1.amount > a2.amount

    def test_roas_weighted_ratio(self):
        result = self.opt.optimize(
            7000, ["c1", "c2"], strategy=AllocationStrategy.ROAS_WEIGHTED
        )
        # ROAS 5:2 → 5000:2000
        a1 = result.get("c1")
        a2 = result.get("c2")
        assert abs(a1.amount - 5000) < 1.0
        assert abs(a2.amount - 2000) < 1.0

    def test_equal_strategy(self):
        result = self.opt.optimize(
            10000, ["c1", "c2"], strategy=AllocationStrategy.EQUAL
        )
        assert abs(result.get("c1").amount - 5000) < 1.0
        assert abs(result.get("c2").amount - 5000) < 1.0

    def test_performance_strategy(self):
        result = self.opt.optimize(
            10000, ["c1", "c2"], strategy=AllocationStrategy.PERFORMANCE
        )
        # c1 CVR=50/500=0.1, c2 CVR=20/300=0.067 → c1 多い
        assert result.get("c1").amount > result.get("c2").amount

    def test_proportional_strategy(self):
        result = self.opt.optimize(
            10000, ["c1", "c2"], strategy=AllocationStrategy.PROPORTIONAL
        )
        # spend 同じ → ほぼ均等
        assert abs(result.get("c1").amount - result.get("c2").amount) < 1.0

    def test_allocated_total(self):
        result = self.opt.optimize(10000, ["c1", "c2"])
        assert abs(result.allocated_total - 10000) < 1.0

    def test_shares_sum_to_one(self):
        result = self.opt.optimize(10000, ["c1", "c2"])
        total_share = sum(a.share for a in result.allocations)
        assert abs(total_share - 1.0) < 1e-6

    def test_empty_campaigns(self):
        result = self.opt.optimize(10000, [])
        assert result.allocations == []
        assert result.unallocated == 10000

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError):
            self.opt.optimize(-100, ["c1"])

    def test_unknown_campaign_zero_roas(self):
        result = self.opt.optimize(
            10000, ["unknown1", "unknown2"], strategy=AllocationStrategy.ROAS_WEIGHTED
        )
        # ROAS 全 0 → 均等フォールバック
        assert abs(result.get("unknown1").amount - 5000) < 1.0

    def test_constraint_applied(self):
        result = self.opt.optimize(
            10000, ["c1", "c2"],
            strategy=AllocationStrategy.ROAS_WEIGHTED,
            constraints={"c1": BudgetConstraint(max_budget=3000)},
        )
        assert result.get("c1").amount <= 3000

    def test_recommend_strategy_roas(self):
        # revenue あり → ROAS_WEIGHTED
        assert self.opt.recommend_strategy(["c1", "c2"]) == AllocationStrategy.ROAS_WEIGHTED

    def test_recommend_strategy_performance(self):
        store = CampaignAnalyticsStore()
        store.record("x", clicks=100, conversions=10)  # revenue なし
        opt = BudgetOptimizer(store=store)
        assert opt.recommend_strategy(["x"]) == AllocationStrategy.PERFORMANCE

    def test_recommend_strategy_proportional(self):
        store = CampaignAnalyticsStore()
        store.record("x", spend=500.0)  # spend のみ
        opt = BudgetOptimizer(store=store)
        assert opt.recommend_strategy(["x"]) == AllocationStrategy.PROPORTIONAL

    def test_recommend_strategy_equal(self):
        store = CampaignAnalyticsStore()
        opt = BudgetOptimizer(store=store)
        assert opt.recommend_strategy(["nope"]) == AllocationStrategy.EQUAL

    def test_result_to_dict(self):
        result = self.opt.optimize(10000, ["c1", "c2"])
        d = result.to_dict()
        assert "allocations" in d
        assert d["strategy"] == "roas_weighted"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 64B: CopyGenerator with morphology
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCopyGeneratorMorphology:
    def test_default_no_morphology(self):
        gen = CopyGenerator(brand="Test")
        assert gen.use_morphology is False

    def test_morphology_enabled(self):
        gen = CopyGenerator(brand="Test", use_morphology=True)
        assert gen.use_morphology is True
        assert gen._analyzer is not None

    def test_morphology_generates_copy(self):
        gen = CopyGenerator(brand="Test", use_morphology=True)
        copy = gen.generate_from_scenario("東京で広告キャンペーンを実施したい")
        assert copy is not None
        assert len(copy.headline) > 0

    def test_morphology_tags_are_list(self):
        gen = CopyGenerator(brand="Test", use_morphology=True)
        copy = gen.generate_from_scenario("コスト削減を実現する企業向けサービス")
        assert isinstance(copy.tags, list)

    def test_morphology_tags_min_length(self):
        gen = CopyGenerator(brand="Test", use_morphology=True)
        copy = gen.generate_from_scenario("広告効果を測定したい")
        # 2 文字以上の名詞のみ
        assert all(len(t) >= 2 for t in copy.tags)

    def test_morphology_backward_compat_signature(self):
        # 既存シグネチャで生成できる
        gen = CopyGenerator(brand="Test", use_morphology=True)
        copy = gen.generate_from_scenario(
            "テスト", objective=AdObjective.CONVERSION
        )
        assert copy.objective == AdObjective.CONVERSION

    def test_custom_analyzer(self):
        from open_mythos.tokenizer_ja import JaMorphologicalAnalyzer, PartOfSpeech
        analyzer = JaMorphologicalAnalyzer()
        analyzer.dictionary.add_word("特別商品", PartOfSpeech.NOUN)
        gen = CopyGenerator(brand="Test", use_morphology=True, analyzer=analyzer)
        copy = gen.generate_from_scenario("特別商品を販売する")
        assert "特別商品" in copy.tags

    def test_rule_based_still_works(self):
        gen = CopyGenerator(brand="Test", use_morphology=False)
        copy = gen.generate_from_scenario("コスト削減したい企業向け")
        assert isinstance(copy.tags, list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 64A: API エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serve.api import app
    return TestClient(app)


class TestABTestApi:
    def _create(self, client):
        return client.post("/v1/abtest/", json={
            "name": "API A/B テスト",
            "variants": [
                {"name": "案A", "content": "コピーA", "weight": 1.0},
                {"name": "案B", "content": "コピーB", "weight": 1.0},
            ],
        })

    def test_create(self, client):
        resp = self._create(client)
        assert resp.status_code == 200
        assert len(resp.json()["variants"]) == 2

    def test_list(self, client):
        self._create(client)
        resp = client.get("/v1/abtest/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get(self, client):
        tid = self._create(client).json()["id"]
        resp = client.get(f"/v1/abtest/{tid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == tid

    def test_get_not_found(self, client):
        assert client.get("/v1/abtest/nope").status_code == 404

    def test_delete(self, client):
        tid = self._create(client).json()["id"]
        assert client.delete(f"/v1/abtest/{tid}").status_code == 200
        assert client.get(f"/v1/abtest/{tid}").status_code == 404

    def test_start(self, client):
        tid = self._create(client).json()["id"]
        resp = client.post(f"/v1/abtest/{tid}/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_record_stats(self, client):
        created = self._create(client).json()
        tid = created["id"]
        vid = created["variants"][0]["id"]
        resp = client.post(
            f"/v1/abtest/{tid}/variant/{vid}/record",
            json={"impressions": 1000, "clicks": 100},
        )
        assert resp.status_code == 200
        assert resp.json()["stats"]["impressions"] == 1000

    def test_record_variant_not_found(self, client):
        tid = self._create(client).json()["id"]
        resp = client.post(
            f"/v1/abtest/{tid}/variant/nope/record",
            json={"impressions": 10},
        )
        assert resp.status_code == 404

    def test_report_json(self, client):
        created = self._create(client).json()
        tid = created["id"]
        vid = created["variants"][0]["id"]
        client.post(f"/v1/abtest/{tid}/variant/{vid}/record",
                    json={"impressions": 1000, "clicks": 100})
        resp = client.get(f"/v1/abtest/{tid}/report")
        assert resp.status_code == 200
        assert "variant_count" in resp.json()

    def test_report_md(self, client):
        tid = self._create(client).json()["id"]
        resp = client.get(f"/v1/abtest/{tid}/report/md")
        assert resp.status_code == 200
        assert "A/B" in resp.text


class TestAnalyticsApi:
    def test_record(self, client):
        resp = client.post("/v1/analytics/camp-a/record", json={
            "impressions": 1000, "clicks": 100, "spend": 200.0, "revenue": 600.0,
        })
        assert resp.status_code == 200
        assert resp.json()["impressions"] == 1000

    def test_kpis(self, client):
        client.post("/v1/analytics/camp-b/record", json={
            "impressions": 1000, "clicks": 100,
        })
        resp = client.get("/v1/analytics/camp-b/kpis")
        assert resp.status_code == 200
        assert resp.json()["ctr"] == 0.1

    def test_kpis_not_found(self, client):
        assert client.get("/v1/analytics/nope-xyz/kpis").status_code == 404

    def test_report_md(self, client):
        client.post("/v1/analytics/camp-c/record", json={
            "impressions": 1000, "clicks": 100, "spend": 200.0, "revenue": 600.0,
        })
        resp = client.get("/v1/analytics/camp-c/report/md")
        assert resp.status_code == 200
        assert "KPI" in resp.text

    def test_report_md_not_found(self, client):
        assert client.get("/v1/analytics/nope-abc/report/md").status_code == 404

    def test_summary(self, client):
        client.post("/v1/analytics/camp-d/record", json={"impressions": 500})
        resp = client.get("/v1/analytics/summary")
        assert resp.status_code == 200
        assert "snapshot" in resp.json()


class TestBudgetApi:
    def test_optimize(self, client):
        client.post("/v1/analytics/bo-1/record", json={
            "impressions": 10000, "clicks": 500, "spend": 1000.0, "revenue": 5000.0,
        })
        client.post("/v1/analytics/bo-2/record", json={
            "impressions": 10000, "clicks": 300, "spend": 1000.0, "revenue": 2000.0,
        })
        resp = client.post("/v1/budget/optimize", json={
            "total_budget": 7000,
            "campaign_ids": ["bo-1", "bo-2"],
            "strategy": "roas_weighted",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["allocations"]) == 2

    def test_optimize_unknown_strategy(self, client):
        resp = client.post("/v1/budget/optimize", json={
            "total_budget": 1000,
            "campaign_ids": ["x"],
            "strategy": "invalid",
        })
        assert resp.status_code == 422

    def test_recommend_strategy(self, client):
        client.post("/v1/analytics/bo-3/record", json={
            "impressions": 1000, "clicks": 100, "spend": 100.0, "revenue": 500.0,
        })
        resp = client.post("/v1/budget/recommend-strategy", json={
            "total_budget": 0,
            "campaign_ids": ["bo-3"],
        })
        assert resp.status_code == 200
        assert resp.json()["recommended_strategy"] == "roas_weighted"

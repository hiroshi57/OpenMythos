"""
Sprint 67 — 異常検知自動停止 + Fusion キャッシュ + 統合ダッシュボード テスト

対象:
  67A: open_mythos/skills/campaign_orchestrator.py
       — FreezeDecision / FrozenBudgetPlan / CampaignOrchestrator.freeze_if_critical
  67B: open_mythos/skills/fusion_cache.py
       — FusionCache / CachedFusionEngine
  67C: serve/api.py
       — /v1/orchestrator/freeze
       — /v1/fusion/cached  /v1/fusion/cache/stats  /v1/fusion/cache/clear
       — /v1/dashboard/summary  /v1/dashboard/campaigns  /v1/dashboard/alerts/critical
"""
from __future__ import annotations

import time
import pytest

from open_mythos.skills.anomaly_detector import (
    Alert, AlertSeverity, AnomalyType, AlertStore, AnomalyDetector,
)
from open_mythos.skills.campaign_analytics import CampaignAnalyticsStore
from open_mythos.skills.campaign_orchestrator import (
    CampaignOrchestrator, OrchestrationConfig,
    FreezeDecision, FrozenBudgetPlan,
)
from open_mythos.skills.fusion import FusionConfig, CandidateSpec, FusionEngineFactory
from open_mythos.skills.fusion_cache import (
    FusionCache, CachedFusionEngine, _make_cache_key,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ヘルパ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _critical_alert(campaign_id: str = "c1") -> Alert:
    return Alert(
        id="a1", campaign_id=campaign_id, metric="clicks",
        anomaly_type=AnomalyType.SPIKE, severity=AlertSeverity.CRITICAL,
        current=1000, baseline=100, z_score=9.0, change_pct=9.0,
        message="急増テスト",
    )


def _warning_alert(campaign_id: str = "c1") -> Alert:
    return Alert(
        id="a2", campaign_id=campaign_id, metric="clicks",
        anomaly_type=AnomalyType.DROP, severity=AlertSeverity.WARNING,
        current=50, baseline=100, z_score=2.5, change_pct=-0.5,
        message="急減テスト",
    )


def _make_analytics_store_with_spike(campaign_id: str) -> CampaignAnalyticsStore:
    store = CampaignAnalyticsStore()
    m = store.get_or_create(campaign_id)
    for _ in range(4):
        m.record(clicks=100)
    m.record(clicks=5000)   # Critical スパイク
    return store


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 67A: FreezeDecision / FrozenBudgetPlan
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFreezeDecision:
    def test_to_dict_keys(self):
        fd = FreezeDecision(
            freeze_campaign_ids=["c1"],
            frozen=True,
            alert_count=2,
            reason="テスト凍結",
        )
        d = fd.to_dict()
        assert "freeze_campaign_ids" in d
        assert "frozen" in d
        assert "alert_count" in d
        assert "reason" in d
        assert "timestamp" in d

    def test_frozen_true_when_ids(self):
        fd = FreezeDecision(freeze_campaign_ids=["c1"], frozen=True, alert_count=1, reason="x")
        assert fd.frozen is True

    def test_frozen_false_when_empty(self):
        fd = FreezeDecision(freeze_campaign_ids=[], frozen=False, alert_count=0, reason="y")
        assert fd.frozen is False


class TestFrozenBudgetPlan:
    def _plan(self) -> FrozenBudgetPlan:
        store = _make_analytics_store_with_spike("c1")
        orch = CampaignOrchestrator(analytics_store=store)
        alert_store = AlertStore()
        alert_store.add(_critical_alert("c1"))
        return orch.freeze_if_critical(
            campaign_ids=["c1", "c2"],
            total_budget=10000,
            alert_store=alert_store,
        )

    def test_returns_frozen_budget_plan(self):
        plan = self._plan()
        assert isinstance(plan, FrozenBudgetPlan)

    def test_c1_frozen(self):
        plan = self._plan()
        assert "c1" in plan.freeze_decision.freeze_campaign_ids

    def test_c2_not_frozen(self):
        plan = self._plan()
        assert "c2" not in plan.freeze_decision.freeze_campaign_ids

    def test_remaining_budget_set(self):
        plan = self._plan()
        assert plan.remaining_budget == 10000

    def test_to_dict_keys(self):
        plan = self._plan()
        d = plan.to_dict()
        assert "freeze_decision" in d
        assert "optimization" in d
        assert "remaining_budget" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 67A: CampaignOrchestrator.freeze_if_critical
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFreezeIfCritical:
    def setup_method(self):
        self.orch = CampaignOrchestrator()

    def _alert_store_with_critical(self, campaign_id: str) -> AlertStore:
        store = AlertStore()
        store.add(_critical_alert(campaign_id))
        return store

    def test_no_critical_no_freeze(self):
        store = AlertStore()
        store.add(_warning_alert("c1"))
        plan = self.orch.freeze_if_critical(
            campaign_ids=["c1", "c2"],
            total_budget=10000,
            alert_store=store,
        )
        assert plan.freeze_decision.frozen is False
        assert plan.freeze_decision.freeze_campaign_ids == []

    def test_critical_freezes_campaign(self):
        store = self._alert_store_with_critical("c1")
        plan = self.orch.freeze_if_critical(
            campaign_ids=["c1", "c2"],
            total_budget=10000,
            alert_store=store,
        )
        assert plan.freeze_decision.frozen is True
        assert "c1" in plan.freeze_decision.freeze_campaign_ids

    def test_non_target_campaign_not_frozen(self):
        # c99 は campaign_ids にない → 対象外
        store = AlertStore()
        store.add(_critical_alert("c99"))
        plan = self.orch.freeze_if_critical(
            campaign_ids=["c1", "c2"],
            total_budget=10000,
            alert_store=store,
        )
        assert plan.freeze_decision.frozen is False

    def test_all_frozen_empty_optimization(self):
        store = AlertStore()
        store.add(_critical_alert("c1"))
        store.add(_critical_alert("c2"))
        plan = self.orch.freeze_if_critical(
            campaign_ids=["c1", "c2"],
            total_budget=10000,
            alert_store=store,
        )
        assert plan.freeze_decision.frozen is True
        assert plan.optimization.allocations == []

    def test_redetect_from_analytics(self):
        # alert_store=None → analytics_store から再検知
        analytics = _make_analytics_store_with_spike("c1")
        orch = CampaignOrchestrator(analytics_store=analytics)
        plan = orch.freeze_if_critical(
            campaign_ids=["c1", "c2"],
            total_budget=10000,
            alert_store=None,
        )
        assert plan.freeze_decision.frozen is True
        assert "c1" in plan.freeze_decision.freeze_campaign_ids

    def test_alert_count_recorded(self):
        store = AlertStore()
        store.add(_critical_alert("c1"))
        store.add(_critical_alert("c1"))
        plan = self.orch.freeze_if_critical(
            campaign_ids=["c1"],
            total_budget=5000,
            alert_store=store,
        )
        assert plan.freeze_decision.alert_count == 2

    def test_reason_not_empty(self):
        store = self._alert_store_with_critical("c1")
        plan = self.orch.freeze_if_critical(
            campaign_ids=["c1"],
            total_budget=5000,
            alert_store=store,
        )
        assert len(plan.freeze_decision.reason) > 0

    def test_active_campaign_gets_budget(self):
        store = self._alert_store_with_critical("c1")
        analytics = CampaignAnalyticsStore()
        analytics.record("c2", spend=500, revenue=2000)
        orch = CampaignOrchestrator(analytics_store=analytics)
        plan = orch.freeze_if_critical(
            campaign_ids=["c1", "c2"],
            total_budget=10000,
            alert_store=store,
        )
        # c2 は非凍結 → optimization に含まれる
        active_ids = [a.campaign_id for a in plan.optimization.allocations]
        assert "c2" in active_ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 67B: FusionCache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMakeCacheKey:
    def test_same_input_same_key(self):
        assert _make_cache_key("Q", None) == _make_cache_key("Q", None)

    def test_different_question_different_key(self):
        assert _make_cache_key("Q1", None) != _make_cache_key("Q2", None)

    def test_different_system_different_key(self):
        assert _make_cache_key("Q", "sys1") != _make_cache_key("Q", "sys2")

    def test_none_vs_empty_system(self):
        assert _make_cache_key("Q", None) == _make_cache_key("Q", "")


class TestFusionCache:
    def setup_method(self):
        self.engine = FusionEngineFactory.rule_based(
            config=FusionConfig(candidates=[
                CandidateSpec(label="c1"), CandidateSpec(label="c2"),
            ])
        )
        self.cache = FusionCache(ttl=300.0, max_size=128)

    def _result(self, q: str = "Q"):
        return self.engine.run(q)

    def test_miss_on_empty(self):
        assert self.cache.get("Q") is None

    def test_put_and_get(self):
        r = self._result()
        self.cache.put("Q", r)
        assert self.cache.get("Q") is not None

    def test_hit_returns_same_result(self):
        r = self._result()
        self.cache.put("Q", r)
        assert self.cache.get("Q").final_answer == r.final_answer

    def test_different_question_miss(self):
        r = self._result()
        self.cache.put("Q1", r)
        assert self.cache.get("Q2") is None

    def test_expired_returns_none(self):
        r = self._result()
        entry = self.cache.put("Q", r)
        entry.created_at = time.time() - 400   # TTL=300 超過
        assert self.cache.get("Q") is None

    def test_stats_hit_rate(self):
        r = self._result()
        self.cache.put("Q", r)
        self.cache.get("Q")   # hit
        self.cache.get("X")   # miss
        s = self.cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5

    def test_clear(self):
        self.cache.put("Q", self._result())
        n = self.cache.clear()
        assert n == 1
        assert len(self.cache) == 0

    def test_evict_expired(self):
        r = self._result()
        entry = self.cache.put("Q", r)
        entry.created_at = time.time() - 400
        evicted = self.cache.evict_expired()
        assert evicted == 1
        assert len(self.cache) == 0

    def test_max_size_evict_oldest(self):
        cache = FusionCache(ttl=300.0, max_size=2)
        r = self._result()
        cache.put("Q1", r)
        cache.put("Q2", r)
        cache.put("Q3", r)   # max_size=2 なので Q1 が evict
        assert len(cache) == 2
        assert cache.get("Q1") is None

    def test_invalidate(self):
        self.cache.put("Q", self._result())
        removed = self.cache.invalidate("Q")
        assert removed is True
        assert self.cache.get("Q") is None

    def test_invalidate_nonexistent(self):
        assert self.cache.invalidate("nope") is False

    def test_stats_keys(self):
        s = self.cache.stats()
        for k in ("size", "hits", "misses", "hit_rate", "ttl", "max_size"):
            assert k in s


class TestCachedFusionEngine:
    def setup_method(self):
        self.engine = FusionEngineFactory.rule_based(
            config=FusionConfig(candidates=[
                CandidateSpec(label="c1"), CandidateSpec(label="c2"),
            ])
        )
        self.cache = FusionCache(ttl=300.0)
        self.cached = CachedFusionEngine(self.engine, cache=self.cache)

    def test_first_call_miss(self):
        self.cached.run("Q")
        assert self.cache.stats()["misses"] == 1

    def test_second_call_hit(self):
        self.cached.run("Q")
        self.cached.run("Q")
        s = self.cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1

    def test_result_consistent(self):
        r1 = self.cached.run("Q")
        r2 = self.cached.run("Q")
        assert r1.final_answer == r2.final_answer

    def test_stream_bypasses_cache(self):
        events = list(self.cached.run_stream("Q"))
        # ストリームはキャッシュ非対象 → ヒット数は変わらない
        assert self.cache.stats()["hits"] == 0

    def test_cache_stats_property(self):
        self.cached.run("Q")
        s = self.cached.cache_stats
        assert "hit_rate" in s


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serve.api import app
    return TestClient(app)


class TestFreezeApi:
    def _setup_critical(self, client):
        """analytics に Critical スパイクを作り、detect で alert_store に蓄積する"""
        for _ in range(4):
            client.post("/v1/analytics/freeze-c1/record", json={"clicks": 100})
        client.post("/v1/analytics/freeze-c1/record", json={"clicks": 5000})
        client.post("/v1/anomaly/freeze-c1/detect", json={"metrics": ["clicks"]})

    def test_freeze_with_alert_store(self, client):
        self._setup_critical(client)
        resp = client.post("/v1/orchestrator/freeze", json={
            "campaign_ids": ["freeze-c1", "freeze-c2"],
            "total_budget": 10000,
            "use_alert_store": True,
        })
        assert resp.status_code == 200
        d = resp.json()
        assert "freeze_decision" in d
        assert d["freeze_decision"]["frozen"] is True

    def test_freeze_no_critical(self, client):
        resp = client.post("/v1/orchestrator/freeze", json={
            "campaign_ids": ["no-crit-x"],
            "total_budget": 5000,
            "use_alert_store": False,
        })
        assert resp.status_code == 200
        # analytics なし → 検知なし → frozen=False
        assert resp.json()["freeze_decision"]["frozen"] is False

    def test_freeze_response_keys(self, client):
        resp = client.post("/v1/orchestrator/freeze", json={
            "campaign_ids": ["x"],
            "total_budget": 1000,
        })
        assert resp.status_code == 200
        d = resp.json()
        assert "freeze_decision" in d
        assert "optimization" in d
        assert "remaining_budget" in d


class TestFusionCacheApi:
    def test_cached_run(self, client):
        resp = client.post("/v1/fusion/cached", json={"question": "テスト質問"})
        assert resp.status_code == 200
        assert "final_answer" in resp.json()

    def test_cache_stats(self, client):
        resp = client.get("/v1/fusion/cache/stats")
        assert resp.status_code == 200
        s = resp.json()
        assert "hit_rate" in s
        assert "size" in s

    def test_cache_clear(self, client):
        client.post("/v1/fusion/cached", json={"question": "キャッシュ用質問"})
        resp = client.delete("/v1/fusion/cache/clear")
        assert resp.status_code == 200
        assert "cleared" in resp.json()

    def test_second_call_hits_cache(self, client):
        client.delete("/v1/fusion/cache/clear")
        q = "キャッシュテスト一意質問xyz"
        client.post("/v1/fusion/cached", json={"question": q})
        client.post("/v1/fusion/cached", json={"question": q})
        s = client.get("/v1/fusion/cache/stats").json()
        assert s["hits"] >= 1

    def test_custom_candidates_no_cache(self, client):
        """candidates 指定時はキャッシュをスキップして毎回生成"""
        resp = client.post("/v1/fusion/cached", json={
            "question": "カスタム候補テスト",
            "candidates": [{"label": "x1"}, {"label": "x2"}],
        })
        assert resp.status_code == 200


class TestDashboardApi:
    def _setup_data(self, client):
        """ダッシュボード用データ準備"""
        client.post("/v1/analytics/dash-c1/record", json={
            "spend": 1000.0, "revenue": 4000.0,
            "impressions": 10000, "clicks": 200, "conversions": 20,
        })
        client.post("/v1/analytics/dash-c2/record", json={
            "spend": 500.0, "revenue": 1500.0,
            "impressions": 5000, "clicks": 80,
        })

    def test_summary_status(self, client):
        resp = client.get("/v1/dashboard/summary")
        assert resp.status_code == 200

    def test_summary_structure(self, client):
        self._setup_data(client)
        d = client.get("/v1/dashboard/summary").json()
        assert "campaigns" in d
        assert "alert_summary" in d
        assert "abtest_summary" in d

    def test_summary_campaigns_list(self, client):
        self._setup_data(client)
        d = client.get("/v1/dashboard/summary").json()
        ids = [c["campaign_id"] for c in d["campaigns"]]
        assert "dash-c1" in ids

    def test_summary_alert_has_critical_count(self, client):
        d = client.get("/v1/dashboard/summary").json()
        assert "critical_count" in d["alert_summary"]

    def test_campaigns_endpoint(self, client):
        self._setup_data(client)
        resp = client.get("/v1/dashboard/campaigns")
        assert resp.status_code == 200
        d = resp.json()
        assert "campaigns" in d
        assert "total" in d

    def test_campaigns_kpi_fields(self, client):
        self._setup_data(client)
        d = client.get("/v1/dashboard/campaigns").json()
        # dash-c1 が含まれる
        entry = next((c for c in d["campaigns"] if c["campaign_id"] == "dash-c1"), None)
        assert entry is not None
        for k in ("ctr", "roas", "spend", "revenue"):
            assert k in entry

    def test_critical_alerts_endpoint(self, client):
        resp = client.get("/v1/dashboard/alerts/critical")
        assert resp.status_code == 200
        d = resp.json()
        assert "critical_count" in d
        assert "alerts" in d

    def test_critical_alerts_only_critical(self, client):
        # Warning アラートを検知させる
        for _ in range(4):
            client.post("/v1/analytics/warn-camp/record", json={"clicks": 100})
        client.post("/v1/analytics/warn-camp/record", json={"clicks": 140})
        client.post("/v1/anomaly/warn-camp/detect", json={"metrics": ["clicks"]})
        d = client.get("/v1/dashboard/alerts/critical").json()
        # 全て critical であることを確認
        for a in d["alerts"]:
            assert a["severity"] == "critical"

"""
Sprint 66 — Fusion streaming + A/B予算連携 + 異常検知 テスト

対象:
  66A: open_mythos/skills/fusion.py — FusionEngine.run_stream
  66B: open_mythos/skills/campaign_orchestrator.py — CampaignOrchestrator
  66C: open_mythos/skills/anomaly_detector.py — AnomalyDetector
"""
from __future__ import annotations

import json
import pytest

from open_mythos.skills.fusion import (
    FusionConfig, CandidateSpec, FusionEngineFactory,
)
from open_mythos.skills.ab_test import ABTest, Variant, VariantStats
from open_mythos.skills.campaign_analytics import CampaignAnalyticsStore
from open_mythos.skills.campaign_orchestrator import (
    OrchestrationConfig, WinnerDecision, ReallocationPlan, CampaignOrchestrator,
)
from open_mythos.skills.anomaly_detector import (
    AlertSeverity, AnomalyType, Alert, DetectorConfig,
    AnomalyDetector, AlertStore, AnomalyReportEngine,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 66A: Fusion streaming
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFusionStreaming:
    def setup_method(self):
        self.engine = FusionEngineFactory.rule_based(
            config=FusionConfig(candidates=[
                CandidateSpec(label="c1"),
                CandidateSpec(label="c2"),
            ])
        )

    def test_run_stream_yields_events(self):
        events = list(self.engine.run_stream("質問"))
        assert len(events) > 0

    def test_stages_present(self):
        events = list(self.engine.run_stream("質問"))
        stages = [e["stage"] for e in events]
        assert "candidates" in stages
        assert "analysis" in stages
        assert "done" in stages

    def test_candidates_stage_first(self):
        events = list(self.engine.run_stream("質問"))
        assert events[0]["stage"] == "candidates"

    def test_done_stage_last(self):
        events = list(self.engine.run_stream("質問"))
        assert events[-1]["stage"] == "done"

    def test_done_contains_final_answer(self):
        events = list(self.engine.run_stream("質問"))
        done = events[-1]
        assert "final_answer" in done["data"]

    def test_analysis_stage_has_ranking(self):
        events = list(self.engine.run_stream("質問"))
        analysis = [e for e in events if e["stage"] == "analysis"][0]
        assert "ranking" in analysis["data"]

    def test_delta_chunks_reconstruct_answer(self):
        events = list(self.engine.run_stream("質問"))
        deltas = [e["data"]["text"] for e in events if e["stage"] == "delta"]
        done = events[-1]["data"]["final_answer"]
        assert "".join(deltas) == done

    def test_stream_with_mock_llm(self):
        responses = ["候補1", "候補2", json.dumps({
            "analyses": [{"label": "c1", "score": 0.9}, {"label": "c2", "score": 0.5}],
            "ranking": ["c1", "c2"], "synthesis_guidance": "g",
        }), "最終合成回答テキスト"]
        engine = FusionEngineFactory.from_mock(responses, config=FusionConfig(
            candidates=[CandidateSpec(label="c1"), CandidateSpec(label="c2")]
        ))
        events = list(engine.run_stream("質問"))
        done = events[-1]["data"]
        assert done["final_answer"] == "最終合成回答テキスト"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 66C: AnomalyDetector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAlertEnums:
    def test_severity(self):
        assert AlertSeverity.INFO.value     == "info"
        assert AlertSeverity.WARNING.value  == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_anomaly_type(self):
        assert AnomalyType.SPIKE.value == "spike"
        assert AnomalyType.DROP.value  == "drop"
        assert AnomalyType.STALE.value == "stale"


class TestDetectorConfig:
    def test_defaults(self):
        c = DetectorConfig()
        assert c.z_warning == 2.0
        assert c.z_critical == 3.0
        assert c.min_samples == 3


class TestAnomalyDetector:
    def setup_method(self):
        self.detector = AnomalyDetector()

    def _metrics(self, clicks_series):
        store = CampaignAnalyticsStore()
        m = store.get_or_create("c1")
        for c in clicks_series:
            m.record(clicks=c)
        return m

    def test_no_alert_insufficient_samples(self):
        m = self._metrics([10, 20])  # 2 点 < min_samples
        assert self.detector.detect(m, "clicks") == []

    def test_no_alert_stable(self):
        m = self._metrics([100, 102, 98, 101, 99])
        alerts = self.detector.detect(m, "clicks")
        assert alerts == []

    def test_spike_detected(self):
        m = self._metrics([100, 100, 100, 100, 500])  # 急増
        alerts = self.detector.detect(m, "clicks")
        assert len(alerts) == 1
        assert alerts[0].anomaly_type == AnomalyType.SPIKE

    def test_drop_detected(self):
        m = self._metrics([100, 100, 100, 100, 5])  # 急減
        alerts = self.detector.detect(m, "clicks")
        assert len(alerts) == 1
        assert alerts[0].anomaly_type == AnomalyType.DROP

    def test_critical_severity(self):
        m = self._metrics([100, 100, 100, 100, 1000])  # 極端なスパイク
        alerts = self.detector.detect(m, "clicks")
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_stale_detected(self):
        m = self._metrics([50, 50, 50, 50])  # 全部同値
        alerts = self.detector.detect(m, "clicks")
        assert len(alerts) == 1
        assert alerts[0].anomaly_type == AnomalyType.STALE

    def test_alert_message_not_empty(self):
        m = self._metrics([100, 100, 100, 100, 500])
        alerts = self.detector.detect(m, "clicks")
        assert len(alerts[0].message) > 0

    def test_alert_to_dict(self):
        m = self._metrics([100, 100, 100, 100, 500])
        alerts = self.detector.detect(m, "clicks")
        d = alerts[0].to_dict()
        assert d["anomaly_type"] == "spike"
        assert "z_score" in d

    def test_detect_multi(self):
        store = CampaignAnalyticsStore()
        m = store.get_or_create("c1")
        for _ in range(4):
            m.record(clicks=100, impressions=1000)
        m.record(clicks=500, impressions=1000)  # clicks スパイク
        alerts = self.detector.detect_multi(m, ["clicks", "impressions"])
        # clicks にアラート、impressions は安定
        assert any(a.metric == "clicks" for a in alerts)

    def test_change_pct_recorded(self):
        m = self._metrics([100, 100, 100, 100, 200])
        alerts = self.detector.detect(m, "clicks")
        if alerts:
            assert alerts[0].change_pct > 0


class TestAlertStore:
    def setup_method(self):
        self.store = AlertStore()

    def _alert(self, severity=AlertSeverity.WARNING, campaign="c1") -> Alert:
        return Alert(
            id="a1", campaign_id=campaign, metric="clicks",
            anomaly_type=AnomalyType.SPIKE, severity=severity,
            current=500, baseline=100, z_score=4.0, change_pct=4.0,
            message="test",
        )

    def test_add(self):
        self.store.add(self._alert())
        assert self.store.count() == 1

    def test_add_many(self):
        self.store.add_many([self._alert(), self._alert()])
        assert self.store.count() == 2

    def test_list_by_severity(self):
        self.store.add(self._alert(AlertSeverity.CRITICAL))
        self.store.add(self._alert(AlertSeverity.WARNING))
        assert len(self.store.list_by_severity(AlertSeverity.CRITICAL)) == 1

    def test_list_by_campaign(self):
        self.store.add(self._alert(campaign="c1"))
        self.store.add(self._alert(campaign="c2"))
        assert len(self.store.list_by_campaign("c1")) == 1

    def test_critical_count(self):
        self.store.add(self._alert(AlertSeverity.CRITICAL))
        self.store.add(self._alert(AlertSeverity.CRITICAL))
        self.store.add(self._alert(AlertSeverity.WARNING))
        assert self.store.critical_count() == 2

    def test_clear(self):
        self.store.add(self._alert())
        self.store.clear()
        assert self.store.count() == 0


class TestAnomalyReportEngine:
    def setup_method(self):
        self.store = AlertStore()
        self.engine = AnomalyReportEngine(self.store)

    def _alert(self, severity=AlertSeverity.CRITICAL) -> Alert:
        return Alert(
            id="a1", campaign_id="c1", metric="clicks",
            anomaly_type=AnomalyType.SPIKE, severity=severity,
            current=500, baseline=100, z_score=4.0, change_pct=4.0,
            message="急増",
        )

    def test_summary_empty(self):
        d = self.engine.summary_json()
        assert d["total_alerts"] == 0

    def test_summary_counts(self):
        self.store.add(self._alert())
        d = self.engine.summary_json()
        assert d["total_alerts"] == 1
        assert d["critical_count"] == 1

    def test_markdown_empty(self):
        md = self.engine.markdown()
        assert "アラートはありません" in md

    def test_markdown_with_alerts(self):
        self.store.add(self._alert())
        md = self.engine.markdown()
        assert "急増" in md
        assert "clicks" in md


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 66B: CampaignOrchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_ab_test(strong_winner=True) -> ABTest:
    t = ABTest(id="t1", name="A/Bテスト")
    v1 = Variant(id="v1", name="案A", content="A")
    v2 = Variant(id="v2", name="案B", content="B")
    if strong_winner:
        v1.stats = VariantStats(impressions=10000, clicks=1000)  # CTR 10%
        v2.stats = VariantStats(impressions=10000, clicks=500)   # CTR 5%
    else:
        v1.stats = VariantStats(impressions=100, clicks=11)
        v2.stats = VariantStats(impressions=100, clicks=10)
    t.add_variant(v1)
    t.add_variant(v2)
    return t


class TestOrchestrationConfig:
    def test_defaults(self):
        c = OrchestrationConfig()
        assert c.winner_metric == "ctr"
        assert c.require_significance is True
        assert c.winner_bonus == 1.5


class TestCampaignOrchestratorWinner:
    def setup_method(self):
        self.orch = CampaignOrchestrator()

    def test_decide_winner_returns_decision(self):
        decision = self.orch.decide_winner(_make_ab_test())
        assert isinstance(decision, WinnerDecision)

    def test_strong_winner_confident(self):
        decision = self.orch.decide_winner(_make_ab_test(strong_winner=True))
        assert decision.winner_label == "案A"
        assert decision.is_significant is True
        assert decision.confident is True

    def test_weak_winner_not_confident(self):
        decision = self.orch.decide_winner(_make_ab_test(strong_winner=False))
        # 有意差なし → require_significance=True なので confident=False
        assert decision.confident is False

    def test_no_data_no_winner(self):
        t = ABTest(id="t1", name="empty")
        t.add_variant(Variant(id="v1", name="A", content="a"))
        t.add_variant(Variant(id="v2", name="B", content="b"))
        decision = self.orch.decide_winner(t)
        assert decision.winner_variant_id is None
        assert decision.confident is False

    def test_no_significance_required(self):
        orch = CampaignOrchestrator(
            config=OrchestrationConfig(require_significance=False)
        )
        decision = orch.decide_winner(_make_ab_test(strong_winner=False))
        # 有意差不要 → confident=True
        assert decision.confident is True

    def test_decision_to_dict(self):
        decision = self.orch.decide_winner(_make_ab_test())
        d = decision.to_dict()
        assert "winner_label" in d
        assert "is_significant" in d


class TestCampaignOrchestratorReallocate:
    def setup_method(self):
        self.store = CampaignAnalyticsStore()
        # c1: ROAS 5, c2: ROAS 2
        self.store.record("c1", spend=1000.0, revenue=5000.0, impressions=10000, clicks=1000)
        self.store.record("c2", spend=1000.0, revenue=2000.0, impressions=10000, clicks=500)
        self.orch = CampaignOrchestrator(analytics_store=self.store)

    def test_reallocate_returns_plan(self):
        plan = self.orch.reallocate(
            _make_ab_test(), 10000, ["c1", "c2"], winner_campaign_id="c1"
        )
        assert isinstance(plan, ReallocationPlan)

    def test_winner_bonus_applied(self):
        plan = self.orch.reallocate(
            _make_ab_test(strong_winner=True), 10000, ["c1", "c2"],
            winner_campaign_id="c1",
        )
        assert plan.applied_bonus is True
        # 勝者 c1 に最低保証
        c1_alloc = plan.optimization.get("c1")
        assert c1_alloc.amount >= 7500 * 0.99  # base 5000 * 1.5 = 7500

    def test_no_bonus_when_not_confident(self):
        plan = self.orch.reallocate(
            _make_ab_test(strong_winner=False), 10000, ["c1", "c2"],
            winner_campaign_id="c1",
        )
        assert plan.applied_bonus is False

    def test_no_bonus_when_winner_not_in_campaigns(self):
        plan = self.orch.reallocate(
            _make_ab_test(strong_winner=True), 10000, ["c1", "c2"],
            winner_campaign_id="c99",  # 対象外
        )
        assert plan.applied_bonus is False

    def test_allocation_shares_sum_to_one(self):
        # min 制約適用時、合計は予算を超え得るが share は常に 1 に正規化される
        plan = self.orch.reallocate(
            _make_ab_test(), 10000, ["c1", "c2"], winner_campaign_id="c1"
        )
        total_share = sum(a.share for a in plan.optimization.allocations)
        assert abs(total_share - 1.0) < 1e-6

    def test_allocations_sum_to_budget_no_bonus(self):
        # bonus 非適用（弱い勝者）なら合計は予算と一致
        plan = self.orch.reallocate(
            _make_ab_test(strong_winner=False), 10000, ["c1", "c2"],
            winner_campaign_id="c1",
        )
        assert plan.applied_bonus is False
        assert abs(plan.optimization.allocated_total - 10000) < 1.0

    def test_run_workflow_dict(self):
        d = self.orch.run_workflow(
            _make_ab_test(), 10000, ["c1", "c2"], winner_campaign_id="c1"
        )
        assert "winner_decision" in d
        assert "optimization" in d

    def test_plan_to_dict(self):
        plan = self.orch.reallocate(
            _make_ab_test(), 10000, ["c1", "c2"], winner_campaign_id="c1"
        )
        d = plan.to_dict()
        assert "winner_decision" in d
        assert d["winner_campaign_id"] == "c1"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serve.api import app
    return TestClient(app)


class TestFusionStreamApi:
    def test_stream_returns_sse(self, client):
        resp = client.post("/v1/fusion/stream", json={"question": "テスト"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_contains_done(self, client):
        resp = client.post("/v1/fusion/stream", json={"question": "テスト"})
        assert "[DONE]" in resp.text

    def test_stream_contains_stages(self, client):
        resp = client.post("/v1/fusion/stream", json={"question": "テスト"})
        assert "event: candidates" in resp.text
        assert "event: done" in resp.text


class TestOrchestratorApi:
    def _setup_abtest(self, client):
        created = client.post("/v1/abtest/", json={
            "name": "orch test",
            "variants": [
                {"name": "案A", "content": "A"},
                {"name": "案B", "content": "B"},
            ],
        }).json()
        tid = created["id"]
        # 強い勝者を作る
        v1, v2 = created["variants"][0]["id"], created["variants"][1]["id"]
        client.post(f"/v1/abtest/{tid}/variant/{v1}/record",
                    json={"impressions": 10000, "clicks": 1000})
        client.post(f"/v1/abtest/{tid}/variant/{v2}/record",
                    json={"impressions": 10000, "clicks": 500})
        return tid

    def test_decide_winner(self, client):
        tid = self._setup_abtest(client)
        resp = client.post(f"/v1/orchestrator/decide-winner/{tid}")
        assert resp.status_code == 200
        assert resp.json()["winner_label"] == "案A"

    def test_decide_winner_not_found(self, client):
        assert client.post("/v1/orchestrator/decide-winner/nope").status_code == 404

    def test_reallocate(self, client):
        tid = self._setup_abtest(client)
        client.post("/v1/analytics/orch-c1/record", json={
            "spend": 1000.0, "revenue": 5000.0,
        })
        client.post("/v1/analytics/orch-c2/record", json={
            "spend": 1000.0, "revenue": 2000.0,
        })
        resp = client.post("/v1/orchestrator/reallocate", json={
            "test_id": tid,
            "total_budget": 10000,
            "campaign_ids": ["orch-c1", "orch-c2"],
            "winner_campaign_id": "orch-c1",
        })
        assert resp.status_code == 200
        assert "optimization" in resp.json()

    def test_reallocate_test_not_found(self, client):
        resp = client.post("/v1/orchestrator/reallocate", json={
            "test_id": "nope", "total_budget": 1000, "campaign_ids": ["x"],
        })
        assert resp.status_code == 404


class TestAnomalyApi:
    def test_detect(self, client):
        # スパイクを作る
        for _ in range(4):
            client.post("/v1/analytics/anom-1/record", json={"clicks": 100})
        client.post("/v1/analytics/anom-1/record", json={"clicks": 500})
        resp = client.post("/v1/anomaly/anom-1/detect", json={"metrics": ["clicks"]})
        assert resp.status_code == 200
        assert resp.json()["detected"] >= 1

    def test_detect_not_found(self, client):
        resp = client.post("/v1/anomaly/nope-xyz/detect", json={})
        assert resp.status_code == 404

    def test_alerts_summary(self, client):
        resp = client.get("/v1/anomaly/alerts")
        assert resp.status_code == 200
        assert "total_alerts" in resp.json()

    def test_alerts_report_md(self, client):
        resp = client.get("/v1/anomaly/alerts/report/md")
        assert resp.status_code == 200
        assert "異常検知" in resp.text

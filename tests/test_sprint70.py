"""
Sprint 70 — 予測アラート統合 + レポート配信 Webhook + NLQ エージェント テスト

対象:
  70A: open_mythos/skills/forecast_alert.py
  70B: open_mythos/skills/report_dispatcher.py
  70C: open_mythos/skills/nlq_agent.py
  serve/api.py
"""
from __future__ import annotations
import pytest

from open_mythos.skills.forecast_alert import (
    AlertThreshold, ForecastAlertRule, ForecastAlertCheck,
    ForecastAlertRuleStore, ForecastAlertEngine,
)
from open_mythos.skills.report_dispatcher import (
    WebhookTarget, DispatchPayload, DispatchResult,
    WebhookStore, ReportDispatcher,
)
from open_mythos.skills.nlq_agent import (
    NLQIntent, NLQQuery, NLQResult, NLQParser, NLQExecutor,
)
from open_mythos.skills.time_series import MockForecaster, ForecastStore, CampaignForecaster
from open_mythos.skills.campaign_analytics import CampaignAnalyticsStore
from open_mythos.skills.anomaly_detector import AlertStore


def _make_forecast_store(campaign_id="c1", metric="clicks"):
    analytics = CampaignAnalyticsStore()
    m = analytics.get_or_create(campaign_id)
    for i in range(10):
        m.record(clicks=100 + i * 10)
    store = ForecastStore()
    cf = CampaignForecaster(MockForecaster(), analytics)
    result = cf.forecast_metric(campaign_id, metric=metric, horizon=5)
    store.save(result)
    return store


def _make_rule_store():
    rs = ForecastAlertRuleStore()
    rs.add(ForecastAlertRule(id="r1", campaign_id="c1",
        threshold=AlertThreshold(metric="clicks", upper_limit=500.0), severity="warning"))
    rs.add(ForecastAlertRule(id="r2", campaign_id="c1",
        threshold=AlertThreshold(metric="clicks", upper_limit=50.0), severity="critical"))
    return rs


# ─── 70A: AlertThreshold ──────────────────────────────────────────

class TestAlertThreshold:
    def test_upper_only(self):
        t = AlertThreshold(metric="clicks", upper_limit=500.0)
        assert t.metric == "clicks" and t.upper_limit == 500.0 and t.lower_limit is None

    def test_lower_only(self):
        t = AlertThreshold(metric="spend", lower_limit=10.0)
        assert t.lower_limit == 10.0 and t.upper_limit is None

    def test_both_limits(self):
        t = AlertThreshold(metric="ctr", upper_limit=1.0, lower_limit=0.01)
        assert t.upper_limit == 1.0 and t.lower_limit == 0.01

    def test_to_dict_keys(self):
        d = AlertThreshold(metric="clicks", upper_limit=100.0).to_dict()
        assert "metric" in d and "upper_limit" in d and "lower_limit" in d


# ─── 70A: ForecastAlertRule ───────────────────────────────────────

class TestForecastAlertRule:
    def test_default_enabled(self):
        rule = ForecastAlertRule(id="r1", campaign_id="c1",
            threshold=AlertThreshold(metric="clicks", upper_limit=500.0), severity="warning")
        assert rule.enabled is True

    def test_disabled_rule(self):
        rule = ForecastAlertRule(id="r2", campaign_id="c2",
            threshold=AlertThreshold(metric="spend", lower_limit=0.0), severity="critical", enabled=False)
        assert rule.enabled is False

    def test_to_dict_keys(self):
        d = ForecastAlertRule(id="r3", campaign_id="c3",
            threshold=AlertThreshold(metric="clicks", upper_limit=100.0), severity="warning").to_dict()
        for k in ("id", "campaign_id", "threshold", "severity", "enabled"):
            assert k in d

    def test_severity_values(self):
        for sev in ("info", "warning", "critical"):
            rule = ForecastAlertRule(id="rx", campaign_id="c1",
                threshold=AlertThreshold(metric="clicks", upper_limit=1.0), severity=sev)
            assert rule.severity == sev


# ─── 70A: ForecastAlertCheck ──────────────────────────────────────

class TestForecastAlertCheck:
    def _check(self, triggered=True):
        return ForecastAlertCheck(rule_id="r1", campaign_id="c1", metric="clicks",
            triggered=triggered, severity="warning" if triggered else None,
            predicted_values=[100.0, 120.0, 140.0],
            violations=[{"step": 1, "value": 600.0, "limit": 500.0}] if triggered else [])

    def test_triggered(self):
        c = self._check(True)
        assert c.triggered is True and c.severity == "warning"

    def test_not_triggered(self):
        c = self._check(False)
        assert c.triggered is False and c.severity is None and len(c.violations) == 0

    def test_to_dict_keys(self):
        d = self._check().to_dict()
        for k in ("rule_id", "campaign_id", "metric", "triggered", "severity",
                  "predicted_values", "violations", "checked_at"):
            assert k in d

    def test_violations_list(self):
        c = self._check()
        assert isinstance(c.violations, list) and c.violations[0]["step"] == 1


# ─── 70A: ForecastAlertRuleStore ──────────────────────────────────

class TestForecastAlertRuleStore:
    def test_add_and_get(self):
        rs = ForecastAlertRuleStore()
        rule = ForecastAlertRule(id="r1", campaign_id="c1",
            threshold=AlertThreshold(metric="clicks", upper_limit=100.0), severity="warning")
        rs.add(rule)
        assert rs.get("r1") is rule

    def test_list_empty(self):
        assert ForecastAlertRuleStore().list() == []

    def test_list_by_campaign(self):
        assert len(_make_rule_store().list_by_campaign("c1")) == 2

    def test_list_by_campaign_empty(self):
        assert _make_rule_store().list_by_campaign("no-such") == []

    def test_delete(self):
        rs = _make_rule_store()
        rs.delete("r1")
        assert rs.get("r1") is None and len(rs.list_by_campaign("c1")) == 1

    def test_delete_nonexistent(self):
        ForecastAlertRuleStore().delete("nonexistent")

    def test_update_enabled(self):
        rs = _make_rule_store()
        rs.set_enabled("r1", False)
        assert rs.get("r1").enabled is False


# ─── 70A: ForecastAlertEngine ─────────────────────────────────────

class TestForecastAlertEngine:
    def _engine(self):
        return ForecastAlertEngine(
            forecast_store=_make_forecast_store("c1", "clicks"),
            rule_store=_make_rule_store())

    def test_check_returns_check_object(self):
        assert isinstance(self._engine().check("r1"), ForecastAlertCheck)

    def test_check_rule_id_in_result(self):
        assert self._engine().check("r1").rule_id == "r1"

    def test_high_upper_no_trigger(self):
        fs = _make_forecast_store("c1", "clicks")
        rs = ForecastAlertRuleStore()
        rs.add(ForecastAlertRule(id="r_high", campaign_id="c1",
            threshold=AlertThreshold(metric="clicks", upper_limit=1e9), severity="warning"))
        result = ForecastAlertEngine(forecast_store=fs, rule_store=rs).check("r_high")
        assert result.triggered is False and result.violations == []

    def test_low_upper_triggers(self):
        fs = _make_forecast_store("c1", "clicks")
        rs = ForecastAlertRuleStore()
        rs.add(ForecastAlertRule(id="r_low", campaign_id="c1",
            threshold=AlertThreshold(metric="clicks", upper_limit=1.0), severity="critical"))
        result = ForecastAlertEngine(forecast_store=fs, rule_store=rs).check("r_low")
        assert result.triggered is True and len(result.violations) > 0

    def test_lower_limit_triggers_when_below(self):
        fs = _make_forecast_store("c1", "clicks")
        rs = ForecastAlertRuleStore()
        rs.add(ForecastAlertRule(id="r_lower", campaign_id="c1",
            threshold=AlertThreshold(metric="clicks", lower_limit=1e9), severity="warning"))
        result = ForecastAlertEngine(forecast_store=fs, rule_store=rs).check("r_lower")
        assert result.triggered is True

    def test_no_forecast_returns_no_trigger(self):
        assert ForecastAlertEngine(
            forecast_store=ForecastStore(), rule_store=_make_rule_store()).check("r1").triggered is False

    def test_check_all_returns_list(self):
        results = self._engine().check_all("c1")
        assert isinstance(results, list) and len(results) == 2

    def test_check_nonexistent_rule(self):
        assert self._engine().check("no-such-rule").triggered is False

    def test_disabled_rule_skipped(self):
        fs = _make_forecast_store("c1", "clicks")
        rs = ForecastAlertRuleStore()
        rs.add(ForecastAlertRule(id="r_off", campaign_id="c1",
            threshold=AlertThreshold(metric="clicks", upper_limit=1.0),
            severity="critical", enabled=False))
        assert ForecastAlertEngine(forecast_store=fs, rule_store=rs).check_all("c1") == []


# ─── 70B: WebhookTarget ───────────────────────────────────────────

class TestWebhookTarget:
    def test_basic(self):
        wh = WebhookTarget(id="w1", name="Slack Dev", url="https://hooks.slack.com/test", type="slack")
        assert wh.id == "w1" and wh.type == "slack" and wh.enabled is True

    def test_generic_type(self):
        assert WebhookTarget(id="w2", name="My hook", url="https://example.com/hook", type="generic").type == "generic"

    def test_disabled(self):
        assert WebhookTarget(id="w3", name="off", url="https://example.com", type="generic", enabled=False).enabled is False

    def test_to_dict_keys(self):
        d = WebhookTarget(id="w4", name="x", url="https://example.com", type="slack").to_dict()
        for k in ("id", "name", "url", "type", "enabled"):
            assert k in d


# ─── 70B: DispatchResult ──────────────────────────────────────────

class TestDispatchResult:
    def test_success(self):
        r = DispatchResult(webhook_id="w1", success=True, status_code=200)
        assert r.success is True and r.error is None

    def test_failure(self):
        r = DispatchResult(webhook_id="w1", success=False, error="Connection refused")
        assert r.success is False and r.status_code is None

    def test_to_dict_keys(self):
        d = DispatchResult(webhook_id="w1", success=True, status_code=200).to_dict()
        for k in ("webhook_id", "success", "status_code", "error", "dispatched_at"):
            assert k in d


# ─── 70B: WebhookStore ────────────────────────────────────────────

class TestWebhookStore:
    def test_add_and_get(self):
        ws = WebhookStore()
        wh = WebhookTarget(id="w1", name="test", url="https://example.com", type="generic")
        ws.add(wh)
        assert ws.get("w1") is wh

    def test_list_empty(self):
        assert WebhookStore().list() == []

    def test_list_returns_all(self):
        ws = WebhookStore()
        ws.add(WebhookTarget(id="w1", name="a", url="https://a.com", type="slack"))
        ws.add(WebhookTarget(id="w2", name="b", url="https://b.com", type="generic"))
        assert len(ws.list()) == 2

    def test_delete(self):
        ws = WebhookStore()
        ws.add(WebhookTarget(id="w1", name="x", url="https://x.com", type="generic"))
        ws.delete("w1")
        assert ws.get("w1") is None

    def test_delete_nonexistent(self):
        WebhookStore().delete("nonexistent")

    def test_list_enabled_only(self):
        ws = WebhookStore()
        ws.add(WebhookTarget(id="w1", name="on", url="https://on.com", type="generic", enabled=True))
        ws.add(WebhookTarget(id="w2", name="off", url="https://off.com", type="generic", enabled=False))
        enabled = ws.list_enabled()
        assert len(enabled) == 1 and enabled[0].id == "w1"


# ─── 70B: ReportDispatcher ────────────────────────────────────────

class TestReportDispatcher:
    def _dispatcher(self):
        ws = WebhookStore()
        ws.add(WebhookTarget(id="w1", name="Test", url="https://example.com/hook", type="generic"))
        analytics = CampaignAnalyticsStore()
        m = analytics.get_or_create("c1")
        for _ in range(3):
            m.record(clicks=100, impressions=1000, spend=10.0, revenue=50.0)
        return ReportDispatcher(webhook_store=ws, analytics_store=analytics)

    def test_build_payload_campaign_summary(self):
        payload = self._dispatcher().build_payload("w1", "campaign_summary", "c1")
        assert isinstance(payload, DispatchPayload)
        assert payload.webhook_id == "w1" and payload.report_type == "campaign_summary"
        assert len(payload.content) > 0

    def test_build_payload_generic(self):
        assert self._dispatcher().build_payload("w1", "generic", campaign_id=None).content

    def test_dispatch_mock_returns_result(self):
        result = self._dispatcher().dispatch_mock("w1", "campaign_summary", "c1")
        assert isinstance(result, DispatchResult) and result.webhook_id == "w1"

    def test_dispatch_mock_success(self):
        assert self._dispatcher().dispatch_mock("w1", "campaign_summary", "c1").success is True

    def test_dispatch_mock_nonexistent(self):
        result = self._dispatcher().dispatch_mock("no-such", "generic", None)
        assert result.success is False and result.error is not None

    def test_dispatch_all_mock(self):
        ws = WebhookStore()
        ws.add(WebhookTarget(id="w1", name="A", url="https://a.com", type="generic"))
        ws.add(WebhookTarget(id="w2", name="B", url="https://b.com", type="slack"))
        ws.add(WebhookTarget(id="w3", name="C", url="https://c.com", type="generic", enabled=False))
        d = ReportDispatcher(webhook_store=ws, analytics_store=CampaignAnalyticsStore())
        assert len(d.dispatch_all_mock("generic", None)) == 2

    def test_dispatch_history(self):
        d = self._dispatcher()
        d.dispatch_mock("w1", "generic", None)
        d.dispatch_mock("w1", "generic", None)
        assert len(d.history()) >= 2


# ─── 70C: NLQIntent ───────────────────────────────────────────────

class TestNLQIntent:
    def test_enum_values(self):
        assert NLQIntent.CAMPAIGN_KPI.value == "campaign_kpi"
        assert NLQIntent.FORECAST.value == "forecast"
        assert NLQIntent.ANOMALY.value == "anomaly"
        assert NLQIntent.BUDGET.value == "budget"
        assert NLQIntent.UNKNOWN.value == "unknown"


# ─── 70C: NLQQuery ────────────────────────────────────────────────

class TestNLQQuery:
    def test_basic(self):
        q = NLQQuery(raw="先週のCTRは？", intent=NLQIntent.CAMPAIGN_KPI, metric="ctr")
        assert q.raw == "先週のCTRは？" and q.intent == NLQIntent.CAMPAIGN_KPI and q.metric == "ctr"

    def test_defaults(self):
        q = NLQQuery(raw="test", intent=NLQIntent.UNKNOWN)
        assert q.campaign_id is None and q.metric is None and q.params == {}

    def test_to_dict_keys(self):
        d = NLQQuery(raw="test", intent=NLQIntent.FORECAST, metric="clicks").to_dict()
        for k in ("raw", "intent", "campaign_id", "metric", "params"):
            assert k in d


# ─── 70C: NLQParser ───────────────────────────────────────────────

class TestNLQParser:
    def setup_method(self):
        self.p = NLQParser()

    def test_kpi_ctr(self):
        q = self.p.parse("先週のCTRを教えて")
        assert q.intent == NLQIntent.CAMPAIGN_KPI and q.metric == "ctr"

    def test_kpi_clicks(self):
        q = self.p.parse("クリック数を確認したい")
        assert q.intent == NLQIntent.CAMPAIGN_KPI and q.metric == "clicks"

    def test_kpi_spend(self):
        assert self.p.parse("今日の消費予算は？").intent == NLQIntent.CAMPAIGN_KPI

    def test_forecast_ja(self):
        assert self.p.parse("来週のクリック数を予測して").intent == NLQIntent.FORECAST

    def test_forecast_en(self):
        assert self.p.parse("forecast clicks for next week").intent == NLQIntent.FORECAST

    def test_anomaly_ja(self):
        assert self.p.parse("異常アラートを確認したい").intent == NLQIntent.ANOMALY

    def test_anomaly_en(self):
        assert self.p.parse("show anomaly alerts for campaign c1").intent == NLQIntent.ANOMALY

    def test_budget_ja(self):
        assert self.p.parse("予算の最適化をして").intent == NLQIntent.BUDGET

    def test_budget_en(self):
        assert self.p.parse("optimize budget allocation").intent == NLQIntent.BUDGET

    def test_unknown(self):
        assert self.p.parse("xyzzy nonsense query foobar").intent == NLQIntent.UNKNOWN

    def test_campaign_id_extracted(self):
        assert self.p.parse("キャンペーン campaign_abc のCTRは？").campaign_id == "campaign_abc"

    def test_returns_nlq_query(self):
        q = self.p.parse("test query")
        assert isinstance(q, NLQQuery) and q.raw == "test query"

    def test_metric_clicks_en(self):
        assert self.p.parse("show me clicks data").metric == "clicks"

    def test_metric_impressions(self):
        assert self.p.parse("インプレッション数は？").metric == "impressions"

    def test_metric_revenue(self):
        assert self.p.parse("売上 revenue を見せて").metric == "revenue"


# ─── 70C: NLQResult ───────────────────────────────────────────────

class TestNLQResult:
    def test_success(self):
        q = NLQQuery(raw="test", intent=NLQIntent.CAMPAIGN_KPI)
        r = NLQResult(query=q, intent=NLQIntent.CAMPAIGN_KPI, data={"kpi": 0.5}, message="OK", success=True)
        assert r.success is True and r.data["kpi"] == 0.5

    def test_failure(self):
        q = NLQQuery(raw="test", intent=NLQIntent.UNKNOWN)
        r = NLQResult(query=q, intent=NLQIntent.UNKNOWN, data=None, message="不明", success=False)
        assert r.success is False and r.data is None

    def test_to_dict_keys(self):
        q = NLQQuery(raw="test", intent=NLQIntent.FORECAST)
        d = NLQResult(query=q, intent=NLQIntent.FORECAST, data={}, message="done", success=True).to_dict()
        for k in ("query", "intent", "data", "message", "success"):
            assert k in d


# ─── 70C: NLQExecutor ─────────────────────────────────────────────

class TestNLQExecutor:
    def _executor(self):
        analytics = CampaignAnalyticsStore()
        m = analytics.get_or_create("c1")
        for i in range(5):
            m.record(clicks=100 + i * 10, impressions=1000, spend=10.0, revenue=50.0)
        return NLQExecutor(
            analytics_store=analytics,
            forecast_store=_make_forecast_store("c1", "clicks"),
            alert_store=AlertStore())

    def test_unknown_returns_failure(self):
        result = self._executor().execute(NLQQuery(raw="xyzzy", intent=NLQIntent.UNKNOWN))
        assert isinstance(result, NLQResult) and result.success is False

    def test_kpi_no_campaign(self):
        result = self._executor().execute(NLQQuery(raw="全体のCTRは？", intent=NLQIntent.CAMPAIGN_KPI))
        assert isinstance(result, NLQResult) and result.success is True

    def test_kpi_with_id(self):
        result = self._executor().execute(
            NLQQuery(raw="c1のCTR", intent=NLQIntent.CAMPAIGN_KPI, campaign_id="c1"))
        assert result.success is True and result.data is not None

    def test_forecast_with_id(self):
        result = self._executor().execute(
            NLQQuery(raw="来週の予測", intent=NLQIntent.FORECAST, campaign_id="c1", metric="clicks"))
        assert isinstance(result, NLQResult) and result.success is True

    def test_forecast_no_data(self):
        result = self._executor().execute(
            NLQQuery(raw="来週", intent=NLQIntent.FORECAST, campaign_id="no-data", metric="clicks"))
        assert isinstance(result, NLQResult) and result.message

    def test_anomaly(self):
        result = self._executor().execute(NLQQuery(raw="アラート", intent=NLQIntent.ANOMALY))
        assert result.success is True

    def test_budget(self):
        result = self._executor().execute(NLQQuery(raw="予算最適化", intent=NLQIntent.BUDGET))
        assert isinstance(result, NLQResult)

    def test_returns_nlq_result(self):
        result = self._executor().execute(NLQQuery(raw="test", intent=NLQIntent.CAMPAIGN_KPI))
        assert isinstance(result, NLQResult)


# ─── API fixture ──────────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serve.api import app
    return TestClient(app)


# ─── API: 70A ForecastAlert ───────────────────────────────────────

class TestForecastAlertRulesApi:
    def test_list_rules_empty(self, client):
        resp = client.get("/v1/forecast/alert/rules")
        assert resp.status_code == 200 and "rules" in resp.json()

    def test_add_rule(self, client):
        resp = client.post("/v1/forecast/alert/rules", json={
            "campaign_id": "api-c1", "metric": "clicks",
            "upper_limit": 1000.0, "severity": "warning"})
        assert resp.status_code == 200 and "rule_id" in resp.json()

    def test_delete_rule(self, client):
        r = client.post("/v1/forecast/alert/rules", json={
            "campaign_id": "del-c1", "metric": "spend",
            "upper_limit": 500.0, "severity": "info"})
        assert client.delete(f"/v1/forecast/alert/rules/{r.json()['rule_id']}").status_code == 200


class TestForecastAlertCheckApi:
    def _setup(self, client, campaign_id="ac1"):
        for _ in range(5):
            client.post(f"/v1/analytics/{campaign_id}/record",
                json={"clicks": 100, "impressions": 1000, "spend": 10.0, "revenue": 50.0})
        client.post(f"/v1/forecast/{campaign_id}", json={"metric": "clicks", "horizon": 5})
        r = client.post("/v1/forecast/alert/rules", json={
            "campaign_id": campaign_id, "metric": "clicks",
            "upper_limit": 1e9, "severity": "warning"})
        return r.json()["rule_id"]

    def test_check_campaign(self, client):
        self._setup(client, "ac2")
        resp = client.get("/v1/forecast/alert/check/ac2")
        assert resp.status_code == 200 and "checks" in resp.json()

    def test_check_no_rules(self, client):
        resp = client.get("/v1/forecast/alert/check/no-rules-campaign")
        assert resp.status_code == 200 and resp.json()["checks"] == []


# ─── API: 70B ReportDispatcher ────────────────────────────────────

class TestWebhooksApi:
    def test_list_empty(self, client):
        resp = client.get("/v1/report/webhooks")
        assert resp.status_code == 200 and "webhooks" in resp.json()

    def test_add_webhook(self, client):
        resp = client.post("/v1/report/webhooks", json={
            "name": "Test Hook", "url": "https://example.com/hook", "type": "generic"})
        assert resp.status_code == 200 and "webhook_id" in resp.json()

    def test_delete_webhook(self, client):
        r = client.post("/v1/report/webhooks", json={
            "name": "Del Hook", "url": "https://del.com/hook", "type": "generic"})
        assert client.delete(f"/v1/report/webhooks/{r.json()['webhook_id']}").status_code == 200


class TestReportDispatchApi:
    def _add_webhook(self, client):
        r = client.post("/v1/report/webhooks", json={
            "name": "Dispatch Test", "url": "https://example.com/dispatch", "type": "generic"})
        return r.json()["webhook_id"]

    def test_dispatch_single(self, client):
        wid = self._add_webhook(client)
        resp = client.post("/v1/report/dispatch",
            json={"webhook_id": wid, "report_type": "generic"})
        assert resp.status_code == 200 and "result" in resp.json()

    def test_dispatch_all(self, client):
        self._add_webhook(client)
        resp = client.post("/v1/report/dispatch/all", json={"report_type": "generic"})
        assert resp.status_code == 200 and "results" in resp.json()

    def test_dispatch_history(self, client):
        wid = self._add_webhook(client)
        client.post("/v1/report/dispatch", json={"webhook_id": wid, "report_type": "generic"})
        resp = client.get("/v1/report/dispatch/history")
        assert resp.status_code == 200 and "history" in resp.json()


# ─── API: 70C NLQ ─────────────────────────────────────────────────

class TestNLQApi:
    def test_query_kpi(self, client):
        resp = client.post("/v1/nlq/query", json={"text": "先週のCTRを教えて"})
        assert resp.status_code == 200
        data = resp.json()
        assert "intent" in data and "result" in data

    def test_query_forecast(self, client):
        resp = client.post("/v1/nlq/query", json={"text": "来週のclicks予測"})
        assert resp.status_code == 200 and resp.json()["intent"] == "forecast"

    def test_query_anomaly(self, client):
        resp = client.post("/v1/nlq/query", json={"text": "異常アラートを見せて"})
        assert resp.status_code == 200 and resp.json()["intent"] == "anomaly"

    def test_query_unknown(self, client):
        resp = client.post("/v1/nlq/query", json={"text": "xyzzy foobar nonsense"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "unknown" and data["result"]["success"] is False

    def test_query_empty_text(self, client):
        assert client.post("/v1/nlq/query", json={"text": ""}).status_code in (200, 422)

    def test_query_response_shape(self, client):
        resp = client.post("/v1/nlq/query", json={"text": "クリック数は？"})
        assert resp.status_code == 200
        data = resp.json()
        assert "intent" in data and "query" in data and "result" in data

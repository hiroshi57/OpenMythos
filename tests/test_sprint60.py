"""
Sprint 60 — 広告キャンペーン管理 テスト (80 tests)

対象:
  open_mythos/skills/campaign_manager.py:
    CampaignStatus / AdChannel / AdObjective
    AdCopy / CampaignBudget / Campaign
    CampaignStore
    CopyGenerator
    CampaignEvaluator / EvalResult
    CampaignWorkflow / WorkflowResult
    CampaignReportEngine
  serve/api.py:
    POST /v1/campaign/
    GET  /v1/campaign/
    GET  /v1/campaign/{id}
    DELETE /v1/campaign/{id}
    POST /v1/campaign/{id}/activate
    POST /v1/campaign/{id}/pause
    POST /v1/campaign/{id}/complete
    GET  /v1/campaign/{id}/report/md
    POST /v1/campaign/workflow
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from open_mythos.skills.campaign_manager import (
    CampaignStatus, AdChannel, AdObjective,
    AdCopy, CampaignBudget, Campaign,
    CampaignStore,
    CopyGenerator,
    CampaignEvaluator, EvalResult,
    CampaignWorkflow, WorkflowResult,
    CampaignReportEngine,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignStatus:
    def test_values(self):
        assert CampaignStatus.DRAFT.value      == "draft"
        assert CampaignStatus.ACTIVE.value     == "active"
        assert CampaignStatus.PAUSED.value     == "paused"
        assert CampaignStatus.COMPLETED.value  == "completed"
        assert CampaignStatus.ARCHIVED.value   == "archived"

    def test_from_value(self):
        assert CampaignStatus("active") == CampaignStatus.ACTIVE


class TestAdChannel:
    def test_values(self):
        assert AdChannel.SEARCH.value  == "search"
        assert AdChannel.SOCIAL.value  == "social"
        assert AdChannel.DISPLAY.value == "display"
        assert AdChannel.EMAIL.value   == "email"
        assert AdChannel.VIDEO.value   == "video"


class TestAdObjective:
    def test_values(self):
        assert AdObjective.AWARENESS.value     == "awareness"
        assert AdObjective.CONSIDERATION.value == "consideration"
        assert AdObjective.CONVERSION.value    == "conversion"
        assert AdObjective.RETENTION.value     == "retention"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AdCopy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_copy(**kwargs) -> AdCopy:
    defaults = dict(
        id="copy-1",
        headline="テスト見出し",
        body="テスト本文です。品質スコアを確認します。",
        cta="今すぐチェック",
        channel=AdChannel.SEARCH,
        objective=AdObjective.AWARENESS,
    )
    defaults.update(kwargs)
    return AdCopy(**defaults)


class TestAdCopy:
    def test_to_dict_keys(self):
        c = make_copy()
        d = c.to_dict()
        assert set(d.keys()) >= {"id", "headline", "body", "cta", "channel", "objective", "score"}

    def test_channel_serialized_as_value(self):
        c = make_copy(channel=AdChannel.SOCIAL)
        assert c.to_dict()["channel"] == "social"

    def test_default_score_zero(self):
        c = make_copy()
        assert c.score == 0.0

    def test_tags_default_empty(self):
        c = make_copy()
        assert c.tags == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignBudget
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignBudget:
    def test_remaining(self):
        b = CampaignBudget(total=100_000, daily=5_000, spent=30_000)
        assert b.remaining == 70_000

    def test_remaining_no_negative(self):
        b = CampaignBudget(total=100_000, daily=5_000, spent=200_000)
        assert b.remaining == 0.0

    def test_to_dict(self):
        b = CampaignBudget(total=50_000, daily=2_500)
        d = b.to_dict()
        assert d["total"]    == 50_000
        assert d["daily"]    == 2_500
        assert d["currency"] == "JPY"
        assert d["remaining"] == 50_000

    def test_currency_custom(self):
        b = CampaignBudget(total=1_000, daily=100, currency="USD")
        assert b.to_dict()["currency"] == "USD"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Campaign
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_campaign(**kwargs) -> Campaign:
    defaults = dict(
        id="camp-1",
        name="テストキャンペーン",
        objective=AdObjective.AWARENESS,
        budget=CampaignBudget(total=100_000, daily=5_000),
    )
    defaults.update(kwargs)
    return Campaign(**defaults)


class TestCampaign:
    def test_default_status_draft(self):
        c = make_campaign()
        assert c.status == CampaignStatus.DRAFT

    def test_activate(self):
        c = make_campaign()
        c.activate()
        assert c.status == CampaignStatus.ACTIVE

    def test_pause(self):
        c = make_campaign()
        c.activate()
        c.pause()
        assert c.status == CampaignStatus.PAUSED

    def test_complete_from_active(self):
        c = make_campaign()
        c.activate()
        c.complete()
        assert c.status == CampaignStatus.COMPLETED

    def test_complete_from_paused(self):
        c = make_campaign()
        c.activate()
        c.pause()
        c.complete()
        assert c.status == CampaignStatus.COMPLETED

    def test_archive(self):
        c = make_campaign()
        c.archive()
        assert c.status == CampaignStatus.ARCHIVED

    def test_activate_invalid_raises(self):
        c = make_campaign()
        c.activate()
        c.complete()
        with pytest.raises(ValueError):
            c.activate()

    def test_pause_from_draft_raises(self):
        c = make_campaign()
        with pytest.raises(ValueError):
            c.pause()

    def test_archive_twice_raises(self):
        c = make_campaign()
        c.archive()
        with pytest.raises(ValueError):
            c.archive()

    def test_best_copy_none_when_empty(self):
        c = make_campaign()
        assert c.best_copy is None

    def test_best_copy_highest_score(self):
        c = make_campaign()
        c1 = make_copy(id="a", score=0.5)
        c2 = make_copy(id="b", score=0.9)
        c.copies = [c1, c2]
        assert c.best_copy.id == "b"

    def test_avg_score_zero_when_no_copies(self):
        c = make_campaign()
        assert c.avg_score == 0.0

    def test_avg_score(self):
        c = make_campaign()
        c.copies = [make_copy(id="a", score=0.4), make_copy(id="b", score=0.8)]
        assert abs(c.avg_score - 0.6) < 1e-6

    def test_to_dict(self):
        c = make_campaign()
        d = c.to_dict()
        assert d["status"] == "draft"
        assert d["objective"] == "awareness"
        assert "budget" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignStore:
    def test_add_and_get(self):
        store = CampaignStore()
        c = make_campaign()
        store.add(c)
        assert store.get("camp-1") is c

    def test_get_missing_returns_none(self):
        store = CampaignStore()
        assert store.get("nope") is None

    def test_list_all(self):
        store = CampaignStore()
        store.add(make_campaign(id="a"))
        store.add(make_campaign(id="b"))
        assert len(store.list_all()) == 2

    def test_delete(self):
        store = CampaignStore()
        store.add(make_campaign(id="del"))
        assert store.delete("del") is True
        assert store.get("del") is None

    def test_delete_missing_returns_false(self):
        store = CampaignStore()
        assert store.delete("no") is False

    def test_list_by_status(self):
        store = CampaignStore()
        c_active = make_campaign(id="a")
        c_active.activate()
        store.add(c_active)
        store.add(make_campaign(id="b"))
        assert len(store.list_by_status(CampaignStatus.ACTIVE)) == 1
        assert len(store.list_by_status(CampaignStatus.DRAFT)) == 1

    def test_find_by_objective(self):
        store = CampaignStore()
        store.add(make_campaign(id="a", objective=AdObjective.AWARENESS))
        store.add(make_campaign(id="b", objective=AdObjective.CONVERSION))
        assert len(store.find_by_objective(AdObjective.CONVERSION)) == 1

    def test_find_by_cep(self):
        store = CampaignStore()
        c = make_campaign(id="a")
        c.cep_ids = ["cep-1", "cep-2"]
        store.add(c)
        store.add(make_campaign(id="b"))
        assert len(store.find_by_cep("cep-1")) == 1
        assert len(store.find_by_cep("cep-99")) == 0

    def test_count(self):
        store = CampaignStore()
        assert store.count() == 0
        store.add(make_campaign(id="x"))
        assert store.count() == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CopyGenerator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCopyGenerator:
    def test_generate_returns_ad_copy(self):
        gen = CopyGenerator(brand="TestBrand")
        copy = gen.generate_from_scenario("暑い夏に冷たい飲み物を探している")
        assert isinstance(copy, AdCopy)

    def test_headline_not_empty(self):
        gen = CopyGenerator()
        copy = gen.generate_from_scenario("問題を解決したい")
        assert len(copy.headline) > 0

    def test_cta_not_empty(self):
        gen = CopyGenerator()
        copy = gen.generate_from_scenario("競合と比べたい")
        assert len(copy.cta) > 0

    def test_headline_max_length(self):
        gen = CopyGenerator()
        copy = gen.generate_from_scenario("非常に長いシナリオ文字列で見出し長さ制限を確認するテストです")
        assert len(copy.headline) <= 40

    def test_brand_in_headline_or_body(self):
        gen = CopyGenerator(brand="MyCoolBrand")
        copy = gen.generate_from_scenario("何か困っているとき", objective=AdObjective.AWARENESS)
        assert "MyCoolBrand" in copy.headline or "MyCoolBrand" in copy.body

    def test_channel_set_correctly(self):
        gen = CopyGenerator()
        copy = gen.generate_from_scenario("テスト", channel=AdChannel.EMAIL)
        assert copy.channel == AdChannel.EMAIL

    def test_objective_set_correctly(self):
        gen = CopyGenerator()
        copy = gen.generate_from_scenario("テスト", objective=AdObjective.CONVERSION)
        assert copy.objective == AdObjective.CONVERSION

    def test_generate_batch_returns_multiple(self):
        gen = CopyGenerator()
        copies = gen.generate_batch(
            "テストシナリオ",
            channels=[AdChannel.SEARCH, AdChannel.SOCIAL, AdChannel.EMAIL],
        )
        assert len(copies) == 3

    def test_generate_batch_default_channels(self):
        gen = CopyGenerator()
        copies = gen.generate_batch("シナリオ")
        assert len(copies) == 2

    def test_tags_extracted(self):
        gen = CopyGenerator()
        copy = gen.generate_from_scenario("コスト削減したい企業担当者向け")
        assert isinstance(copy.tags, list)

    def test_all_objectives_generate(self):
        gen = CopyGenerator()
        for obj in AdObjective:
            copy = gen.generate_from_scenario("テスト", objective=obj)
            assert copy.objective == obj


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignEvaluator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignEvaluator:
    def setup_method(self):
        self.ev = CampaignEvaluator()

    def test_evaluate_returns_eval_result(self):
        c = make_copy()
        result = self.ev.evaluate(c)
        assert isinstance(result, EvalResult)

    def test_total_score_between_0_and_1(self):
        c = make_copy()
        result = self.ev.evaluate(c)
        assert 0.0 <= result.total_score <= 1.0

    def test_score_written_to_copy(self):
        c = make_copy()
        assert c.score == 0.0
        self.ev.evaluate(c)
        assert c.score > 0.0

    def test_short_headline_penalized(self):
        c = make_copy(headline="短")
        result = self.ev.evaluate(c)
        assert result.headline_score < 1.0
        assert any("見出し" in n for n in result.notes)

    def test_long_headline_penalized(self):
        c = make_copy(headline="a" * 50)
        result = self.ev.evaluate(c)
        assert result.headline_score < 1.0

    def test_short_body_penalized(self):
        c = make_copy(body="短い")
        result = self.ev.evaluate(c)
        assert result.body_score < 0.8

    def test_empty_cta_zero_score(self):
        c = make_copy(cta="")
        result = self.ev.evaluate(c)
        assert result.cta_score == 0.0

    def test_action_cta_full_score(self):
        c = make_copy(cta="今すぐ試す")
        result = self.ev.evaluate(c)
        assert result.cta_score == 1.0

    def test_alignment_score_with_no_scenario(self):
        c = make_copy()
        result = self.ev.evaluate(c, scenario=None)
        assert result.alignment_score == 0.8

    def test_alignment_score_with_scenario(self):
        c = make_copy(headline="コスト削減の方法", body="コスト削減を実現するためのサービスです。", cta="今すぐ試す")
        result = self.ev.evaluate(c, scenario="コスト削減したいとき")
        assert result.alignment_score >= 0.5

    def test_evaluate_batch(self):
        copies = [make_copy(id=str(i)) for i in range(3)]
        results = self.ev.evaluate_batch(copies)
        assert len(results) == 3

    def test_eval_result_to_dict(self):
        c = make_copy()
        r = self.ev.evaluate(c)
        d = r.to_dict()
        assert "total_score" in d
        assert "notes" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignWorkflow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignWorkflow:
    def setup_method(self):
        self.wf = CampaignWorkflow()

    def test_run_returns_workflow_result(self):
        result = self.wf.run(
            name="テストキャンペーン",
            scenario="暑い夏に冷たい飲み物を探している",
        )
        assert isinstance(result, WorkflowResult)

    def test_campaign_stored(self):
        result = self.wf.run(name="stored", scenario="テスト")
        assert self.wf.store.get(result.campaign.id) is not None

    def test_copies_generated(self):
        result = self.wf.run(name="copies", scenario="テスト")
        assert result.total_copies > 0

    def test_best_copy_is_highest_score(self):
        result = self.wf.run(name="best", scenario="テスト")
        if result.best_copy and len(result.campaign.copies) > 1:
            assert result.best_copy.score == max(c.score for c in result.campaign.copies)

    def test_avg_score_positive(self):
        result = self.wf.run(name="avg", scenario="テスト")
        assert result.avg_score >= 0.0

    def test_custom_budget(self):
        budget = CampaignBudget(total=200_000, daily=10_000)
        result = self.wf.run(name="budget", scenario="テスト", budget=budget)
        assert result.campaign.budget.total == 200_000

    def test_custom_channels(self):
        result = self.wf.run(
            name="ch",
            scenario="テスト",
            channels=[AdChannel.EMAIL],
        )
        assert result.total_copies == 1

    def test_cep_ids_stored(self):
        result = self.wf.run(
            name="cep",
            scenario="テスト",
            cep_ids=["cep-1", "cep-2"],
        )
        assert result.campaign.cep_ids == ["cep-1", "cep-2"]

    def test_to_dict(self):
        result = self.wf.run(name="dict", scenario="テスト")
        d = result.to_dict()
        assert "campaign" in d
        assert "eval_results" in d
        assert "avg_score" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignReportEngine:
    def setup_method(self):
        self.store = CampaignStore()
        self.engine = CampaignReportEngine(self.store)

    def _add_campaign(self, campaign_id: str = "c1") -> Campaign:
        c = make_campaign(id=campaign_id)
        c.copies = [make_copy(id="cp1", score=0.8)]
        self.store.add(c)
        return c

    def test_summary_json_empty(self):
        d = self.engine.summary_json()
        assert d["total_campaigns"] == 0

    def test_summary_json_counts(self):
        self._add_campaign("a")
        self._add_campaign("b")
        d = self.engine.summary_json()
        assert d["total_campaigns"] == 2

    def test_summary_json_has_campaigns_list(self):
        self._add_campaign()
        d = self.engine.summary_json()
        assert len(d["campaigns"]) == 1

    def test_campaign_markdown_not_found(self):
        md = self.engine.campaign_markdown("nonexistent")
        assert "エラー" in md

    def test_campaign_markdown_contains_name(self):
        c = self._add_campaign()
        md = self.engine.campaign_markdown("c1")
        assert c.name in md

    def test_campaign_markdown_contains_budget(self):
        self._add_campaign()
        md = self.engine.campaign_markdown("c1")
        assert "予算" in md

    def test_campaign_markdown_contains_copies(self):
        self._add_campaign()
        md = self.engine.campaign_markdown("c1")
        assert "広告コピー" in md

    def test_avg_copy_score_in_summary(self):
        self._add_campaign()
        d = self.engine.summary_json()
        assert "avg_copy_score" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serve.api import app
    return TestClient(app)


class TestCampaignApi:
    def test_create_campaign(self, client):
        resp = client.post("/v1/campaign/", json={
            "name": "API テストキャンペーン",
            "objective": "awareness",
            "budget_total": 100000,
            "budget_daily": 5000,
        })
        assert resp.status_code == 200
        assert "id" in resp.json()

    def test_list_campaigns(self, client):
        client.post("/v1/campaign/", json={
            "name": "list test", "objective": "conversion",
            "budget_total": 50000, "budget_daily": 2000,
        })
        resp = client.get("/v1/campaign/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_campaign(self, client):
        create_resp = client.post("/v1/campaign/", json={
            "name": "get test", "objective": "consideration",
            "budget_total": 80000, "budget_daily": 4000,
        })
        cid = create_resp.json()["id"]
        resp = client.get(f"/v1/campaign/{cid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == cid

    def test_get_campaign_not_found(self, client):
        resp = client.get("/v1/campaign/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_campaign(self, client):
        create_resp = client.post("/v1/campaign/", json={
            "name": "del test", "objective": "awareness",
            "budget_total": 10000, "budget_daily": 1000,
        })
        cid = create_resp.json()["id"]
        del_resp = client.delete(f"/v1/campaign/{cid}")
        assert del_resp.status_code == 200
        assert client.get(f"/v1/campaign/{cid}").status_code == 404

    def test_activate_campaign(self, client):
        create_resp = client.post("/v1/campaign/", json={
            "name": "act test", "objective": "awareness",
            "budget_total": 10000, "budget_daily": 1000,
        })
        cid = create_resp.json()["id"]
        resp = client.post(f"/v1/campaign/{cid}/activate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_pause_campaign(self, client):
        create_resp = client.post("/v1/campaign/", json={
            "name": "pause test", "objective": "awareness",
            "budget_total": 10000, "budget_daily": 1000,
        })
        cid = create_resp.json()["id"]
        client.post(f"/v1/campaign/{cid}/activate")
        resp = client.post(f"/v1/campaign/{cid}/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_complete_campaign(self, client):
        create_resp = client.post("/v1/campaign/", json={
            "name": "complete test", "objective": "awareness",
            "budget_total": 10000, "budget_daily": 1000,
        })
        cid = create_resp.json()["id"]
        client.post(f"/v1/campaign/{cid}/activate")
        resp = client.post(f"/v1/campaign/{cid}/complete")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_report_markdown(self, client):
        create_resp = client.post("/v1/campaign/", json={
            "name": "report test", "objective": "awareness",
            "budget_total": 10000, "budget_daily": 1000,
        })
        cid = create_resp.json()["id"]
        resp = client.get(f"/v1/campaign/{cid}/report/md")
        assert resp.status_code == 200
        assert "キャンペーン" in resp.text

    def test_workflow_endpoint(self, client):
        resp = client.post("/v1/campaign/workflow", json={
            "name": "workflow test",
            "scenario": "コスト削減したい担当者向け広告",
            "objective": "conversion",
            "budget_total": 200000,
            "budget_daily": 10000,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "campaign" in data
        assert "avg_score" in data

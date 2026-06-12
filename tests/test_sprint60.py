"""
Sprint 60 — 広告キャンペーン管理 (60-A) + A/Bテスト基盤 (60-C) テスト

対象:
  open_mythos/skills/campaign_manager.py:
    CampaignStatus / AdFormat
    CopyRequest / AdCopy
    CampaignMetrics / CampaignEntry
    CampaignStore / CopyGenerator / CampaignEvaluator / CopyScore
    CampaignWorkflow / WorkflowResult
    CampaignReportEngine

  open_mythos/skills/ab_test.py:
    VariantStatus / TestStatus / StatMethod
    Variant / ABTest
    ABTestStore / StatEngine / StatResult
    ABTestAnalyzer / AnalysisResult
    ABTestRunner / ABReportEngine

  serve/api.py:
    POST /v1/campaign
    GET  /v1/campaign
    GET  /v1/campaign/{id}
    POST /v1/campaign/{id}/run
    POST /v1/campaign/{id}/metrics
    GET  /v1/campaign/{id}/report
    GET  /v1/campaign/{id}/report/md
    PATCH /v1/campaign/{id}/status
    DELETE /v1/campaign/{id}

    POST /v1/ab/test
    GET  /v1/ab/test
    GET  /v1/ab/test/{id}
    POST /v1/ab/test/{id}/variant
    POST /v1/ab/test/{id}/start
    POST /v1/ab/test/{id}/record
    GET  /v1/ab/test/{id}/analyze
    POST /v1/ab/test/{id}/stop
    GET  /v1/ab/test/{id}/report
    GET  /v1/ab/test/{id}/report/md
    DELETE /v1/ab/test/{id}
"""
from __future__ import annotations

import sys
import pytest
import torch
from unittest.mock import MagicMock

from open_mythos.skills.campaign_manager import (
    AdFormat, CampaignStatus,
    CopyRequest, AdCopy,
    CampaignMetrics, CampaignEntry,
    CampaignStore, CopyGenerator, CampaignEvaluator, CopyScore,
    CampaignWorkflow, WorkflowResult,
    CampaignReportEngine,
)
from open_mythos.skills.ab_test import (
    VariantStatus, TestStatus, StatMethod,
    Variant, ABTest,
    ABTestStore, StatEngine, StatResult,
    ABTestAnalyzer, AnalysisResult,
    ABTestRunner, ABReportEngine,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.agents import OpenMythosLLM
    import serve.api as api_mod

    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.eos_token_id = 50256
    tok.decode.side_effect = lambda ids, **kw: "ok " * max(len(ids), 1)
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    cfg = MythosConfig(
        vocab_size=50257, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=128, max_loop_iters=4,
    )
    model = OpenMythos(cfg)
    llm = OpenMythosLLM(model=model, tokenizer=tok)

    api_mod._llm = llm
    api_mod._campaign_store    = CampaignStore()
    api_mod._campaign_workflow = CampaignWorkflow(store=api_mod._campaign_store)
    api_mod._ab_store  = ABTestStore()
    api_mod._ab_runner = ABTestRunner(store=api_mod._ab_store)

    return TestClient(api_mod.app)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 60-A: campaign_manager.py ユニットテスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignEnums:
    def test_campaign_status_values(self):
        assert CampaignStatus.DRAFT.value == "draft"
        assert CampaignStatus.ACTIVE.value == "active"
        assert CampaignStatus.COMPLETED.value == "completed"

    def test_ad_format_values(self):
        assert AdFormat.BANNER.value == "banner"
        assert AdFormat.SEARCH.value == "search"
        assert len(AdFormat) == 6


class TestCampaignMetrics:
    def test_ctr_zero_impressions(self):
        m = CampaignMetrics()
        assert m.ctr == 0.0

    def test_ctr_calculation(self):
        m = CampaignMetrics(impressions=1000, clicks=50)
        assert m.ctr == pytest.approx(0.05)

    def test_cvr_calculation(self):
        m = CampaignMetrics(clicks=100, conversions=10)
        assert m.cvr == pytest.approx(0.1)

    def test_roas_calculation(self):
        m = CampaignMetrics(spend=10000.0, revenue=35000.0)
        assert m.roas == pytest.approx(3.5)

    def test_cpa_calculation(self):
        m = CampaignMetrics(spend=5000.0, conversions=10)
        assert m.cpa == pytest.approx(500.0)

    def test_update_accumulates(self):
        m = CampaignMetrics()
        m.update(impressions=100, clicks=5, conversions=1, spend=1000.0, revenue=3000.0)
        m.update(impressions=200, clicks=10, conversions=2, spend=2000.0, revenue=6000.0)
        assert m.impressions == 300
        assert m.clicks == 15
        assert m.spend == pytest.approx(3000.0)

    def test_to_dict_keys(self):
        m = CampaignMetrics(impressions=100, clicks=5)
        d = m.to_dict()
        assert set(d.keys()) >= {"impressions", "clicks", "ctr", "cvr", "roas", "cpa"}


class TestCopyGenerator:
    def test_generate_returns_n_copies(self):
        gen = CopyGenerator()
        req = CopyRequest(campaign_id="c1", brief="高品質な日本茶")
        copies = gen.generate(req, n=3)
        assert len(copies) == 3

    def test_all_copies_have_content(self):
        gen = CopyGenerator()
        req = CopyRequest(campaign_id="c1", brief="健康食品", keywords=["オーガニック"])
        for copy in gen.generate(req, n=2):
            assert copy.headline
            assert copy.body
            assert copy.cta

    def test_copy_format_matches_request(self):
        gen = CopyGenerator()
        req = CopyRequest(campaign_id="c1", brief="動画広告", format=AdFormat.VIDEO)
        copies = gen.generate(req, n=1)
        assert copies[0].format == AdFormat.VIDEO

    def test_headline_respects_max_chars(self):
        gen = CopyGenerator()
        req = CopyRequest(campaign_id="c1", brief="テスト", max_chars=20)
        copies = gen.generate(req, n=1)
        assert len(copies[0].headline) <= 20

    def test_different_tones_produce_different_headlines(self):
        gen = CopyGenerator()
        req_friendly = CopyRequest(campaign_id="c1", brief="商品", tone="friendly")
        req_urgent   = CopyRequest(campaign_id="c1", brief="商品", tone="urgent")
        h_friendly = gen.generate(req_friendly, n=1)[0].headline
        h_urgent   = gen.generate(req_urgent, n=1)[0].headline
        assert h_friendly != h_urgent


class TestCampaignEvaluator:
    def test_score_returns_copy_score(self):
        ev = CampaignEvaluator()
        copy = AdCopy(id="x", request_id="r", headline="今すぐ試す健康食品",
                      body="毎日続けられるサプリ。キーワード: 健康。",
                      cta="今すぐ試す", format=AdFormat.TEXT)
        score = ev.score(copy, keywords=["健康"])
        assert isinstance(score, CopyScore)
        assert 0.0 <= score.overall <= 1.0

    def test_strong_cta_boosts_score(self):
        ev = CampaignEvaluator()
        copy_strong = AdCopy(id="a", request_id="r", headline="商品",
                             body="詳細。", cta="今すぐ無料で試す", format=AdFormat.TEXT)
        copy_weak   = AdCopy(id="b", request_id="r", headline="商品",
                             body="詳細。", cta="クリック", format=AdFormat.TEXT)
        s1 = ev.score(copy_strong).cta_strength
        s2 = ev.score(copy_weak).cta_strength
        assert s1 > s2

    def test_relevance_zero_without_keywords(self):
        ev = CampaignEvaluator()
        copy = AdCopy(id="x", request_id="r", headline="HL", body="body",
                      cta="CTA", format=AdFormat.TEXT)
        score = ev.score(copy, keywords=[])
        assert score.relevance == pytest.approx(0.5)

    def test_score_batch(self):
        ev = CampaignEvaluator()
        copies = [
            AdCopy(id=f"c{i}", request_id="r", headline=f"HL{i}", body="body",
                   cta="詳細を見る", format=AdFormat.TEXT)
            for i in range(4)
        ]
        scores = ev.score_batch(copies)
        assert len(scores) == 4

    def test_predict_ctr_in_range(self):
        ev = CampaignEvaluator()
        copy = AdCopy(id="x", request_id="r", headline="今すぐ無料",
                      body="body", cta="今すぐ試す", format=AdFormat.TEXT)
        ctr = ev.predict_ctr(copy)
        assert 0.0 <= ctr <= 0.15


class TestCampaignStore:
    def test_create_and_get(self):
        store = CampaignStore()
        entry = store.create(name="テストキャンペーン", budget=100000.0)
        fetched = store.get(entry.id)
        assert fetched is not None
        assert fetched.name == "テストキャンペーン"

    def test_list_all(self):
        store = CampaignStore()
        store.create(name="A")
        store.create(name="B")
        assert len(store.list_all()) == 2

    def test_list_filter_by_status(self):
        store = CampaignStore()
        e1 = store.create(name="Draft")
        e2 = store.create(name="Active")
        store.update_status(e2.id, CampaignStatus.ACTIVE)
        active = store.list_all(status=CampaignStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].id == e2.id

    def test_delete(self):
        store = CampaignStore()
        e = store.create(name="To Delete")
        assert store.delete(e.id) is True
        assert store.get(e.id) is None

    def test_delete_nonexistent(self):
        store = CampaignStore()
        assert store.delete("nonexistent") is False

    def test_record_metrics(self):
        store = CampaignStore()
        e = store.create(name="Metrics Test")
        metrics = store.record_metrics(e.id, impressions=500, clicks=25, spend=5000.0)
        assert metrics is not None
        assert metrics.impressions == 500
        assert metrics.ctr == pytest.approx(0.05)


class TestCampaignWorkflow:
    def test_workflow_run_returns_result(self):
        store = CampaignStore()
        wf = CampaignWorkflow(store=store)
        entry = store.create(name="Workflow Test")
        result = wf.run(entry.id, brief="日本茶の魅力", keywords=["茶", "健康"])
        assert isinstance(result, WorkflowResult)
        assert result.campaign_id == entry.id
        assert len(result.copies) == 3

    def test_workflow_copies_attached_to_campaign(self):
        store = CampaignStore()
        wf = CampaignWorkflow(store=store)
        entry = store.create(name="Attach Test")
        wf.run(entry.id, brief="テスト商品")
        updated = store.get(entry.id)
        assert len(updated.copies) == 3

    def test_workflow_best_copy_id_set(self):
        store = CampaignStore()
        wf = CampaignWorkflow(store=store)
        entry = store.create(name="Best Test")
        result = wf.run(entry.id, brief="最高のコピー", n_copies=5)
        assert result.best_copy_id is not None
        best_score = max(c.score for c in result.copies)
        best = next(c for c in result.copies if c.id == result.best_copy_id)
        assert best.score == pytest.approx(best_score)


class TestCampaignReportEngine:
    def test_to_markdown_contains_name(self):
        engine = CampaignReportEngine()
        store  = CampaignStore()
        entry  = store.create(name="サマーキャンペーン")
        md = engine.to_markdown(entry)
        assert "サマーキャンペーン" in md

    def test_to_markdown_contains_metrics(self):
        engine = CampaignReportEngine()
        store  = CampaignStore()
        entry  = store.create(name="X")
        entry.metrics.update(impressions=1000, clicks=50, spend=10000.0, revenue=30000.0)
        md = engine.to_markdown(entry)
        assert "CTR" in md
        assert "ROAS" in md

    def test_to_json_structure(self):
        engine = CampaignReportEngine()
        store  = CampaignStore()
        entry  = store.create(name="JSON Test")
        j = engine.to_json(entry)
        assert j["report_type"] == "campaign"
        assert "summary" in j
        assert "total_copies" in j["summary"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 60-C: ab_test.py ユニットテスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVariant:
    def test_ctr_calculation(self):
        v = Variant(id="v1", name="A", copy="コピーA", impressions=1000, clicks=50)
        assert v.ctr == pytest.approx(0.05)

    def test_cvr_calculation(self):
        v = Variant(id="v1", name="A", copy="コピーA", clicks=100, conversions=10)
        assert v.cvr == pytest.approx(0.1)

    def test_record_accumulates(self):
        v = Variant(id="v1", name="A", copy="コピーA")
        v.record(impressions=100, clicks=5, conversions=1)
        v.record(impressions=200, clicks=10, conversions=2)
        assert v.impressions == 300
        assert v.clicks == 15
        assert v.conversions == 3

    def test_record_ignores_negative(self):
        v = Variant(id="v1", name="A", copy="コピーA", impressions=100)
        v.record(impressions=-50)
        assert v.impressions == 100

    def test_to_dict_keys(self):
        v = Variant(id="v1", name="A", copy="コピーA")
        d = v.to_dict()
        assert set(d.keys()) >= {"id", "name", "copy", "status", "impressions", "ctr", "cvr"}


class TestABTest:
    def test_add_variant(self):
        test = ABTest(id="t1", name="Test")
        v = test.add_variant("A", "コピーA")
        assert len(test.variants) == 1
        assert v.name == "A"

    def test_get_variant(self):
        test = ABTest(id="t1", name="Test")
        v = test.add_variant("A", "コピーA")
        fetched = test.get_variant(v.id)
        assert fetched is not None
        assert fetched.id == v.id

    def test_get_variant_not_found(self):
        test = ABTest(id="t1", name="Test")
        assert test.get_variant("xxx") is None

    def test_is_ready_false_when_not_enough_samples(self):
        test = ABTest(id="t1", name="Test", min_samples=100)
        test.add_variant("A", "コピーA")
        test.add_variant("B", "コピーB")
        assert test.is_ready() is False

    def test_is_ready_true_when_enough_samples(self):
        test = ABTest(id="t1", name="Test", min_samples=10)
        a = test.add_variant("A", "コピーA")
        b = test.add_variant("B", "コピーB")
        a.record(impressions=100)
        b.record(impressions=100)
        assert test.is_ready() is True


class TestStatEngine:
    def _make_variants(self, n_a, c_a, n_b, c_b):
        a = Variant(id="a", name="A", copy="A", impressions=n_a,
                    clicks=int(n_a * 0.1), conversions=c_a)
        b = Variant(id="b", name="B", copy="B", impressions=n_b,
                    clicks=int(n_b * 0.1), conversions=c_b)
        return a, b

    def test_chi_square_no_data(self):
        eng = StatEngine()
        a = Variant(id="a", name="A", copy="A")
        b = Variant(id="b", name="B", copy="B")
        r = eng.chi_square_test(a, b)
        assert r.significant is False
        assert r.p_value == pytest.approx(1.0)

    def test_chi_square_significant(self):
        eng = StatEngine()
        # A: 10000 impressions, 500 conv (5%); B: 10000 impressions, 250 conv (2.5%)
        a, b = self._make_variants(10000, 500, 10000, 250)
        r = eng.chi_square_test(a, b, confidence=0.95)
        assert r.significant is True
        assert r.p_value < 0.05

    def test_chi_square_not_significant(self):
        eng = StatEngine()
        # 小差分: A=5%, B=4.8%
        a, b = self._make_variants(100, 5, 100, 4)
        r = eng.chi_square_test(a, b, confidence=0.95)
        assert r.significant is False

    def test_welch_t_test_significant(self):
        eng = StatEngine()
        a = Variant(id="a", name="A", copy="A", impressions=5000, clicks=500)
        b = Variant(id="b", name="B", copy="B", impressions=5000, clicks=100)
        r = eng.welch_t_test(a, b)
        assert r.significant is True

    def test_welch_t_test_insufficient_data(self):
        eng = StatEngine()
        a = Variant(id="a", name="A", copy="A", impressions=1, clicks=1)
        b = Variant(id="b", name="B", copy="B", impressions=1, clicks=0)
        r = eng.welch_t_test(a, b)
        assert r.significant is False

    def test_bayesian_test_returns_result(self):
        eng = StatEngine()
        a = Variant(id="a", name="A", copy="A", impressions=1000, clicks=100, conversions=50)
        b = Variant(id="b", name="B", copy="B", impressions=1000, clicks=100, conversions=20)
        r = eng.bayesian_test(a, b)
        assert isinstance(r, StatResult)
        assert 0.0 <= r.p_value <= 1.0

    def test_normal_cdf_bounds(self):
        eng = StatEngine()
        assert eng._normal_cdf(-10) == pytest.approx(0.0, abs=1e-6)
        assert eng._normal_cdf(10)  == pytest.approx(1.0, abs=1e-6)
        assert eng._normal_cdf(0)   == pytest.approx(0.5, abs=1e-4)


class TestABTestStore:
    def test_create_and_get(self):
        store = ABTestStore()
        test = store.create("テストA")
        assert store.get(test.id) is not None

    def test_list_all(self):
        store = ABTestStore()
        store.create("X")
        store.create("Y")
        assert len(store.list_all()) == 2

    def test_list_filter_status(self):
        store = ABTestStore()
        t1 = store.create("Running Test")
        t1.status = TestStatus.RUNNING
        store.create("Draft Test")
        running = store.list_all(status=TestStatus.RUNNING)
        assert len(running) == 1

    def test_delete(self):
        store = ABTestStore()
        t = store.create("Delete Me")
        assert store.delete(t.id) is True
        assert store.get(t.id) is None


class TestABTestRunner:
    def _setup_test(self, store: ABTestStore, n: int = 200) -> ABTest:
        test = store.create("Runner Test", min_samples=100)
        a = test.add_variant("A", "コピーA")
        b = test.add_variant("B", "コピーB")
        a.record(impressions=n, clicks=int(n * 0.12), conversions=int(n * 0.05))
        b.record(impressions=n, clicks=int(n * 0.08), conversions=int(n * 0.02))
        return test

    def test_start_sets_running(self):
        store = ABTestStore()
        runner = ABTestRunner(store=store)
        test = store.create("Start Test")
        test.add_variant("A", "A")
        test.add_variant("B", "B")
        result = runner.start(test.id)
        assert result.status == TestStatus.RUNNING

    def test_start_requires_2_variants(self):
        store = ABTestStore()
        runner = ABTestRunner(store=store)
        test = store.create("Only One")
        test.add_variant("A", "A")
        with pytest.raises(ValueError):
            runner.start(test.id)

    def test_record_updates_variant(self):
        store = ABTestStore()
        runner = ABTestRunner(store=store)
        test = self._setup_test(store)
        v = test.variants[0]
        before = v.impressions
        runner.record(test.id, v.id, impressions=50)
        assert v.impressions == before + 50

    def test_analyze_returns_result(self):
        store = ABTestStore()
        runner = ABTestRunner(store=store)
        test = self._setup_test(store)
        result = runner.analyze(test.id)
        assert isinstance(result, AnalysisResult)
        assert result.test_id == test.id

    def test_stop_sets_stopped(self):
        store = ABTestStore()
        runner = ABTestRunner(store=store)
        test = self._setup_test(store)
        stopped = runner.stop(test.id)
        assert stopped.status == TestStatus.STOPPED

    def test_auto_analyze_completes_when_ready(self):
        store = ABTestStore()
        runner = ABTestRunner(store=store)
        test = self._setup_test(store, n=500)
        result = runner.auto_analyze_and_complete(test.id)
        # サンプル十分なので None にならない
        assert result is not None

    def test_auto_analyze_returns_none_when_not_ready(self):
        store = ABTestStore()
        runner = ABTestRunner(store=store)
        test = store.create("Not Ready", min_samples=10000)
        test.add_variant("A", "A")
        test.add_variant("B", "B")
        result = runner.auto_analyze_and_complete(test.id)
        assert result is None


class TestABReportEngine:
    def _make_test_with_data(self) -> ABTest:
        store = ABTestStore()
        test = store.create("Report Test")
        a = test.add_variant("バリアントA", "コピーA")
        b = test.add_variant("バリアントB", "コピーB")
        a.record(impressions=500, clicks=50, conversions=25)
        b.record(impressions=500, clicks=40, conversions=15)
        return test

    def test_to_markdown_contains_variant_names(self):
        engine = ABReportEngine()
        test = self._make_test_with_data()
        md = engine.to_markdown(test)
        assert "バリアントA" in md
        assert "バリアントB" in md

    def test_to_markdown_contains_analysis(self):
        engine = ABReportEngine()
        analyzer = ABTestAnalyzer()
        test = self._make_test_with_data()
        analysis = analyzer.analyze(test)
        md = engine.to_markdown(test, analysis=analysis)
        assert "p 値" in md or "recommendation" in md.lower() or "推奨" in md

    def test_to_json_structure(self):
        engine = ABReportEngine()
        test = self._make_test_with_data()
        j = engine.to_json(test)
        assert j["report_type"] == "ab_test"
        assert "test" in j


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 60 API 統合テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignAPI:
    def test_create_campaign(self, client):
        r = client.post("/v1/campaign", json={"name": "秋キャンペーン", "budget": 50000.0})
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "秋キャンペーン"
        assert d["status"] == "draft"

    def test_list_campaigns(self, client):
        client.post("/v1/campaign", json={"name": "List Test A"})
        client.post("/v1/campaign", json={"name": "List Test B"})
        r = client.get("/v1/campaign")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_campaign(self, client):
        r = client.post("/v1/campaign", json={"name": "Get Test"})
        cid = r.json()["id"]
        r2 = client.get(f"/v1/campaign/{cid}")
        assert r2.status_code == 200
        assert r2.json()["id"] == cid

    def test_get_campaign_not_found(self, client):
        r = client.get("/v1/campaign/nonexistent")
        assert r.status_code == 404

    def test_run_workflow(self, client):
        r = client.post("/v1/campaign", json={"name": "Workflow Test"})
        cid = r.json()["id"]
        r2 = client.post(f"/v1/campaign/{cid}/run", json={
            "brief": "健康食品の魅力を伝える",
            "format": "text",
            "keywords": ["健康", "オーガニック"],
            "n_copies": 3,
        })
        assert r2.status_code == 200
        d = r2.json()
        assert d["campaign_id"] == cid
        assert len(d["copies"]) == 3

    def test_record_metrics(self, client):
        r = client.post("/v1/campaign", json={"name": "Metrics Test"})
        cid = r.json()["id"]
        r2 = client.post(f"/v1/campaign/{cid}/metrics", json={
            "impressions": 1000, "clicks": 50, "conversions": 5,
            "spend": 10000.0, "revenue": 30000.0,
        })
        assert r2.status_code == 200
        d = r2.json()
        assert d["impressions"] == 1000
        assert d["ctr"] == pytest.approx(0.05)

    def test_campaign_report_json(self, client):
        r = client.post("/v1/campaign", json={"name": "Report JSON"})
        cid = r.json()["id"]
        r2 = client.get(f"/v1/campaign/{cid}/report")
        assert r2.status_code == 200
        assert r2.json()["report_type"] == "campaign"

    def test_campaign_report_md(self, client):
        r = client.post("/v1/campaign", json={"name": "Report MD"})
        cid = r.json()["id"]
        r2 = client.get(f"/v1/campaign/{cid}/report/md")
        assert r2.status_code == 200

    def test_update_status(self, client):
        r = client.post("/v1/campaign", json={"name": "Status Test"})
        cid = r.json()["id"]
        r2 = client.patch(f"/v1/campaign/{cid}/status?status=active")
        assert r2.status_code == 200
        assert r2.json()["status"] == "active"

    def test_delete_campaign(self, client):
        r = client.post("/v1/campaign", json={"name": "Delete Test"})
        cid = r.json()["id"]
        r2 = client.delete(f"/v1/campaign/{cid}")
        assert r2.status_code == 200
        assert client.get(f"/v1/campaign/{cid}").status_code == 404


class TestABTestAPI:
    def test_create_test(self, client):
        r = client.post("/v1/ab/test", json={"name": "CTR テスト", "method": "chi_square"})
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "CTR テスト"
        assert d["status"] == "draft"

    def test_list_tests(self, client):
        client.post("/v1/ab/test", json={"name": "List A"})
        client.post("/v1/ab/test", json={"name": "List B"})
        r = client.get("/v1/ab/test")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_test(self, client):
        r = client.post("/v1/ab/test", json={"name": "Get Test"})
        tid = r.json()["id"]
        r2 = client.get(f"/v1/ab/test/{tid}")
        assert r2.status_code == 200
        assert r2.json()["id"] == tid

    def test_get_test_not_found(self, client):
        r = client.get("/v1/ab/test/nonexistent")
        assert r.status_code == 404

    def test_add_variant(self, client):
        r = client.post("/v1/ab/test", json={"name": "Variant Test"})
        tid = r.json()["id"]
        r2 = client.post(f"/v1/ab/test/{tid}/variant", json={"name": "A案", "copy_text": "コピーA"})
        assert r2.status_code == 200
        assert r2.json()["name"] == "A案"

    def test_start_test(self, client):
        r = client.post("/v1/ab/test", json={"name": "Start Test"})
        tid = r.json()["id"]
        client.post(f"/v1/ab/test/{tid}/variant", json={"name": "A", "copy_text": "A"})
        client.post(f"/v1/ab/test/{tid}/variant", json={"name": "B", "copy_text": "B"})
        r2 = client.post(f"/v1/ab/test/{tid}/start")
        assert r2.status_code == 200
        assert r2.json()["status"] == "running"

    def test_record_and_analyze(self, client):
        r = client.post("/v1/ab/test", json={"name": "Record Analyze", "min_samples": 50})
        tid = r.json()["id"]
        rv_a = client.post(f"/v1/ab/test/{tid}/variant", json={"name": "A", "copy_text": "A"})
        rv_b = client.post(f"/v1/ab/test/{tid}/variant", json={"name": "B", "copy_text": "B"})
        vid_a = rv_a.json()["id"]
        vid_b = rv_b.json()["id"]

        client.post(f"/v1/ab/test/{tid}/record", json={"variant_id": vid_a, "impressions": 1000, "clicks": 80, "conversions": 40})
        client.post(f"/v1/ab/test/{tid}/record", json={"variant_id": vid_b, "impressions": 1000, "clicks": 50, "conversions": 20})

        r2 = client.get(f"/v1/ab/test/{tid}/analyze")
        assert r2.status_code == 200
        d = r2.json()
        assert "stat_result" in d
        assert "recommendation" in d

    def test_stop_test(self, client):
        r = client.post("/v1/ab/test", json={"name": "Stop Test"})
        tid = r.json()["id"]
        r2 = client.post(f"/v1/ab/test/{tid}/stop")
        assert r2.status_code == 200
        assert r2.json()["status"] == "stopped"

    def test_report_json(self, client):
        r = client.post("/v1/ab/test", json={"name": "Report JSON Test"})
        tid = r.json()["id"]
        client.post(f"/v1/ab/test/{tid}/variant", json={"name": "A", "copy_text": "A"})
        client.post(f"/v1/ab/test/{tid}/variant", json={"name": "B", "copy_text": "B"})
        r2 = client.get(f"/v1/ab/test/{tid}/report")
        assert r2.status_code == 200
        assert r2.json()["report_type"] == "ab_test"

    def test_report_md(self, client):
        r = client.post("/v1/ab/test", json={"name": "Report MD Test"})
        tid = r.json()["id"]
        client.post(f"/v1/ab/test/{tid}/variant", json={"name": "A", "copy_text": "A"})
        client.post(f"/v1/ab/test/{tid}/variant", json={"name": "B", "copy_text": "B"})
        r2 = client.get(f"/v1/ab/test/{tid}/report/md")
        assert r2.status_code == 200

    def test_delete_test(self, client):
        r = client.post("/v1/ab/test", json={"name": "Delete Test"})
        tid = r.json()["id"]
        r2 = client.delete(f"/v1/ab/test/{tid}")
        assert r2.status_code == 200
        assert client.get(f"/v1/ab/test/{tid}").status_code == 404

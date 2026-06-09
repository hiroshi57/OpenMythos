"""
Sprint 58 — LLMO ダッシュボード・CEP管理・競合分析 テスト (54 tests)

対象:
  open_mythos/skills/llmo_dashboard.py:
    CepCategory / CepEntry / CepStore
    MentionSnapshot / CompetitorEntry / CompetitorAnalysis
    LlmoDashboard / LlmoReportEngine
  serve/api.py:
    POST /v1/cep, GET /v1/cep, DELETE /v1/cep/{id}
    POST /v1/llmo/snapshot
    GET  /v1/llmo/dashboard/{brand}
    GET  /v1/llmo/dashboard/{brand}/report
    GET  /v1/llmo/dashboard/{brand}/trend
    POST /v1/llmo/competitor
    POST /v1/llmo/competitor/analyze
"""
from __future__ import annotations

import sys
import pytest
import torch
from unittest.mock import MagicMock

from open_mythos.skills.llmo_dashboard import (
    CepCategory, CepEntry, CepStore,
    MentionSnapshot, CompetitorEntry, CompetitorAnalysis,
    LlmoDashboard, LlmoReportEngine,
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
        prelude_layers=1, coda_layers=1,
        n_experts=1, n_shared_experts=0, n_experts_per_tok=1,
        expert_dim=32,
    )
    model = OpenMythos(cfg)
    api_mod.state.model = model
    api_mod.state.tokenizer = tok
    api_mod.state.llm = OpenMythosLLM(model=model, tokenizer=tok)
    api_mod.state.llm.stream = lambda p: iter(["test"])

    return TestClient(api_mod.app)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CepCategory (4 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCepCategory:
    def test_problem_value(self):
        assert CepCategory.PROBLEM.value == "problem"

    def test_comparison_value(self):
        assert CepCategory.COMPARISON.value == "comparison"

    def test_recommend_value(self):
        assert CepCategory.RECOMMEND.value == "recommend"

    def test_from_string(self):
        assert CepCategory("how_to") == CepCategory.HOW_TO


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CepStore (8 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCepStore:
    def _store(self):
        return CepStore()

    def test_initially_empty(self):
        assert self._store().count() == 0

    def test_add_entry(self):
        s = self._store()
        e = s.add("日焼けが気になって検索する30代女性")
        assert e.id
        assert s.count() == 1

    def test_add_with_category(self):
        s = self._store()
        e = s.add("おすすめを聞く", category=CepCategory.RECOMMEND)
        assert e.category == CepCategory.RECOMMEND

    def test_get_existing(self):
        s = self._store()
        e = s.add("テスト")
        found = s.get(e.id)
        assert found is not None
        assert found.scenario == "テスト"

    def test_get_nonexistent(self):
        assert self._store().get("nonexistent") is None

    def test_list_all(self):
        s = self._store()
        s.add("A", priority=2)
        s.add("B", priority=1)
        entries = s.list_all()
        assert entries[0].priority <= entries[1].priority

    def test_by_category(self):
        s = self._store()
        s.add("問題", category=CepCategory.PROBLEM)
        s.add("推薦", category=CepCategory.RECOMMEND)
        problems = s.by_category(CepCategory.PROBLEM)
        assert len(problems) == 1

    def test_delete(self):
        s = self._store()
        e = s.add("削除テスト")
        assert s.delete(e.id) is True
        assert s.count() == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MentionSnapshot (5 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMentionSnapshot:
    def _snap(self, m=0.5, c=0.3, r=0.2):
        return MentionSnapshot(id="s1", brand_name="Brand", mention_rate=m,
                               citation_rate=c, reference_rate=r)

    def test_overall_score_formula(self):
        snap = self._snap(m=1.0, c=1.0, r=1.0)
        assert snap.overall_score == pytest.approx(1.0, abs=0.001)

    def test_overall_score_zero(self):
        snap = self._snap(m=0, c=0, r=0)
        assert snap.overall_score == 0.0

    def test_overall_score_partial(self):
        snap = self._snap(m=0.5, c=0.0, r=0.0)
        assert snap.overall_score == pytest.approx(0.25, abs=0.001)

    def test_to_dict_keys(self):
        d = self._snap().to_dict()
        assert "mention_rate" in d
        assert "overall_score" in d

    def test_overall_in_range(self):
        snap = self._snap(m=0.6, c=0.4, r=0.2)
        assert 0.0 <= snap.overall_score <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CompetitorAnalysis (5 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCompetitorAnalysis:
    def _analysis(self, our=0.6, comp=0.4):
        return CompetitorAnalysis(
            brand_name="Us", competitor_name="Them",
            prompt="test", our_mention=our, comp_mention=comp,
            gap=round(our - comp, 4),
        )

    def test_is_winning_true(self):
        assert self._analysis(0.6, 0.4).is_winning

    def test_is_winning_false(self):
        assert not self._analysis(0.3, 0.5).is_winning

    def test_gap_label_big_lead(self):
        a = self._analysis(0.8, 0.4)
        assert a.gap_label == "大幅リード"

    def test_gap_label_losing(self):
        a = self._analysis(0.2, 0.6)
        assert "劣後" in a.gap_label

    def test_to_dict(self):
        d = self._analysis().to_dict()
        assert "gap" in d
        assert "is_winning" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LlmoDashboard (10 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLlmoDashboard:
    def _db(self):
        return LlmoDashboard(brand_name="SunGuard")

    def test_initial_snapshot_count(self):
        assert self._db().snapshot_count() == 0

    def test_add_snapshot(self):
        db = self._db()
        db.add_snapshot("テスト", mention_rate=0.4)
        assert db.snapshot_count() == 1

    def test_latest_snapshot(self):
        db = self._db()
        db.add_snapshot("A", mention_rate=0.3)
        db.add_snapshot("B", mention_rate=0.7)
        assert db.latest_snapshot().prompt == "B"

    def test_latest_score(self):
        db = self._db()
        db.add_snapshot("x", mention_rate=0.5, citation_rate=0.3, reference_rate=0.2)
        assert 0.0 < db.latest_score() <= 1.0

    def test_avg_mention_rate(self):
        db = self._db()
        db.add_snapshot("a", mention_rate=0.4)
        db.add_snapshot("b", mention_rate=0.6)
        assert db.avg_mention_rate() == pytest.approx(0.5, abs=0.001)

    def test_trend_length(self):
        db = self._db()
        db.add_snapshot("a", mention_rate=0.3)
        db.add_snapshot("b", mention_rate=0.5)
        assert len(db.trend()) == 2

    def test_trend_delta_positive(self):
        db = self._db()
        db.add_snapshot("a", mention_rate=0.3)
        db.add_snapshot("b", mention_rate=0.9)
        assert db.trend_delta() > 0

    def test_trend_delta_single(self):
        db = self._db()
        db.add_snapshot("only", mention_rate=0.5)
        assert db.trend_delta() == 0.0

    def test_add_competitor(self):
        db = self._db()
        comp = db.add_competitor("Rival Brand")
        assert comp.name == "Rival Brand"
        assert len(db.competitors()) == 1

    def test_analyze_competitor(self):
        db = self._db()
        comp = db.add_competitor("Rival")
        result = db.analyze_competitor(
            competitor_id=comp.id, prompt="test",
            our_mention=0.6, comp_mention=0.4,
        )
        assert result is not None
        assert result.is_winning


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LlmoReportEngine (4 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLlmoReportEngine:
    def _engine(self):
        db = LlmoDashboard("TestBrand")
        db.add_snapshot("test", mention_rate=0.5, citation_rate=0.3)
        return LlmoReportEngine(db)

    def test_summary_keys(self):
        s = self._engine().summary()
        assert "brand_name" in s
        assert "latest_score" in s
        assert "snapshots" in s

    def test_summary_brand_name(self):
        assert self._engine().summary()["brand_name"] == "TestBrand"

    def test_to_markdown_header(self):
        md = self._engine().to_markdown()
        assert "TestBrand" in md

    def test_to_markdown_metrics(self):
        md = self._engine().to_markdown()
        assert "言及" in md


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API エンドポイント (18 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLlmoDashboardAPI:
    # ── CEP ──────────────────────────────────────────────────────

    def test_cep_create_ok(self, client):
        r = client.post("/v1/cep", json={
            "scenario": "日焼けが気になって検索する30代女性",
            "category": "problem",
            "priority": 1,
        })
        assert r.status_code == 200
        data = r.json()
        assert "id" in data

    def test_cep_create_invalid_category(self, client):
        r = client.post("/v1/cep", json={
            "scenario": "テスト",
            "category": "invalid_cat",
        })
        assert r.status_code == 400

    def test_cep_list_ok(self, client):
        r = client.get("/v1/cep")
        assert r.status_code == 200
        data = r.json()
        assert "entries" in data
        assert "total" in data

    def test_cep_list_by_category(self, client):
        r = client.get("/v1/cep?category=problem")
        assert r.status_code == 200

    def test_cep_delete_ok(self, client):
        # 先に作成
        r = client.post("/v1/cep", json={"scenario": "削除テスト"})
        cep_id = r.json()["id"]
        # 削除
        rd = client.delete(f"/v1/cep/{cep_id}")
        assert rd.status_code == 200

    def test_cep_delete_not_found(self, client):
        r = client.delete("/v1/cep/nonexistent_id")
        assert r.status_code == 404

    # ── スナップショット ──────────────────────────────────────────

    def test_snapshot_add_ok(self, client):
        r = client.post("/v1/llmo/snapshot", json={
            "brand_name": "TestBrand",
            "prompt": "アウトドア向け日焼け止め",
            "mention_rate": 0.45,
            "citation_rate": 0.25,
            "reference_rate": 0.20,
        })
        assert r.status_code == 200
        data = r.json()
        assert "overall_score" in data

    def test_snapshot_rate_out_of_range(self, client):
        r = client.post("/v1/llmo/snapshot", json={
            "brand_name": "B",
            "prompt": "x",
            "mention_rate": 1.5,  # > 1.0
        })
        assert r.status_code == 422

    # ── ダッシュボード ────────────────────────────────────────────

    def test_dashboard_ok(self, client):
        client.post("/v1/llmo/snapshot", json={
            "brand_name": "DashBrand",
            "prompt": "test",
            "mention_rate": 0.5,
        })
        r = client.get("/v1/llmo/dashboard/DashBrand")
        assert r.status_code == 200
        data = r.json()
        assert data["brand_name"] == "DashBrand"

    def test_dashboard_report_ok(self, client):
        r = client.get("/v1/llmo/dashboard/DashBrand/report")
        assert r.status_code == 200
        assert "markdown" in r.json()

    def test_dashboard_trend_ok(self, client):
        r = client.get("/v1/llmo/dashboard/DashBrand/trend")
        assert r.status_code == 200
        data = r.json()
        assert "trend" in data
        assert "trend_delta" in data

    # ── 競合 ─────────────────────────────────────────────────────

    def test_competitor_add_ok(self, client):
        r = client.post("/v1/llmo/competitor", json={
            "brand_name": "TestBrand",
            "name": "RivalCo",
            "category": "cosmetics",
        })
        assert r.status_code == 200
        assert r.json()["name"] == "RivalCo"

    def test_competitor_analyze_ok(self, client):
        # 競合登録
        comp_r = client.post("/v1/llmo/competitor", json={
            "brand_name": "AnalyzeBrand",
            "name": "Competitor X",
        })
        comp_id = comp_r.json()["id"]

        r = client.post("/v1/llmo/competitor/analyze", json={
            "brand_name": "AnalyzeBrand",
            "competitor_id": comp_id,
            "prompt": "日焼け止めのおすすめは？",
            "our_mention": 0.55,
            "comp_mention": 0.40,
        })
        assert r.status_code == 200
        data = r.json()
        assert "gap" in data
        assert "is_winning" in data

    def test_competitor_analyze_not_found(self, client):
        r = client.post("/v1/llmo/competitor/analyze", json={
            "brand_name": "NoBrand",
            "competitor_id": "nonexistent",
            "prompt": "test",
            "our_mention": 0.5,
            "comp_mention": 0.3,
        })
        assert r.status_code == 404

    def test_competitor_analyze_winning(self, client):
        comp_r = client.post("/v1/llmo/competitor", json={
            "brand_name": "WinBrand", "name": "LoseBrand"
        })
        comp_id = comp_r.json()["id"]
        r = client.post("/v1/llmo/competitor/analyze", json={
            "brand_name": "WinBrand",
            "competitor_id": comp_id,
            "prompt": "x", "our_mention": 0.8, "comp_mention": 0.2,
        })
        assert r.json()["is_winning"] is True

    def test_competitor_analyze_losing(self, client):
        comp_r = client.post("/v1/llmo/competitor", json={
            "brand_name": "LoseBrand2", "name": "WinBrand2"
        })
        comp_id = comp_r.json()["id"]
        r = client.post("/v1/llmo/competitor/analyze", json={
            "brand_name": "LoseBrand2",
            "competitor_id": comp_id,
            "prompt": "x", "our_mention": 0.2, "comp_mention": 0.8,
        })
        assert r.json()["is_winning"] is False

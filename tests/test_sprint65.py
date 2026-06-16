"""
Sprint 65 — Fusion マルチモデル融合 テスト

対象:
  open_mythos/skills/fusion.py:
    FusionRole / CandidateSpec / FusionConfig
    CandidateResponse / CandidateAnalysis / FusionAnalysis / FusionResult
    FusionAnalysisParser
    JudgeAnalyzer
    FusionEngine / FusionEngineFactory
"""
from __future__ import annotations

import json
import pytest

from open_mythos.skills.fusion import (
    FusionRole, CandidateSpec, FusionConfig,
    CandidateResponse, CandidateAnalysis, FusionAnalysis, FusionResult,
    FusionAnalysisParser,
    JudgeAnalyzer,
    FusionEngine, FusionEngineFactory,
    _heuristic_score,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum / 設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFusionRole:
    def test_values(self):
        assert FusionRole.CANDIDATE.value == "candidate"
        assert FusionRole.JUDGE.value     == "judge"
        assert FusionRole.CALLER.value    == "caller"


class TestCandidateSpec:
    def test_defaults(self):
        s = CandidateSpec(label="c1")
        assert s.temperature == 0.7
        assert s.preferred_provider is None

    def test_custom(self):
        s = CandidateSpec(label="c1", preferred_provider="claude", temperature=0.3)
        assert s.preferred_provider == "claude"
        assert s.temperature == 0.3


class TestFusionConfig:
    def test_default_has_3_candidates(self):
        cfg = FusionConfig.default()
        assert len(cfg.candidates) == 3

    def test_judge_temperature_low(self):
        cfg = FusionConfig.default()
        assert cfg.judge_temperature < 0.5

    def test_custom_config(self):
        cfg = FusionConfig(candidates=[CandidateSpec(label="x")])
        assert len(cfg.candidates) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCandidateResponse:
    def test_success(self):
        c = CandidateResponse(label="c1", text="回答")
        assert c.success is True

    def test_failure_on_error(self):
        c = CandidateResponse(label="c1", text="", error="timeout")
        assert c.success is False

    def test_failure_on_empty(self):
        c = CandidateResponse(label="c1", text="")
        assert c.success is False

    def test_to_dict(self):
        c = CandidateResponse(label="c1", text="回答", provider_used="claude")
        d = c.to_dict()
        assert d["label"] == "c1"
        assert d["success"] is True


class TestCandidateAnalysis:
    def test_to_dict(self):
        ca = CandidateAnalysis(label="c1", score=0.8, strengths=["明確"])
        d = ca.to_dict()
        assert d["score"] == 0.8
        assert d["strengths"] == ["明確"]


class TestFusionAnalysis:
    def _make(self):
        return FusionAnalysis(
            candidate_analyses=[
                CandidateAnalysis(label="c1", score=0.9),
                CandidateAnalysis(label="c2", score=0.5),
            ],
            ranking=["c1", "c2"],
            synthesis_guidance="c1 をベースに",
        )

    def test_best_label(self):
        assert self._make().best_label() == "c1"

    def test_best_label_empty(self):
        a = FusionAnalysis(candidate_analyses=[], ranking=[])
        assert a.best_label() is None

    def test_get_analysis(self):
        a = self._make()
        assert a.get_analysis("c1").score == 0.9
        assert a.get_analysis("nope") is None

    def test_to_dict(self):
        d = self._make().to_dict()
        assert d["ranking"] == ["c1", "c2"]
        assert len(d["candidate_analyses"]) == 2


class TestFusionResult:
    def test_to_dict(self):
        analysis = FusionAnalysis(candidate_analyses=[], ranking=[])
        r = FusionResult(
            final_answer="最終回答",
            analysis=analysis,
            candidates=[],
        )
        d = r.to_dict()
        assert d["final_answer"] == "最終回答"
        assert "analysis" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _heuristic_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHeuristicScore:
    def test_empty_zero(self):
        assert _heuristic_score("") == 0.0

    def test_longer_higher(self):
        short = _heuristic_score("短い。")
        long = _heuristic_score("これは長い回答です。" * 10)
        assert long > short

    def test_bounded(self):
        score = _heuristic_score("文。" * 1000)
        assert 0.0 <= score <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FusionAnalysisParser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_VALID_ANALYSIS = json.dumps({
    "analyses": [
        {"label": "c1", "score": 0.9, "strengths": ["正確"], "weaknesses": [], "key_points": ["要点A"]},
        {"label": "c2", "score": 0.6, "strengths": [], "weaknesses": ["冗長"], "key_points": []},
    ],
    "ranking": ["c1", "c2"],
    "synthesis_guidance": "c1 を主軸に c2 の要点を補完",
})


class TestFusionAnalysisParser:
    def setup_method(self):
        self.parser = FusionAnalysisParser()

    def test_parse_valid_json(self):
        a = self.parser.parse(_VALID_ANALYSIS, ["c1", "c2"])
        assert a.ranking == ["c1", "c2"]
        assert a.get_analysis("c1").score == 0.9

    def test_parse_synthesis_guidance(self):
        a = self.parser.parse(_VALID_ANALYSIS, ["c1", "c2"])
        assert "c1" in a.synthesis_guidance

    def test_parse_code_block(self):
        raw = f"```json\n{_VALID_ANALYSIS}\n```"
        a = self.parser.parse(raw, ["c1", "c2"])
        assert a.get_analysis("c1").score == 0.9

    def test_parse_fallback_on_garbage(self):
        a = self.parser.parse("解析不能なテキスト", ["c1", "c2"])
        # フォールバック: 均等スコア
        assert len(a.candidate_analyses) == 2
        assert all(ca.score == 0.5 for ca in a.candidate_analyses)

    def test_parse_score_clamped(self):
        data = json.dumps({"analyses": [{"label": "c1", "score": 5.0}], "ranking": ["c1"]})
        a = self.parser.parse(data, ["c1"])
        assert a.get_analysis("c1").score == 1.0

    def test_parse_ranking_auto_generated(self):
        # ranking 欠如 → スコア降順で生成
        data = json.dumps({"analyses": [
            {"label": "c1", "score": 0.3},
            {"label": "c2", "score": 0.8},
        ]})
        a = self.parser.parse(data, ["c1", "c2"])
        assert a.ranking[0] == "c2"

    def test_parse_empty_analyses_fallback(self):
        data = json.dumps({"analyses": [], "ranking": []})
        a = self.parser.parse(data, ["c1", "c2"])
        assert len(a.candidate_analyses) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JudgeAnalyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJudgeAnalyzerHeuristic:
    def setup_method(self):
        self.judge = JudgeAnalyzer(router=None)

    def test_heuristic_analysis(self):
        candidates = [
            CandidateResponse(label="c1", text="これは詳しい回答です。要点が複数あります。具体例も豊富です。"),
            CandidateResponse(label="c2", text="短い。"),
        ]
        analysis = self.judge.analyze("質問", candidates, FusionConfig.default())
        # 長い c1 が上位
        assert analysis.ranking[0] == "c1"

    def test_heuristic_provider(self):
        candidates = [CandidateResponse(label="c1", text="回答テキスト。")]
        analysis = self.judge.analyze("質問", candidates, FusionConfig.default())
        assert analysis.judge_provider == "heuristic"

    def test_all_candidates_analyzed(self):
        candidates = [
            CandidateResponse(label="c1", text="回答1。"),
            CandidateResponse(label="c2", text="回答2。"),
            CandidateResponse(label="c3", text="回答3。"),
        ]
        analysis = self.judge.analyze("質問", candidates, FusionConfig.default())
        assert len(analysis.candidate_analyses) == 3


class TestJudgeAnalyzerLLM:
    def test_llm_analysis(self):
        from open_mythos.skills.fusion import _MockFusionRouter
        router = _MockFusionRouter([_VALID_ANALYSIS])
        judge = JudgeAnalyzer(router=router)
        candidates = [
            CandidateResponse(label="c1", text="回答1"),
            CandidateResponse(label="c2", text="回答2"),
        ]
        analysis = judge.analyze("質問", candidates, FusionConfig.default())
        assert analysis.ranking == ["c1", "c2"]
        assert analysis.get_analysis("c1").score == 0.9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FusionEngine — rule-based
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFusionEngineRuleBased:
    def setup_method(self):
        self.engine = FusionEngineFactory.rule_based()

    def test_has_llm_false(self):
        assert self.engine.has_llm is False

    def test_run_returns_result(self):
        result = self.engine.run("テスト質問")
        assert isinstance(result, FusionResult)

    def test_run_produces_candidates(self):
        result = self.engine.run("テスト質問")
        assert len(result.candidates) == 3  # デフォルト 3 候補

    def test_run_fallback_used(self):
        result = self.engine.run("テスト質問")
        assert result.fallback_used is True

    def test_final_answer_not_empty(self):
        result = self.engine.run("テスト質問")
        assert len(result.final_answer) > 0

    def test_analysis_present(self):
        result = self.engine.run("テスト質問")
        assert len(result.analysis.candidate_analyses) == 3

    def test_custom_config(self):
        cfg = FusionConfig(candidates=[
            CandidateSpec(label="solo"),
        ])
        engine = FusionEngineFactory.rule_based(config=cfg)
        result = engine.run("質問")
        assert len(result.candidates) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FusionEngine — LLM モック (フルパイプライン)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFusionEngineWithMock:
    def setup_method(self):
        # 候補3つ + 審査JSON + 最終回答
        responses = [
            "候補1の回答",
            "候補2の回答",
            "候補3の回答",
            _VALID_ANALYSIS,
            "これが合成された最終回答です。",
        ]
        cfg = FusionConfig(candidates=[
            CandidateSpec(label="c1"),
            CandidateSpec(label="c2"),
            CandidateSpec(label="c3"),
        ])
        self.engine = FusionEngineFactory.from_mock(responses, config=cfg)

    def test_has_llm_true(self):
        assert self.engine.has_llm is True

    def test_full_pipeline(self):
        result = self.engine.run("質問")
        assert isinstance(result, FusionResult)

    def test_candidates_from_llm(self):
        result = self.engine.run("質問")
        assert result.candidates[0].text == "候補1の回答"
        assert result.candidates[0].provider_used == "mock"

    def test_final_answer_synthesized(self):
        result = self.engine.run("質問")
        assert result.final_answer == "これが合成された最終回答です。"

    def test_not_fallback(self):
        result = self.engine.run("質問")
        assert result.fallback_used is False

    def test_analysis_from_judge(self):
        result = self.engine.run("質問")
        assert result.analysis.ranking == ["c1", "c2"]

    def test_to_dict(self):
        result = self.engine.run("質問")
        d = result.to_dict()
        assert "final_answer" in d
        assert "analysis" in d
        assert "candidates" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FusionEngineFactory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFusionEngineFactory:
    def test_rule_based(self):
        engine = FusionEngineFactory.rule_based()
        assert isinstance(engine, FusionEngine)
        assert engine.has_llm is False

    def test_from_mock(self):
        engine = FusionEngineFactory.from_mock(["a", "b"])
        assert isinstance(engine, FusionEngine)
        assert engine.has_llm is True

    def test_from_env(self):
        # API キーなしでも構築できる
        engine = FusionEngineFactory.from_env()
        assert isinstance(engine, FusionEngine)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serve.api import app
    return TestClient(app)


class TestFusionApi:
    def test_status(self, client):
        resp = client.get("/v1/fusion/status")
        assert resp.status_code == 200
        assert "has_llm" in resp.json()
        assert resp.json()["default_candidates"] == 3

    def test_run_default(self, client):
        resp = client.post("/v1/fusion/run", json={"question": "テスト質問です"})
        assert resp.status_code == 200
        data = resp.json()
        assert "final_answer" in data
        assert "analysis" in data
        assert "candidates" in data

    def test_run_custom_candidates(self, client):
        resp = client.post("/v1/fusion/run", json={
            "question": "Pythonの利点は？",
            "candidates": [
                {"label": "a", "temperature": 0.5},
                {"label": "b", "temperature": 0.9},
            ],
        })
        assert resp.status_code == 200
        assert len(resp.json()["candidates"]) == 2

    def test_run_produces_answer(self, client):
        resp = client.post("/v1/fusion/run", json={"question": "質問"})
        assert len(resp.json()["final_answer"]) > 0

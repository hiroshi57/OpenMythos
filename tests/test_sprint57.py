"""
Sprint 57 — LLM 評価フレームワーク テスト (54 tests)

対象:
  open_mythos/skills/evaluation.py:
    EvalMetric / EvalSample / EvalResult / BenchmarkReport
    TextEvaluator / AdEvaluator / BenchmarkRunner / EvalLeaderboard
  serve/api.py:
    POST /v1/eval/benchmark
    POST /v1/eval/benchmark/md
    POST /v1/eval/leaderboard
"""
from __future__ import annotations

import sys
import pytest
import torch
from unittest.mock import MagicMock

from open_mythos.skills.evaluation import (
    EvalMetric, EvalSample, EvalResult, BenchmarkReport,
    TextEvaluator, AdEvaluator, BenchmarkRunner, EvalLeaderboard,
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
    api_mod.state.llm.stream = lambda p: iter(["test ", "stream"])

    return TestClient(api_mod.app)


def _sample(idx=0, pred="hello world", ref="hello"):
    return EvalSample(id=f"s{idx}", input="test", prediction=pred, reference=ref)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EvalMetric (4 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEvalMetric:
    def test_default_direction(self):
        m = EvalMetric(name="bleu")
        assert m.is_higher_better()

    def test_lower_direction(self):
        m = EvalMetric(name="perplexity", direction="lower")
        assert not m.is_higher_better()

    def test_weight_default(self):
        assert EvalMetric(name="x").weight == 1.0

    def test_custom_weight(self):
        m = EvalMetric(name="y", weight=0.5)
        assert m.weight == 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EvalSample / EvalResult (5 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEvalDataClasses:
    def test_sample_fields(self):
        s = EvalSample(id="s1", input="Q", prediction="A", reference="ref")
        assert s.id == "s1"
        assert s.prediction == "A"

    def test_sample_no_reference(self):
        s = EvalSample(id="s2", input="Q", prediction="A")
        assert s.reference is None

    def test_result_top_metric(self):
        r = EvalResult(sample_id="s1", scores={"a": 0.8, "b": 0.3}, overall=0.55)
        assert r.top_metric() == "a"

    def test_result_top_metric_empty(self):
        r = EvalResult(sample_id="s1")
        assert r.top_metric() is None

    def test_result_overall_range(self):
        r = EvalResult(sample_id="s1", overall=0.75)
        assert 0.0 <= r.overall <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BenchmarkReport (7 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBenchmarkReport:
    def _report(self):
        r1 = EvalResult(sample_id="s1", scores={"bleu": 0.8}, overall=0.8)
        r2 = EvalResult(sample_id="s2", scores={"bleu": 0.6}, overall=0.6)
        return BenchmarkReport(
            run_id="abc", model_name="test",
            total_samples=2, results=[r1, r2],
            metrics_used=["bleu"],
        )

    def test_avg_overall(self):
        r = self._report()
        assert r.avg_overall == pytest.approx(0.7, abs=1e-4)

    def test_avg_overall_empty(self):
        r = BenchmarkReport(run_id="x", model_name="m", total_samples=0)
        assert r.avg_overall == 0.0

    def test_metric_avg(self):
        r = self._report()
        assert r.metric_avg("bleu") == pytest.approx(0.7, abs=1e-4)

    def test_to_dict_keys(self):
        d = self._report().to_dict()
        assert "avg_overall" in d
        assert "model_name" in d
        assert "total_samples" in d

    def test_to_markdown_contains_header(self):
        md = self._report().to_markdown()
        assert "Benchmark Report" in md

    def test_to_markdown_contains_metric(self):
        md = self._report().to_markdown()
        assert "bleu" in md

    def test_avg_latency(self):
        r1 = EvalResult(sample_id="s1", overall=0.5, latency_ms=10.0)
        r2 = EvalResult(sample_id="s2", overall=0.7, latency_ms=20.0)
        report = BenchmarkReport(
            run_id="x", model_name="m",
            total_samples=2, results=[r1, r2],
        )
        assert report.avg_latency_ms == pytest.approx(15.0, abs=0.1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TextEvaluator (10 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTextEvaluator:
    def _ev(self):
        return TextEvaluator()

    def test_evaluate_returns_result(self):
        r = self._ev().evaluate(_sample())
        assert isinstance(r, EvalResult)

    def test_overall_in_range(self):
        r = self._ev().evaluate(_sample())
        assert 0.0 <= r.overall <= 1.0

    def test_scores_have_all_metrics(self):
        r = self._ev().evaluate(_sample())
        for m in ["bleu_1", "rouge_1", "length", "diversity"]:
            assert m in r.scores

    def test_bleu1_identical(self):
        score = TextEvaluator._bleu1("hello", "hello")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_bleu1_no_overlap(self):
        score = TextEvaluator._bleu1("abc", "xyz")
        assert score == 0.0

    def test_bleu1_no_reference(self):
        score = TextEvaluator._bleu1("hello", "")
        assert score == 0.5

    def test_rouge1_identical(self):
        score = TextEvaluator._rouge1("hello", "hello")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_length_score_golden_zone(self):
        score = TextEvaluator._length_score("0123456789012345")  # 16文字
        assert score == 1.0

    def test_length_score_too_short(self):
        score = TextEvaluator._length_score("hi")  # 2文字
        assert score < 0.5

    def test_ttr_all_unique(self):
        score = TextEvaluator._ttr("abcde")
        assert score == pytest.approx(1.0, abs=0.01)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AdEvaluator (8 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAdEvaluator:
    def _ev(self, kws=None):
        return AdEvaluator(brand_keywords=kws)

    def test_evaluate_returns_result(self):
        r = self._ev().evaluate(_sample(pred="塗って、忘れて、思い切り楽しむ。"))
        assert isinstance(r, EvalResult)

    def test_ad_metrics_present(self):
        r = self._ev().evaluate(_sample(pred="テスト広告"))
        for m in ["llmo_total", "ctr_potential", "brand_fit"]:
            assert m in r.scores

    def test_brand_fit_with_keyword(self):
        ev = self._ev(kws=["日焼け止め"])
        s = EvalSample(id="x", input="", prediction="日焼け止めを選ぼう")
        r = ev.evaluate(s)
        assert r.scores["brand_fit"] == pytest.approx(1.0, abs=0.01)

    def test_brand_fit_without_keyword(self):
        ev = self._ev()
        r = ev.evaluate(_sample(pred="テスト"))
        assert r.scores["brand_fit"] == pytest.approx(0.5, abs=0.01)

    def test_ctr_with_action_word(self):
        ev = self._ev()
        s = EvalSample(id="x", input="", prediction="今すぐ試してみよう")
        r = ev.evaluate(s)
        assert r.scores["ctr_potential"] > 0.3

    def test_llmo_total_range(self):
        r = self._ev().evaluate(_sample(pred="素肌のままで、全部やる。"))
        assert 0.0 <= r.scores["llmo_total"] <= 1.0

    def test_overall_in_range(self):
        r = self._ev().evaluate(_sample(pred="アウトドア派の、静かな自信。"))
        assert 0.0 <= r.overall <= 1.0

    def test_multiple_keywords(self):
        ev = AdEvaluator(brand_keywords=["日焼け止め", "アウトドア"])
        s = EvalSample(id="x", input="", prediction="日焼け止めでアウトドアを楽しもう")
        r = ev.evaluate(s)
        assert r.scores["brand_fit"] == pytest.approx(1.0, abs=0.01)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BenchmarkRunner (6 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBenchmarkRunner:
    def _samples(self):
        return [
            EvalSample(id="s1", input="Q", prediction="塗って忘れて楽しむ"),
            EvalSample(id="s2", input="Q", prediction="素肌のままで全部やる"),
        ]

    def test_run_returns_report(self):
        runner = BenchmarkRunner(evaluator=AdEvaluator())
        report = runner.run(self._samples(), model_name="test")
        assert isinstance(report, BenchmarkReport)

    def test_report_total_samples(self):
        runner = BenchmarkRunner()
        report = runner.run(self._samples())
        assert report.total_samples == 2

    def test_report_has_results(self):
        runner = BenchmarkRunner()
        report = runner.run(self._samples())
        assert len(report.results) == 2

    def test_compare_returns_leaderboard(self):
        runner = BenchmarkRunner()
        samples = self._samples()
        board = runner.compare(
            samples,
            model_names=["modelA", "modelB"],
            predictions_by_model={
                "modelA": ["prediction A1", "prediction A2"],
                "modelB": ["prediction B1", "prediction B2"],
            },
        )
        assert isinstance(board, EvalLeaderboard)

    def test_leaderboard_has_winner(self):
        runner = BenchmarkRunner()
        board = runner.compare(
            self._samples(),
            model_names=["A", "B"],
            predictions_by_model={"A": ["塗って忘れて楽しむ", "ok"], "B": ["test", "ok"]},
        )
        assert board.winner() in ["A", "B"]

    def test_leaderboard_rankings_sorted(self):
        runner = BenchmarkRunner()
        board = runner.compare(
            self._samples(),
            model_names=["A", "B"],
            predictions_by_model={"A": ["x", "x"], "B": ["y", "y"]},
        )
        r = board.rankings()
        scores = [e["avg_overall"] for e in r]
        assert scores == sorted(scores, reverse=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EvalLeaderboard (4 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEvalLeaderboard:
    def _board(self):
        r1 = BenchmarkReport("x","A",1,[EvalResult("s1",overall=0.8)],["bleu_1"])
        r2 = BenchmarkReport("y","B",1,[EvalResult("s2",overall=0.5)],["bleu_1"])
        return EvalLeaderboard({"A": r1, "B": r2})

    def test_winner(self):
        assert self._board().winner() == "A"

    def test_rankings_length(self):
        assert len(self._board().rankings()) == 2

    def test_rankings_first_highest(self):
        ranks = self._board().rankings()
        assert ranks[0]["avg_overall"] >= ranks[1]["avg_overall"]

    def test_to_markdown(self):
        md = self._board().to_markdown()
        assert "Leaderboard" in md
        assert "A" in md


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API エンドポイント (10 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_DATA = [
    {"id": "s1", "input": "日焼け止めの広告を作って",
     "prediction": "塗って、忘れて、思い切り楽しむ。", "reference": "守られてる感ゼロ"},
    {"id": "s2", "input": "テスト",
     "prediction": "素肌のままで、全部やる。", "reference": None},
]


class TestEvalAPI:
    def test_benchmark_text_ok(self, client):
        r = client.post("/v1/eval/benchmark", json={
            "samples": SAMPLE_DATA,
            "model_name": "test_model",
            "evaluator_type": "text",
        })
        assert r.status_code == 200

    def test_benchmark_ad_ok(self, client):
        r = client.post("/v1/eval/benchmark", json={
            "samples": SAMPLE_DATA,
            "model_name": "test_model",
            "evaluator_type": "ad",
            "brand_keywords": ["日焼け止め", "アウトドア"],
        })
        assert r.status_code == 200

    def test_benchmark_response_fields(self, client):
        r = client.post("/v1/eval/benchmark", json={
            "samples": SAMPLE_DATA,
            "model_name": "m",
        })
        data = r.json()
        assert "avg_overall" in data
        assert "total_samples" in data
        assert data["total_samples"] == 2

    def test_benchmark_avg_overall_range(self, client):
        r = client.post("/v1/eval/benchmark", json={"samples": SAMPLE_DATA})
        data = r.json()
        assert 0.0 <= data["avg_overall"] <= 1.0

    def test_benchmark_md_ok(self, client):
        r = client.post("/v1/eval/benchmark/md", json={
            "samples": SAMPLE_DATA,
            "model_name": "test",
        })
        assert r.status_code == 200
        data = r.json()
        assert "markdown" in data
        assert "avg_overall" in data

    def test_benchmark_md_contains_header(self, client):
        r = client.post("/v1/eval/benchmark/md", json={"samples": SAMPLE_DATA})
        md = r.json()["markdown"]
        assert "Benchmark Report" in md

    def test_leaderboard_ok(self, client):
        r = client.post("/v1/eval/leaderboard", json={
            "samples": SAMPLE_DATA,
            "models": ["modelA", "modelB"],
            "predictions": {
                "modelA": ["塗って忘れて楽しむ", "ok"],
                "modelB": ["test text here", "more text"],
            },
        })
        assert r.status_code == 200

    def test_leaderboard_has_winner(self, client):
        r = client.post("/v1/eval/leaderboard", json={
            "samples": SAMPLE_DATA,
            "models": ["A", "B"],
            "predictions": {"A": ["hi there", "ok"], "B": ["test", "x"]},
        })
        data = r.json()
        assert "winner" in data
        assert data["winner"] in ["A", "B"]

    def test_leaderboard_rankings(self, client):
        r = client.post("/v1/eval/leaderboard", json={
            "samples": SAMPLE_DATA,
            "models": ["A", "B"],
            "predictions": {"A": ["a", "b"], "B": ["c", "d"]},
        })
        data = r.json()
        assert "rankings" in data
        assert len(data["rankings"]) == 2

    def test_benchmark_empty_samples_fails(self, client):
        r = client.post("/v1/eval/benchmark", json={"samples": []})
        assert r.status_code == 422

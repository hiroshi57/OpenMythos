"""
Sprint 9 テスト

9.1.1  scripts/eval_marketing.py — CTR/CVR/コンテンツ品質/ペルソナ/広告 Tier 評価
9.1.2  serve/ab_router.py — _significance_test / /ab/stats p_value フィールド
9.2.1  serve/api.py — POST /v1/batch バッチ推論
9.3.1  pyproject.toml 0.13.0
"""

from __future__ import annotations

import math

import pytest
import torch

# ===========================================================================
# 9.1.1  scripts.eval_marketing — 評価メトリクス
# ===========================================================================


class TestBasicMetrics:
    def test_mae_correct(self):
        from scripts.eval_marketing import _mae

        assert _mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)
        assert _mae([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)

    def test_rmse_correct(self):
        from scripts.eval_marketing import _rmse

        # [0,2] vs [0,0] → errors=[0,2] → mse=2 → rmse=√2
        assert _rmse([0.0, 2.0], [0.0, 0.0]) == pytest.approx(math.sqrt(2))

    def test_spearman_perfect(self):
        from scripts.eval_marketing import _spearman_rho

        rho = _spearman_rho([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        assert rho == pytest.approx(1.0, abs=1e-9)

    def test_spearman_reversed(self):
        from scripts.eval_marketing import _spearman_rho

        rho = _spearman_rho([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
        assert rho == pytest.approx(-1.0, abs=1e-9)

    def test_accuracy_all_correct(self):
        from scripts.eval_marketing import _accuracy

        assert _accuracy(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_accuracy_none_correct(self):
        from scripts.eval_marketing import _accuracy

        assert _accuracy(["a", "a"], ["b", "b"]) == pytest.approx(0.0)

    def test_precision_recall_f1_basic(self):
        from scripts.eval_marketing import _precision_recall_f1

        # TP=2, FP=1, FN=1
        preds = [1, 1, 1, 0]
        actuals = [1, 1, 0, 1]
        p, r, f1 = _precision_recall_f1(preds, actuals)
        assert p == pytest.approx(2 / 3, abs=1e-6)
        assert r == pytest.approx(2 / 3, abs=1e-6)
        assert f1 == pytest.approx(2 / 3, abs=1e-6)

    def test_mae_empty_returns_nan(self):
        from scripts.eval_marketing import _mae

        assert math.isnan(_mae([], []))


class TestEvaluateCtrPrediction:
    def test_returns_dict(self):
        from scripts.eval_marketing import evaluate_ctr_prediction

        records = [
            {"predicted_ctr": 0.03, "actual_ctr": 0.028},
            {"predicted_ctr": 0.05, "actual_ctr": 0.045},
        ]
        result = evaluate_ctr_prediction(records)
        assert isinstance(result, dict)
        assert "ctr_mae" in result
        assert "ctr_rmse" in result
        assert "ctr_spearman" in result

    def test_n_correct(self):
        from scripts.eval_marketing import evaluate_ctr_prediction

        records = [{"predicted_ctr": 0.03, "actual_ctr": 0.03}] * 5
        result = evaluate_ctr_prediction(records)
        assert result["n"] == 5

    def test_perfect_prediction_mae_zero(self):
        from scripts.eval_marketing import evaluate_ctr_prediction

        records = [{"predicted_ctr": v, "actual_ctr": v} for v in [0.01, 0.02, 0.03]]
        result = evaluate_ctr_prediction(records)
        assert result["ctr_mae"] == pytest.approx(0.0, abs=1e-9)

    def test_cvr_optional(self):
        from scripts.eval_marketing import evaluate_ctr_prediction

        # CVR なしでもエラーにならない
        records = [{"predicted_ctr": 0.03, "actual_ctr": 0.03}]
        result = evaluate_ctr_prediction(records)
        assert "cvr_mae" not in result

    def test_with_cvr_and_roas(self):
        from scripts.eval_marketing import evaluate_ctr_prediction

        records = [
            {
                "predicted_ctr": 0.03,
                "actual_ctr": 0.028,
                "predicted_cvr": 0.012,
                "actual_cvr": 0.011,
                "predicted_roas": 4.5,
                "actual_roas": 4.2,
            }
        ]
        result = evaluate_ctr_prediction(records)
        assert "cvr_mae" in result
        assert "roas_mae" in result


class TestEvaluateContentQuality:
    def test_score_mae(self):
        from scripts.eval_marketing import evaluate_content_quality

        records = [
            {"predicted_score": 4.0, "actual_score": 4.0},
            {"predicted_score": 3.0, "actual_score": 4.0},
        ]
        result = evaluate_content_quality(records)
        assert result["score_mae"] == pytest.approx(0.5, abs=1e-9)

    def test_llmo_optional(self):
        from scripts.eval_marketing import evaluate_content_quality

        records = [{"predicted_score": 4.0, "actual_score": 3.5}]
        result = evaluate_content_quality(records)
        assert "llmo_mae" not in result


class TestEvaluatePersonaClassification:
    def test_accuracy_correct(self):
        from scripts.eval_marketing import evaluate_persona_classification

        records = [
            {"predicted_segment": "A", "actual_segment": "A"},
            {"predicted_segment": "B", "actual_segment": "B"},
            {"predicted_segment": "A", "actual_segment": "B"},
        ]
        result = evaluate_persona_classification(records)
        assert result["accuracy"] == pytest.approx(2 / 3, abs=1e-9)
        assert result["n_classes"] == 2.0  # actual に A と B

    def test_n_classes(self):
        from scripts.eval_marketing import evaluate_persona_classification

        records = [
            {"predicted_segment": "X", "actual_segment": "X"},
            {"predicted_segment": "Y", "actual_segment": "Y"},
            {"predicted_segment": "Z", "actual_segment": "Z"},
        ]
        result = evaluate_persona_classification(records)
        assert result["n_classes"] == 3.0


class TestEvaluateAdPerformanceTier:
    def test_all_correct(self):
        from scripts.eval_marketing import evaluate_ad_performance_tier

        records = [
            {"predicted_tier": "high", "actual_tier": "high"},
            {"predicted_tier": "medium", "actual_tier": "medium"},
        ]
        result = evaluate_ad_performance_tier(records)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_high_tier_f1_all_correct(self):
        from scripts.eval_marketing import evaluate_ad_performance_tier

        records = [
            {"predicted_tier": "high", "actual_tier": "high"},
            {"predicted_tier": "low", "actual_tier": "low"},
        ]
        result = evaluate_ad_performance_tier(records)
        assert result["high_tier_f1"] == pytest.approx(1.0)

    def test_task_evaluators_dict(self):
        from scripts.eval_marketing import TASK_EVALUATORS

        assert set(TASK_EVALUATORS.keys()) == {
            "ctr_prediction",
            "content_quality",
            "persona_classification",
            "ad_performance_tier",
        }


class TestRunEvaluation:
    def test_saves_csv(self, tmp_path):
        from scripts.eval_marketing import run_evaluation

        records = [
            {"predicted_ctr": 0.03, "actual_ctr": 0.028, "task": "ctr_prediction"},
        ]
        results = run_evaluation("ctr_prediction", records, tmp_path)
        assert "ctr_prediction" in results
        csv_file = tmp_path / "marketing_eval_ctr_prediction.csv"
        assert csv_file.exists()

    def test_all_tasks_run(self, tmp_path):
        from scripts.eval_marketing import run_evaluation

        records = [
            {"predicted_ctr": 0.03, "actual_ctr": 0.03},
            {"predicted_score": 4.0, "actual_score": 4.0},
            {
                "predicted_segment": "A",
                "actual_segment": "A",
            },
            {"predicted_tier": "high", "actual_tier": "high"},
        ]
        results = run_evaluation("all", records, tmp_path)
        assert len(results) == 4


# ===========================================================================
# 9.1.2  serve.ab_router — _significance_test
# ===========================================================================


class TestSignificanceTest:
    def test_identical_groups_not_significant(self):
        from serve.ab_router import _significance_test

        a = [0.8] * 20
        b = [0.8] * 20
        result = _significance_test(a, b)
        assert result["significant"] is False
        assert result["p_value"] > 0.05

    def test_very_different_groups_significant(self):
        from serve.ab_router import _significance_test

        a = [0.9] * 50
        b = [0.1] * 50
        result = _significance_test(a, b)
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_small_sample_not_significant(self):
        from serve.ab_router import _significance_test

        # n < 2 → p=1.0, not significant
        result = _significance_test([0.8], [0.2])
        assert result["significant"] is False
        assert result["p_value"] == pytest.approx(1.0)

    def test_empty_group_not_significant(self):
        from serve.ab_router import _significance_test

        result = _significance_test([], [0.8, 0.9])
        assert result["significant"] is False
        assert result["p_value"] == pytest.approx(1.0)

    def test_returns_means(self):
        from serve.ab_router import _significance_test

        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = _significance_test(a, b)
        assert result["mean_a"] == pytest.approx(2.0, abs=1e-5)
        assert result["mean_b"] == pytest.approx(5.0, abs=1e-5)

    def test_returns_n(self):
        from serve.ab_router import _significance_test

        result = _significance_test([0.5] * 10, [0.6] * 8)
        assert result["n_a"] == 10
        assert result["n_b"] == 8

    def test_p_value_in_range(self):
        from serve.ab_router import _significance_test

        import random

        rng = random.Random(42)
        a = [rng.gauss(0.7, 0.1) for _ in range(30)]
        b = [rng.gauss(0.5, 0.1) for _ in range(30)]
        result = _significance_test(a, b)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_ab_stats_has_significance(self):
        """_Stats に 2 件以上データがあれば significance_test が含まれること。"""
        from serve.ab_router import _Stats, _significance_test

        s = _Stats()
        s.scores["openmythos"] = [0.8, 0.85, 0.9]
        s.scores["existing_ml"] = [0.6, 0.65, 0.7]
        sig = _significance_test(s.scores["openmythos"], s.scores["existing_ml"])
        assert "p_value" in sig
        assert "significant" in sig


# ===========================================================================
# 9.2.1  serve.api — POST /v1/batch
# ===========================================================================


class TestBatchInference:
    @pytest.fixture
    def mock_state(self):
        """serve.api の global state を小さいモデルで差し替える。"""
        import serve.api as api
        from open_mythos.main import MythosConfig, OpenMythos
        from transformers import AutoTokenizer

        cfg = MythosConfig(
            vocab_size=50257,
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            max_loop_iters=4,
            prelude_layers=1,
            coda_layers=1,
            attn_type="gqa",
            n_experts=2,
            n_shared_experts=1,
            n_experts_per_tok=1,
            expert_dim=32,
            act_threshold=0.99,
            lora_rank=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
        model = OpenMythos(cfg)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        api.state.model = model
        api.state.tokenizer = tokenizer
        api.state.device = torch.device("cpu")
        api.state.n_params = sum(p.numel() for p in model.parameters())
        return api

    def test_batch_single_item(self, mock_state):
        from serve.api import BatchRequest, BatchItem, batch_infer

        req = BatchRequest(items=[BatchItem(text="hello world")])
        resp = batch_infer(req)
        assert resp.n_items == 1
        assert len(resp.results) == 1
        assert resp.results[0].index == 0

    def test_batch_multiple_items(self, mock_state):
        from serve.api import BatchRequest, BatchItem, batch_infer

        items = [BatchItem(text=f"text {i}") for i in range(5)]
        resp = batch_infer(BatchRequest(items=items))
        assert resp.n_items == 5
        assert len(resp.results) == 5
        # index は 0-4
        indices = [r.index for r in resp.results]
        assert indices == list(range(5))

    def test_batch_response_fields(self, mock_state):
        from serve.api import BatchRequest, BatchItem, batch_infer

        req = BatchRequest(items=[BatchItem(text="test", task="fraud_detect")])
        resp = batch_infer(req)
        item = resp.results[0]
        assert 0.0 <= item.score <= 1.0
        assert item.label in {0, 1}
        assert item.latency_ms >= 0
        assert item.task == "fraud_detect"

    def test_batch_total_latency(self, mock_state):
        from serve.api import BatchRequest, BatchItem, batch_infer

        items = [BatchItem(text=f"item {i}") for i in range(3)]
        resp = batch_infer(BatchRequest(items=items))
        assert resp.total_latency_ms >= 0

    def test_batch_request_defaults(self):
        from serve.api import BatchItem

        item = BatchItem(text="hello")
        assert item.task == "general"

    def test_batch_request_schema_validation(self):
        from serve.api import BatchRequest
        from pydantic import ValidationError

        # items が空だと ValidationError
        with pytest.raises(ValidationError):
            BatchRequest(items=[])

    def test_batch_task_specific_loops(self, mock_state):
        """fraud_detect タスクは高ループ数で実行されること。"""
        from serve.api import BatchRequest, BatchItem, batch_infer, TASK_LOOPS

        req = BatchRequest(items=[BatchItem(text="fraud check", task="fraud_detect")])
        resp = batch_infer(req)
        # loops_used は TASK_LOOPS["fraud_detect"] 以下であること
        assert resp.results[0].loops_used <= TASK_LOOPS["fraud_detect"]

    def test_batch_mixed_tasks(self, mock_state):
        from serve.api import BatchRequest, BatchItem, batch_infer

        items = [
            BatchItem(text="ad copy", task="ad_performance"),
            BatchItem(text="content", task="content_quality"),
            BatchItem(text="general", task="general"),
        ]
        resp = batch_infer(BatchRequest(items=items))
        tasks = [r.task for r in resp.results]
        assert tasks == ["ad_performance", "content_quality", "general"]


# ===========================================================================
# 9.3.1  pyproject.toml — v0.13.0
# ===========================================================================


class TestVersionBump:
    def test_version_is_013_or_later(self):
        """Sprint 9 で 0.13.0 以上にバンプされていることを確認 (Sprint 10 以降も PASS)。"""
        import re
        from pathlib import Path

        content = (Path(__file__).parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        m = re.search(r'^version = "(\d+)\.(\d+)\.(\d+)"', content, re.MULTILINE)
        assert m is not None, "pyproject.toml に version フィールドが見つからない"
        major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert (major, minor, patch) >= (0, 13, 0), (
            f"version {major}.{minor}.{patch} は 0.13.0 以上であること"
        )

    def test_changelog_has_013(self):
        from pathlib import Path

        content = (Path(__file__).parent.parent / "CHANGELOG.md").read_text(
            encoding="utf-8"
        )
        assert "0.13.0" in content
        assert "Sprint 9" in content

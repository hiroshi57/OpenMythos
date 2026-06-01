"""
Sprint 25 テスト — 継続的自己蒸留 SelfDistillLoop

- TestDistillSample        : サンプルデータ構造
- TestDistillDataset       : add / mean_score / to_jsonl
- TestOutputFilter         : スコア閾値・多様性フィルタ
- TestSelfDistillCollector : 推論→スコア収集
- TestSelfDistillConfig    : 設定値
- TestSelfDistillLoop      : Collect→Filter→SFT→Eval サイクル
- TestDistillAPIEndpoint   : FastAPI /v1/distill/* (静的検査)
"""

from __future__ import annotations

import json
import pathlib

_ROOT = pathlib.Path(__file__).parent.parent


# ===========================================================================
# TestDistillSample
# ===========================================================================


class TestDistillSample:
    def test_to_jsonl_dict_keys(self):
        from open_mythos.self_distill import DistillSample

        s = DistillSample(prompt="p", output="o", score=0.8)
        d = s.to_jsonl_dict()
        assert "prompt" in d
        assert "output" in d
        assert "score" in d

    def test_sample_id_unique(self):
        from open_mythos.self_distill import DistillSample

        s1 = DistillSample(prompt="p", output="o", score=0.5)
        s2 = DistillSample(prompt="p", output="o", score=0.5)
        assert s1.sample_id != s2.sample_id

    def test_round_num_assigned(self):
        from open_mythos.self_distill import DistillSample

        s = DistillSample(prompt="p", output="o", score=0.7, round_num=2)
        assert s.round_num == 2


# ===========================================================================
# TestDistillDataset
# ===========================================================================


class TestDistillDataset:
    def test_add_increments_total(self):
        from open_mythos.self_distill import DistillDataset, DistillSample

        ds = DistillDataset()
        ds.add(DistillSample(prompt="p", output="o", score=0.7))
        assert ds.total == 1

    def test_len_equals_total(self):
        from open_mythos.self_distill import DistillDataset, DistillSample

        ds = DistillDataset()
        ds.add(DistillSample(prompt="p", output="o", score=0.5))
        assert len(ds) == 1

    def test_mean_score(self):
        from open_mythos.self_distill import DistillDataset, DistillSample

        ds = DistillDataset()
        ds.add(DistillSample(prompt="p", output="o", score=0.8))
        ds.add(DistillSample(prompt="p", output="o", score=0.6))
        assert abs(ds.mean_score - 0.7) < 1e-6

    def test_empty_mean_score_zero(self):
        from open_mythos.self_distill import DistillDataset

        ds = DistillDataset()
        assert ds.mean_score == 0.0

    def test_to_jsonl_valid(self):
        from open_mythos.self_distill import DistillDataset, DistillSample

        ds = DistillDataset()
        ds.add(DistillSample(prompt="p", output="o", score=0.7))
        jsonl = ds.to_jsonl()
        obj = json.loads(jsonl.split("\n")[0])
        assert obj["prompt"] == "p"

    def test_samples_above_threshold(self):
        from open_mythos.self_distill import DistillDataset, DistillSample

        ds = DistillDataset()
        ds.add(DistillSample(prompt="p", output="o", score=0.4))
        ds.add(DistillSample(prompt="p", output="o", score=0.8))
        above = ds.samples_above(0.6)
        assert len(above) == 1

    def test_max_size_limit(self):
        from open_mythos.self_distill import DistillDataset, DistillSample

        ds = DistillDataset(max_size=3)
        for i in range(5):
            ds.add(DistillSample(prompt="p", output="o", score=float(i) / 10))
        assert ds.total <= 3

    def test_add_batch(self):
        from open_mythos.self_distill import DistillDataset, DistillSample

        ds = DistillDataset()
        batch = [DistillSample(prompt="p", output="o", score=0.5) for _ in range(3)]
        ds.add_batch(batch)
        assert ds.total == 3


# ===========================================================================
# TestOutputFilter
# ===========================================================================


class TestOutputFilter:
    def test_filter_removes_low_score(self):
        from open_mythos.self_distill import OutputFilter, DistillSample

        f = OutputFilter(score_threshold=0.7)
        samples = [
            DistillSample(prompt="p", output="long enough output text", score=0.5),
            DistillSample(prompt="p", output="long enough output text", score=0.8),
        ]
        result = f.filter(samples)
        assert all(s.score >= 0.7 for s in result)

    def test_filter_removes_short_output(self):
        from open_mythos.self_distill import OutputFilter, DistillSample

        f = OutputFilter(score_threshold=0.0, diversity_min_len=20)
        samples = [
            DistillSample(prompt="p", output="short", score=0.9),
            DistillSample(prompt="p", output="this is long enough output text here", score=0.9),
        ]
        result = f.filter(samples)
        assert all(len(s.output) >= 20 for s in result)

    def test_filter_dedup_identical(self):
        from open_mythos.self_distill import OutputFilter, DistillSample

        f = OutputFilter(score_threshold=0.0, diversity_min_len=0, max_similar_ratio=0.95)
        identical = "same output text repeated here"
        samples = [
            DistillSample(prompt="p", output=identical, score=0.9),
            DistillSample(prompt="p", output=identical, score=0.9),
        ]
        result = f.filter(samples)
        assert len(result) == 1

    def test_filter_keeps_diverse(self):
        from open_mythos.self_distill import OutputFilter, DistillSample

        f = OutputFilter(score_threshold=0.0, diversity_min_len=0)
        samples = [
            DistillSample(prompt="p", output="apple banana cherry", score=0.9),
            DistillSample(prompt="p", output="python java golang", score=0.9),
        ]
        result = f.filter(samples)
        assert len(result) == 2

    def test_filter_empty_input(self):
        from open_mythos.self_distill import OutputFilter

        f = OutputFilter()
        assert f.filter([]) == []


# ===========================================================================
# TestSelfDistillCollector
# ===========================================================================


class TestSelfDistillCollector:
    def test_collect_returns_samples(self):
        from open_mythos.self_distill import SelfDistillCollector

        collector = SelfDistillCollector(
            generate_fn=lambda p: f"output for {p}",
            score_fn=lambda t: 0.7,
        )
        samples = collector.collect(["prompt1", "prompt2"])
        assert len(samples) == 2

    def test_collect_scores_assigned(self):
        from open_mythos.self_distill import SelfDistillCollector

        collector = SelfDistillCollector(
            generate_fn=lambda p: "output",
            score_fn=lambda t: 0.75,
        )
        samples = collector.collect(["p"])
        assert samples[0].score == 0.75

    def test_collect_round_num(self):
        from open_mythos.self_distill import SelfDistillCollector

        collector = SelfDistillCollector(
            generate_fn=lambda p: "out",
            score_fn=lambda t: 0.5,
        )
        samples = collector.collect(["p"], round_num=3)
        assert samples[0].round_num == 3

    def test_collect_handles_exception(self):
        from open_mythos.self_distill import SelfDistillCollector

        def bad_gen(p):
            raise RuntimeError("error")

        collector = SelfDistillCollector(generate_fn=bad_gen, score_fn=lambda t: 0.5)
        samples = collector.collect(["p"])
        assert samples[0].score == 0.0


# ===========================================================================
# TestSelfDistillConfig
# ===========================================================================


class TestSelfDistillConfig:
    def test_defaults(self):
        from open_mythos.self_distill import SelfDistillConfig

        cfg = SelfDistillConfig()
        assert cfg.n_rounds == 3
        assert cfg.score_threshold == 0.6
        assert cfg.early_stop_score == 0.85

    def test_custom_values(self):
        from open_mythos.self_distill import SelfDistillConfig

        cfg = SelfDistillConfig(n_rounds=5, score_threshold=0.75)
        assert cfg.n_rounds == 5
        assert cfg.score_threshold == 0.75


# ===========================================================================
# TestSelfDistillLoop
# ===========================================================================


class TestSelfDistillLoop:
    def _make_loop(self, n_rounds=2, score_threshold=0.5):
        from open_mythos.self_distill import SelfDistillLoop, SelfDistillConfig

        cfg = SelfDistillConfig(
            n_rounds=n_rounds,
            score_threshold=score_threshold,
            sft_simulate=True,
        )
        return SelfDistillLoop(
            cfg=cfg,
            generate_fn=lambda p: f"SEO対策とは重要なマーケティング施策です。{p}",
            score_fn=lambda t: min(len(t) / 200, 1.0),
        )

    def test_run_returns_result(self):
        from open_mythos.self_distill import SelfDistillResult

        loop = self._make_loop()
        result = loop.run(["SEO対策とは？"])
        assert isinstance(result, SelfDistillResult)

    def test_rounds_completed_positive(self):
        loop = self._make_loop(n_rounds=2)
        result = loop.run(["SEO"])
        assert result.rounds_completed >= 1

    def test_total_samples_non_negative(self):
        loop = self._make_loop()
        result = loop.run(["SEO", "LLMO"])
        assert result.total_samples >= 0

    def test_mean_score_improvement_float(self):
        loop = self._make_loop()
        result = loop.run(["SEO"])
        assert isinstance(result.mean_score_improvement, float)

    def test_early_stopped_bool(self):
        loop = self._make_loop()
        result = loop.run(["SEO"])
        assert isinstance(result.early_stopped, bool)

    def test_total_latency_positive(self):
        loop = self._make_loop()
        result = loop.run(["SEO"])
        assert result.total_latency_ms > 0

    def test_round_results_list(self):
        loop = self._make_loop(n_rounds=2)
        result = loop.run(["SEO"])
        assert isinstance(result.round_results, list)

    def test_dataset_accessible(self):
        loop = self._make_loop()
        result = loop.run(["SEO"])
        assert hasattr(result.dataset, "total")

    def test_early_stop_when_high_score(self):
        from open_mythos.self_distill import SelfDistillLoop, SelfDistillConfig

        cfg = SelfDistillConfig(n_rounds=5, score_threshold=0.0, early_stop_score=0.01)
        loop = SelfDistillLoop(
            cfg=cfg,
            generate_fn=lambda p: "long output text for scoring" * 5,
            score_fn=lambda t: 1.0,
        )
        result = loop.run(["SEO"])
        assert result.early_stopped is True
        assert result.rounds_completed < 5

    def test_round_result_fields(self):
        loop = self._make_loop(n_rounds=1)
        result = loop.run(["SEO"])
        if result.round_results:
            r = result.round_results[0]
            assert hasattr(r, "collected")
            assert hasattr(r, "filtered")
            assert hasattr(r, "mean_score")

    def test_sft_result_generated(self):
        loop = self._make_loop(n_rounds=1, score_threshold=0.0)
        result = loop.run(["SEO最適化とはコンテンツを検索エンジン向けに最適化すること"])
        if result.round_results and result.round_results[0].sft_result:
            sft = result.round_results[0].sft_result
            assert sft.n_samples >= 0

    def test_initial_mean_score_float(self):
        loop = self._make_loop()
        result = loop.run(["SEO"])
        assert isinstance(result.initial_mean_score, float)

    def test_final_mean_score_float(self):
        loop = self._make_loop()
        result = loop.run(["SEO"])
        assert isinstance(result.final_mean_score, float)

    def test_default_generate_fn(self):
        from open_mythos.self_distill import SelfDistillLoop, SelfDistillConfig, SelfDistillResult

        cfg = SelfDistillConfig(n_rounds=1, score_threshold=0.0)
        loop = SelfDistillLoop(cfg=cfg)
        result = loop.run(["テスト"])
        assert isinstance(result, SelfDistillResult)


# ===========================================================================
# TestDistillAPIEndpoint (静的ソース検査)
# ===========================================================================


class TestDistillAPIEndpoint:
    def _src(self) -> str:
        return (_ROOT / "serve" / "api.py").read_text(encoding="utf-8")

    def test_distill_run_route_exists(self):
        assert '"/v1/distill/run"' in self._src()

    def test_distill_status_route_exists(self):
        assert '"/v1/distill/status"' in self._src()

    def test_distill_run_post_method(self):
        src = self._src()
        idx = src.index('"/v1/distill/run"')
        snippet = src[max(0, idx - 60):idx]
        assert "@app.post" in snippet

    def test_distill_tag(self):
        src = self._src()
        idx = src.index('"/v1/distill/run"')
        snippet = src[idx:idx + 200]
        assert 'tags=["distill"]' in snippet

    def test_distill_run_request_model(self):
        assert "DistillRunRequest" in self._src()

    def test_rounds_completed_key(self):
        assert '"rounds_completed"' in self._src()

    def test_mean_score_improvement_key(self):
        assert '"mean_score_improvement"' in self._src()

    def test_early_stopped_key(self):
        assert '"early_stopped"' in self._src()

    def test_self_distill_loop_used(self):
        assert "SelfDistillLoop" in self._src()

    def test_self_distill_config_used(self):
        assert "SelfDistillConfig" in self._src()

    def test_verify_api_key_distill(self):
        src = self._src()
        idx = src.index("def distill_run")
        snippet = src[max(0, idx - 200):idx + 100]
        assert "verify_api_key" in snippet

    def test_total_samples_key(self):
        assert '"total_samples"' in self._src()

    def test_final_mean_score_key(self):
        assert '"final_mean_score"' in self._src()

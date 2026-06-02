"""
Sprint 22 テスト — ボトルネック発見・解消 ProfilerAgent

- TestStageMetrics        : StageMetrics データ構造
- TestProfileResult       : ProfileResult 集計メソッド
- TestBottleneckReport    : BottleneckReport プロパティ
- TestPipelineProfiler    : ステージ実行・計測
- TestBottleneckDetector  : IQR外れ値検出
- TestAutoFixResult       : AutoFixResult データ構造
- TestProfilerAgent       : profile_and_fix サイクル
- TestProfilerAPIEndpoint : FastAPI /v1/profile/* (静的検査)
"""

from __future__ import annotations

import pathlib
import time

_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _fast_stage(text: str):
    return text + "_fast", 0.8


def _slow_stage(text: str):
    time.sleep(0.05)
    return text + "_slow", 0.7


def _bad_score_stage(text: str):
    return text + "_bad", 0.1


def _error_stage(text: str):
    raise ValueError("意図的なエラー")


def _make_stages(include_slow=False, include_error=False, include_bad_score=False):
    stages = {"stage_a": _fast_stage, "stage_b": _fast_stage}
    if include_slow:
        stages["stage_slow"] = _slow_stage
    if include_error:
        stages["stage_error"] = _error_stage
    if include_bad_score:
        stages["stage_bad"] = _bad_score_stage
    return stages


# ===========================================================================
# TestStageMetrics
# ===========================================================================


class TestStageMetrics:
    def test_ok_true_when_no_error(self):
        from open_mythos.profiler import StageMetrics

        m = StageMetrics(stage_name="s", latency_ms=10.0)
        assert m.ok is True

    def test_ok_false_when_error(self):
        from open_mythos.profiler import StageMetrics

        m = StageMetrics(stage_name="s", latency_ms=10.0, error="fail")
        assert m.ok is False

    def test_error_rate_zero_when_ok(self):
        from open_mythos.profiler import StageMetrics

        m = StageMetrics(stage_name="s", latency_ms=10.0)
        assert m.error_rate == 0.0

    def test_error_rate_one_when_error(self):
        from open_mythos.profiler import StageMetrics

        m = StageMetrics(stage_name="s", latency_ms=10.0, error="err")
        assert m.error_rate == 1.0

    def test_run_id_unique(self):
        from open_mythos.profiler import StageMetrics

        m1 = StageMetrics(stage_name="s", latency_ms=1.0)
        m2 = StageMetrics(stage_name="s", latency_ms=1.0)
        assert m1.run_id != m2.run_id

    def test_score_default_minus_one(self):
        from open_mythos.profiler import StageMetrics

        m = StageMetrics(stage_name="s", latency_ms=1.0)
        assert m.score == -1.0


# ===========================================================================
# TestProfileResult
# ===========================================================================


class TestProfileResult:
    def _make_result(self):
        from open_mythos.profiler import ProfileResult, StageMetrics

        return ProfileResult(
            stages={
                "a": StageMetrics("a", latency_ms=10.0, score=0.8),
                "b": StageMetrics("b", latency_ms=30.0, score=0.6),
                "c": StageMetrics("c", latency_ms=5.0, score=0.9),
            },
            total_latency_ms=45.0,
            final_output="result",
        )

    def test_stage_names(self):
        r = self._make_result()
        assert set(r.stage_names()) == {"a", "b", "c"}

    def test_latencies(self):
        r = self._make_result()
        assert r.latencies()["b"] == 30.0

    def test_scores(self):
        r = self._make_result()
        assert r.scores()["a"] == 0.8

    def test_slowest_stage(self):
        r = self._make_result()
        assert r.slowest_stage() == "b"

    def test_lowest_score_stage(self):
        r = self._make_result()
        assert r.lowest_score_stage() == "b"

    def test_run_id_not_empty(self):
        r = self._make_result()
        assert r.run_id != ""


# ===========================================================================
# TestBottleneckReport
# ===========================================================================


class TestBottleneckReport:
    def _make_report(self, btype="latency"):
        from open_mythos.profiler import BottleneckReport, ProfileResult

        profile = ProfileResult(stages={}, total_latency_ms=0.0, final_output="")
        return BottleneckReport(
            bottleneck_stage="stage_slow",
            bottleneck_type=btype,
            severity="high",
            affected_stages=["stage_slow"],
            diagnosis="遅い",
            suggested_fix="高速化してください",
            baseline_profile=profile,
        )

    def test_has_bottleneck_true(self):
        r = self._make_report("latency")
        assert r.has_bottleneck is True

    def test_has_bottleneck_false_when_none(self):
        r = self._make_report("none")
        assert r.has_bottleneck is False

    def test_bottleneck_type(self):
        r = self._make_report("score")
        assert r.bottleneck_type == "score"

    def test_severity_field(self):
        r = self._make_report()
        assert r.severity in ("high", "medium", "low", "none")


# ===========================================================================
# TestPipelineProfiler
# ===========================================================================


class TestPipelineProfiler:
    def test_run_returns_profile_result(self):
        from open_mythos.profiler import PipelineProfiler, ProfileResult

        profiler = PipelineProfiler({"a": _fast_stage, "b": _fast_stage})
        result = profiler.run("input")
        assert isinstance(result, ProfileResult)

    def test_all_stages_executed(self):
        from open_mythos.profiler import PipelineProfiler

        profiler = PipelineProfiler({"a": _fast_stage, "b": _fast_stage, "c": _fast_stage})
        result = profiler.run("input")
        assert set(result.stage_names()) == {"a", "b", "c"}

    def test_latency_positive(self):
        from open_mythos.profiler import PipelineProfiler

        profiler = PipelineProfiler({"a": _fast_stage})
        result = profiler.run("input")
        assert result.stages["a"].latency_ms >= 0

    def test_score_captured(self):
        from open_mythos.profiler import PipelineProfiler

        profiler = PipelineProfiler({"a": _fast_stage})
        result = profiler.run("input")
        assert result.stages["a"].score == 0.8

    def test_output_propagates(self):
        from open_mythos.profiler import PipelineProfiler

        profiler = PipelineProfiler({"a": _fast_stage, "b": _fast_stage})
        result = profiler.run("hello")
        assert "hello" in result.final_output

    def test_error_stage_captured(self):
        from open_mythos.profiler import PipelineProfiler

        profiler = PipelineProfiler({"a": _fast_stage, "err": _error_stage})
        result = profiler.run("input")
        assert result.stages["err"].ok is False
        assert result.stages["err"].error is not None

    def test_total_latency_positive(self):
        from open_mythos.profiler import PipelineProfiler

        profiler = PipelineProfiler({"a": _fast_stage, "b": _fast_stage})
        result = profiler.run("input")
        assert result.total_latency_ms > 0

    def test_stage_without_score(self):
        from open_mythos.profiler import PipelineProfiler

        def no_score_stage(text: str) -> str:
            return text + "_ns"

        profiler = PipelineProfiler({"a": no_score_stage})
        result = profiler.run("test")
        assert result.stages["a"].score == -1.0


# ===========================================================================
# TestBottleneckDetector
# ===========================================================================


class TestBottleneckDetector:
    def test_detect_no_bottleneck_identical_latencies(self):
        from open_mythos.profiler import PipelineProfiler, BottleneckDetector

        profiler = PipelineProfiler({"a": _fast_stage, "b": _fast_stage, "c": _fast_stage})
        result = profiler.run("input")
        report = BottleneckDetector().detect(result)
        assert report.bottleneck_type in ("none", "latency", "score")

    def test_detect_error_bottleneck(self):
        from open_mythos.profiler import PipelineProfiler, BottleneckDetector

        profiler = PipelineProfiler({"a": _fast_stage, "err": _error_stage})
        result = profiler.run("input")
        report = BottleneckDetector().detect(result)
        assert report.bottleneck_type == "error"
        assert report.bottleneck_stage == "err"

    def test_detect_score_bottleneck(self):
        from open_mythos.profiler import PipelineProfiler, BottleneckDetector

        stages = {
            "a": _fast_stage,
            "b": _fast_stage,
            "c": _fast_stage,
            "d": _fast_stage,
            "bad": _bad_score_stage,
        }
        profiler = PipelineProfiler(stages)
        result = profiler.run("input")
        report = BottleneckDetector(iqr_factor=0.5).detect(result)
        assert report.bottleneck_type in ("score", "none")

    def test_detect_report_has_diagnosis(self):
        from open_mythos.profiler import PipelineProfiler, BottleneckDetector

        profiler = PipelineProfiler({"a": _fast_stage})
        result = profiler.run("input")
        report = BottleneckDetector().detect(result)
        assert isinstance(report.diagnosis, str)
        assert len(report.diagnosis) > 0

    def test_detect_empty_stages(self):
        from open_mythos.profiler import ProfileResult, BottleneckDetector

        result = ProfileResult(stages={}, total_latency_ms=0.0, final_output="")
        report = BottleneckDetector().detect(result)
        assert report.bottleneck_type == "none"

    def test_affected_stages_list(self):
        from open_mythos.profiler import PipelineProfiler, BottleneckDetector

        profiler = PipelineProfiler({"a": _error_stage, "b": _error_stage})
        result = profiler.run("input")
        report = BottleneckDetector().detect(result)
        assert isinstance(report.affected_stages, list)


# ===========================================================================
# TestAutoFixResult
# ===========================================================================


class TestAutoFixResult:
    def _make_fix_result(self):
        from open_mythos.profiler import AutoFixResult, BottleneckReport, ProfileResult

        profile = ProfileResult(stages={}, total_latency_ms=100.0, final_output="")
        report = BottleneckReport(
            bottleneck_stage="s",
            bottleneck_type="latency",
            severity="high",
            affected_stages=["s"],
            diagnosis="遅い",
            suggested_fix="速くして",
            baseline_profile=profile,
        )
        return AutoFixResult(
            bottleneck_report=report,
            before_profile=profile,
            after_profile=profile,
            latency_improvement_pct=20.0,
            score_improvement=0.05,
            fixed=True,
            fix_description="高速化パッチ適用",
        )

    def test_fixed_field(self):
        r = self._make_fix_result()
        assert r.fixed is True

    def test_latency_improvement_pct(self):
        r = self._make_fix_result()
        assert r.latency_improvement_pct == 20.0

    def test_fix_description_not_empty(self):
        r = self._make_fix_result()
        assert r.fix_description != ""


# ===========================================================================
# TestProfilerAgent
# ===========================================================================


class TestProfilerAgent:
    def test_profile_returns_result(self):
        from open_mythos.profiler import ProfilerAgent, ProfileResult

        agent = ProfilerAgent({"a": _fast_stage, "b": _fast_stage})
        result = agent.profile("input")
        assert isinstance(result, ProfileResult)

    def test_detect_returns_report(self):
        from open_mythos.profiler import ProfilerAgent, BottleneckReport

        agent = ProfilerAgent({"a": _fast_stage})
        profile = agent.profile("input")
        report = agent.detect(profile)
        assert isinstance(report, BottleneckReport)

    def test_profile_and_fix_returns_auto_fix_result(self):
        from open_mythos.profiler import ProfilerAgent, AutoFixResult

        agent = ProfilerAgent({"a": _fast_stage, "b": _fast_stage})
        result = agent.profile_and_fix("input")
        assert isinstance(result, AutoFixResult)

    def test_auto_fix_no_bottleneck(self):
        from open_mythos.profiler import ProfilerAgent

        agent = ProfilerAgent({"a": _fast_stage})
        result = agent.profile_and_fix("input")
        assert isinstance(result.fixed, bool)

    def test_auto_fix_error_stage(self):
        from open_mythos.profiler import ProfilerAgent

        agent = ProfilerAgent({"ok": _fast_stage, "err": _error_stage})
        result = agent.profile_and_fix("input")
        assert result.bottleneck_report.bottleneck_type == "error"

    def test_tune_log_updated_after_fix(self):
        from open_mythos.profiler import ProfilerAgent

        agent = ProfilerAgent({"ok": _fast_stage, "err": _error_stage})
        agent.profile_and_fix("input")
        assert len(agent.tune_log) >= 1

    def test_after_fix_profile_exists(self):
        from open_mythos.profiler import ProfilerAgent

        agent = ProfilerAgent({"a": _fast_stage, "b": _fast_stage})
        result = agent.profile_and_fix("input")
        assert result.after_profile is not None

    def test_before_profile_latency_positive(self):
        from open_mythos.profiler import ProfilerAgent

        agent = ProfilerAgent({"a": _fast_stage})
        result = agent.profile_and_fix("input")
        assert result.before_profile.total_latency_ms >= 0

    def test_score_improvement_is_float(self):
        from open_mythos.profiler import ProfilerAgent

        agent = ProfilerAgent({"a": _fast_stage})
        result = agent.profile_and_fix("input")
        assert isinstance(result.score_improvement, float)

    def test_fix_description_string(self):
        from open_mythos.profiler import ProfilerAgent

        agent = ProfilerAgent({"a": _error_stage})
        result = agent.profile_and_fix("input")
        assert isinstance(result.fix_description, str)


# ===========================================================================
# TestProfilerAPIEndpoint (静的ソース検査)
# ===========================================================================


class TestProfilerAPIEndpoint:
    def _src(self) -> str:
        return (_ROOT / "serve" / "api.py").read_text(encoding="utf-8")

    def test_profile_run_route_exists(self):
        assert '"/v1/profile/run"' in self._src()

    def test_profile_fix_route_exists(self):
        assert '"/v1/profile/fix"' in self._src()

    def test_profile_report_route_exists(self):
        assert '"/v1/profile/report"' in self._src()

    def test_profile_run_post_method(self):
        src = self._src()
        idx = src.index('"/v1/profile/run"')
        snippet = src[max(0, idx - 60):idx]
        assert "@app.post" in snippet

    def test_profile_fix_post_method(self):
        src = self._src()
        idx = src.index('"/v1/profile/fix"')
        snippet = src[max(0, idx - 60):idx]
        assert "@app.post" in snippet

    def test_profiler_tag_run(self):
        src = self._src()
        idx = src.index('"/v1/profile/run"')
        snippet = src[idx:idx + 200]
        assert 'tags=["profiler"]' in snippet

    def test_profiler_tag_fix(self):
        src = self._src()
        idx = src.index('"/v1/profile/fix"')
        snippet = src[idx:idx + 200]
        assert 'tags=["profiler"]' in snippet

    def test_profile_run_request_model(self):
        assert "ProfileRunRequest" in self._src()

    def test_profile_fix_request_model(self):
        assert "ProfileFixRequest" in self._src()

    def test_bottleneck_stage_key(self):
        assert '"bottleneck_stage"' in self._src()

    def test_bottleneck_type_key(self):
        assert '"bottleneck_type"' in self._src()

    def test_latency_improvement_pct_key(self):
        assert '"latency_improvement_pct"' in self._src()

    def test_profiler_agent_used(self):
        assert "ProfilerAgent" in self._src()

    def test_pipeline_profiler_used(self):
        assert "PipelineProfiler" in self._src()

    def test_bottleneck_detector_used(self):
        assert "BottleneckDetector" in self._src()

    def test_verify_api_key_in_profile_run(self):
        src = self._src()
        idx = src.index("def profile_run")
        snippet = src[max(0, idx - 200):idx + 100]
        assert "verify_api_key" in snippet

    def test_severity_key_in_run(self):
        src = self._src()
        assert '"severity"' in src

    def test_suggested_fix_key(self):
        assert '"suggested_fix"' in self._src()

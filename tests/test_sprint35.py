"""
Sprint 35 — GrowingAI ベンチマーク (benchmark/growing_ai_bench.py) テストスイート (40 tests)

テスト対象:
    PatternBenchResult  — 1パターン結果データクラス
    bench_pN functions  — P1〜P10 個別ベンチマーク関数
    GrowingAIBenchmark  — 全パターン一括実行クラス
    BenchmarkReport     — 集計レポートデータクラス
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

# benchmark フォルダを path に追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.growing_ai_bench import (
    BenchmarkReport,
    GrowingAIBenchmark,
    PatternBenchResult,
    bench_p1_debate,
    bench_p2_kpi,
    bench_p3_profiler,
    bench_p4_signal,
    bench_p5_guard,
    bench_p6_distill,
    bench_p7_memory,
    bench_p8_ensemble,
    bench_p9_evolution,
    bench_p10_planner,
    _PATTERN_FUNCS,
)


# ===========================================================================
# 1. PatternBenchResult — データクラス (5 tests)
# ===========================================================================

def test_pattern_bench_result_improvement_computed():
    r = PatternBenchResult(
        pattern_id="px", pattern_name="Test",
        baseline_score=0.50, final_score=0.65,
    )
    assert abs(r.improvement - 0.15) < 1e-6


def test_pattern_bench_result_improvement_pct_computed():
    r = PatternBenchResult(
        pattern_id="px", pattern_name="Test",
        baseline_score=0.50, final_score=0.65,
    )
    assert abs(r.improvement_pct - 30.0) < 1e-3


def test_pattern_bench_result_negative_improvement():
    r = PatternBenchResult(
        pattern_id="px", pattern_name="Test",
        baseline_score=0.60, final_score=0.55,
    )
    assert r.improvement < 0


def test_pattern_bench_result_zero_baseline_no_division_error():
    """baseline=0 でも ZeroDivision が起きない。"""
    r = PatternBenchResult(
        pattern_id="px", pattern_name="Test",
        baseline_score=0.0, final_score=0.10,
    )
    assert math.isfinite(r.improvement_pct)


def test_pattern_bench_result_success_default_true():
    r = PatternBenchResult(
        pattern_id="px", pattern_name="Test",
        baseline_score=0.5, final_score=0.5,
    )
    assert r.success is True


# ===========================================================================
# 2. P1 — bench_p1_debate (3 tests)
# ===========================================================================

def test_p1_returns_pattern_bench_result():
    result = bench_p1_debate()
    assert isinstance(result, PatternBenchResult)


def test_p1_pattern_id():
    result = bench_p1_debate()
    assert result.pattern_id == "p1"


def test_p1_latency_positive():
    result = bench_p1_debate()
    assert result.latency_ms >= 0


# ===========================================================================
# 3. P2 — bench_p2_kpi (3 tests)
# ===========================================================================

def test_p2_returns_pattern_bench_result():
    result = bench_p2_kpi()
    assert isinstance(result, PatternBenchResult)


def test_p2_pattern_id():
    result = bench_p2_kpi()
    assert result.pattern_id == "p2"


def test_p2_baseline_score_positive():
    result = bench_p2_kpi()
    assert result.baseline_score >= 0


# ===========================================================================
# 4. P3 — bench_p3_profiler (3 tests)
# ===========================================================================

def test_p3_returns_pattern_bench_result():
    result = bench_p3_profiler()
    assert isinstance(result, PatternBenchResult)


def test_p3_pattern_id():
    result = bench_p3_profiler()
    assert result.pattern_id == "p3"


def test_p3_latency_positive():
    result = bench_p3_profiler()
    assert result.latency_ms >= 0


# ===========================================================================
# 5. P4 — bench_p4_signal (3 tests)
# ===========================================================================

def test_p4_returns_pattern_bench_result():
    result = bench_p4_signal()
    assert isinstance(result, PatternBenchResult)


def test_p4_pattern_id():
    result = bench_p4_signal()
    assert result.pattern_id == "p4"


def test_p4_notes_nonempty():
    result = bench_p4_signal()
    assert isinstance(result.notes, str)


# ===========================================================================
# 6. P5 — bench_p5_guard (3 tests)
# ===========================================================================

def test_p5_returns_pattern_bench_result():
    result = bench_p5_guard()
    assert isinstance(result, PatternBenchResult)


def test_p5_pattern_id():
    result = bench_p5_guard()
    assert result.pattern_id == "p5"


def test_p5_pattern_score_in_range():
    """guard_score は F1 的合成なので [0, 1] の範囲内。"""
    result = bench_p5_guard()
    assert 0.0 <= result.pattern_score <= 1.0


# ===========================================================================
# 7. P6 — bench_p6_distill (3 tests)
# ===========================================================================

def test_p6_returns_pattern_bench_result():
    result = bench_p6_distill()
    assert isinstance(result, PatternBenchResult)


def test_p6_pattern_id():
    result = bench_p6_distill()
    assert result.pattern_id == "p6"


def test_p6_success():
    result = bench_p6_distill()
    assert result.success is True


# ===========================================================================
# 8. P7 — bench_p7_memory (3 tests)
# ===========================================================================

def test_p7_returns_pattern_bench_result():
    result = bench_p7_memory()
    assert isinstance(result, PatternBenchResult)


def test_p7_pattern_id():
    result = bench_p7_memory()
    assert result.pattern_id == "p7"


def test_p7_baseline_score_finite():
    result = bench_p7_memory()
    assert math.isfinite(result.baseline_score)


# ===========================================================================
# 9. P8 — bench_p8_ensemble (3 tests)
# ===========================================================================

def test_p8_returns_pattern_bench_result():
    result = bench_p8_ensemble()
    assert isinstance(result, PatternBenchResult)


def test_p8_pattern_id():
    result = bench_p8_ensemble()
    assert result.pattern_id == "p8"


def test_p8_final_score_positive():
    result = bench_p8_ensemble()
    assert result.final_score >= 0


# ===========================================================================
# 10. P9 — bench_p9_evolution (3 tests)
# ===========================================================================

def test_p9_returns_pattern_bench_result():
    result = bench_p9_evolution()
    assert isinstance(result, PatternBenchResult)


def test_p9_pattern_id():
    result = bench_p9_evolution()
    assert result.pattern_id == "p9"


def test_p9_latency_positive():
    result = bench_p9_evolution()
    assert result.latency_ms >= 0


# ===========================================================================
# 11. P10 — bench_p10_planner (3 tests)
# ===========================================================================

def test_p10_returns_pattern_bench_result():
    result = bench_p10_planner()
    assert isinstance(result, PatternBenchResult)


def test_p10_pattern_id():
    result = bench_p10_planner()
    assert result.pattern_id == "p10"


def test_p10_success():
    result = bench_p10_planner()
    assert result.success is True


# ===========================================================================
# 12. GrowingAIBenchmark — run_all (3 tests)
# ===========================================================================

def test_run_all_returns_benchmark_report():
    bench = GrowingAIBenchmark(patterns=["p5", "p8"])
    report = bench.run_all()
    assert isinstance(report, BenchmarkReport)


def test_run_all_result_count_matches_patterns():
    bench = GrowingAIBenchmark(patterns=["p2", "p7", "p10"])
    report = bench.run_all()
    assert report.n_patterns == 3
    assert len(report.results) == 3


def test_run_all_success_rate_in_range():
    bench = GrowingAIBenchmark(patterns=["p5", "p6"])
    report = bench.run_all()
    assert 0.0 <= report.success_rate <= 1.0


# ===========================================================================
# 13. BenchmarkReport — 集計 (5 tests)
# ===========================================================================

def test_benchmark_report_avg_improvement_computed():
    r1 = PatternBenchResult(pattern_id="p1", pattern_name="P1", baseline_score=0.5, final_score=0.6)
    r2 = PatternBenchResult(pattern_id="p2", pattern_name="P2", baseline_score=0.5, final_score=0.7)
    report = BenchmarkReport(
        timestamp="2026-06-02T00:00:00",
        results=[r1, r2],
        n_patterns=2,
        n_success=2,
        avg_improvement=round((r1.improvement + r2.improvement) / 2, 4),
        avg_improvement_pct=round((r1.improvement_pct + r2.improvement_pct) / 2, 2),
        avg_latency_ms=0.0,
        total_latency_ms=0.0,
    )
    assert abs(report.avg_improvement - 0.15) < 1e-3


def test_benchmark_report_success_rate():
    r1 = PatternBenchResult(pattern_id="p1", pattern_name="P1", baseline_score=0.5, final_score=0.6, success=True)
    r2 = PatternBenchResult(pattern_id="p2", pattern_name="P2", baseline_score=0.5, final_score=0.5, success=False)
    report = BenchmarkReport(
        timestamp="2026-06-02T00:00:00",
        results=[r1, r2], n_patterns=2, n_success=1,
        avg_improvement=0.0, avg_improvement_pct=0.0,
        avg_latency_ms=0.0, total_latency_ms=0.0,
    )
    assert report.success_rate == pytest.approx(0.5, abs=1e-4)


def test_benchmark_report_to_dict():
    r = PatternBenchResult(pattern_id="p5", pattern_name="P5", baseline_score=0.5, final_score=0.6)
    report = BenchmarkReport(
        timestamp="2026-06-02T00:00:00",
        results=[r], n_patterns=1, n_success=1,
        avg_improvement=r.improvement, avg_improvement_pct=r.improvement_pct,
        avg_latency_ms=0.0, total_latency_ms=0.0,
    )
    d = report.to_dict()
    assert "results" in d
    assert isinstance(d["results"], list)
    assert d["n_patterns"] == 1


def test_format_summary_returns_string():
    bench = GrowingAIBenchmark(patterns=["p5"])
    report = bench.run_all()
    summary = bench.format_summary(report)
    assert isinstance(summary, str)
    assert "GrowingAIBench" in summary


def test_pattern_funcs_has_10_patterns():
    assert len(_PATTERN_FUNCS) == 10
    for i in range(1, 10):
        assert f"p{i}" in _PATTERN_FUNCS
    assert "p10" in _PATTERN_FUNCS


# ===========================================================================
# 14. save / load (2 tests)
# ===========================================================================

def test_save_creates_json_file():
    bench = GrowingAIBenchmark(patterns=["p5"])
    report = bench.run_all()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test_bench.json"
        saved = bench.save(report, str(out))
        assert saved.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "results" in data
        assert data["n_patterns"] == 1


def test_load_roundtrips_report():
    bench = GrowingAIBenchmark(patterns=["p5", "p6"])
    report = bench.run_all()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = str(Path(tmpdir) / "bench.json")
        bench.save(report, out)
        loaded = bench.load(out)
        assert loaded.n_patterns == report.n_patterns
        assert len(loaded.results) == len(report.results)
        assert loaded.results[0].pattern_id == report.results[0].pattern_id

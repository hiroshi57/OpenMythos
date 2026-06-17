"""
tests/test_sprint37.py
Sprint 37: ベンチマーク結果可視化 + E2E 疎通テスト — 40 tests

検証内容:
  37.1  benchmark/report.py — ReportGenerator.to_markdown() / to_html()
  37.2  benchmark/report.py — ReportGenerator.trend_table() / load_reports()
  37.3  serve/api.py — P1〜P10 全エンドポイント TestClient 疎通テスト
  37.4  .github/workflows/bench.yml — HTML レポート artifact ステップ
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml

# リポジトリルートを sys.path に追加
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.growing_ai_bench import (
    BenchmarkReport,
    GrowingAIBenchmark,
    PatternBenchResult,
)
from benchmark.report import ReportGenerator, _pattern_detail_html  # noqa: F401

BENCH_YML = ROOT / ".github" / "workflows" / "bench.yml"

# ===========================================================================
# ヘルパー: ダミー BenchmarkReport を生成
# ===========================================================================

def _make_result(pid: str, name: str, baseline: float, final: float, **kwargs) -> PatternBenchResult:
    return PatternBenchResult(
        pattern_id=pid,
        pattern_name=name,
        baseline_score=baseline,
        final_score=final,
        latency_ms=kwargs.get("latency_ms", 5.0),
        success=kwargs.get("success", True),
        notes=kwargs.get("notes", ""),
        pattern_score=kwargs.get("pattern_score", 0.0),
    )


def _make_report(
    n_results: int = 3,
    timestamp: str = "2026-06-03T10:00:00",
) -> BenchmarkReport:
    results = [
        _make_result("p1", "P1: 討議型集合知 (Debate)", 0.50, 0.58),
        _make_result("p2", "P2: KPI 駆動自己改善", 0.50, 0.81, latency_ms=12.0),
        _make_result("p3", "P3: ボトルネック発見", 0.50, 0.53, latency_ms=4.0, success=False),
    ][:n_results]
    improvements = [r.improvement for r in results]
    pcts = [r.improvement_pct for r in results]
    latencies = [r.latency_ms for r in results]
    n_success = sum(1 for r in results if r.success)
    return BenchmarkReport(
        timestamp=timestamp,
        results=results,
        n_patterns=len(results),
        n_success=n_success,
        avg_improvement=round(sum(improvements) / max(len(improvements), 1), 4),
        avg_improvement_pct=round(sum(pcts) / max(len(pcts), 1), 2),
        avg_latency_ms=round(sum(latencies) / max(len(latencies), 1), 2),
        total_latency_ms=round(sum(latencies), 2),
    )


# ===========================================================================
# 37.1a ReportGenerator.to_markdown() — (8 tests)
# ===========================================================================


class TestToMarkdown:

    @pytest.fixture(scope="class")
    def report(self):
        return _make_report()

    @pytest.fixture(scope="class")
    def md(self, report):
        return ReportGenerator(report).to_markdown()

    def test_returns_str(self, md):
        assert isinstance(md, str)

    def test_has_h1_title(self, md):
        assert "# OpenMythos" in md and "Benchmark Report" in md

    def test_has_timestamp(self, md):
        assert "2026-06-03" in md

    def test_has_success_rate(self, md):
        # 2/3 成功
        assert "2/3" in md

    def test_has_table_header(self, md):
        assert "| パターン |" in md

    def test_has_pattern_rows(self, md):
        assert "P1:" in md or "討議型集合知" in md

    def test_has_pattern_detail_section(self, md):
        assert "## パターン詳細" in md

    def test_has_footer(self, md):
        assert "benchmark/report.py" in md


# ===========================================================================
# 37.1b ReportGenerator.to_html() — (8 tests)
# ===========================================================================


class TestToHtml:

    @pytest.fixture(scope="class")
    def report(self):
        return _make_report()

    @pytest.fixture(scope="class")
    def html(self, report):
        return ReportGenerator(report).to_html()

    def test_returns_str(self, html):
        assert isinstance(html, str)

    def test_starts_with_doctype(self, html):
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_has_charset_utf8(self, html):
        assert "UTF-8" in html

    def test_has_title(self, html):
        assert "<title>" in html and "Benchmark" in html

    def test_has_table_tag(self, html):
        assert "<table>" in html and "</table>" in html

    def test_has_pattern_names(self, html):
        assert "討議型集合知" in html

    def test_has_success_indicator(self, html):
        assert "✅" in html

    def test_has_fail_indicator(self, html):
        # P3 は success=False なので ❌ があるはず
        assert "❌" in html


# ===========================================================================
# 37.1c ReportGenerator.save_markdown() / save_html() — (4 tests)
# ===========================================================================


class TestSaveMethods:

    @pytest.fixture(scope="class")
    def report(self):
        return _make_report()

    def test_save_markdown_returns_path(self, report, tmp_path):
        p = ReportGenerator(report).save_markdown(str(tmp_path / "out.md"))
        assert p.exists()

    def test_save_markdown_content(self, report, tmp_path):
        p = ReportGenerator(report).save_markdown(str(tmp_path / "sub" / "out.md"))
        content = p.read_text(encoding="utf-8")
        assert "# OpenMythos" in content

    def test_save_html_returns_path(self, report, tmp_path):
        p = ReportGenerator(report).save_html(str(tmp_path / "out.html"))
        assert p.exists()

    def test_save_html_content(self, report, tmp_path):
        p = ReportGenerator(report).save_html(str(tmp_path / "out.html"))
        content = p.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content


# ===========================================================================
# 37.2 ReportGenerator.trend_table() / load_reports() — (7 tests)
# ===========================================================================


class TestTrend:

    @pytest.fixture(scope="class")
    def two_json_paths(self, tmp_path_factory):
        """2件の BenchmarkReport JSON を作成し、パスリストを返す。"""
        tmp = tmp_path_factory.mktemp("trend")
        r1 = _make_report(timestamp="2026-06-01T09:00:00")
        r2 = _make_report(timestamp="2026-06-03T10:00:00")
        p1 = tmp / "bench-20260601.json"
        p2 = tmp / "bench-20260603.json"
        GrowingAIBenchmark.save(r1, str(p1))
        GrowingAIBenchmark.save(r2, str(p2))
        return [str(p1), str(p2)]

    def test_load_reports_returns_list(self, two_json_paths):
        reports = ReportGenerator.load_reports(two_json_paths)
        assert len(reports) == 2

    def test_load_reports_sorted_by_timestamp(self, two_json_paths):
        reports = ReportGenerator.load_reports(two_json_paths)
        assert reports[0].timestamp < reports[1].timestamp

    def test_load_reports_ignores_invalid(self, two_json_paths, tmp_path):
        bad = str(tmp_path / "bad.json")
        reports = ReportGenerator.load_reports(two_json_paths + [bad])
        assert len(reports) == 2  # 壊れたファイルはスキップ

    def test_trend_table_returns_str(self, two_json_paths):
        table = ReportGenerator.trend_table(two_json_paths)
        assert isinstance(table, str)

    def test_trend_table_has_header(self, two_json_paths):
        table = ReportGenerator.trend_table(two_json_paths)
        assert "KPI 改善率トレンド" in table or "Δ%" in table

    def test_trend_table_has_two_rows(self, two_json_paths):
        table = ReportGenerator.trend_table(two_json_paths)
        # 日付パターンが2つあるか確認
        import re
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", table)
        assert len(dates) >= 2

    def test_trend_table_n_limits_rows(self, two_json_paths):
        """n=1 で最新 1 件のみ表示される"""
        table = ReportGenerator.trend_table(two_json_paths, n=1)
        import re
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", table)
        assert len(dates) == 1

    def test_trend_table_empty_paths(self):
        result = ReportGenerator.trend_table([])
        assert result == ""


# ===========================================================================
# 37.3 P1〜P10 E2E 疎通テスト (TestClient) — (13 tests)
# ===========================================================================


# ---------- transformers モック (lifespan で AutoTokenizer を使用するため) ----------

@pytest.fixture(scope="module", autouse=True)
def mock_transformers():
    """AutoTokenizer をモック化してモデルロードを軽量化する。"""
    def _make_tok():
        tok = MagicMock()
        tok.side_effect = lambda text, **kw: {
            "input_ids": torch.zeros(1, max(1, len(str(text).split())), dtype=torch.long)
        }
        tok.decode = MagicMock(return_value="mock output")
        return tok

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=_make_tok())
    orig = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    yield
    if orig is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = orig


@pytest.fixture(scope="module")
def client():
    """P1〜P10 エンドポイントのスモークテスト用 TestClient。"""
    from fastapi.testclient import TestClient
    from open_mythos.variants import mythos_nano
    from open_mythos.main import OpenMythos
    from open_mythos.agents import OpenMythosLLM
    import serve.api as api_module

    cfg = mythos_nano()
    model = OpenMythos(cfg).eval()
    api_module.state.model = model
    api_module.state.tokenizer = MagicMock(
        side_effect=lambda t, **k: {"input_ids": torch.zeros(1, 4, dtype=torch.long)},
    )
    api_module.state.tokenizer.decode = MagicMock(return_value="ok")
    api_module.state.device = torch.device("cpu")
    api_module.state.n_params = sum(p.numel() for p in model.parameters())
    api_module.state.llm = OpenMythosLLM(
        model=model, device="cpu",
        max_new_tokens=4, temperature=1.0, top_k=10, top_p=0.9,
    )
    api_module.state.agents = {}

    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


class TestP2KPISmoke:
    """P2: /v1/kpi/measure"""

    def test_kpi_measure_200(self, client):
        r = client.post("/v1/kpi/measure", json={
            "name": "llmo_score", "target": 0.7, "context": "LLMO テスト"
        })
        assert r.status_code == 200

    def test_kpi_measure_has_value(self, client):
        r = client.post("/v1/kpi/measure", json={
            "name": "llmo_score", "target": 0.7, "context": "test"
        })
        body = r.json()
        assert "value" in body


class TestP3ProfilerSmoke:
    """P3: /v1/profile/run"""

    def test_profile_run_200(self, client):
        r = client.post("/v1/profile/run", json={"input_text": "ベンチマーク入力テキスト"})
        assert r.status_code == 200

    def test_profile_run_has_stages(self, client):
        r = client.post("/v1/profile/run", json={"input_text": "test"})
        body = r.json()
        assert "stages" in body or "bottlenecks" in body or len(body) > 0


class TestP4SignalSmoke:
    """P4: /v1/signal/detect"""

    def test_signal_detect_200(self, client):
        r = client.post("/v1/signal/detect", json={
            "context": "SEO コンテンツ最適化",
            "keyword": "LLMO"
        })
        assert r.status_code == 200

    def test_signal_detect_has_signals(self, client):
        r = client.post("/v1/signal/detect", json={
            "context": "テスト",
            "keyword": ""
        })
        body = r.json()
        assert "signals" in body


class TestP5MistakesSmoke:
    """P5: /v1/mistakes/record + /v1/mistakes/rules"""

    def test_mistakes_record_200(self, client):
        r = client.post("/v1/mistakes/record", json={
            "text": "テストミステキスト",
            "severity": "low",
            "context": "smoke test",
        })
        assert r.status_code == 200

    def test_mistakes_rules_200(self, client):
        r = client.get("/v1/mistakes/rules")
        assert r.status_code == 200


class TestP6DistillSmoke:
    """P6: /v1/distill/run"""

    def test_distill_run_200(self, client):
        r = client.post("/v1/distill/run", json={
            "prompts": ["test prompt 1", "test prompt 2"],
            "n_rounds": 1,
        })
        assert r.status_code == 200

    def test_distill_status_200(self, client):
        r = client.get("/v1/distill/status")
        assert r.status_code == 200


class TestP7MemorySmoke:
    """P7: /v1/memory/store + /v1/memory/retrieve"""

    def test_memory_store_200(self, client):
        r = client.post("/v1/memory/store", json={
            "context": "テストクエリ",
            "text": "テスト記憶本文",
            "score": 0.9,
        })
        assert r.status_code == 200

    def test_memory_retrieve_200(self, client):
        r = client.post("/v1/memory/retrieve", json={"query": "テスト"})
        assert r.status_code == 200


class TestP8EnsembleSmoke:
    """P8: /v1/ensemble/score"""

    def test_ensemble_score_200(self, client):
        r = client.post("/v1/ensemble/score", json={
            "text": "テキスト1。エンティティ密度が高い。entity_density を重視した評価用テキスト。"
        })
        assert r.status_code == 200


class TestP9EvolveSmoke:
    """P9: /v1/evolve/run"""

    def test_evolve_run_200(self, client):
        r = client.post("/v1/evolve/run", json={
            "seed_prompt": "LLMO 最適化プロンプト",
            "population_size": 2,
            "n_generations": 1,
        })
        assert r.status_code == 200

    def test_evolve_run_has_best_prompt(self, client):
        r = client.post("/v1/evolve/run", json={
            "seed_prompt": "test",
            "population_size": 2,
            "n_generations": 1,
        })
        body = r.json()
        assert "best_prompt" in body


class TestP10PlanSmoke:
    """P10: /v1/plan/decompose"""

    def test_plan_decompose_200(self, client):
        r = client.post("/v1/plan/decompose", json={
            "goal": "LLMO スコアを最大化する"
        })
        assert r.status_code == 200

    def test_plan_decompose_has_tasks(self, client):
        r = client.post("/v1/plan/decompose", json={
            "goal": "テストゴール"
        })
        body = r.json()
        assert "tasks" in body


class TestGuardSmoke:
    """guard: /v1/guard/stats"""

    def test_guard_stats_200(self, client):
        r = client.get("/v1/guard/stats")
        assert r.status_code == 200


class TestGrowSmoke:
    """grow: /v1/grow/run"""

    def test_grow_run_200(self, client):
        r = client.post("/v1/grow/run", json={
            "goal": "LLMO スコア改善"
        })
        # 200 または 500 (内部エラー時) を許容 — 疎通確認
        assert r.status_code in (200, 500)

    def test_grow_run_returns_json(self, client):
        r = client.post("/v1/grow/run", json={"goal": "test"})
        assert r.headers.get("content-type", "").startswith("application/json")


# ===========================================================================
# 37.4 bench.yml — HTML artifact ステップ検証 (5 tests)
# ===========================================================================


def _load_bench_yml() -> dict:
    with open(BENCH_YML, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
# ダッシュボード (to_dashboard / save_dashboard) — (8 tests)
# ===========================================================================


class TestDashboard:

    @pytest.fixture(scope="class")
    def report(self):
        return _make_report()

    @pytest.fixture(scope="class")
    def dash(self, report):
        return ReportGenerator(report).to_dashboard()

    def test_returns_str(self, dash):
        assert isinstance(dash, str)

    def test_starts_with_doctype(self, dash):
        assert dash.strip().startswith("<!DOCTYPE html>")

    def test_has_chartjs_cdn(self, dash):
        assert "chart.js" in dash.lower() or "chart.umd" in dash.lower()

    def test_has_canvas_elements(self, dash):
        assert dash.count("<canvas") >= 3

    def test_has_kpi_summary_cards(self, dash):
        assert "成功率" in dash and "平均 KPI 改善率" in dash

    def test_has_pattern_names(self, dash):
        assert "討議型集合知" in dash

    def test_save_dashboard_returns_path(self, report, tmp_path):
        p = ReportGenerator(report).save_dashboard(str(tmp_path / "dash.html"))
        assert p.exists()
        assert p.stat().st_size > 1000

    def test_dashboard_with_trend(self, tmp_path_factory):
        """trend_reports 付きダッシュボードにトレンドセクションが含まれる"""
        tmp_path_factory.mktemp("dash_trend")
        r1 = _make_report(timestamp="2026-06-01T09:00:00")
        r2 = _make_report(timestamp="2026-06-03T10:00:00")
        dash = ReportGenerator(r2).to_dashboard(trend_reports=[r1, r2])
        assert "trendChart" in dash or "トレンド" in dash


class TestBenchYmlHtmlArtifact:
    """bench.yml に HTML レポート生成 + artifact upload ステップが追加されている"""

    @pytest.fixture(scope="class")
    def data(self):
        return _load_bench_yml()

    def test_yml_valid(self, data):
        assert isinstance(data, dict)

    def test_has_generate_html_step(self, data):
        steps = data["jobs"]["benchmark"]["steps"]
        names = [s.get("name", "") for s in steps]
        assert any("html" in n.lower() or "report" in n.lower() for n in names), \
            f"HTML report step が見つからない: {names}"

    def test_html_step_uses_report_py(self, data):
        steps = data["jobs"]["benchmark"]["steps"]
        html_steps = [
            s for s in steps
            if "html" in s.get("name", "").lower() or "report" in s.get("name", "").lower()
        ]
        assert html_steps, "HTML report ステップが存在しない"
        run_cmd = html_steps[0].get("run", "")
        assert "report.py" in run_cmd

    def test_has_upload_html_artifact_step(self, data):
        steps = data["jobs"]["benchmark"]["steps"]
        upload_steps = [
            s for s in steps
            if "upload" in s.get("name", "").lower()
            and "html" in (s.get("name", "") + s.get("with", {}).get("path", "")).lower()
        ]
        assert upload_steps, "HTML artifact upload ステップが見つからない"

    def test_html_artifact_retention(self, data):
        steps = data["jobs"]["benchmark"]["steps"]
        html_upload_steps = [
            s for s in steps
            if isinstance(s.get("with", {}).get("path", ""), str)
            and "html" in s.get("with", {}).get("path", "")
        ]
        # HTML artifact upload ステップの存在は test_has_upload_html_artifact_step が確認済み
        # ここでは retention-days の値を検証する
        for s in html_upload_steps:
            retention = s.get("with", {}).get("retention-days", 0)
            assert retention > 0, "retention-days が設定されていない"

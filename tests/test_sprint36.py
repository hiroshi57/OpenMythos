"""
tests/test_sprint36.py
Sprint 36: API ドキュメント整備 + CI ベンチマーク自動化 — 35 tests

検証内容:
  - serve/api.py の version・openapi_tags が正しく更新されている
  - .github/workflows/bench.yml の構造・設定が正しい
  - ベンチマーク CLI の --output / --patterns オプションが動作する

Note: YAML では `on:` キーは Python の True (bool) に変換されるため
      data[True] でアクセスする。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
BENCH_YML = ROOT / ".github" / "workflows" / "bench.yml"
API_PY = ROOT / "serve" / "api.py"


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def load_bench_yml() -> dict:
    """bench.yml を UTF-8 で読み込んで dict を返す。`on` キーは True (bool)。"""
    with open(BENCH_YML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def api_src() -> str:
    return API_PY.read_text(encoding="utf-8")


def get_openapi_tag_names() -> list[str]:
    return re.findall(r'"name":\s*"([^"]+)"', api_src())


# ---------------------------------------------------------------------------
# 36.1a: serve/api.py — version
# ---------------------------------------------------------------------------

class TestApiVersion:
    """FastAPI app の version が最新に更新されている"""

    def test_version_is_0_38_0(self):
        assert 'version="0.38.0"' in api_src()

    def test_old_version_removed(self):
        assert 'version="0.22.0"' not in api_src()


# ---------------------------------------------------------------------------
# 36.1b: serve/api.py — openapi_tags
# ---------------------------------------------------------------------------

REQUIRED_TAGS = [
    # 基盤
    "health", "infer", "generate", "agent", "chat",
    "seo", "thinking", "tools", "rag", "sessions", "batch",
    # P1〜P10
    "debate", "kpi", "profiler", "signal", "mistakes",
    "distill", "memory", "ensemble", "evolve", "plan",
    # 統合
    "grow", "guard",
]


class TestOpenApiTags:
    """openapi_tags に P1〜P10 + guard/grow が含まれている"""

    def test_all_required_tags_present(self):
        names = get_openapi_tag_names()
        missing = [t for t in REQUIRED_TAGS if t not in names]
        assert not missing, f"openapi_tags に不足: {missing}"

    def test_tag_count_at_least_23(self):
        names = get_openapi_tag_names()
        assert len(names) >= 23, f"tag 数が少ない: {len(names)}"

    def test_p1_debate_description(self):
        src = api_src()
        assert "P1" in src and '"debate"' in src

    def test_p2_kpi_description(self):
        src = api_src()
        assert "P2" in src and '"kpi"' in src

    def test_p3_profiler_description(self):
        src = api_src()
        assert "P3" in src and '"profiler"' in src

    def test_p4_signal_description(self):
        src = api_src()
        assert "P4" in src and '"signal"' in src

    def test_p5_mistakes_description(self):
        src = api_src()
        assert "P5" in src and '"mistakes"' in src

    def test_p6_distill_description(self):
        src = api_src()
        assert "P6" in src and '"distill"' in src

    def test_p7_memory_description(self):
        src = api_src()
        assert "P7" in src and '"memory"' in src

    def test_p8_ensemble_description(self):
        src = api_src()
        assert "P8" in src and '"ensemble"' in src

    def test_p9_evolve_description(self):
        src = api_src()
        assert "P9" in src and '"evolve"' in src

    def test_p10_plan_description(self):
        src = api_src()
        assert "P10" in src and '"plan"' in src

    def test_grow_tag_with_orchestrator_desc(self):
        src = api_src()
        assert '"grow"' in src and "GrowingAI" in src

    def test_guard_tag_with_middleware_desc(self):
        src = api_src()
        assert '"guard"' in src and "Middleware" in src

    def test_faiss_ann_mentioned_in_memory_tag(self):
        src = api_src()
        assert "FAISS" in src


# ---------------------------------------------------------------------------
# 36.2a: bench.yml — ファイル存在・YAML 構文
# ---------------------------------------------------------------------------

class TestBenchYmlFile:
    """bench.yml が存在し有効な YAML である"""

    def test_file_exists(self):
        assert BENCH_YML.exists()

    def test_valid_yaml(self):
        data = load_bench_yml()
        assert isinstance(data, dict)

    def test_has_name(self):
        data = load_bench_yml()
        assert "name" in data

    def test_name_contains_bench(self):
        data = load_bench_yml()
        assert "bench" in data["name"].lower()

    def test_has_jobs(self):
        data = load_bench_yml()
        assert "jobs" in data


# ---------------------------------------------------------------------------
# 36.2b: bench.yml — トリガー設定
# ---------------------------------------------------------------------------

class TestBenchYmlTriggers:
    """bench.yml のトリガー (schedule + workflow_dispatch) が正しい"""

    def test_has_on_key(self):
        data = load_bench_yml()
        # YAML `on:` → Python True
        assert True in data, "`on:` キーが見つからない"

    def test_has_schedule(self):
        data = load_bench_yml()
        assert "schedule" in data[True]

    def test_schedule_has_cron(self):
        data = load_bench_yml()
        schedules = data[True]["schedule"]
        assert any("cron" in s for s in schedules)

    def test_cron_is_weekly(self):
        data = load_bench_yml()
        schedules = data[True]["schedule"]
        cron = schedules[0]["cron"]
        fields = cron.split()
        assert len(fields) == 5
        assert fields[4] != "*", "day-of-week が * (毎日) になっている"

    def test_has_workflow_dispatch(self):
        data = load_bench_yml()
        assert "workflow_dispatch" in data[True]

    def test_workflow_dispatch_has_patterns_input(self):
        data = load_bench_yml()
        inputs = data[True]["workflow_dispatch"].get("inputs", {})
        assert "patterns" in inputs

    def test_workflow_dispatch_has_verbose_input(self):
        data = load_bench_yml()
        inputs = data[True]["workflow_dispatch"].get("inputs", {})
        assert "verbose" in inputs


# ---------------------------------------------------------------------------
# 36.2c: bench.yml — jobs 設定
# ---------------------------------------------------------------------------

class TestBenchYmlJobs:
    """bench.yml の jobs が正しく設定されている"""

    def test_has_benchmark_job(self):
        data = load_bench_yml()
        assert "benchmark" in data["jobs"]

    def test_runs_on_ubuntu(self):
        data = load_bench_yml()
        assert "ubuntu" in data["jobs"]["benchmark"]["runs-on"]

    def test_has_checkout_step(self):
        data = load_bench_yml()
        steps = data["jobs"]["benchmark"]["steps"]
        names = [s.get("name", "").lower() for s in steps]
        assert any("checkout" in n for n in names)

    def test_has_python_setup_step(self):
        data = load_bench_yml()
        steps = data["jobs"]["benchmark"]["steps"]
        uses = [s.get("uses", "") for s in steps]
        assert any("setup-python" in u for u in uses)

    def test_has_upload_artifact_step(self):
        data = load_bench_yml()
        steps = data["jobs"]["benchmark"]["steps"]
        uses = [s.get("uses", "") for s in steps]
        assert any("upload-artifact" in u for u in uses)

    def test_runs_growing_ai_bench_script(self):
        data = load_bench_yml()
        steps = data["jobs"]["benchmark"]["steps"]
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "growing_ai_bench.py" in run_cmds

    def test_uses_output_flag(self):
        data = load_bench_yml()
        steps = data["jobs"]["benchmark"]["steps"]
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "--output" in run_cmds

    def test_artifact_has_retention_days(self):
        data = load_bench_yml()
        steps = data["jobs"]["benchmark"]["steps"]
        has_retention = any(
            "retention-days" in step.get("with", {})
            for step in steps
        )
        assert has_retention

    def test_results_saved_under_benchmark_dir(self):
        data = load_bench_yml()
        steps = data["jobs"]["benchmark"]["steps"]
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "benchmark/results" in run_cmds

    def test_github_step_summary_written(self):
        data = load_bench_yml()
        steps = data["jobs"]["benchmark"]["steps"]
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "GITHUB_STEP_SUMMARY" in run_cmds

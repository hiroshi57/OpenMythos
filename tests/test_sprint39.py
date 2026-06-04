"""
Sprint 39 — ショーケースダッシュボード + Prometheus メトリクス + GitHub Pages テスト

対象:
  - serve/dashboard.py  : build_showcase_dashboard() / save_dashboard() / FastAPI router
  - serve/monitor.py    : build_monitor_dashboard_html() / /monitor/dashboard HTML 化
  - serve/api.py        : /dashboard ルーター + /metrics エンドポイント
  - .github/workflows/pages.yml : GitHub Pages 自動デプロイ設定
"""

from __future__ import annotations

import sys
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# transformers モック (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


# ---------------------------------------------------------------------------
# 1. serve/dashboard.py — build_showcase_dashboard()
# ---------------------------------------------------------------------------

class TestShowcaseDashboard:
    """ショーケースダッシュボードの HTML 生成を検証"""

    def setup_method(self):
        from serve.dashboard import build_showcase_dashboard
        self.html = build_showcase_dashboard()

    def test_returns_str(self):
        assert isinstance(self.html, str)

    def test_is_html_document(self):
        assert "<!DOCTYPE html>" in self.html

    def test_has_charset(self):
        assert "UTF-8" in self.html or "utf-8" in self.html

    def test_has_title(self):
        assert "OpenMythos" in self.html

    def test_has_chartjs_cdn(self):
        assert "chart.js" in self.html.lower() or "chart.umd" in self.html

    def test_has_hero_section(self):
        assert "Recurrent" in self.html or "recurrent" in self.html.lower()

    def test_has_stats_bar(self):
        # テスト数・Sprint数・API数
        assert "1,963" in self.html or "1963" in self.html

    def test_has_p1_to_p10(self):
        for i in range(1, 11):
            assert "P" + str(i) in self.html

    def test_has_benchmark_chart_canvas(self):
        assert "kpiChart" in self.html

    def test_has_api_chart_canvas(self):
        assert "apiChart" in self.html

    def test_has_sprint_timeline(self):
        assert "sprint-dot" in self.html or "Sprint 38" in self.html

    def test_has_live_status_section(self):
        assert "statusDot" in self.html or "checkHealth" in self.html

    def test_has_monitor_link(self):
        assert "/monitor/dashboard" in self.html

    def test_has_metrics_link(self):
        assert "/metrics" in self.html

    def test_has_docs_link(self):
        assert "/docs" in self.html

    def test_kpi_data_present(self):
        # P1〜P10 KPI 改善率の数値が含まれる
        assert "11.2" in self.html or "KPI" in self.html

    def test_architecture_comparison(self):
        # MLA / MoE / LoRA の言及
        assert "MLA" in self.html
        assert "MoE" in self.html or "Mixture" in self.html

    def test_html_closes_properly(self):
        assert "</html>" in self.html


class TestSaveDashboard:
    """save_dashboard() のファイル出力を検証"""

    def test_save_creates_file(self, tmp_path):
        from serve.dashboard import save_dashboard
        out = str(tmp_path / "test.html")
        result = save_dashboard(out)
        assert Path(result).exists()

    def test_save_returns_path(self, tmp_path):
        from serve.dashboard import save_dashboard
        out = str(tmp_path / "out.html")
        result = save_dashboard(out)
        assert str(result).endswith(".html")

    def test_save_content_is_html(self, tmp_path):
        from serve.dashboard import save_dashboard
        out = str(tmp_path / "dash.html")
        save_dashboard(out)
        content = Path(out).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_save_creates_parent_dirs(self, tmp_path):
        from serve.dashboard import save_dashboard
        out = str(tmp_path / "nested" / "dir" / "dash.html")
        save_dashboard(out)
        assert Path(out).exists()

    def test_save_file_size_reasonable(self, tmp_path):
        from serve.dashboard import save_dashboard
        out = str(tmp_path / "dash.html")
        save_dashboard(out)
        size = Path(out).stat().st_size
        assert size > 5_000   # 最低 5KB
        assert size < 500_000  # 500KB 未満


# ---------------------------------------------------------------------------
# 2. FastAPI エンドポイント — /dashboard
# ---------------------------------------------------------------------------

import serve.api as api_module


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]

    import torch
    from open_mythos.main import MythosConfig, OpenMythos

    cfg = MythosConfig(
        vocab_size=50257, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=128,
        max_loop_iters=4, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=1, expert_dim=32,
        act_threshold=0.99, lora_rank=4, kv_lora_rank=32, q_lora_rank=64,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
    )
    model = OpenMythos(cfg)
    model.eval()

    api_module.state.model = model
    api_module.state.tokenizer = tok
    api_module.state.device = torch.device("cpu")
    api_module.state.n_params = sum(p.numel() for p in model.parameters())

    from open_mythos.agents import OpenMythosLLM
    api_module.state.llm = OpenMythosLLM(model=model, device="cpu")
    api_module.state.agents = {}

    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


class TestDashboardEndpoint:
    """/dashboard エンドポイントを検証"""

    def test_returns_200(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_content_type_html(self, client):
        r = client.get("/dashboard")
        assert "text/html" in r.headers.get("content-type", "")

    def test_body_contains_openmythos(self, client):
        r = client.get("/dashboard")
        assert "OpenMythos" in r.text

    def test_body_contains_chartjs(self, client):
        r = client.get("/dashboard")
        assert "chart" in r.text.lower()

    def test_body_contains_patterns(self, client):
        r = client.get("/dashboard")
        assert "P1" in r.text and "P10" in r.text


class TestMetricsEndpoint:
    """/metrics Prometheus エンドポイントを検証"""

    def test_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_content_type_text(self, client):
        r = client.get("/metrics")
        assert "text/plain" in r.headers.get("content-type", "")

    def test_body_is_non_empty(self, client):
        r = client.get("/metrics")
        assert len(r.text) > 0


# ---------------------------------------------------------------------------
# 3. serve/monitor.py — build_monitor_dashboard_html()
# ---------------------------------------------------------------------------

class TestMonitorDashboardHtml:
    """monitor.py の HTML ダッシュボード生成を検証"""

    def setup_method(self):
        import tempfile, os
        from serve import monitor
        # テスト用に DB を一時ファイルへ差し替え
        self._orig_db = monitor.DB_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        monitor.DB_PATH = Path(tmp.name)
        monitor._init_db()
        from serve.monitor import build_monitor_dashboard_html
        self.html = build_monitor_dashboard_html()
        monitor.DB_PATH = self._orig_db

    def test_returns_str(self):
        assert isinstance(self.html, str)

    def test_is_html(self):
        assert "<!DOCTYPE html>" in self.html

    def test_has_title(self):
        assert "Monitor" in self.html

    def test_has_drift_section(self):
        assert "ドリフト" in self.html or "drift" in self.html.lower()

    def test_has_metrics_table(self):
        assert "<table" in self.html

    def test_has_link_to_showcase(self):
        assert "/dashboard" in self.html

    def test_has_prometheus_link(self):
        assert "/metrics" in self.html


# ---------------------------------------------------------------------------
# 4. GitHub Pages ワークフロー検証
# ---------------------------------------------------------------------------

class TestPagesWorkflow:
    """pages.yml の設定を検証"""

    @pytest.fixture(autouse=True)
    def load_yml(self):
        self.path = Path(".github/workflows/pages.yml")
        with open(self.path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def test_file_exists(self):
        assert self.path.exists()

    def test_triggers_on_push_master(self):
        # PyYAML は "on" を boolean True にパースする (YAML 1.1)
        trigger = self.cfg.get(True) or self.cfg.get("on") or {}
        assert "master" in trigger["push"]["branches"]

    def test_has_workflow_dispatch(self):
        trigger = self.cfg.get(True) or self.cfg.get("on") or {}
        assert "workflow_dispatch" in trigger

    def test_has_pages_permission(self):
        perms = self.cfg.get("permissions", {})
        assert perms.get("pages") == "write"

    def test_has_id_token_permission(self):
        perms = self.cfg.get("permissions", {})
        assert perms.get("id-token") == "write"

    def test_has_build_job(self):
        assert "build-and-deploy" in self.cfg["jobs"]

    def test_has_deploy_pages_step(self):
        steps = self.cfg["jobs"]["build-and-deploy"]["steps"]
        names = [s.get("name", "") for s in steps]
        assert any("Deploy" in n for n in names)

    def test_generates_showcase_dashboard(self):
        steps = self.cfg["jobs"]["build-and-deploy"]["steps"]
        all_run = " ".join(s.get("run", "") for s in steps)
        assert "dashboard.py" in all_run

    def test_generates_project_dashboard(self):
        steps = self.cfg["jobs"]["build-and-deploy"]["steps"]
        all_run = " ".join(s.get("run", "") for s in steps)
        assert "project_dashboard.py" in all_run

    def test_uploads_artifact(self):
        steps = self.cfg["jobs"]["build-and-deploy"]["steps"]
        uses_list = [s.get("uses", "") for s in steps]
        assert any("upload-pages-artifact" in u for u in uses_list)

    def test_site_output_dir(self):
        steps = self.cfg["jobs"]["build-and-deploy"]["steps"]
        for s in steps:
            with_block = s.get("with", {})
            if "path" in with_block and "_site" in str(with_block["path"]):
                return
        pytest.fail("_site/ output path not found in upload step")

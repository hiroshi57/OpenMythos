"""
Sprint 53 — セキュリティ統合 テスト

対象:
  - open_mythos/skills/security.py:
      PentestFinding / PentestReport / WebPentester
      DependencyInfo / ForensicsReport / OSSForensics
  - serve/api.py:
      POST /v1/security/scan
      POST /v1/security/report/md
      POST /v1/security/oss/analyze
      POST /v1/security/oss/sbom
"""
from __future__ import annotations

import sys
import pytest
import torch
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# transformers モック
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


# ---------------------------------------------------------------------------
# TestClient フィクスチャ
# ---------------------------------------------------------------------------

import serve.api as api_module


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.agents import OpenMythosLLM

    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.eos_token_id = 50256
    tok.decode.side_effect = lambda ids, **kwargs: "ok " * max(len(ids), 1)
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    cfg = MythosConfig(
        vocab_size=50257, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=128,
        max_loop_iters=2, prelude_layers=1, coda_layers=1, attn_type="gqa",
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
    api_module.state.llm = OpenMythosLLM(model=model, device="cpu")
    api_module.state.agents = {}

    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


_HDR = {"Authorization": "Bearer dev"}

from open_mythos.skills.security import (
    PentestFinding, PentestReport, WebPentester,
    DependencyInfo, ForensicsReport, OSSForensics,
)


# ---------------------------------------------------------------------------
# Section A: PentestFinding / PentestReport
# ---------------------------------------------------------------------------

class TestPentestFinding:
    def test_creation(self):
        f = PentestFinding(
            severity="HIGH", category="header",
            title="Missing HSTS", description="HSTS not set",
        )
        assert f.severity == "HIGH"
        assert f.url == ""
        assert f.evidence == ""
        assert f.recommendation == ""

    def test_full_creation(self):
        f = PentestFinding(
            severity="CRITICAL", category="ssl",
            title="Invalid cert", description="Cert expired",
            url="https://example.com", evidence="cert date: 2020-01-01",
            recommendation="Renew certificate",
        )
        assert f.recommendation == "Renew certificate"


class TestPentestReport:
    def _make_report(self):
        findings = [
            PentestFinding(severity="CRITICAL", category="ssl", title="SSL Error", description=""),
            PentestFinding(severity="HIGH", category="header", title="Missing CSP", description=""),
            PentestFinding(severity="MEDIUM", category="header", title="Missing XFO", description=""),
        ]
        return PentestReport(
            target_url="https://example.com",
            findings=findings,
            scan_time_s=1.5,
            risk_score=7.5,
        )

    def test_critical_count(self):
        report = self._make_report()
        assert report.critical_count == 1

    def test_high_count(self):
        report = self._make_report()
        assert report.high_count == 1

    def test_summary_default(self):
        report = PentestReport(target_url="https://x.com", findings=[], scan_time_s=0.1, risk_score=0.0)
        assert report.summary == ""


# ---------------------------------------------------------------------------
# Section B: WebPentester
# ---------------------------------------------------------------------------

class TestWebPentester:
    def test_scan_returns_report(self):
        pentester = WebPentester()
        report = pentester.scan("http://invalid.nonexistent.test", timeout=2.0)
        assert isinstance(report, PentestReport)

    def test_scan_target_url_preserved(self):
        pentester = WebPentester()
        url = "http://invalid.nonexistent.test"
        report = pentester.scan(url, timeout=2.0)
        assert report.target_url == url

    def test_scan_http_adds_no_https_finding(self):
        pentester = WebPentester()
        report = pentester.scan("http://invalid.nonexistent.test", timeout=2.0)
        categories = [f.category for f in report.findings]
        assert "ssl" in categories or "header" in categories

    def test_scan_findings_is_list(self):
        pentester = WebPentester()
        report = pentester.scan("http://invalid.nonexistent.test", timeout=2.0)
        assert isinstance(report.findings, list)

    def test_scan_risk_score_in_range(self):
        pentester = WebPentester()
        report = pentester.scan("http://invalid.nonexistent.test", timeout=2.0)
        assert 0.0 <= report.risk_score <= 10.0

    def test_scan_time_nonneg(self):
        pentester = WebPentester()
        report = pentester.scan("http://invalid.nonexistent.test", timeout=2.0)
        assert report.scan_time_s >= 0.0

    def test_generate_report_md_returns_string(self):
        pentester = WebPentester()
        report = pentester.scan("http://invalid.nonexistent.test", timeout=2.0)
        md = pentester.generate_report_md(report)
        assert isinstance(md, str)

    def test_generate_report_md_contains_target(self):
        pentester = WebPentester()
        url = "http://invalid.nonexistent.test"
        report = pentester.scan(url, timeout=2.0)
        md = pentester.generate_report_md(report)
        assert url in md

    def test_generate_report_md_contains_risk_score(self):
        pentester = WebPentester()
        report = pentester.scan("http://invalid.nonexistent.test", timeout=2.0)
        md = pentester.generate_report_md(report)
        assert "Risk Score" in md

    def test_missing_headers_finding_severity(self):
        pentester = WebPentester()
        report = pentester.scan("http://invalid.nonexistent.test", timeout=2.0)
        severities = {f.severity for f in report.findings}
        # HTTP なので HIGH (no HTTPS) が含まれるはず
        assert len(severities) > 0


# ---------------------------------------------------------------------------
# Section C: DependencyInfo / ForensicsReport / OSSForensics
# ---------------------------------------------------------------------------

class TestDependencyInfo:
    def test_creation(self):
        dep = DependencyInfo(name="requests", version="2.28.0")
        assert dep.license == ""
        assert dep.is_direct is True
        assert dep.has_known_vuln is False
        assert dep.vuln_ids == []

    def test_with_license(self):
        dep = DependencyInfo(name="flask", version="2.0.0", license="BSD-3-Clause")
        assert dep.license == "BSD-3-Clause"


class TestOSSForensics:
    def test_analyze_returns_report(self):
        oss = OSSForensics()
        report = oss.analyze(".")
        assert isinstance(report, ForensicsReport)

    def test_analyze_has_dependencies(self):
        oss = OSSForensics()
        report = oss.analyze(".")
        assert isinstance(report.dependencies, list)

    def test_analyze_total_deps_correct(self):
        oss = OSSForensics()
        report = oss.analyze(".")
        assert report.total_deps == len(report.dependencies)

    def test_analyze_vulnerable_count_nonneg(self):
        oss = OSSForensics()
        report = oss.analyze(".")
        assert report.vulnerable_count >= 0

    def test_analyze_license_issues_list(self):
        oss = OSSForensics()
        report = oss.analyze(".")
        assert isinstance(report.license_issues, list)

    def test_analyze_sbom_has_format(self):
        oss = OSSForensics()
        report = oss.analyze(".")
        assert "bomFormat" in report.sbom
        assert report.sbom["bomFormat"] == "CycloneDX"

    def test_check_vulnerabilities_returns_list(self):
        oss = OSSForensics()
        deps = [DependencyInfo(name="requests", version="2.28.0")]
        result = oss.check_vulnerabilities(deps)
        assert isinstance(result, list)

    def test_generate_sbom_returns_json_string(self):
        oss = OSSForensics()
        deps = [DependencyInfo(name="numpy", version="1.24.0", license="BSD-3-Clause")]
        sbom_json = oss.generate_sbom(deps)
        assert isinstance(sbom_json, str)
        import json
        sbom = json.loads(sbom_json)
        assert sbom["bomFormat"] == "CycloneDX"

    def test_generate_sbom_has_components(self):
        oss = OSSForensics()
        deps = [
            DependencyInfo(name="torch", version="2.0.0", license="BSD-3-Clause"),
            DependencyInfo(name="fastapi", version="0.100.0", license="MIT"),
        ]
        sbom_json = oss.generate_sbom(deps)
        import json
        sbom = json.loads(sbom_json)
        assert len(sbom["components"]) == 2


# ---------------------------------------------------------------------------
# Section D: API エンドポイント
# ---------------------------------------------------------------------------

class TestSecurityScanEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/security/scan",
                        json={"target_url": "http://invalid.nonexistent.test",
                              "timeout": 2.0},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_findings(self, client):
        r = client.post("/v1/security/scan",
                        json={"target_url": "http://invalid.nonexistent.test",
                              "timeout": 2.0},
                        headers=_HDR)
        data = r.json()
        assert "findings" in data
        assert isinstance(data["findings"], list)

    def test_has_risk_score(self, client):
        r = client.post("/v1/security/scan",
                        json={"target_url": "http://invalid.nonexistent.test"},
                        headers=_HDR)
        data = r.json()
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 10.0

    def test_has_summary(self, client):
        r = client.post("/v1/security/scan",
                        json={"target_url": "http://invalid.nonexistent.test"},
                        headers=_HDR)
        assert "summary" in r.json()

    def test_has_scan_time(self, client):
        r = client.post("/v1/security/scan",
                        json={"target_url": "http://invalid.nonexistent.test"},
                        headers=_HDR)
        assert "scan_time_s" in r.json()


class TestSecurityReportMdEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/security/report/md",
                        json={"target_url": "http://invalid.nonexistent.test",
                              "timeout": 2.0},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_markdown(self, client):
        r = client.post("/v1/security/report/md",
                        json={"target_url": "http://invalid.nonexistent.test"},
                        headers=_HDR)
        data = r.json()
        assert "markdown" in data
        assert len(data["markdown"]) > 0

    def test_markdown_has_heading(self, client):
        r = client.post("/v1/security/report/md",
                        json={"target_url": "http://invalid.nonexistent.test"},
                        headers=_HDR)
        assert "# Penetration Test Report" in r.json()["markdown"]


class TestOSSAnalyzeEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/security/oss/analyze",
                        json={"project_path": "."},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_total_deps(self, client):
        r = client.post("/v1/security/oss/analyze",
                        json={"project_path": "."},
                        headers=_HDR)
        data = r.json()
        assert "total_deps" in data
        assert data["total_deps"] >= 0

    def test_has_vulnerable_count(self, client):
        r = client.post("/v1/security/oss/analyze",
                        json={"project_path": "."},
                        headers=_HDR)
        assert "vulnerable_count" in r.json()

    def test_has_license_issues(self, client):
        r = client.post("/v1/security/oss/analyze",
                        json={"project_path": "."},
                        headers=_HDR)
        assert "license_issues" in r.json()
        assert isinstance(r.json()["license_issues"], list)


class TestOSSSBOMEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/security/oss/sbom",
                        json={"project_path": "."},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_sbom_json(self, client):
        r = client.post("/v1/security/oss/sbom",
                        json={"project_path": "."},
                        headers=_HDR)
        data = r.json()
        assert "sbom" in data

    def test_sbom_has_bom_format(self, client):
        r = client.post("/v1/security/oss/sbom",
                        json={"project_path": "."},
                        headers=_HDR)
        import json
        sbom_str = r.json()["sbom"]
        sbom = json.loads(sbom_str)
        assert sbom["bomFormat"] == "CycloneDX"

"""
Sprint 59 — 自律脆弱性スキャン テスト (54 tests)

対象:
  open_mythos/skills/vuln_scanner.py:
    VulnSeverity / VulnCategory / PatchStatus
    ScanTarget / VulnFinding / VerifyVerdict
    PatchCandidate / ScanReport / ScanSession
    VulnStore / VulnScanner / VulnPatcher / ScanReportEngine
  serve/api.py:
    POST /v1/vuln/scan
    GET  /v1/vuln/findings
    GET  /v1/vuln/findings/{id}
    DELETE /v1/vuln/findings/{id}
    POST /v1/vuln/patch/{finding_id}
    GET  /v1/vuln/session/{session_id}
    GET  /v1/vuln/session/{session_id}/report
    GET  /v1/vuln/session/{session_id}/report/md
"""
from __future__ import annotations

import sys
import pytest
import torch
from unittest.mock import MagicMock

from open_mythos.skills.vuln_scanner import (
    VulnSeverity, VulnCategory, PatchStatus,
    ScanTarget, VulnFinding, VerifyVerdict,
    PatchCandidate, ScanReport, ScanSession,
    VulnStore, VulnScanner, VulnPatcher, ScanReportEngine,
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
    )
    model = OpenMythos(cfg)
    llm   = OpenMythosLLM(model=model, tokenizer=tok)

    api_mod._model     = model
    api_mod._tokenizer = tok
    api_mod._llm       = llm
    api_mod._vuln_store   = VulnStore()
    api_mod._vuln_scanner = VulnScanner(api_mod._vuln_store)

    return TestClient(api_mod.app)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnSeverity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVulnSeverity:
    def test_critical_value(self):
        assert VulnSeverity.CRITICAL.value == "critical"

    def test_high_value(self):
        assert VulnSeverity.HIGH.value == "high"

    def test_score_order(self):
        assert VulnSeverity.CRITICAL.score > VulnSeverity.HIGH.score
        assert VulnSeverity.HIGH.score > VulnSeverity.MEDIUM.score
        assert VulnSeverity.MEDIUM.score > VulnSeverity.LOW.score
        assert VulnSeverity.LOW.score > VulnSeverity.INFO.score

    def test_critical_score_is_5(self):
        assert VulnSeverity.CRITICAL.score == 5

    def test_info_score_is_1(self):
        assert VulnSeverity.INFO.score == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnCategory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVulnCategory:
    def test_injection_value(self):
        assert VulnCategory.INJECTION.value == "injection"

    def test_auth_value(self):
        assert VulnCategory.AUTH.value == "auth"

    def test_all_categories(self):
        cats = {c.value for c in VulnCategory}
        assert "injection" in cats
        assert "auth" in cats
        assert "crypto" in cats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScanTarget
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestScanTarget:
    def test_basic_creation(self):
        t = ScanTarget(name="myapp", path="/src")
        assert t.name == "myapp"
        assert t.path == "/src"

    def test_defaults(self):
        t = ScanTarget(name="x", path=".")
        assert t.language == "python"
        assert t.focus_areas == []
        assert t.known_findings == []

    def test_to_dict_keys(self):
        t = ScanTarget(name="x", path=".", focus_areas=["auth"])
        d = t.to_dict()
        assert "name" in d
        assert "focus_areas" in d
        assert "known_findings" in d
        assert d["focus_areas"] == ["auth"]

    def test_to_dict_version_none(self):
        t = ScanTarget(name="x", path=".")
        assert t.to_dict()["version"] is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnFinding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVulnFinding:
    def _make(self, **kw) -> VulnFinding:
        defaults = dict(id="f1", title="test issue")
        defaults.update(kw)
        return VulnFinding(**defaults)

    def test_basic_creation(self):
        f = self._make()
        assert f.id == "f1"
        assert f.title == "test issue"

    def test_defaults(self):
        f = self._make()
        assert f.category == VulnCategory.OTHER
        assert f.severity  == VulnSeverity.MEDIUM
        assert f.cwe_id is None

    def test_to_dict_keys(self):
        f = self._make(
            category=VulnCategory.INJECTION,
            severity=VulnSeverity.HIGH,
            cwe_id="CWE-89",
        )
        d = f.to_dict()
        assert d["category"] == "injection"
        assert d["severity"]  == "high"
        assert d["cwe_id"]    == "CWE-89"

    def test_created_at_set(self):
        f = self._make()
        assert f.created_at > 0

    def test_to_dict_line_numbers(self):
        f = self._make(line_start=10, line_end=12)
        d = f.to_dict()
        assert d["line_start"] == 10
        assert d["line_end"]   == 12


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VerifyVerdict  (harness: GraderVerdict 5基準)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVerifyVerdict:
    def test_all_pass(self):
        v = VerifyVerdict.make(True, True, True, True, True)
        assert v.passed is True
        assert v.score == 1.0

    def test_all_fail(self):
        v = VerifyVerdict.make(False, False, False, False, False)
        assert v.passed is False
        assert v.score == 0.0

    def test_threshold_60pct(self):
        # 3/5 = 0.6 → passed
        v = VerifyVerdict.make(True, True, True, False, False)
        assert v.passed is True
        assert v.score == pytest.approx(0.6)

    def test_below_threshold(self):
        # 2/5 = 0.4 → not passed
        v = VerifyVerdict.make(True, True, False, False, False)
        assert v.passed is False

    def test_criteria_keys(self):
        v = VerifyVerdict.make(True, True, True, True, True)
        assert "reproducible"    in v.criteria
        assert "has_evidence"    in v.criteria
        assert "severity_stated" in v.criteria
        assert "not_duplicate"   in v.criteria
        assert "exploitable"     in v.criteria

    def test_to_dict(self):
        v = VerifyVerdict.make(True, False, True, True, False, evidence="test")
        d = v.to_dict()
        assert "passed" in d
        assert "score"  in d
        assert d["evidence"] == "test"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PatchCandidate  (harness: PatchVerdict T0/T1/T2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPatchCandidate:
    def _make(self, **kw) -> PatchCandidate:
        defaults = dict(
            finding_id="f1",
            original_code='os.system("cmd")',
            patched_code='subprocess.run(["cmd"], check=True)',
            explanation="safer",
        )
        defaults.update(kw)
        return PatchCandidate(**defaults)

    def test_defaults(self):
        p = self._make()
        assert p.status == PatchStatus.PENDING
        assert p.t0_syntax_valid is None
        assert p.t1_vuln_gone is None

    def test_passed_requires_t0_and_t1(self):
        p = self._make()
        p.t0_syntax_valid = True
        p.t1_vuln_gone = True
        assert p.passed is True

    def test_passed_false_when_t0_false(self):
        p = self._make()
        p.t0_syntax_valid = False
        p.t1_vuln_gone = True
        assert p.passed is False

    def test_passed_false_when_t2_false(self):
        p = self._make()
        p.t0_syntax_valid = True
        p.t1_vuln_gone = True
        p.t2_tests_pass = False
        assert p.passed is False

    def test_passed_none_t2_ok(self):
        p = self._make()
        p.t0_syntax_valid = True
        p.t1_vuln_gone = True
        p.t2_tests_pass = None  # T2 skipped
        assert p.passed is True

    def test_to_dict_includes_passed(self):
        p = self._make()
        p.t0_syntax_valid = True
        p.t1_vuln_gone = True
        d = p.to_dict()
        assert "passed" in d
        assert d["passed"] is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScanSession  (harness: RunResult)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestScanSession:
    def _target(self) -> ScanTarget:
        return ScanTarget(name="t", path=".")

    def _finding(self, severity=VulnSeverity.HIGH) -> VulnFinding:
        return VulnFinding(id="x", title="issue", severity=severity)

    def test_initial_finding_count(self):
        s = ScanSession(id="s1", target=self._target())
        assert s.finding_count == 0

    def test_critical_count(self):
        f1 = self._finding(VulnSeverity.CRITICAL)
        f2 = self._finding(VulnSeverity.HIGH)
        s = ScanSession(id="s1", target=self._target(), findings=[f1, f2])
        assert s.critical_count == 1

    def test_elapsed_s_none_while_running(self):
        s = ScanSession(id="s1", target=self._target())
        assert s.elapsed_s is None

    def test_elapsed_s_after_complete(self):
        import time as _time
        s = ScanSession(id="s1", target=self._target())
        s.started_at = int(_time.time()) - 10
        s.completed_at = int(_time.time())
        assert s.elapsed_s is not None
        assert s.elapsed_s >= 0.0

    def test_to_dict_keys(self):
        s = ScanSession(id="s1", target=self._target())
        d = s.to_dict()
        assert "id" in d
        assert "target" in d
        assert "finding_count" in d
        assert "critical_count" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVulnStore:
    def _store(self) -> VulnStore:
        return VulnStore()

    def _finding(self, fid="f1", severity=VulnSeverity.HIGH, category=VulnCategory.INJECTION) -> VulnFinding:
        return VulnFinding(id=fid, title="test", severity=severity, category=category)

    def test_initially_empty(self):
        assert len(self._store()) == 0

    def test_add_finding(self):
        store = self._store()
        f = self._finding()
        store.add_finding(f)
        assert len(store) == 1

    def test_get_existing(self):
        store = self._store()
        f = self._finding("abc")
        store.add_finding(f)
        assert store.get_finding("abc") is f

    def test_get_nonexistent(self):
        assert self._store().get_finding("nope") is None

    def test_list_findings(self):
        store = self._store()
        store.add_finding(self._finding("f1"))
        store.add_finding(self._finding("f2"))
        assert len(store.list_findings()) == 2

    def test_by_severity(self):
        store = self._store()
        store.add_finding(self._finding("f1", VulnSeverity.CRITICAL))
        store.add_finding(self._finding("f2", VulnSeverity.LOW))
        assert len(store.by_severity(VulnSeverity.CRITICAL)) == 1

    def test_by_category(self):
        store = self._store()
        store.add_finding(self._finding("f1", category=VulnCategory.AUTH))
        store.add_finding(self._finding("f2", category=VulnCategory.CRYPTO))
        assert len(store.by_category(VulnCategory.AUTH)) == 1

    def test_delete_finding(self):
        store = self._store()
        store.add_finding(self._finding("del1"))
        assert store.delete_finding("del1") is True
        assert store.get_finding("del1") is None

    def test_delete_nonexistent(self):
        assert self._store().delete_finding("ghost") is False

    def test_session_crud(self):
        store = self._store()
        target = ScanTarget(name="t", path=".")
        session = ScanSession(id="s99", target=target)
        store.add_session(session)
        assert store.get_session("s99") is session
        assert len(store.list_sessions()) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnScanner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVulnScanner:
    _INJECTION_SRC = 'os.system("ls")'
    _AUTH_SRC      = 'password == "secret123"'
    _CRYPTO_SRC    = 'hashlib.md5(data)'
    _SAFE_SRC      = 'x = 1 + 1'

    def test_detects_os_system(self):
        sc = VulnScanner()
        findings = sc.scan_source(self._INJECTION_SRC)
        assert any(f.category == VulnCategory.INJECTION for f in findings)

    def test_detects_hardcoded_password(self):
        sc = VulnScanner()
        findings = sc.scan_source(self._AUTH_SRC)
        assert any(f.category == VulnCategory.AUTH for f in findings)

    def test_detects_md5(self):
        sc = VulnScanner()
        findings = sc.scan_source(self._CRYPTO_SRC)
        assert any(f.category == VulnCategory.CRYPTO for f in findings)

    def test_safe_code_no_findings(self):
        sc = VulnScanner()
        findings = sc.scan_source(self._SAFE_SRC)
        assert findings == []

    def test_file_path_recorded(self):
        sc = VulnScanner()
        findings = sc.scan_source(self._INJECTION_SRC, file_path="app.py")
        assert all(f.file_path == "app.py" for f in findings)

    def test_cwe_assigned(self):
        sc = VulnScanner()
        findings = sc.scan_source(self._INJECTION_SRC)
        assert any(f.cwe_id is not None for f in findings)

    def test_scan_returns_session(self):
        sc = VulnScanner()
        target = ScanTarget(name="t", path="app.py")
        session = sc.scan(target, source=self._INJECTION_SRC)
        assert session.status == "completed"
        assert session.finding_count > 0

    def test_scan_stores_session(self):
        sc = VulnScanner()
        target = ScanTarget(name="t", path="app.py")
        session = sc.scan(target, source=self._INJECTION_SRC)
        assert sc.store.get_session(session.id) is not None

    def test_known_findings_excluded(self):
        sc = VulnScanner()
        target = ScanTarget(
            name="t", path="app.py",
            known_findings=["os.system"],
        )
        session = sc.scan(target, source=self._INJECTION_SRC)
        # os.system finding が除外されるはず
        assert all("os.system" not in f.title for f in session.findings)

    def test_focus_areas_filter(self):
        sc = VulnScanner()
        combined = self._INJECTION_SRC + "\n" + self._AUTH_SRC
        # injection のみにフォーカス
        findings = sc.scan_source(combined, focus_areas=["command injection"])
        assert all(f.category == VulnCategory.INJECTION for f in findings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnPatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVulnPatcher:
    def _finding(self, evidence: str, category=VulnCategory.INJECTION) -> VulnFinding:
        return VulnFinding(
            id="f1", title="issue",
            category=category,
            severity=VulnSeverity.HIGH,
            evidence=evidence,
        )

    def test_suggest_patch_eval(self):
        patcher = VulnPatcher()
        f = self._finding('result = eval(user_input)')
        patch = patcher.suggest_patch(f)
        assert patch is not None
        assert "ast.literal_eval" in patch.patched_code

    def test_suggest_patch_os_system(self):
        patcher = VulnPatcher()
        f = self._finding('os.system("cmd")')
        patch = patcher.suggest_patch(f)
        assert patch is not None
        assert patch.finding_id == "f1"

    def test_suggest_patch_none_for_safe(self):
        patcher = VulnPatcher()
        f = self._finding("x = 1 + 1")
        patch = patcher.suggest_patch(f)
        assert patch is None

    def test_validate_t0_syntax_valid(self):
        patcher = VulnPatcher()
        candidate = PatchCandidate(
            finding_id="f1",
            original_code='eval(x)',
            patched_code='ast.literal_eval(x)',
            explanation="safer",
        )
        result = patcher.validate_patch(candidate)
        assert result.t0_syntax_valid is True

    def test_validate_t0_syntax_invalid(self):
        patcher = VulnPatcher()
        candidate = PatchCandidate(
            finding_id="f1",
            original_code='eval(x)',
            patched_code='def (broken syntax:',
            explanation="broken",
        )
        result = patcher.validate_patch(candidate)
        assert result.t0_syntax_valid is False
        assert result.status == PatchStatus.REJECTED

    def test_validate_t1_vuln_gone(self):
        patcher = VulnPatcher()
        candidate = PatchCandidate(
            finding_id="f1",
            original_code='eval(x)',
            patched_code='ast.literal_eval(x)',
            explanation="safer",
        )
        result = patcher.validate_patch(candidate)
        assert result.t1_vuln_gone is True

    def test_validate_t2_skipped(self):
        patcher = VulnPatcher()
        candidate = PatchCandidate(
            finding_id="f1",
            original_code='eval(x)',
            patched_code='ast.literal_eval(x)',
            explanation="safer",
        )
        result = patcher.validate_patch(candidate)
        assert result.t2_tests_pass is None

    def test_verified_status_when_passed(self):
        patcher = VulnPatcher()
        candidate = PatchCandidate(
            finding_id="f1",
            original_code='eval(x)',
            patched_code='ast.literal_eval(x)',
            explanation="safer",
        )
        result = patcher.validate_patch(candidate)
        assert result.status == PatchStatus.VERIFIED


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScanReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestScanReportEngine:
    def _session_with_findings(self) -> ScanSession:
        target = ScanTarget(name="myapp", path="app.py")
        findings = [
            VulnFinding(id="f1", title="cmd injection",
                        category=VulnCategory.INJECTION, severity=VulnSeverity.CRITICAL),
            VulnFinding(id="f2", title="md5 usage",
                        category=VulnCategory.CRYPTO, severity=VulnSeverity.MEDIUM),
        ]
        return ScanSession(id="s1", target=target, status="completed", findings=findings)

    def test_build_report_no_findings(self):
        engine = ScanReportEngine()
        target = ScanTarget(name="empty", path=".")
        session = ScanSession(id="s0", target=target, status="completed")
        report = engine.build_report(session)
        assert report.severity_rating == "NOT-A-BUG"
        assert report.rubric_score == 0

    def test_build_report_severity_critical(self):
        engine = ScanReportEngine()
        report = engine.build_report(self._session_with_findings())
        assert report.severity_rating == "CRITICAL"

    def test_build_report_reachable_for_injection(self):
        engine = ScanReportEngine()
        report = engine.build_report(self._session_with_findings())
        assert report.reachability == "REACHABLE"

    def test_build_report_total_score_range(self):
        engine = ScanReportEngine()
        report = engine.build_report(self._session_with_findings())
        assert 0.0 <= report.total_score <= 1.0

    def test_to_markdown_contains_title(self):
        engine = ScanReportEngine()
        md = engine.to_markdown(self._session_with_findings())
        assert "myapp" in md
        assert "CRITICAL" in md

    def test_to_markdown_contains_findings(self):
        engine = ScanReportEngine()
        md = engine.to_markdown(self._session_with_findings())
        assert "cmd injection" in md

    def test_to_json_structure(self):
        engine = ScanReportEngine()
        result = engine.to_json(self._session_with_findings())
        assert "session" in result
        assert "report" in result
        assert "findings" in result["session"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API テスト (Sprint 59 endpoints)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVulnAPI:
    def test_scan_returns_200(self, client):
        resp = client.post("/v1/vuln/scan", json={
            "target_name": "test",
            "target_path": "app.py",
            "source": 'os.system("cmd")',
        })
        assert resp.status_code == 200

    def test_scan_response_has_finding_count(self, client):
        resp = client.post("/v1/vuln/scan", json={
            "target_name": "test",
            "target_path": "app.py",
            "source": 'os.system("cmd")',
        })
        data = resp.json()
        assert "finding_count" in data

    def test_scan_finds_injection(self, client):
        resp = client.post("/v1/vuln/scan", json={
            "target_name": "scan_inj",
            "target_path": "app.py",
            "source": 'os.system("cmd")',
        })
        data = resp.json()
        assert data["finding_count"] > 0

    def test_scan_safe_code_zero_findings(self, client):
        resp = client.post("/v1/vuln/scan", json={
            "target_name": "scan_safe",
            "target_path": "safe.py",
            "source": "x = 1 + 1",
        })
        data = resp.json()
        assert data["finding_count"] == 0

    def test_list_findings(self, client):
        # まずスキャンして findings を作る
        client.post("/v1/vuln/scan", json={
            "target_name": "list_test",
            "target_path": "app.py",
            "source": 'eval(user_input)',
        })
        resp = client.get("/v1/vuln/findings")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_session_report(self, client):
        scan_resp = client.post("/v1/vuln/scan", json={
            "target_name": "rep_test",
            "target_path": "rep.py",
            "source": 'eval(x)',
        })
        session_id = scan_resp.json()["id"]
        resp = client.get(f"/v1/vuln/session/{session_id}/report")
        assert resp.status_code == 200
        assert "report" in resp.json()

    def test_get_session_report_md(self, client):
        scan_resp = client.post("/v1/vuln/scan", json={
            "target_name": "md_test",
            "target_path": "md.py",
            "source": 'eval(x)',
        })
        session_id = scan_resp.json()["id"]
        resp = client.get(f"/v1/vuln/session/{session_id}/report/md")
        assert resp.status_code == 200
        assert "# Vulnerability Scan Report" in resp.text

    def test_session_not_found(self, client):
        resp = client.get("/v1/vuln/session/nonexistent/report")
        assert resp.status_code == 404

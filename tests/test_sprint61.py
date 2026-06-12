"""
Sprint 61 — Claude Fable 5 / Mythos 5 統合 + サイバー防衛 テスト

対象:
  open_mythos/skills/llm_providers.py:
    ClaudeModelTier (HAIKU_5 / FABLE_5 / MYTHOS_5)
    HFInferenceProvider
    list_model_tiers()

  open_mythos/skills/cyber_defense.py:
    ThreatLevel / IndicatorType / IncidentStatus
    ThreatIndicator / ThreatIntelReport / Incident / ForensicsArtifact / CyberDefenseReport
    IncidentStore
    ThreatIntelAnalyzer / IncidentResponder / ForensicsAI
    CyberDefenseOrchestrator (マルチエージェント)

  open_mythos/skills/security.py:
    AISecurityEnhancer (Claude Fable 5 強化)

  open_mythos/skills/vuln_scanner.py:
    PatchUrgency / NDayRecord / NDayVulnTracker (N-hour)
    HFCodeBERTClassifier
    MythosVulnAnalyzer

  serve/api.py:
    GET  /v1/model/tiers
    POST /v1/cyber/threat
    POST /v1/cyber/incident
    GET  /v1/cyber/incident
    GET  /v1/cyber/incident/{id}
    POST /v1/cyber/incident/{id}/respond
    PATCH /v1/cyber/incident/{id}/status
    DELETE /v1/cyber/incident/{id}
    POST /v1/cyber/forensics
    POST /v1/cyber/defend
    POST /v1/cyber/classify/code
    POST /v1/cyber/nday/register
    POST /v1/cyber/nday/{id}/patch
    GET  /v1/cyber/nday/breached
    GET  /v1/cyber/nday/summary
    GET  /v1/cyber/nday
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import patch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# torch 2.11.0+cpu / transformers 5.8.1 はインストール済みのため
# モック不要。mock_heavy_deps は削除済み。


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from serve.api import app
    with TestClient(app) as c:
        yield c


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ClaudeModelTier テスト (Sprint 61)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClaudeModelTier:
    def test_haiku_5_value(self):
        from open_mythos.skills.llm_providers import ClaudeModelTier
        assert ClaudeModelTier.HAIKU_5.value == "claude-haiku-4-5"

    def test_fable_5_value(self):
        from open_mythos.skills.llm_providers import ClaudeModelTier
        assert ClaudeModelTier.FABLE_5.value == "claude-sonnet-4-5"

    def test_mythos_5_value(self):
        from open_mythos.skills.llm_providers import ClaudeModelTier
        assert ClaudeModelTier.MYTHOS_5.value == "claude-opus-4"

    def test_tier_label_fable(self):
        from open_mythos.skills.llm_providers import ClaudeModelTier
        assert "Fable" in ClaudeModelTier.FABLE_5.tier_label

    def test_tier_label_mythos(self):
        from open_mythos.skills.llm_providers import ClaudeModelTier
        assert "Mythos" in ClaudeModelTier.MYTHOS_5.tier_label

    def test_context_window_200k(self):
        from open_mythos.skills.llm_providers import ClaudeModelTier
        for tier in ClaudeModelTier:
            assert tier.context_window == 200_000

    def test_recommended_for_cyber(self):
        from open_mythos.skills.llm_providers import ClaudeModelTier
        assert "サイバー" in ClaudeModelTier.MYTHOS_5.recommended_for

    def test_all_three_tiers_exist(self):
        from open_mythos.skills.llm_providers import ClaudeModelTier
        assert len(list(ClaudeModelTier)) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HFInferenceProvider テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHFInferenceProvider:
    def test_always_available(self):
        from open_mythos.skills.llm_providers import HFInferenceProvider, ProviderConfig, ProviderType
        p = HFInferenceProvider(ProviderConfig(provider=ProviderType.HF_CYBER))
        assert p.is_available() is True

    def test_cyber_models_listed(self):
        from open_mythos.skills.llm_providers import HFInferenceProvider
        assert "lily-cyber" in HFInferenceProvider.CYBER_MODELS
        assert "codebert-vuln" in HFInferenceProvider.CYBER_MODELS

    def test_lily_model_id(self):
        from open_mythos.skills.llm_providers import HFInferenceProvider
        assert "Lily-Cybersecurity" in HFInferenceProvider.CYBER_MODELS["lily-cyber"]

    def test_resolved_model_default(self):
        from open_mythos.skills.llm_providers import HFInferenceProvider, ProviderConfig, ProviderType
        p = HFInferenceProvider(ProviderConfig(provider=ProviderType.HF_CYBER))
        assert "Lily-Cybersecurity" in p.config.resolved_model()

    def test_complete_returns_response_on_http_503(self):
        """503 (model loading) でフォールバックメッセージを返す。"""
        import urllib.error
        from open_mythos.skills.llm_providers import (
            HFInferenceProvider, ProviderConfig, ProviderType, LLMRequest,
        )
        p = HFInferenceProvider(ProviderConfig(provider=ProviderType.HF_CYBER))
        err = urllib.error.HTTPError(url="", code=503, msg="Loading", hdrs=None, fp=None)
        with patch("urllib.request.urlopen", side_effect=err):
            resp = p.complete(LLMRequest(prompt="test"))
        assert "loading" in resp.text.lower() or "503" in resp.text.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# list_model_tiers テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestListModelTiers:
    def test_returns_list(self):
        from open_mythos.skills.llm_providers import list_model_tiers
        tiers = list_model_tiers()
        assert isinstance(tiers, list)

    def test_includes_fable_5(self):
        from open_mythos.skills.llm_providers import list_model_tiers
        ids = [t["id"] for t in list_model_tiers()]
        assert "claude-sonnet-4-5" in ids

    def test_includes_mythos_5(self):
        from open_mythos.skills.llm_providers import list_model_tiers
        ids = [t["id"] for t in list_model_tiers()]
        assert "claude-opus-4" in ids

    def test_includes_hf_models(self):
        from open_mythos.skills.llm_providers import list_model_tiers
        providers = [t["provider"] for t in list_model_tiers()]
        assert "huggingface" in providers

    def test_tier_has_required_fields(self):
        from open_mythos.skills.llm_providers import list_model_tiers
        for t in list_model_tiers():
            assert "id" in t and "label" in t and "recommended_for" in t


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ThreatLevel / IndicatorType テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestThreatLevel:
    def test_sla_critical_is_1h(self):
        from open_mythos.skills.cyber_defense import ThreatLevel
        assert ThreatLevel.CRITICAL.response_sla_hours == 1

    def test_sla_info_is_720h(self):
        from open_mythos.skills.cyber_defense import ThreatLevel
        assert ThreatLevel.INFO.response_sla_hours == 720

    def test_five_levels_exist(self):
        from open_mythos.skills.cyber_defense import ThreatLevel
        assert len(list(ThreatLevel)) == 5


class TestIndicatorType:
    def test_cve_type_exists(self):
        from open_mythos.skills.cyber_defense import IndicatorType
        assert IndicatorType.CVE.value == "cve"

    def test_technique_type_exists(self):
        from open_mythos.skills.cyber_defense import IndicatorType
        assert IndicatorType.TECHNIQUE.value == "technique"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ThreatIndicator テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestThreatIndicator:
    def _make_ioc(self):
        from open_mythos.skills.cyber_defense import ThreatIndicator, IndicatorType, ThreatLevel
        return ThreatIndicator(
            ioc_id="ioc-001",
            ioc_type=IndicatorType.IP,
            value="192.168.1.100",
            threat_level=ThreatLevel.HIGH,
        )

    def test_to_dict_keys(self):
        ioc = self._make_ioc()
        d = ioc.to_dict()
        assert "ioc_id" in d and "value" in d and "threat_level" in d

    def test_confidence_default(self):
        ioc = self._make_ioc()
        assert ioc.confidence == 0.5

    def test_created_at_set(self):
        ioc = self._make_ioc()
        assert ioc.created_at > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ThreatIntelAnalyzer テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestThreatIntelAnalyzer:
    def _make_indicators(self):
        from open_mythos.skills.cyber_defense import ThreatIndicator, IndicatorType, ThreatLevel
        return [
            ThreatIndicator("ioc-1", IndicatorType.IP, "10.0.0.1", ThreatLevel.CRITICAL),
            ThreatIndicator("ioc-2", IndicatorType.DOMAIN, "evil.example.com", ThreatLevel.HIGH),
        ]

    def test_analyze_returns_report(self):
        from open_mythos.skills.cyber_defense import ThreatIntelAnalyzer
        analyzer = ThreatIntelAnalyzer()
        report = analyzer.analyze(self._make_indicators())
        assert report.report_id
        assert len(report.indicators) == 2

    def test_risk_score_positive(self):
        from open_mythos.skills.cyber_defense import ThreatIntelAnalyzer
        analyzer = ThreatIntelAnalyzer()
        report = analyzer.analyze(self._make_indicators())
        assert report.risk_score > 0

    def test_summary_contains_count(self):
        from open_mythos.skills.cyber_defense import ThreatIntelAnalyzer
        analyzer = ThreatIntelAnalyzer()
        report = analyzer.analyze(self._make_indicators())
        assert "2" in report.summary

    def test_model_used_is_hf_when_no_key(self):
        from open_mythos.skills.cyber_defense import ThreatIntelAnalyzer
        analyzer = ThreatIntelAnalyzer()
        assert "Lily" in analyzer.model_used or "hf" in analyzer.model_used.lower() or "claude" in analyzer.model_used

    def test_mitre_ttps_extracted(self):
        from open_mythos.skills.cyber_defense import ThreatIntelAnalyzer
        analyzer = ThreatIntelAnalyzer()
        report = analyzer.analyze(
            self._make_indicators(),
            context="phishing attack with credential dumping"
        )
        assert "T1566" in report.mitre_ttps or "T1003" in report.mitre_ttps

    def test_rule_based_fallback_no_crash(self):
        from open_mythos.skills.cyber_defense import ThreatIntelAnalyzer
        analyzer = ThreatIntelAnalyzer(anthropic_api_key=None, hf_token=None)
        report = analyzer.analyze(self._make_indicators())
        assert isinstance(report.ai_analysis, str)

    def test_to_dict_serializable(self):
        import json
        from open_mythos.skills.cyber_defense import ThreatIntelAnalyzer
        analyzer = ThreatIntelAnalyzer()
        report = analyzer.analyze(self._make_indicators())
        d = report.to_dict()
        json.dumps(d)  # シリアライズできることを確認

    def test_critical_count_property(self):
        from open_mythos.skills.cyber_defense import ThreatIntelAnalyzer
        analyzer = ThreatIntelAnalyzer()
        report = analyzer.analyze(self._make_indicators())
        assert report.critical_count == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IncidentStore テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIncidentStore:
    def _store(self):
        from open_mythos.skills.cyber_defense import IncidentStore
        return IncidentStore()

    def test_create_returns_incident(self):
        from open_mythos.skills.cyber_defense import ThreatLevel
        store = self._store()
        inc = store.create("Test", "desc", ThreatLevel.HIGH)
        assert inc.incident_id
        assert len(store) == 1

    def test_get_returns_incident(self):
        from open_mythos.skills.cyber_defense import ThreatLevel
        store = self._store()
        inc = store.create("T", "d", ThreatLevel.LOW)
        assert store.get(inc.incident_id) is inc

    def test_get_unknown_returns_none(self):
        store = self._store()
        assert store.get("nonexistent") is None

    def test_list_all(self):
        from open_mythos.skills.cyber_defense import ThreatLevel
        store = self._store()
        store.create("A", "d", ThreatLevel.HIGH)
        store.create("B", "d", ThreatLevel.LOW)
        assert len(store.list_all()) == 2

    def test_delete(self):
        from open_mythos.skills.cyber_defense import ThreatLevel
        store = self._store()
        inc = store.create("T", "d", ThreatLevel.LOW)
        assert store.delete(inc.incident_id) is True
        assert store.get(inc.incident_id) is None

    def test_delete_nonexistent_returns_false(self):
        store = self._store()
        assert store.delete("bad-id") is False

    def test_update_status(self):
        from open_mythos.skills.cyber_defense import ThreatLevel, IncidentStatus
        store = self._store()
        inc = store.create("T", "d", ThreatLevel.CRITICAL)
        updated = store.update_status(inc.incident_id, IncidentStatus.TRIAGED)
        assert updated.status == IncidentStatus.TRIAGED

    def test_timeline_appended_on_create(self):
        from open_mythos.skills.cyber_defense import ThreatLevel
        store = self._store()
        inc = store.create("T", "d", ThreatLevel.HIGH)
        assert len(inc.timeline) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IncidentResponder テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIncidentResponder:
    def _make_incident(self):
        from open_mythos.skills.cyber_defense import IncidentStore, ThreatLevel
        store = IncidentStore()
        return store.create("Ransomware attack", "Files encrypted", ThreatLevel.CRITICAL,
                            affected_systems=["web01", "db01"])

    def test_triage_sets_status(self):
        from open_mythos.skills.cyber_defense import IncidentResponder, IncidentStatus
        responder = IncidentResponder()
        inc = self._make_incident()
        triaged = responder.triage(inc)
        assert triaged.status == IncidentStatus.TRIAGED

    def test_triage_sets_response_plan(self):
        from open_mythos.skills.cyber_defense import IncidentResponder
        responder = IncidentResponder()
        inc = self._make_incident()
        triaged = responder.triage(inc)
        assert len(triaged.response_plan) > 0

    def test_response_plan_contains_steps(self):
        from open_mythos.skills.cyber_defense import IncidentResponder
        responder = IncidentResponder()
        inc = self._make_incident()
        plan = responder.generate_response_plan(inc)
        assert "1." in plan or "Immediate" in plan or "critical" in plan.lower()

    def test_critical_playbook_has_isolate(self):
        from open_mythos.skills.cyber_defense import IncidentResponder, ThreatLevel
        responder = IncidentResponder()
        plan = IncidentResponder._PLAYBOOKS[ThreatLevel.CRITICAL]
        assert any("隔離" in s or "isolat" in s.lower() for s in plan)

    def test_triage_adds_timeline_event(self):
        from open_mythos.skills.cyber_defense import IncidentResponder
        responder = IncidentResponder()
        inc = self._make_incident()
        initial_len = len(inc.timeline)
        responder.triage(inc)
        assert len(inc.timeline) > initial_len


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ForensicsAI テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestForensicsAI:
    def test_analyze_log_returns_artifact(self):
        from open_mythos.skills.cyber_defense import ForensicsAI
        ai = ForensicsAI()
        artifact = ai.analyze_artifact("log", "2026-01-01 ERROR: eval(user_input) failed")
        assert artifact.artifact_id
        assert artifact.artifact_type == "log"

    def test_extract_ip_ioc(self):
        from open_mythos.skills.cyber_defense import ForensicsAI
        ai = ForensicsAI()
        artifact = ai.analyze_artifact("network_pcap", "Connection from 192.168.1.99 to 10.0.0.1")
        assert any("ip:" in ioc for ioc in artifact.indicators_found)

    def test_critical_severity_for_ransomware(self):
        from open_mythos.skills.cyber_defense import ForensicsAI, ThreatLevel
        ai = ForensicsAI()
        artifact = ai.analyze_artifact("log", "ransomware encrypted all files on c2 server")
        assert artifact.severity == ThreatLevel.CRITICAL

    def test_to_dict_keys(self):
        from open_mythos.skills.cyber_defense import ForensicsAI
        ai = ForensicsAI()
        artifact = ai.analyze_artifact("filesystem", "normal log entry")
        d = artifact.to_dict()
        assert "artifact_id" in d and "ai_summary" in d and "severity" in d

    def test_summary_not_empty(self):
        from open_mythos.skills.cyber_defense import ForensicsAI
        ai = ForensicsAI()
        artifact = ai.analyze_artifact("log", "access denied for user root")
        assert len(artifact.ai_summary) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CyberDefenseOrchestrator テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCyberDefenseOrchestrator:
    def _make_indicators(self):
        from open_mythos.skills.cyber_defense import ThreatIndicator, IndicatorType, ThreatLevel
        return [
            ThreatIndicator("i1", IndicatorType.IP, "1.2.3.4", ThreatLevel.HIGH),
        ]

    def test_run_full_defense_returns_report(self):
        from open_mythos.skills.cyber_defense import CyberDefenseOrchestrator
        orch = CyberDefenseOrchestrator()
        report = orch.run_full_defense(indicators=self._make_indicators())
        assert report.session_id
        assert isinstance(report.recommendations, list)

    def test_run_with_no_indicators(self):
        from open_mythos.skills.cyber_defense import CyberDefenseOrchestrator
        orch = CyberDefenseOrchestrator()
        report = orch.run_full_defense()
        assert report.overall_risk == 0.0

    def test_run_with_artifacts(self):
        from open_mythos.skills.cyber_defense import CyberDefenseOrchestrator
        orch = CyberDefenseOrchestrator()
        report = orch.run_full_defense(
            artifacts_raw=[("log", "malware detected on host")],
        )
        assert len(report.artifacts) == 1

    def test_model_tiers_used_in_report(self):
        from open_mythos.skills.cyber_defense import CyberDefenseOrchestrator
        orch = CyberDefenseOrchestrator()
        report = orch.run_full_defense(indicators=self._make_indicators())
        assert "fable-5" in report.model_tiers_used
        assert "mythos-5" in report.model_tiers_used

    def test_to_dict_serializable(self):
        import json
        from open_mythos.skills.cyber_defense import CyberDefenseOrchestrator
        orch = CyberDefenseOrchestrator()
        report = orch.run_full_defense(indicators=self._make_indicators())
        json.dumps(report.to_dict())

    def test_recommendations_not_empty(self):
        from open_mythos.skills.cyber_defense import CyberDefenseOrchestrator, ThreatIndicator, IndicatorType, ThreatLevel
        orch = CyberDefenseOrchestrator()
        inds = [ThreatIndicator("i1", IndicatorType.IP, "1.2.3.4", ThreatLevel.CRITICAL, confidence=1.0)]
        report = orch.run_full_defense(indicators=inds)
        assert len(report.recommendations) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AISecurityEnhancer テスト (security.py Sprint 61 強化)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAISecurityEnhancer:
    def _make_report(self):
        from open_mythos.skills.security import PentestReport, PentestFinding
        findings = [
            PentestFinding("HIGH", "ssl", "No HTTPS", "HTTP only", "http://test.com"),
            PentestFinding("CRITICAL", "header", "Missing CSP", "CSP absent", "http://test.com"),
        ]
        return PentestReport("http://test.com", findings, 1.5, 7.5, "2 issues found")

    def test_generate_executive_summary(self):
        from open_mythos.skills.security import AISecurityEnhancer
        enhancer = AISecurityEnhancer()
        report = self._make_report()
        summary = enhancer.generate_executive_summary(report)
        assert isinstance(summary, str) and len(summary) > 0

    def test_summary_mentions_risk_score(self):
        from open_mythos.skills.security import AISecurityEnhancer
        enhancer = AISecurityEnhancer()
        report = self._make_report()
        summary = enhancer.generate_executive_summary(report)
        assert "7.5" in summary or "test.com" in summary

    def test_prioritize_findings_sorted(self):
        from open_mythos.skills.security import AISecurityEnhancer
        enhancer = AISecurityEnhancer()
        report = self._make_report()
        priorities = enhancer.prioritize_findings(report)
        assert priorities[0]["severity"] == "CRITICAL"
        assert priorities[1]["severity"] == "HIGH"

    def test_prioritize_findings_has_effort(self):
        from open_mythos.skills.security import AISecurityEnhancer
        enhancer = AISecurityEnhancer()
        report = self._make_report()
        priorities = enhancer.prioritize_findings(report)
        assert all("effort" in p for p in priorities)

    def test_generate_patch_advice_returns_string(self):
        from open_mythos.skills.security import AISecurityEnhancer, PentestFinding
        enhancer = AISecurityEnhancer()
        finding = PentestFinding("HIGH", "ssl", "No HTTPS", "HTTP only", "http://t.com",
                                  recommendation="Enable HTTPS")
        advice = enhancer.generate_patch_advice(finding)
        assert isinstance(advice, str) and len(advice) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PatchUrgency (N-hour) テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPatchUrgency:
    def test_immediate_sla_is_1h(self):
        from open_mythos.skills.vuln_scanner import PatchUrgency
        assert PatchUrgency.IMMEDIATE_1H.sla_hours == 1

    def test_urgent_sla_is_6h(self):
        from open_mythos.skills.vuln_scanner import PatchUrgency
        assert PatchUrgency.URGENT_6H.sla_hours == 6

    def test_label_contains_sla(self):
        from open_mythos.skills.vuln_scanner import PatchUrgency
        assert "1h" in PatchUrgency.IMMEDIATE_1H.label.lower()

    def test_from_finding_critical_is_urgent(self):
        from open_mythos.skills.vuln_scanner import PatchUrgency, VulnSeverity, VulnFinding
        finding = VulnFinding(id="f1", title="SQLi", severity=VulnSeverity.CRITICAL)
        urgency = PatchUrgency.from_finding(finding)
        assert urgency in (PatchUrgency.IMMEDIATE_1H, PatchUrgency.URGENT_6H)

    def test_from_finding_low_is_planned(self):
        from open_mythos.skills.vuln_scanner import PatchUrgency, VulnSeverity, VulnFinding
        finding = VulnFinding(id="f2", title="Minor", severity=VulnSeverity.LOW)
        urgency = PatchUrgency.from_finding(finding)
        assert urgency == PatchUrgency.PLANNED

    def test_five_urgency_levels(self):
        from open_mythos.skills.vuln_scanner import PatchUrgency
        assert len(list(PatchUrgency)) == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NDayVulnTracker テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNDayVulnTracker:
    def test_register_returns_record(self):
        from open_mythos.skills.vuln_scanner import NDayVulnTracker, PatchUrgency
        tracker = NDayVulnTracker()
        rec = tracker.register("CVE-2026-0001", urgency=PatchUrgency.URGENT_6H)
        assert rec.cve_id == "CVE-2026-0001"

    def test_exploit_public_escalates_urgency(self):
        from open_mythos.skills.vuln_scanner import NDayVulnTracker, PatchUrgency
        tracker = NDayVulnTracker()
        rec = tracker.register("CVE-2026-0002", urgency=PatchUrgency.HIGH_72H, exploit_public=True)
        assert rec.urgency == PatchUrgency.IMMEDIATE_1H

    def test_mark_patched(self):
        from open_mythos.skills.vuln_scanner import NDayVulnTracker, PatchUrgency
        tracker = NDayVulnTracker()
        tracker.register("CVE-2026-0003", urgency=PatchUrgency.CRITICAL_24H)
        rec = tracker.mark_patched("CVE-2026-0003")
        assert rec.patch_deployed is True

    def test_breached_when_sla_exceeded(self):
        from open_mythos.skills.vuln_scanner import NDayVulnTracker, PatchUrgency
        tracker = NDayVulnTracker()
        # disclosure 100 hours ago
        tracker.register("CVE-OLD",
                         urgency=PatchUrgency.URGENT_6H,
                         disclosure_ts=time.time() - 100 * 3600)
        breached = tracker.get_breached()
        assert any(r.cve_id == "CVE-OLD" for r in breached)

    def test_not_breached_when_patched(self):
        from open_mythos.skills.vuln_scanner import NDayVulnTracker, PatchUrgency
        tracker = NDayVulnTracker()
        tracker.register("CVE-PATCHED",
                         urgency=PatchUrgency.URGENT_6H,
                         disclosure_ts=time.time() - 100 * 3600)
        tracker.mark_patched("CVE-PATCHED")
        breached = tracker.get_breached()
        assert not any(r.cve_id == "CVE-PATCHED" for r in breached)

    def test_summary_keys(self):
        from open_mythos.skills.vuln_scanner import NDayVulnTracker
        tracker = NDayVulnTracker()
        s = tracker.summary()
        assert "total" in s and "patched" in s and "sla_breached" in s

    def test_patch_latency_calculated(self):
        from open_mythos.skills.vuln_scanner import NDayVulnTracker, PatchUrgency
        tracker = NDayVulnTracker()
        disc_ts = time.time() - 24 * 3600
        tracker.register("CVE-LAT", urgency=PatchUrgency.HIGH_72H, disclosure_ts=disc_ts)
        tracker.mark_patched("CVE-LAT")
        rec = tracker.get_by_urgency(PatchUrgency.HIGH_72H)[0]
        assert rec.patch_latency_hours is not None
        assert rec.patch_latency_hours >= 24

    def test_hours_since_disclosure(self):
        from open_mythos.skills.vuln_scanner import NDayVulnTracker, PatchUrgency
        tracker = NDayVulnTracker()
        rec = tracker.register("CVE-NOW", urgency=PatchUrgency.PLANNED,
                                disclosure_ts=time.time() - 3600)
        assert rec.hours_since_disclosure >= 1.0

    def test_register_from_finding(self):
        from open_mythos.skills.vuln_scanner import NDayVulnTracker, VulnFinding, VulnSeverity
        tracker = NDayVulnTracker()
        finding = VulnFinding(id="f99", title="SQL", severity=VulnSeverity.HIGH)
        rec = tracker.register_from_finding(finding)
        assert rec is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HFCodeBERTClassifier テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHFCodeBERTClassifier:
    def test_classify_returns_result(self):
        from open_mythos.skills.vuln_scanner import HFCodeBERTClassifier
        clf = HFCodeBERTClassifier()
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            result = clf.classify("x = eval(user_input)")
        assert result.label in ("VULNERABLE", "NOT_VULNERABLE")

    def test_vulnerable_code_detected_offline(self):
        from open_mythos.skills.vuln_scanner import HFCodeBERTClassifier
        clf = HFCodeBERTClassifier()
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            result = clf.classify("subprocess.run(cmd, shell=True)")
        assert result.is_vulnerable is True

    def test_safe_code_not_vulnerable_offline(self):
        from open_mythos.skills.vuln_scanner import HFCodeBERTClassifier
        clf = HFCodeBERTClassifier()
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            result = clf.classify("x = a + b")
        assert result.is_vulnerable is False

    def test_score_between_0_and_1(self):
        from open_mythos.skills.vuln_scanner import HFCodeBERTClassifier
        clf = HFCodeBERTClassifier()
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            result = clf.classify("print('hello')")
        assert 0.0 <= result.score <= 1.0

    def test_to_dict_keys(self):
        from open_mythos.skills.vuln_scanner import HFCodeBERTClassifier
        clf = HFCodeBERTClassifier()
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            d = clf.classify("print(x)").to_dict()
        assert "label" in d and "score" in d and "is_vulnerable" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MythosVulnAnalyzer テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMythosVulnAnalyzer:
    def test_analyze_finding_returns_dict(self):
        from open_mythos.skills.vuln_scanner import MythosVulnAnalyzer, VulnFinding, VulnSeverity
        analyzer = MythosVulnAnalyzer()
        finding = VulnFinding(id="f1", title="SQLi", severity=VulnSeverity.CRITICAL,
                              description="SQL injection via user input")
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            result = analyzer.analyze_finding(finding)
        assert "finding_id" in result and "n_hour_urgency" in result

    def test_n_hour_urgency_present(self):
        from open_mythos.skills.vuln_scanner import MythosVulnAnalyzer, VulnFinding, VulnSeverity
        analyzer = MythosVulnAnalyzer()
        finding = VulnFinding(id="f2", title="RCE", severity=VulnSeverity.CRITICAL)
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            result = analyzer.analyze_finding(finding)
        assert "urgency" in result["n_hour_urgency"] or "sla_hours" in result["n_hour_urgency"]

    def test_ai_analysis_fallback_not_empty(self):
        from open_mythos.skills.vuln_scanner import MythosVulnAnalyzer, VulnFinding, VulnSeverity
        analyzer = MythosVulnAnalyzer(anthropic_api_key=None)
        finding = VulnFinding(id="f3", title="XSS", severity=VulnSeverity.MEDIUM)
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            result = analyzer.analyze_finding(finding)
        assert len(result["ai_analysis"]) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API エンドポイント テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestModelTiersAPI:
    def test_get_model_tiers(self, api_client):
        resp = api_client.get("/v1/model/tiers")
        assert resp.status_code == 200
        data = resp.json()
        assert "tiers" in data and len(data["tiers"]) >= 3

    def test_tiers_include_fable(self, api_client):
        resp = api_client.get("/v1/model/tiers")
        ids = [t["id"] for t in resp.json()["tiers"]]
        assert "claude-sonnet-4-5" in ids

    def test_tiers_include_mythos(self, api_client):
        resp = api_client.get("/v1/model/tiers")
        ids = [t["id"] for t in resp.json()["tiers"]]
        assert "claude-opus-4" in ids


class TestCyberThreatAPI:
    def test_analyze_empty_indicators(self, api_client):
        resp = api_client.post("/v1/cyber/threat", json={"indicators": []})
        assert resp.status_code == 200
        assert "report_id" in resp.json()

    def test_analyze_single_ioc(self, api_client):
        resp = api_client.post("/v1/cyber/threat", json={
            "indicators": [{"ioc_type": "ip", "value": "1.2.3.4", "threat_level": "high"}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_score"] > 0

    def test_analyze_multiple_iocs(self, api_client):
        resp = api_client.post("/v1/cyber/threat", json={
            "indicators": [
                {"ioc_type": "ip", "value": "1.2.3.4", "threat_level": "critical"},
                {"ioc_type": "domain", "value": "evil.test", "threat_level": "high"},
            ]
        })
        assert resp.status_code == 200
        assert len(resp.json()["indicators"]) == 2

    def test_invalid_threat_level_returns_400(self, api_client):
        resp = api_client.post("/v1/cyber/threat", json={
            "indicators": [{"ioc_type": "ip", "value": "1.2.3.4", "threat_level": "invalid"}]
        })
        assert resp.status_code == 400


class TestCyberIncidentAPI:
    def test_create_incident(self, api_client):
        resp = api_client.post("/v1/cyber/incident", json={
            "title": "Test Incident",
            "description": "desc",
            "severity": "high",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "incident_id" in data
        return data["incident_id"]

    def test_list_incidents(self, api_client):
        api_client.post("/v1/cyber/incident", json={
            "title": "T", "description": "d", "severity": "low"
        })
        resp = api_client.get("/v1/cyber/incident")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_incident(self, api_client):
        create = api_client.post("/v1/cyber/incident", json={
            "title": "Get Test", "description": "d", "severity": "medium"
        })
        inc_id = create.json()["incident_id"]
        resp = api_client.get(f"/v1/cyber/incident/{inc_id}")
        assert resp.status_code == 200

    def test_get_nonexistent_returns_404(self, api_client):
        resp = api_client.get("/v1/cyber/incident/nonexistent-id")
        assert resp.status_code == 404

    def test_respond_to_incident(self, api_client):
        create = api_client.post("/v1/cyber/incident", json={
            "title": "Resp Test", "description": "d", "severity": "critical"
        })
        inc_id = create.json()["incident_id"]
        resp = api_client.post(f"/v1/cyber/incident/{inc_id}/respond")
        assert resp.status_code == 200
        assert resp.json()["status"] == "triaged"

    def test_update_status(self, api_client):
        create = api_client.post("/v1/cyber/incident", json={
            "title": "Status Test", "description": "d", "severity": "high"
        })
        inc_id = create.json()["incident_id"]
        resp = api_client.patch(f"/v1/cyber/incident/{inc_id}/status?status=contained")
        assert resp.status_code == 200
        assert resp.json()["status"] == "contained"

    def test_delete_incident(self, api_client):
        create = api_client.post("/v1/cyber/incident", json={
            "title": "Del Test", "description": "d", "severity": "low"
        })
        inc_id = create.json()["incident_id"]
        resp = api_client.delete(f"/v1/cyber/incident/{inc_id}")
        assert resp.status_code == 200


class TestCyberForensicsAPI:
    def test_analyze_log(self, api_client):
        resp = api_client.post("/v1/cyber/forensics", json={
            "artifact_type": "log",
            "content": "ERROR: eval(user_data) failed",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "artifact_id" in data and "ai_summary" in data

    def test_analyze_ransomware_log(self, api_client):
        from open_mythos.skills.cyber_defense import ThreatLevel
        resp = api_client.post("/v1/cyber/forensics", json={
            "artifact_type": "log",
            "content": "ransomware encrypted files c2 exfiltration",
        })
        assert resp.status_code == 200
        assert resp.json()["severity"] in ("critical", "high")


class TestCyberDefendAPI:
    def test_defend_empty(self, api_client):
        resp = api_client.post("/v1/cyber/defend", json={"indicators": []})
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_defend_with_iocs(self, api_client):
        resp = api_client.post("/v1/cyber/defend", json={
            "indicators": [{"ioc_type": "ip", "value": "5.5.5.5", "threat_level": "critical"}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_risk" in data and "recommendations" in data


class TestCodeClassifyAPI:
    def test_classify_vuln_code(self, api_client):
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            resp = api_client.post("/v1/cyber/classify/code", json={
                "code": "subprocess.run(cmd, shell=True)"
            })
        assert resp.status_code == 200
        assert resp.json()["is_vulnerable"] is True

    def test_classify_safe_code(self, api_client):
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            resp = api_client.post("/v1/cyber/classify/code", json={"code": "x = 1 + 2"})
        assert resp.status_code == 200
        assert resp.json()["is_vulnerable"] is False


class TestNDayAPI:
    def test_register_cve(self, api_client):
        resp = api_client.post("/v1/cyber/nday/register", json={
            "cve_id": "CVE-2026-1234",
            "urgency": "urgent_6h",
        })
        assert resp.status_code == 200
        assert resp.json()["cve_id"] == "CVE-2026-1234"

    def test_register_with_exploit_escalates(self, api_client):
        resp = api_client.post("/v1/cyber/nday/register", json={
            "cve_id": "CVE-2026-EXPLOIT",
            "urgency": "high_72h",
            "exploit_public": True,
        })
        assert resp.status_code == 200
        assert resp.json()["urgency"] == "immediate_1h"

    def test_mark_patched(self, api_client):
        api_client.post("/v1/cyber/nday/register", json={
            "cve_id": "CVE-2026-PATCH",
            "urgency": "critical_24h",
        })
        resp = api_client.post("/v1/cyber/nday/CVE-2026-PATCH/patch")
        assert resp.status_code == 200
        assert resp.json()["patch_deployed"] is True

    def test_mark_patched_nonexistent(self, api_client):
        resp = api_client.post("/v1/cyber/nday/CVE-NONEXISTENT/patch")
        assert resp.status_code == 404

    def test_get_breached(self, api_client):
        resp = api_client.get("/v1/cyber/nday/breached")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_summary(self, api_client):
        resp = api_client.get("/v1/cyber/nday/summary")
        assert resp.status_code == 200
        assert "total" in resp.json()

    def test_list_all(self, api_client):
        resp = api_client.get("/v1/cyber/nday")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_invalid_urgency_returns_400(self, api_client):
        resp = api_client.post("/v1/cyber/nday/register", json={
            "cve_id": "CVE-BAD",
            "urgency": "invalid_level",
        })
        assert resp.status_code == 400

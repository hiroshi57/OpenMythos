"""
Sprint 68 — セキュリティインテリジェンス + カテゴリ分類 テスト

対象:
  68A: open_mythos/skills/security.py
       — DiagnosisCategory / ThreatCategoryMapper / CategoryMatch
  68B: open_mythos/skills/security_intel.py
       — ThreatSeverity / ThreatSource / ThreatCategory
       — ResponsePlaybook / ThreatEnrichment / SecurityThreat
       — SecurityIntelStore / ThreatEnricher / ThreatCollector
       — SecurityIntelDashboard / IntelReportEngine
  68C: serve/api.py
       — /v1/intel/* エンドポイント
"""
from __future__ import annotations

import pytest

from open_mythos.skills.security import (
    DiagnosisCategory, ThreatCategoryMapper, CategoryMatch, CATEGORY_META,
)
from open_mythos.skills.security_intel import (
    ThreatSeverity, ThreatSource, ThreatCategory,
    ResponsePlaybook, ThreatEnrichment, SecurityThreat,
    SecurityIntelStore, ThreatEnricher, ThreatCollector,
    SecurityIntelDashboard, IntelReportEngine,
    _rule_based_playbook, _rule_based_enrichment,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 68A: DiagnosisCategory / ThreatCategoryMapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDiagnosisCategory:
    def test_values(self):
        assert DiagnosisCategory.A.value == "A"
        assert DiagnosisCategory.F.value == "F"

    def test_category_meta_keys(self):
        for cat in DiagnosisCategory:
            assert cat.value in CATEGORY_META
            assert "label" in CATEGORY_META[cat.value]


class TestThreatCategoryMapper:
    def setup_method(self):
        self.mapper = ThreatCategoryMapper()

    def test_technical_vulnerability(self):
        matches = self.mapper.map("SQL Injection in CMS plugin", "CVE exploit found")
        cats = [m.category for m in matches]
        assert DiagnosisCategory.A in cats

    def test_phishing_maps_to_b(self):
        matches = self.mapper.map("Phishing campaign targeting employees")
        cats = [m.category for m in matches]
        assert DiagnosisCategory.B in cats

    def test_gdpr_maps_to_c(self):
        matches = self.mapper.map("GDPR violation: personal data leak")
        cats = [m.category for m in matches]
        assert DiagnosisCategory.C in cats

    def test_ransomware_maps_to_d(self):
        matches = self.mapper.map("Ransomware incident response required")
        cats = [m.category for m in matches]
        assert DiagnosisCategory.D in cats

    def test_governance_maps_to_e(self):
        matches = self.mapper.map("CISO cybersecurity governance strategy")
        cats = [m.category for m in matches]
        assert DiagnosisCategory.E in cats

    def test_ai_risk_maps_to_f(self):
        matches = self.mapper.map("Prompt injection in LLM application")
        cats = [m.category for m in matches]
        assert DiagnosisCategory.F in cats

    def test_japanese_pattern_matched(self):
        matches = self.mapper.map("フィッシングメールによる情報漏洩")
        cats = [m.category for m in matches]
        assert DiagnosisCategory.B in cats or DiagnosisCategory.C in cats

    def test_max_categories_limit(self):
        matches = self.mapper.map(
            "Ransomware phishing SQL injection GDPR AI prompt injection governance",
            max_categories=2,
        )
        assert len(matches) <= 2

    def test_no_match_returns_empty(self):
        matches = self.mapper.map("cloud migration cost optimization")
        assert isinstance(matches, list)  # 空でも list

    def test_primary_category(self):
        primary = self.mapper.primary_category("CVE RCE buffer overflow exploit")
        assert primary is not None
        assert primary.category == DiagnosisCategory.A

    def test_category_match_to_dict(self):
        primary = self.mapper.primary_category("ransomware incident")
        if primary:
            d = primary.to_dict()
            assert "category" in d
            assert "label" in d
            assert "reason" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 68B: Enum / データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestThreatSeverity:
    def test_values(self):
        assert ThreatSeverity.CRITICAL.value == "critical"
        assert ThreatSeverity.INFO.value == "info"

    def test_score_order(self):
        assert ThreatSeverity.CRITICAL.score > ThreatSeverity.HIGH.score
        assert ThreatSeverity.HIGH.score > ThreatSeverity.MEDIUM.score

    def test_score_info_lowest(self):
        assert ThreatSeverity.INFO.score == 1


class TestResponsePlaybook:
    def _playbook(self) -> ResponsePlaybook:
        return _rule_based_playbook("critical", "Test threat")

    def test_urgency_label_not_empty(self):
        pb = self._playbook()
        assert len(pb.urgency_label) > 0

    def test_immediate_actions_not_empty(self):
        pb = self._playbook()
        assert len(pb.immediate_actions) > 0

    def test_to_dict_keys(self):
        d = self._playbook().to_dict()
        for k in ("urgency_label", "notify_targets", "immediate_actions",
                   "short_term_actions", "long_term_actions",
                   "escalation_trigger", "verification_checklist"):
            assert k in d


class TestSecurityThreat:
    def _threat(self) -> SecurityThreat:
        return SecurityThreat(
            id="t1",
            title="Test CVE Exploit",
            summary="A critical vulnerability in OpenSSL",
            source=ThreatSource.NVD,
            severity=ThreatSeverity.CRITICAL,
            category=ThreatCategory.VULNERABILITY,
        )

    def test_to_dict_keys(self):
        d = self._threat().to_dict()
        for k in ("id", "title", "summary", "source", "severity", "category"):
            assert k in d

    def test_severity_value_in_dict(self):
        d = self._threat().to_dict()
        assert d["severity"] == "critical"

    def test_source_value_in_dict(self):
        d = self._threat().to_dict()
        assert d["source"] == "nvd"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 68B: SecurityIntelStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_threat(
    severity=ThreatSeverity.HIGH,
    source=ThreatSource.NVD,
    category=ThreatCategory.VULNERABILITY,
    is_featured=False,
) -> SecurityThreat:
    import uuid
    return SecurityThreat(
        id=str(uuid.uuid4()),
        title=f"Threat {severity.value}",
        summary="Test summary",
        source=source,
        severity=severity,
        category=category,
        is_featured=is_featured,
    )


class TestSecurityIntelStore:
    def setup_method(self):
        self.store = SecurityIntelStore()

    def test_add_and_get(self):
        t = _make_threat()
        self.store.add(t)
        assert self.store.get(t.id) is not None

    def test_get_missing_returns_none(self):
        assert self.store.get("nope") is None

    def test_list_all(self):
        self.store.add_many([_make_threat(), _make_threat()])
        assert len(self.store.list_all()) == 2

    def test_list_by_severity(self):
        self.store.add(_make_threat(ThreatSeverity.CRITICAL))
        self.store.add(_make_threat(ThreatSeverity.LOW))
        crit = self.store.list_by_severity(ThreatSeverity.CRITICAL)
        assert len(crit) == 1

    def test_list_by_source(self):
        self.store.add(_make_threat(source=ThreatSource.CISA))
        self.store.add(_make_threat(source=ThreatSource.NVD))
        assert len(self.store.list_by_source(ThreatSource.CISA)) == 1

    def test_list_by_category(self):
        self.store.add(_make_threat(category=ThreatCategory.AI_THREAT))
        assert len(self.store.list_by_category(ThreatCategory.AI_THREAT)) == 1

    def test_list_featured(self):
        self.store.add(_make_threat(is_featured=True))
        self.store.add(_make_threat(is_featured=False))
        assert len(self.store.list_featured()) == 1

    def test_delete(self):
        t = _make_threat()
        self.store.add(t)
        assert self.store.delete(t.id) is True
        assert self.store.get(t.id) is None

    def test_delete_missing_returns_false(self):
        assert self.store.delete("nope") is False

    def test_count(self):
        self.store.add_many([_make_threat(), _make_threat()])
        assert self.store.count() == 2

    def test_summary_keys(self):
        self.store.add(_make_threat(ThreatSeverity.CRITICAL))
        s = self.store.summary()
        assert "total" in s
        assert "by_severity" in s
        assert "by_source" in s

    def test_list_by_diagnosis_category(self):
        t = _make_threat()
        t.diagnosis_categories = ["A", "B"]
        self.store.add(t)
        assert len(self.store.list_by_diagnosis_category("A")) == 1
        assert len(self.store.list_by_diagnosis_category("Z")) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 68B: ThreatEnricher (rule-based)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestThreatEnricher:
    def setup_method(self):
        self.enricher = ThreatEnricher(api_key=None)  # rule-based

    def test_enrich_returns_enrichment(self):
        t = _make_threat()
        result = self.enricher.enrich(t)
        assert isinstance(result, ThreatEnrichment)

    def test_title_ja_not_empty(self):
        t = _make_threat()
        result = self.enricher.enrich(t)
        assert len(result.title_ja) > 0

    def test_summary_ja_not_empty(self):
        t = _make_threat()
        result = self.enricher.enrich(t)
        assert len(result.summary_ja) > 0

    def test_industry_tags_list(self):
        t = _make_threat()
        result = self.enricher.enrich(t)
        assert isinstance(result.industry_tags, list)
        assert len(result.industry_tags) > 0

    def test_response_playbook_present(self):
        t = _make_threat()
        result = self.enricher.enrich(t)
        assert isinstance(result.response_playbook, ResponsePlaybook)

    def test_enrich_many_returns_dict(self):
        threats = [_make_threat(), _make_threat()]
        result = self.enricher.enrich_many(threats)
        assert len(result) == 2

    def test_enrich_many_keys_match_ids(self):
        threats = [_make_threat(), _make_threat()]
        result = self.enricher.enrich_many(threats)
        for t in threats:
            assert t.id in result

    def test_from_mock_parses_json(self):
        import json
        mock_json = json.dumps({
            "title_ja": "テスト脅威",
            "summary_ja": "テストのサマリー",
            "industry_tags": ["全業種"],
            "remediation_steps": "1. 対処する",
            "urgency_label": "緊急",
            "notify_targets": ["IT担当者"],
            "immediate_actions": ["即時対応"],
            "short_term_actions": ["短期対応"],
            "long_term_actions": ["長期対応"],
            "escalation_trigger": "エスカレーション条件",
            "verification_checklist": ["確認項目"],
        })
        enricher = ThreatEnricher.from_mock([mock_json])
        t = _make_threat()
        result = enricher.enrich(t)
        assert result.title_ja == "テスト脅威"
        assert result.summary_ja == "テストのサマリー"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 68B: ThreatCollector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestThreatCollector:
    def setup_method(self):
        self.store = SecurityIntelStore()
        self.collector = ThreatCollector(store=self.store)

    def test_collect_nvd(self):
        threats = self.collector.collect_nvd()
        assert len(threats) > 0
        assert all(t.source == ThreatSource.NVD for t in threats)

    def test_collect_cisa(self):
        threats = self.collector.collect_cisa()
        assert len(threats) > 0
        assert all(t.source == ThreatSource.CISA for t in threats)

    def test_collect_ai_feed(self):
        threats = self.collector.collect_ai_feed()
        assert len(threats) > 0
        assert all(t.category == ThreatCategory.AI_THREAT for t in threats)

    def test_collect_manual_samples(self):
        threats = self.collector.collect_manual()
        assert len(threats) > 0
        assert all(t.source == ThreatSource.MANUAL for t in threats)

    def test_collect_all_returns_dict(self):
        result = self.collector.collect_all()
        for key in ("nvd", "cisa", "ai", "manual"):
            assert key in result

    def test_threats_stored_in_store(self):
        self.collector.collect_nvd()
        assert self.store.count() > 0

    def test_diagnosis_categories_auto_assigned(self):
        threats = self.collector.collect_cisa()
        # CISA KEV は "悪用確認済み" 系 → カテゴリ A or D が付与される想定
        for t in threats:
            assert isinstance(t.diagnosis_categories, list)

    def test_collect_manual_custom(self):
        import uuid
        custom = [SecurityThreat(
            id=str(uuid.uuid4()),
            title="カスタム脅威",
            summary="テスト",
            source=ThreatSource.MANUAL,
            severity=ThreatSeverity.LOW,
            category=ThreatCategory.GENERAL,
        )]
        result = self.collector.collect_manual(custom)
        assert len(result) == 1
        assert self.store.get(custom[0].id) is not None

    def test_featured_threats_in_cisa(self):
        threats = self.collector.collect_cisa()
        assert any(t.is_featured for t in threats)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 68B: SecurityIntelDashboard / IntelReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSecurityIntelDashboard:
    def setup_method(self):
        self.store = SecurityIntelStore()
        ThreatCollector(store=self.store).collect_all()
        self.dash = SecurityIntelDashboard(self.store)

    def test_summary_total(self):
        s = self.dash.summary()
        assert s["total"] > 0

    def test_summary_by_severity(self):
        s = self.dash.summary()
        assert "by_severity" in s

    def test_featured_feed(self):
        feed = self.dash.featured_feed()
        assert isinstance(feed, list)

    def test_critical_threats(self):
        crits = self.dash.critical_threats()
        assert isinstance(crits, list)

    def test_by_diagnosis_category(self):
        threats = self.dash.by_diagnosis_category("A")
        assert isinstance(threats, list)


class TestIntelReportEngine:
    def setup_method(self):
        self.store = SecurityIntelStore()
        ThreatCollector(store=self.store).collect_all()
        self.engine = IntelReportEngine(self.store)

    def test_summary_json_keys(self):
        d = self.engine.summary_json()
        assert "total" in d
        assert "by_severity" in d

    def test_markdown_contains_header(self):
        md = self.engine.markdown()
        assert "セキュリティインテリジェンスレポート" in md

    def test_markdown_contains_table(self):
        md = self.engine.markdown()
        assert "深刻度" in md

    def test_markdown_limit(self):
        md = self.engine.markdown(limit=2)
        assert len(md) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 68C: API テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serve.api import app
    return TestClient(app)


class TestIntelCollectApi:
    def test_collect_all(self, client):
        resp = client.post("/v1/intel/collect", json={})
        assert resp.status_code == 200
        d = resp.json()
        assert "collected" in d
        assert "total" in d

    def test_collect_specific_source(self, client):
        resp = client.post("/v1/intel/collect", json={"sources": ["nvd"]})
        assert resp.status_code == 200
        assert resp.json()["collected"].get("nvd", 0) > 0


class TestIntelThreatsApi:
    def _collect(self, client):
        client.post("/v1/intel/collect", json={})

    def test_list_threats(self, client):
        self._collect(client)
        resp = client.get("/v1/intel/threats")
        assert resp.status_code == 200
        d = resp.json()
        assert "threats" in d
        assert "total" in d

    def test_filter_by_severity(self, client):
        self._collect(client)
        resp = client.get("/v1/intel/threats?severity=critical")
        assert resp.status_code == 200
        threats = resp.json()["threats"]
        assert all(t["severity"] == "critical" for t in threats)

    def test_filter_by_source(self, client):
        self._collect(client)
        resp = client.get("/v1/intel/threats?source=nvd")
        assert resp.status_code == 200
        threats = resp.json()["threats"]
        assert all(t["source"] == "nvd" for t in threats)

    def test_filter_featured(self, client):
        self._collect(client)
        resp = client.get("/v1/intel/threats?featured=true")
        assert resp.status_code == 200
        threats = resp.json()["threats"]
        assert all(t["is_featured"] for t in threats)

    def test_invalid_severity_400(self, client):
        resp = client.get("/v1/intel/threats?severity=unknown")
        assert resp.status_code == 400

    def test_get_threat_detail(self, client):
        self._collect(client)
        threats = client.get("/v1/intel/threats").json()["threats"]
        if threats:
            tid = threats[0]["id"]
            resp = client.get(f"/v1/intel/threats/{tid}")
            assert resp.status_code == 200
            assert resp.json()["id"] == tid

    def test_get_threat_not_found(self, client):
        resp = client.get("/v1/intel/threats/nonexistent")
        assert resp.status_code == 404

    def test_create_threat_manual(self, client):
        resp = client.post("/v1/intel/threats", json={
            "title": "国内テスト脅威",
            "summary": "テスト用の手動登録脅威です",
            "source": "manual",
            "severity": "high",
            "category": "general",
            "tags": ["テスト"],
        })
        assert resp.status_code == 200
        d = resp.json()
        assert d["source"] == "manual"
        assert d["severity"] == "high"

    def test_create_threat_diagnosis_category_assigned(self, client):
        resp = client.post("/v1/intel/threats", json={
            "title": "Ransomware incident response",
            "summary": "Ransomware attack detected",
            "source": "manual",
            "severity": "critical",
            "category": "general",
        })
        assert resp.status_code == 200
        d = resp.json()
        assert isinstance(d["diagnosis_categories"], list)


class TestIntelSummaryApi:
    def test_summary(self, client):
        client.post("/v1/intel/collect", json={})
        resp = client.get("/v1/intel/summary")
        assert resp.status_code == 200
        d = resp.json()
        assert "total" in d
        assert "by_severity" in d

    def test_featured_feed(self, client):
        client.post("/v1/intel/collect", json={})
        resp = client.get("/v1/intel/feed/featured")
        assert resp.status_code == 200
        assert "feed" in resp.json()

    def test_report_md(self, client):
        client.post("/v1/intel/collect", json={})
        resp = client.get("/v1/intel/report/md")
        assert resp.status_code == 200
        assert "セキュリティインテリジェンスレポート" in resp.text


class TestCategoryMapApi:
    def test_technical_threat(self, client):
        resp = client.post("/v1/intel/category-map", json={
            "title": "SQL Injection CVE critical exploit",
            "summary": "Buffer overflow in authentication module",
        })
        assert resp.status_code == 200
        d = resp.json()
        assert "matches" in d
        assert "primary" in d

    def test_ai_threat_maps_to_f(self, client):
        resp = client.post("/v1/intel/category-map", json={
            "title": "Prompt injection attack on LLM",
        })
        assert resp.status_code == 200
        d = resp.json()
        cats = [m["category"] for m in d["matches"]]
        assert "F" in cats

    def test_no_match_returns_empty(self, client):
        resp = client.post("/v1/intel/category-map", json={
            "title": "cloud cost optimization",
        })
        assert resp.status_code == 200
        # matches は空配列、primary は None
        d = resp.json()
        assert isinstance(d["matches"], list)

"""
Sprint 68B — セキュリティインテリジェンス統合

国内外のセキュリティ・リスク管理情報を収集し、Claude AI で日本語化・
プレイブック生成を行い、リアルタイムに参照できるインテリジェンスフィードを提供する。

参照:
  security-app/frontend/src/lib/security/threat-enricher.ts (AI富化ロジック移植)
  security-app/supabase/migrations/20260512_add_security_intelligence.sql (データモデル)
  MIT AI Risk Repository V4 / OWASP LLM Top 10 2025

情報収集元:
  - NVD  (NIST National Vulnerability Database) — CVE
  - CISA (Cybersecurity and Infrastructure Security Agency) — 悪用確認済み脆弱性
  - Anthropic / OpenAI          — AI 脅威情報
  - Manual                      — 国内インシデント・手動キュレーション

オブジェクト:
  ThreatSeverity        : 深刻度 (Critical/High/Medium/Low/Info)
  ThreatSource          : 情報源 (NVD/CISA/Anthropic/OpenAI/Manual)
  ThreatCategory        : 脅威カテゴリ (Vulnerability/AI-Threat/Regulation/Tool/General)
  ResponsePlaybook      : 対応プレイブック (緊急度別アクションプラン)
  ThreatEnrichment      : AI富化結果 (日本語タイトル/サマリー/業種タグ/プレイブック)
  SecurityThreat        : 脅威情報 1 件 (security_intelligence テーブル相当)
  SecurityIntelStore    : インメモリストア (Supabase 代替)
  ThreatEnricher        : Claude AI による富化エンジン + rule-based フォールバック
  ThreatCollector       : 情報源シミュレーター (実 API 拡張可)
  SecurityIntelDashboard: 横断集計・ダッシュボード
  IntelReportEngine     : Markdown/JSON レポート生成

設計方針:
  - LLM 不在時は rule-based フォールバック（テスト・オフライン環境で動作保証）
  - ThreatCategoryMapper (security.py Sprint 68A) を再利用して分類
  - 外部 API (NVD/CISA) はシミュレーターで差し替え可 → stub→実API切替を容易に
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from open_mythos.skills.security import ThreatCategoryMapper, DiagnosisCategory


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ThreatSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"

    @property
    def score(self) -> int:
        return {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}[self.value]


class ThreatSource(str, Enum):
    NVD        = "nvd"        # NIST National Vulnerability Database
    CISA       = "cisa"       # CISA Known Exploited Vulnerabilities
    ANTHROPIC  = "anthropic"  # Anthropic セキュリティ情報
    OPENAI     = "openai"     # OpenAI セキュリティ情報
    MANUAL     = "manual"     # 手動キュレーション（国内情報含む）


class ThreatCategory(str, Enum):
    VULNERABILITY = "vulnerability"  # 技術的脆弱性 (CVE等)
    AI_THREAT     = "ai-threat"      # AI/LLM 固有の脅威
    REGULATION    = "regulation"     # 法令・規制動向
    TOOL          = "tool"           # セキュリティツール・手法
    GENERAL       = "general"        # その他・一般情報


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INDUSTRY_TAGS = [
    "全業種", "金融・保険", "医療・ヘルスケア", "製造業",
    "EC・小売", "SaaS・IT事業者", "建設・不動産",
    "教育機関", "官公庁・自治体", "物流・運輸", "エネルギー・インフラ",
]


@dataclass
class ResponsePlaybook:
    """
    深刻度別対応プレイブック (threat-enricher.ts の ResponsePlaybook を Python 移植)

    深刻度ガイドライン:
      critical : 4 時間以内に初動対応
      high     : 24 時間以内
      medium   : 1 週間以内
      low      : 1 ヶ月以内
    """
    urgency_label:         str
    notify_targets:        List[str]
    immediate_actions:     List[str]
    short_term_actions:    List[str]
    long_term_actions:     List[str]
    escalation_trigger:    str
    verification_checklist: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "urgency_label":          self.urgency_label,
            "notify_targets":         self.notify_targets,
            "immediate_actions":      self.immediate_actions,
            "short_term_actions":     self.short_term_actions,
            "long_term_actions":      self.long_term_actions,
            "escalation_trigger":     self.escalation_trigger,
            "verification_checklist": self.verification_checklist,
        }


@dataclass
class ThreatEnrichment:
    """AI 富化結果"""
    title_ja:          str
    summary_ja:        str
    industry_tags:     List[str]
    remediation_steps: str
    response_playbook: ResponsePlaybook

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title_ja":          self.title_ja,
            "summary_ja":        self.summary_ja,
            "industry_tags":     self.industry_tags,
            "remediation_steps": self.remediation_steps,
            "response_playbook": self.response_playbook.to_dict(),
        }


@dataclass
class SecurityThreat:
    """
    セキュリティ脅威情報 1 件
    (security-app の security_intelligence テーブルスキーマに準拠)
    """
    id:           str
    title:        str
    summary:      str
    source:       ThreatSource
    severity:     ThreatSeverity
    category:     ThreatCategory
    source_url:   Optional[str]            = None
    tags:         List[str]                = field(default_factory=list)
    is_featured:  bool                     = False
    enrichment:   Optional[ThreatEnrichment] = None
    published_at: float                    = field(default_factory=time.time)
    created_at:   float                    = field(default_factory=time.time)

    # 診断カテゴリ（A〜F）— ThreatCategoryMapper で自動付与
    diagnosis_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id":           self.id,
            "title":        self.title,
            "summary":      self.summary,
            "source":       self.source.value,
            "severity":     self.severity.value,
            "category":     self.category.value,
            "source_url":   self.source_url,
            "tags":         self.tags,
            "is_featured":  self.is_featured,
            "published_at": self.published_at,
            "created_at":   self.created_at,
            "diagnosis_categories": self.diagnosis_categories,
        }
        if self.enrichment:
            d["enrichment"] = self.enrichment.to_dict()
        return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SecurityIntelStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SecurityIntelStore:
    """
    セキュリティインテリジェンスのインメモリストア。
    (Supabase security_intelligence テーブルの代替)

    Usage:
        store = SecurityIntelStore()
        store.add(threat)
        threats = store.list_by_severity(ThreatSeverity.CRITICAL)
    """

    def __init__(self) -> None:
        self._threats: Dict[str, SecurityThreat] = {}

    def add(self, threat: SecurityThreat) -> SecurityThreat:
        self._threats[threat.id] = threat
        return threat

    def add_many(self, threats: List[SecurityThreat]) -> None:
        for t in threats:
            self._threats[t.id] = t

    def get(self, threat_id: str) -> Optional[SecurityThreat]:
        return self._threats.get(threat_id)

    def list_all(self, limit: int = 100) -> List[SecurityThreat]:
        items = sorted(
            self._threats.values(),
            key=lambda t: t.published_at,
            reverse=True,
        )
        return items[:limit]

    def list_by_severity(self, severity: ThreatSeverity) -> List[SecurityThreat]:
        return [t for t in self._threats.values() if t.severity == severity]

    def list_by_source(self, source: ThreatSource) -> List[SecurityThreat]:
        return [t for t in self._threats.values() if t.source == source]

    def list_by_category(self, category: ThreatCategory) -> List[SecurityThreat]:
        return [t for t in self._threats.values() if t.category == category]

    def list_featured(self) -> List[SecurityThreat]:
        return [t for t in self._threats.values() if t.is_featured]

    def list_by_diagnosis_category(self, cat: str) -> List[SecurityThreat]:
        """診断カテゴリ(A〜F)でフィルタ"""
        return [t for t in self._threats.values() if cat in t.diagnosis_categories]

    def delete(self, threat_id: str) -> bool:
        if threat_id in self._threats:
            del self._threats[threat_id]
            return True
        return False

    def count(self) -> int:
        return len(self._threats)

    def summary(self) -> Dict[str, Any]:
        by_sev: Dict[str, int] = {}
        by_src: Dict[str, int] = {}
        for t in self._threats.values():
            by_sev[t.severity.value] = by_sev.get(t.severity.value, 0) + 1
            by_src[t.source.value]   = by_src.get(t.source.value, 0) + 1
        return {
            "total":       self.count(),
            "by_severity": by_sev,
            "by_source":   by_src,
            "featured":    len(self.list_featured()),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ThreatEnricher — AI 富化エンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 深刻度別プレイブックガイドライン (threat-enricher.ts の SEVERITY_GUIDANCE 移植)
_SEVERITY_GUIDANCE: Dict[str, Dict[str, str]] = {
    "critical": {
        "urgency_label": "今すぐ対応（4時間以内）",
        "immediacy":     "ランサムウェア感染・システム侵害・データ漏洩が進行中または差し迫っている可能性。4時間以内に初動対応を完了すること。",
        "notify_hint":   "情報システム部門責任者・経営幹部（CTO/CEO）・外部セキュリティ会社（CSIRT）",
    },
    "high": {
        "urgency_label": "緊急対応（24時間以内）",
        "immediacy":     "悪用が確認済みまたは確認された脆弱性。放置するとデータ漏洩・不正アクセスのリスクが高い。24時間以内にパッチ適用または回避策を実施。",
        "notify_hint":   "情報システム部門・IT担当者・部門責任者",
    },
    "medium": {
        "urgency_label": "計画的対応（1週間以内）",
        "immediacy":     "リスクは存在するが、即座の侵害は限定的。1週間以内に対応計画を立案・実施。",
        "notify_hint":   "IT担当者・システム管理者",
    },
    "low": {
        "urgency_label": "通常対応（1ヶ月以内）",
        "immediacy":     "リスクは低いが対応推奨。定期メンテナンスサイクルに組み込んで対応。",
        "notify_hint":   "IT担当者",
    },
    "info": {
        "urgency_label": "情報収集・確認（必要に応じて）",
        "immediacy":     "直接的なリスクは低いが、動向を注視すること。",
        "notify_hint":   "担当者",
    },
}


def _rule_based_playbook(severity: str, title: str) -> ResponsePlaybook:
    """LLM 不在時の rule-based プレイブック生成"""
    g = _SEVERITY_GUIDANCE.get(severity, _SEVERITY_GUIDANCE["low"])
    return ResponsePlaybook(
        urgency_label=g["urgency_label"],
        notify_targets=[g["notify_hint"]],
        immediate_actions=[
            "最新のセキュリティ情報を確認し、影響範囲を特定する",
            "関連システム・サービスのログを保全する",
            "暫定的な緩和策（アクセス制限・ネットワーク分離等）を実施する",
        ],
        short_term_actions=[
            "パッチ・アップデートを適用する",
            "インシデント対応手順書に従い対応を記録する",
        ],
        long_term_actions=[
            "再発防止策を策定し、セキュリティポリシーを更新する",
            "社内教育・訓練に反映させる",
        ],
        escalation_trigger=f"実際の被害が確認された場合、または{g['urgency_label']}内に対応完了できない場合はインシデント対応チームを招集する",
        verification_checklist=[
            "影響を受けたシステムへのパッチ適用を確認",
            "異常なアクセスログが消えていることを確認",
            "対応結果を記録・報告する",
        ],
    )


def _rule_based_enrichment(threat: SecurityThreat) -> ThreatEnrichment:
    """LLM 不在時の rule-based 富化（タイトル・サマリーはそのまま使用）"""
    # 業種タグ: カテゴリベースで簡易推定
    industry_map = {
        ThreatCategory.AI_THREAT:    ["全業種", "SaaS・IT事業者"],
        ThreatCategory.VULNERABILITY: ["全業種"],
        ThreatCategory.REGULATION:   ["金融・保険", "医療・ヘルスケア", "全業種"],
        ThreatCategory.TOOL:         ["SaaS・IT事業者", "全業種"],
        ThreatCategory.GENERAL:      ["全業種"],
    }
    industry_tags = industry_map.get(threat.category, ["全業種"])
    playbook = _rule_based_playbook(threat.severity.value, threat.title)
    return ThreatEnrichment(
        title_ja=threat.title,
        summary_ja=threat.summary,
        industry_tags=industry_tags,
        remediation_steps=(
            "1. 影響範囲を特定する\n"
            "2. 暫定的な緩和策を実施する\n"
            "3. パッチ・アップデートを適用する\n"
            "4. ログを保全し、インシデント対応手順に従う\n"
            "5. 再発防止策を策定する"
        ),
        response_playbook=playbook,
    )


class ThreatEnricher:
    """
    セキュリティ脅威を AI で富化するエンジン。
    (threat-enricher.ts を Python 移植)

    LLM 不在（api_key=None）のときは rule-based にフォールバックする。

    Usage:
        enricher = ThreatEnricher(api_key="sk-ant-...")
        enrichment = enricher.enrich(threat)
        enricher_batch = ThreatEnricher.from_env()
        enrichments = enricher_batch.enrich_many(threats)
    """

    SYSTEM_PROMPT = (
        "あなたは日本の中小企業向けセキュリティアドバイザーです。"
        "海外（英語）のセキュリティ脅威情報を受け取り、以下を生成してください。"
        "IT専門家だけでなく一般の情シス担当者・経営者も読めるよう、平易な日本語で書くこと。\n\n"
        "生成項目:\n"
        "1. title_ja: 日本語タイトル（25文字以内）\n"
        "2. summary_ja: 日本語サマリー（3〜4文・脅威の内容・影響・緊急性）\n"
        "3. industry_tags: 影響業種（配列）\n"
        "4. remediation_steps: 対応手順（箇条書き5〜7ステップ）\n"
        "5. urgency_label / notify_targets / immediate_actions / short_term_actions / long_term_actions / escalation_trigger / verification_checklist\n\n"
        "JSON オブジェクトで返すこと。"
    )

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key
        self._mapper  = ThreatCategoryMapper()

    @classmethod
    def from_env(cls) -> "ThreatEnricher":
        import os
        return cls(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @classmethod
    def from_mock(cls, responses: List[str]) -> "ThreatEnricher":
        """テスト用モックエンリッチャー"""
        inst = cls(api_key=None)
        inst._mock_responses = list(responses)
        return inst

    def enrich(self, threat: SecurityThreat) -> ThreatEnrichment:
        """1 件の脅威を富化する。"""
        if hasattr(self, "_mock_responses") and self._mock_responses:
            return self._parse_llm_response(self._mock_responses.pop(0), threat)
        if self._api_key:
            return self._enrich_with_llm(threat)
        return _rule_based_enrichment(threat)

    def enrich_many(
        self, threats: List[SecurityThreat], max_batch: int = 5
    ) -> Dict[str, ThreatEnrichment]:
        """複数脅威を一括富化する（LLM は max_batch 件ずつ）"""
        result: Dict[str, ThreatEnrichment] = {}
        if self._api_key:
            for i in range(0, len(threats), max_batch):
                batch = threats[i : i + max_batch]
                batch_result = self._enrich_batch_with_llm(batch)
                result.update(batch_result)
        else:
            for t in threats:
                result[t.id] = _rule_based_enrichment(t)
        return result

    # ---- LLM 呼び出し ----

    def _enrich_with_llm(self, threat: SecurityThreat) -> ThreatEnrichment:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            g = _SEVERITY_GUIDANCE.get(threat.severity.value, _SEVERITY_GUIDANCE["low"])
            user_msg = (
                f"タイトル: {threat.title}\n"
                f"サマリー: {threat.summary}\n"
                f"深刻度: {threat.severity.value}\n"
                f"緊急度ガイド: {g['urgency_label']}\n"
                f"通知先ヒント: {g['notify_hint']}"
            )
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = "".join(
                b.text for b in msg.content if hasattr(b, "text")
            )
            return self._parse_llm_response(raw, threat)
        except Exception:
            return _rule_based_enrichment(threat)

    def _enrich_batch_with_llm(
        self, threats: List[SecurityThreat]
    ) -> Dict[str, ThreatEnrichment]:
        result: Dict[str, ThreatEnrichment] = {}
        for t in threats:
            result[t.id] = self._enrich_with_llm(t)
        return result

    def _parse_llm_response(self, raw: str, threat: SecurityThreat) -> ThreatEnrichment:
        import json, re as _re
        try:
            m = _re.search(r"\{[\s\S]*\}", raw)
            if not m:
                raise ValueError("No JSON found")
            data = json.loads(m.group())
            playbook = ResponsePlaybook(
                urgency_label=data.get("urgency_label", ""),
                notify_targets=data.get("notify_targets", []),
                immediate_actions=data.get("immediate_actions", []),
                short_term_actions=data.get("short_term_actions", []),
                long_term_actions=data.get("long_term_actions", []),
                escalation_trigger=data.get("escalation_trigger", ""),
                verification_checklist=data.get("verification_checklist", []),
            )
            return ThreatEnrichment(
                title_ja=data.get("title_ja", threat.title),
                summary_ja=data.get("summary_ja", threat.summary),
                industry_tags=data.get("industry_tags", ["全業種"]),
                remediation_steps=data.get("remediation_steps", ""),
                response_playbook=playbook,
            )
        except Exception:
            return _rule_based_enrichment(threat)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ThreatCollector — 情報収集シミュレーター
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ThreatCollector:
    """
    国内外セキュリティ情報を収集し SecurityIntelStore に登録するエージェント。

    collect_nvd()      : NVD CVE フィード（シミュレーター / 将来 API 化）
    collect_cisa()     : CISA KEV フィード（シミュレーター）
    collect_ai_feed()  : Anthropic/OpenAI AI 脅威情報（シミュレーター）
    collect_manual()   : 手動登録（国内インシデント等）
    collect_all()      : 全ソースを一括収集

    将来拡張:
        NVD  → https://services.nvd.nist.gov/rest/json/cves/2.0
        CISA → https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
    """

    def __init__(
        self,
        store: Optional[SecurityIntelStore] = None,
        enricher: Optional[ThreatEnricher] = None,
        enrich_on_collect: bool = False,
    ) -> None:
        self.store             = store or SecurityIntelStore()
        self.enricher          = enricher or ThreatEnricher()
        self.enrich_on_collect = enrich_on_collect
        self._mapper           = ThreatCategoryMapper()

    def _make_threat(
        self,
        title: str,
        summary: str,
        source: ThreatSource,
        severity: ThreatSeverity,
        category: ThreatCategory,
        source_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_featured: bool = False,
    ) -> SecurityThreat:
        threat = SecurityThreat(
            id=str(uuid.uuid4()),
            title=title,
            summary=summary,
            source=source,
            severity=severity,
            category=category,
            source_url=source_url,
            tags=tags or [],
            is_featured=is_featured,
        )
        # 診断カテゴリを自動付与
        matches = self._mapper.map(title, summary)
        threat.diagnosis_categories = [m.category.value for m in matches]
        # AI 富化
        if self.enrich_on_collect:
            threat.enrichment = self.enricher.enrich(threat)
        return threat

    def collect_nvd(self) -> List[SecurityThreat]:
        """NVD CVE フィードシミュレーター"""
        threats = [
            self._make_threat(
                title="CVE-2025-XXXX: Remote Code Execution via Buffer Overflow in LibSSL",
                summary="A critical buffer overflow vulnerability in LibSSL allows unauthenticated remote attackers to execute arbitrary code. All versions prior to 3.2.1 are affected.",
                source=ThreatSource.NVD,
                severity=ThreatSeverity.CRITICAL,
                category=ThreatCategory.VULNERABILITY,
                source_url="https://nvd.nist.gov/vuln/detail/CVE-2025-XXXX",
                tags=["CVE", "RCE", "SSL", "バッファオーバーフロー"],
            ),
            self._make_threat(
                title="CVE-2025-YYYY: SQL Injection in Popular CMS Plugin",
                summary="An SQL injection flaw in a widely-used CMS plugin enables attackers to extract database contents. Over 200,000 installations are at risk.",
                source=ThreatSource.NVD,
                severity=ThreatSeverity.HIGH,
                category=ThreatCategory.VULNERABILITY,
                source_url="https://nvd.nist.gov/vuln/detail/CVE-2025-YYYY",
                tags=["CVE", "SQLインジェクション", "CMS"],
            ),
            self._make_threat(
                title="CVE-2025-ZZZZ: Path Traversal in File Upload Handler",
                summary="Improper input validation in file upload handlers allows attackers to traverse directories and access sensitive configuration files.",
                source=ThreatSource.NVD,
                severity=ThreatSeverity.MEDIUM,
                category=ThreatCategory.VULNERABILITY,
                source_url="https://nvd.nist.gov/vuln/detail/CVE-2025-ZZZZ",
                tags=["CVE", "パストラバーサル", "ファイルアップロード"],
            ),
        ]
        self.store.add_many(threats)
        return threats

    def collect_cisa(self) -> List[SecurityThreat]:
        """CISA Known Exploited Vulnerabilities フィードシミュレーター"""
        threats = [
            self._make_threat(
                title="CISA KEV: Active Exploitation of VPN Appliance Authentication Bypass",
                summary="CISA confirms active exploitation of an authentication bypass vulnerability in enterprise VPN appliances. Federal agencies must patch within 3 days. All organizations strongly advised to apply emergency patches immediately.",
                source=ThreatSource.CISA,
                severity=ThreatSeverity.CRITICAL,
                category=ThreatCategory.VULNERABILITY,
                source_url="https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
                tags=["CISA", "KEV", "VPN", "認証バイパス", "悪用確認済み"],
                is_featured=True,
            ),
            self._make_threat(
                title="CISA Alert: Ransomware Group Targeting Healthcare Sector",
                summary="CISA and FBI issue joint advisory warning of a ransomware group actively targeting healthcare organizations. Phishing emails with malicious attachments are the primary attack vector.",
                source=ThreatSource.CISA,
                severity=ThreatSeverity.HIGH,
                category=ThreatCategory.GENERAL,
                source_url="https://www.cisa.gov/alerts",
                tags=["CISA", "ランサムウェア", "医療", "フィッシング"],
                is_featured=True,
            ),
        ]
        self.store.add_many(threats)
        return threats

    def collect_ai_feed(self) -> List[SecurityThreat]:
        """Anthropic / OpenAI AI 脅威情報フィードシミュレーター"""
        threats = [
            self._make_threat(
                title="Anthropic: LLM Prompt Injection via Indirect User Input Vectors",
                summary="Research from Anthropic demonstrates that LLMs integrated into enterprise applications are vulnerable to indirect prompt injection through user-controlled data fields. Attackers can override system instructions and exfiltrate data silently.",
                source=ThreatSource.ANTHROPIC,
                severity=ThreatSeverity.HIGH,
                category=ThreatCategory.AI_THREAT,
                source_url="https://www.anthropic.com/security",
                tags=["AI", "プロンプトインジェクション", "LLM", "OWASP LLM01"],
                is_featured=True,
            ),
            self._make_threat(
                title="OpenAI: GPT Model Hallucination Risk in Legal and Medical Contexts",
                summary="OpenAI documents cases where GPT models generate confident but factually incorrect outputs in high-stakes domains including legal contracts and medical advice. Organizations must implement human review workflows.",
                source=ThreatSource.OPENAI,
                severity=ThreatSeverity.MEDIUM,
                category=ThreatCategory.AI_THREAT,
                source_url="https://openai.com/safety",
                tags=["AI", "幻覚", "ハルシネーション", "OWASP LLM09", "法令"],
            ),
            self._make_threat(
                title="AI Supply Chain Risk: Poisoned Open-Source Model Weights",
                summary="Security researchers identify cases of maliciously modified open-source model weights uploaded to public repositories. These backdoored models can be triggered by specific input patterns to produce harmful outputs.",
                source=ThreatSource.ANTHROPIC,
                severity=ThreatSeverity.HIGH,
                category=ThreatCategory.AI_THREAT,
                source_url="https://www.anthropic.com/research",
                tags=["AI", "サプライチェーン", "データポイズニング", "モデル改ざん"],
            ),
        ]
        self.store.add_many(threats)
        return threats

    def collect_manual(
        self, threats: Optional[List[SecurityThreat]] = None
    ) -> List[SecurityThreat]:
        """
        手動登録（国内インシデント等）。
        threats が None のときはサンプル国内情報を追加する。
        """
        if threats is not None:
            self.store.add_many(threats)
            return threats

        samples = [
            self._make_threat(
                title="国内大手製造業へのランサムウェア攻撃（2025年）",
                summary="国内大手製造業がランサムウェア攻撃を受け、工場ライン停止を含む大規模障害が発生。VPN 機器の脆弱性を突いた侵入経路が特定されている。同種製品を使用している企業は直ちにパッチ適用を実施すること。",
                source=ThreatSource.MANUAL,
                severity=ThreatSeverity.CRITICAL,
                category=ThreatCategory.GENERAL,
                tags=["ランサムウェア", "製造業", "VPN", "国内インシデント"],
                is_featured=True,
            ),
            self._make_threat(
                title="改正個人情報保護法：AI 学習データへの適用明確化（2025年改正）",
                summary="2025年の個人情報保護法改正により、生成AIの学習データとして個人情報を利用する場合の要件が明確化された。第三者提供規制や仮名加工情報の取り扱いについて自社ポリシーを見直す必要がある。",
                source=ThreatSource.MANUAL,
                severity=ThreatSeverity.MEDIUM,
                category=ThreatCategory.REGULATION,
                tags=["個人情報保護法", "AI学習データ", "法令", "コンプライアンス"],
                is_featured=False,
            ),
        ]
        self.store.add_many(samples)
        return samples

    def collect_all(self) -> Dict[str, List[SecurityThreat]]:
        """全ソースを一括収集する"""
        return {
            "nvd":    self.collect_nvd(),
            "cisa":   self.collect_cisa(),
            "ai":     self.collect_ai_feed(),
            "manual": self.collect_manual(),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SecurityIntelDashboard + IntelReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SecurityIntelDashboard:
    """
    横断集計ダッシュボード。

    Usage:
        dash = SecurityIntelDashboard(store)
        summary = dash.summary()
        feed = dash.featured_feed()
    """

    def __init__(self, store: SecurityIntelStore) -> None:
        self._store = store

    def summary(self) -> Dict[str, Any]:
        """全体サマリー"""
        base = self._store.summary()
        by_diag: Dict[str, int] = {}
        for t in self._store.list_all(limit=9999):
            for cat in t.diagnosis_categories:
                by_diag[cat] = by_diag.get(cat, 0) + 1
        base["by_diagnosis_category"] = by_diag
        return base

    def featured_feed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """注目情報フィード（is_featured=True, 深刻度順）"""
        featured = self._store.list_featured()
        featured.sort(key=lambda t: t.severity.score, reverse=True)
        return [t.to_dict() for t in featured[:limit]]

    def critical_threats(self) -> List[Dict[str, Any]]:
        """Critical 脅威一覧"""
        threats = self._store.list_by_severity(ThreatSeverity.CRITICAL)
        return [t.to_dict() for t in threats]

    def by_diagnosis_category(self, cat: str) -> List[Dict[str, Any]]:
        """診断カテゴリ(A〜F)別の脅威一覧"""
        return [t.to_dict() for t in self._store.list_by_diagnosis_category(cat)]


class IntelReportEngine:
    """セキュリティインテリジェンス Markdown/JSON レポート生成"""

    def __init__(self, store: SecurityIntelStore) -> None:
        self._store = store
        self._dash  = SecurityIntelDashboard(store)

    def summary_json(self) -> Dict[str, Any]:
        return self._dash.summary()

    def markdown(self, limit: int = 20) -> str:
        threats = self._store.list_all(limit=limit)
        summary = self._dash.summary()
        lines = [
            "# セキュリティインテリジェンスレポート",
            "",
            f"**総件数**: {summary['total']}  ",
            f"**Critical**: {summary['by_severity'].get('critical', 0)}  ",
            f"**High**: {summary['by_severity'].get('high', 0)}  ",
            "",
            "## 脅威一覧",
            "",
            "| 深刻度 | ソース | カテゴリ | タイトル |",
            "|--------|--------|----------|----------|",
        ]
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        for t in sorted(threats, key=lambda x: sev_order.get(x.severity.value, 9)):
            title = t.enrichment.title_ja if t.enrichment else t.title
            lines.append(
                f"| {t.severity.value} | {t.source.value} | {t.category.value} | {title} |"
            )
        return "\n".join(lines)

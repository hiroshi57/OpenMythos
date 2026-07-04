"""
serve/routers/security.py — セキュリティドメイン API (Sprint 53 / 59 / 68)

セキュリティ統合 (InputGuard/OutputGuard) / 自律脆弱性スキャン /
セキュリティインテリジェンス + リスクカテゴリ分類。
serve/api.py のモノリスから分割 (認証は app 全体の verify_api_key に委譲)。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from serve.auth import verify_api_key

router = APIRouter()

# ============================================================================
# Sprint 53 — セキュリティ統合 API
# ============================================================================

from open_mythos.skills.security import (  # noqa: E402
    WebPentester as _WebPentester,
    OSSForensics as _OSSForensics,
)

# ── リクエストモデル ────────────────────────────────────────────────────────────


class _SecurityScanRequest(BaseModel):
    target_url: str
    checks: Optional[List[str]] = None
    timeout: float = 10.0


class _OSSAnalyzeRequest(BaseModel):
    project_path: str = "."


# ── エンドポイント ─────────────────────────────────────────────────────────────


@router.post(
    "/v1/security/scan",
    tags=["security"],
    summary="WebPentester — パッシブセキュリティスキャン (Sprint 53)",
    dependencies=[Depends(verify_api_key)],
)
def security_scan(req: _SecurityScanRequest):
    """ターゲット URL をスキャンし OWASP ベースの脆弱性を報告する。"""
    pentester = _WebPentester()
    report = pentester.scan(req.target_url, checks=req.checks, timeout=req.timeout)
    return {
        "target_url": report.target_url,
        "risk_score": report.risk_score,
        "scan_time_s": report.scan_time_s,
        "summary": report.summary,
        "critical_count": report.critical_count,
        "high_count": report.high_count,
        "findings": [
            {
                "severity": f.severity,
                "category": f.category,
                "title": f.title,
                "description": f.description,
                "url": f.url,
                "recommendation": f.recommendation,
            }
            for f in report.findings
        ],
    }


@router.post(
    "/v1/security/report/md",
    tags=["security"],
    summary="WebPentester — Markdown レポート生成 (Sprint 53)",
    dependencies=[Depends(verify_api_key)],
)
def security_report_md(req: _SecurityScanRequest):
    """セキュリティスキャンを実行し Markdown レポートを返す。"""
    pentester = _WebPentester()
    report = pentester.scan(req.target_url, checks=req.checks, timeout=req.timeout)
    md = pentester.generate_report_md(report)
    return {
        "markdown": md,
        "risk_score": report.risk_score,
        "n_findings": len(report.findings),
    }


@router.post(
    "/v1/security/oss/analyze",
    tags=["security"],
    summary="OSSForensics — 依存関係・ライセンス分析 (Sprint 53)",
    dependencies=[Depends(verify_api_key)],
)
def oss_analyze(req: _OSSAnalyzeRequest):
    """プロジェクトの OSS 依存関係とライセンスを分析する。"""
    oss = _OSSForensics()
    report = oss.analyze(req.project_path)
    return {
        "project_path": report.project_path,
        "total_deps": report.total_deps,
        "vulnerable_count": report.vulnerable_count,
        "license_issues": report.license_issues,
        "dependencies": [
            {
                "name": d.name,
                "version": d.version,
                "license": d.license,
                "has_known_vuln": d.has_known_vuln,
                "vuln_ids": d.vuln_ids,
            }
            for d in report.dependencies[:20]  # 最大20件
        ],
    }


@router.post(
    "/v1/security/oss/sbom",
    tags=["security"],
    summary="OSSForensics — SBOM (CycloneDX) 生成 (Sprint 53)",
    dependencies=[Depends(verify_api_key)],
)
def oss_sbom(req: _OSSAnalyzeRequest):
    """プロジェクトの SBOM (Software Bill of Materials) を CycloneDX 形式で生成する。"""
    oss = _OSSForensics()
    report = oss.analyze(req.project_path)
    sbom_json = oss.generate_sbom(report.dependencies)
    return {
        "sbom": sbom_json,
        "format": "CycloneDX",
        "n_components": len(report.dependencies),
    }



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 59 — 自律脆弱性スキャン エンドポイント
# harness: TargetConfig→ScanTarget / CrashArtifact→VulnFinding /
#          GraderVerdict→VerifyVerdict / PatchVerdict→PatchCandidate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.vuln_scanner import (  # noqa: E402
    VulnStore      as _VulnStore,
    VulnScanner    as _VulnScanner,
    VulnPatcher    as _VulnPatcher,
    ScanTarget     as _ScanTarget,
    ScanReportEngine as _ScanReportEngine,
)

# サービス全体で共有するシングルトン (インプロセス)
_vuln_store:   _VulnStore   = _VulnStore()
_vuln_scanner: _VulnScanner = _VulnScanner(_vuln_store)


class _VulnScanReq(BaseModel):
    target_name:     str
    target_path:     str
    source:          str           = ""
    language:        str           = "python"
    focus_areas:     Optional[list[str]] = None
    known_findings:  Optional[list[str]] = None


@router.post("/v1/vuln/scan", tags=["vuln-scanner"], summary="脆弱性スキャン実行 — Sprint 59")
def vuln_scan(req: _VulnScanReq):
    """ソースコードをスキャンして VulnFinding を検出する。
    harness の run_find() + run_recon() に相当。
    """
    target = _ScanTarget(
        name=req.target_name,
        path=req.target_path,
        language=req.language,
        focus_areas=req.focus_areas or [],
        known_findings=req.known_findings or [],
    )
    session = _vuln_scanner.scan(target, source=req.source)
    return session.to_dict()


@router.get("/v1/vuln/findings", tags=["vuln-scanner"], summary="全 Finding 一覧 — Sprint 59")
def vuln_list_findings():
    """スキャンで検出した全 Finding を返す。"""
    return [f.to_dict() for f in _vuln_store.list_findings()]


@router.get("/v1/vuln/findings/{finding_id}", tags=["vuln-scanner"], summary="Finding 詳細 — Sprint 59")
def vuln_get_finding(finding_id: str):
    """Finding 1 件を返す。"""
    f = _vuln_store.get_finding(finding_id)
    if f is None:
        raise HTTPException(404, f"Finding not found: {finding_id}")
    return f.to_dict()


@router.delete("/v1/vuln/findings/{finding_id}", tags=["vuln-scanner"], summary="Finding 削除 — Sprint 59")
def vuln_delete_finding(finding_id: str):
    """Finding を削除する。"""
    deleted = _vuln_store.delete_finding(finding_id)
    if not deleted:
        raise HTTPException(404, f"Finding not found: {finding_id}")
    return {"deleted": finding_id}


class _VulnPatchReq(BaseModel):
    finding_id: str


@router.post("/v1/vuln/patch/{finding_id}", tags=["vuln-scanner"], summary="パッチ候補生成+検証 — Sprint 59")
def vuln_patch(finding_id: str):
    """Finding に対してパッチ候補を生成し T0/T1/T2 ラダーで検証する。
    harness の patch + patch_grade に相当。
    """
    f = _vuln_store.get_finding(finding_id)
    if f is None:
        raise HTTPException(404, f"Finding not found: {finding_id}")
    patcher = _VulnPatcher()
    candidate = patcher.suggest_patch(f)
    if candidate is None:
        raise HTTPException(422, f"Auto-patch not available for finding: {finding_id}")
    validated = patcher.validate_patch(candidate)
    return validated.to_dict()


@router.get("/v1/vuln/session/{session_id}", tags=["vuln-scanner"], summary="スキャンセッション詳細 — Sprint 59")
def vuln_get_session(session_id: str):
    """ScanSession を返す。harness の RunResult に相当。"""
    session = _vuln_store.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")
    return session.to_dict()


@router.get("/v1/vuln/session/{session_id}/report", tags=["vuln-scanner"], summary="脆弱性レポート JSON — Sprint 59")
def vuln_session_report(session_id: str):
    """ScanSession の JSON レポートを返す。harness の ReportVerdict に相当。"""
    session = _vuln_store.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")
    engine = _ScanReportEngine()
    return engine.to_json(session)


@router.get("/v1/vuln/session/{session_id}/report/md", tags=["vuln-scanner"], summary="脆弱性レポート Markdown — Sprint 59")
def vuln_session_report_md(session_id: str):
    """ScanSession の Markdown レポートを返す。"""
    session = _vuln_store.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")
    engine = _ScanReportEngine()
    return Response(content=engine.to_markdown(session), media_type="text/markdown")



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 68 — セキュリティインテリジェンス + カテゴリ分類
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.security_intel import (  # noqa: E402
    ThreatSeverity as _ThreatSeverity,
    ThreatSource   as _ThreatSource,
    ThreatCategory as _ThreatCategory,
    SecurityThreat as _SecurityThreat,
    SecurityIntelStore  as _SecurityIntelStore,
    ThreatEnricher      as _ThreatEnricher,
    ThreatCollector     as _ThreatCollector,
    SecurityIntelDashboard as _SecurityIntelDashboard,
    IntelReportEngine   as _IntelReportEngine,
)
from open_mythos.skills.security import ThreatCategoryMapper as _ThreatCategoryMapper  # noqa: E402

# シングルトン
_intel_store   = _SecurityIntelStore()
_intel_enricher = _ThreatEnricher()   # LLM なしなら rule-based
_intel_collector = _ThreatCollector(store=_intel_store, enricher=_intel_enricher)
_intel_dashboard = _SecurityIntelDashboard(_intel_store)
_intel_report    = _IntelReportEngine(_intel_store)
_threat_mapper   = _ThreatCategoryMapper()


# ---- リクエストモデル ----

class _ThreatCreateReq(BaseModel):
    title:      str
    summary:    str
    source:     str = "manual"
    severity:   str = "medium"
    category:   str = "general"
    source_url: Optional[str] = None
    tags:       list = []
    is_featured: bool = False


class _CollectReq(BaseModel):
    sources: list = []   # 空 = 全ソース。例: ["nvd", "cisa", "ai", "manual"]
    enrich:  bool = False


class _CategoryMapReq(BaseModel):
    title:   str
    summary: str = ""


# ---- エンドポイント ----

@router.get("/v1/intel/threats", tags=["intel"], summary="脅威情報一覧 — Sprint 68")
def intel_list_threats(
    severity: Optional[str] = None,
    source:   Optional[str] = None,
    category: Optional[str] = None,
    featured: Optional[bool] = None,
    limit:    int = 50,
):
    """脅威情報をフィルタ付きで一覧取得する。"""
    if featured:
        threats = _intel_store.list_featured()
    elif severity:
        try:
            sev = _ThreatSeverity(severity)
            threats = _intel_store.list_by_severity(sev)
        except ValueError:
            raise HTTPException(400, f"Invalid severity: {severity}")
    elif source:
        try:
            src = _ThreatSource(source)
            threats = _intel_store.list_by_source(src)
        except ValueError:
            raise HTTPException(400, f"Invalid source: {source}")
    elif category:
        try:
            cat = _ThreatCategory(category)
            threats = _intel_store.list_by_category(cat)
        except ValueError:
            raise HTTPException(400, f"Invalid category: {category}")
    else:
        threats = _intel_store.list_all(limit=limit)
    return {"threats": [t.to_dict() for t in threats[:limit]], "total": len(threats)}


@router.get("/v1/intel/threats/{threat_id}", tags=["intel"], summary="脅威詳細 — Sprint 68")
def intel_get_threat(threat_id: str):
    t = _intel_store.get(threat_id)
    if t is None:
        raise HTTPException(404, f"Threat not found: {threat_id}")
    return t.to_dict()


@router.post("/v1/intel/threats", tags=["intel"], summary="脅威情報手動登録 — Sprint 68")
def intel_create_threat(req: _ThreatCreateReq):
    """手動で脅威情報を登録し、診断カテゴリを自動付与する。"""
    try:
        source   = _ThreatSource(req.source)
        severity = _ThreatSeverity(req.severity)
        category = _ThreatCategory(req.category)
    except ValueError as e:
        raise HTTPException(400, str(e))

    import uuid as _uuid
    threat = _SecurityThreat(
        id=str(_uuid.uuid4()),
        title=req.title,
        summary=req.summary,
        source=source,
        severity=severity,
        category=category,
        source_url=req.source_url,
        tags=req.tags,
        is_featured=req.is_featured,
    )
    # 診断カテゴリ自動付与
    matches = _threat_mapper.map(req.title, req.summary)
    threat.diagnosis_categories = [m.category.value for m in matches]

    _intel_store.add(threat)
    return threat.to_dict()


@router.post("/v1/intel/collect", tags=["intel"], summary="情報収集トリガー — Sprint 68")
def intel_collect(req: _CollectReq):
    """
    指定ソースから脅威情報を収集して登録する。
    sources=[] のときは全ソース (nvd/cisa/ai/manual) を収集する。
    enrich=True のときは AI 富化も実行する（LLM API キーが必要）。
    """
    collector = _ThreatCollector(
        store=_intel_store,
        enricher=_intel_enricher,
        enrich_on_collect=req.enrich,
    )
    sources = req.sources or ["nvd", "cisa", "ai", "manual"]
    collected: Dict[str, int] = {}
    if "nvd"    in sources: collected["nvd"]    = len(collector.collect_nvd())
    if "cisa"   in sources: collected["cisa"]   = len(collector.collect_cisa())
    if "ai"     in sources: collected["ai"]     = len(collector.collect_ai_feed())
    if "manual" in sources: collected["manual"] = len(collector.collect_manual())
    return {"collected": collected, "total": sum(collected.values())}


@router.post("/v1/intel/threats/{threat_id}/enrich", tags=["intel"], summary="個別AI富化 — Sprint 68")
def intel_enrich_threat(threat_id: str):
    """指定した脅威を AI で富化する（LLM 不在時は rule-based）。"""
    t = _intel_store.get(threat_id)
    if t is None:
        raise HTTPException(404, f"Threat not found: {threat_id}")
    t.enrichment = _intel_enricher.enrich(t)
    return t.to_dict()


@router.get("/v1/intel/summary", tags=["intel"], summary="インテリジェンスサマリー — Sprint 68")
def intel_summary():
    return _intel_dashboard.summary()


@router.get("/v1/intel/feed/featured", tags=["intel"], summary="注目脅威フィード — Sprint 68")
def intel_featured_feed(limit: int = 10):
    return {"feed": _intel_dashboard.featured_feed(limit=limit)}


@router.get("/v1/intel/report/md", tags=["intel"], summary="インテルレポート Markdown — Sprint 68")
def intel_report_md(limit: int = 20):
    return Response(content=_intel_report.markdown(limit=limit), media_type="text/markdown")


@router.post("/v1/intel/category-map", tags=["intel"], summary="脅威→診断カテゴリ判定 — Sprint 68")
def intel_category_map(req: _CategoryMapReq):
    """
    タイトル・サマリーから診断カテゴリ(A〜F)を判定する。
    (security-app の threat-category-map.ts 相当)
    """
    matches = _threat_mapper.map(req.title, req.summary)
    return {
        "title":   req.title,
        "matches": [m.to_dict() for m in matches],
        "primary": matches[0].to_dict() if matches else None,
    }



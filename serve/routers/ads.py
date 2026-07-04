"""
serve/routers/ads.py — 広告運用ドメイン API (Sprint 60 / 63〜67 / 69 / 70)

広告キャンペーン管理 / A/B テスト + 分析ダッシュボード / 予算最適化 /
Fusion マルチモデル融合 + キャッシュ / オーケストレーター (勝者判定・凍結) /
KPI 異常検知 / 時系列予測 / 予測アラート / レポート Webhook / NLQ。
serve/api.py のモノリスから分割 (認証は app 全体の verify_api_key に委譲)。
"""

from __future__ import annotations

import json as _json_mod
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from serve.state import state

router = APIRouter()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 60 — 広告キャンペーン管理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.campaign_manager import (  # noqa: E402
    AdObjective as _AdObjective,
    AdChannel as _AdChannel,
    CampaignBudget as _CampaignBudget,
    CampaignStore as _CampaignStore,
    CampaignWorkflow as _CampaignWorkflow,
    CampaignReportEngine as _CampaignReportEngine,
)

_campaign_store    = _CampaignStore()
_campaign_workflow = _CampaignWorkflow(store=_campaign_store)
_campaign_report   = _CampaignReportEngine(_campaign_store)


class _CampaignCreateReq(BaseModel):
    name:         str
    objective:    str   = "awareness"
    budget_total: float = 100_000
    budget_daily: float = 10_000
    currency:     str   = "JPY"
    channels:     list  = []
    cep_ids:      list  = []
    tags:         list  = []
    description:  str   = ""


class _CampaignWorkflowReq(BaseModel):
    name:         str
    scenario:     str
    objective:    str   = "awareness"
    budget_total: float = 100_000
    budget_daily: float = 10_000
    currency:     str   = "JPY"
    channels:     list  = []
    cep_ids:      list  = []
    tags:         list  = []
    description:  str   = ""
    extra:        dict  = {}


@router.post("/v1/campaign/workflow", tags=["campaign"], summary="CEP→コピー生成→評価フロー — Sprint 60")
def campaign_workflow(req: _CampaignWorkflowReq):
    """シナリオからコピーを生成・評価してキャンペーンを登録する。"""
    try:
        obj = _AdObjective(req.objective)
    except ValueError:
        raise HTTPException(422, f"Unknown objective: {req.objective}")
    channels = [_AdChannel(ch) for ch in req.channels] if req.channels else None
    budget = _CampaignBudget(
        total=req.budget_total,
        daily=req.budget_daily,
        currency=req.currency,
    )
    result = _campaign_workflow.run(
        name=req.name,
        scenario=req.scenario,
        objective=obj,
        budget=budget,
        channels=channels,
        cep_ids=req.cep_ids or [],
        tags=req.tags or [],
        description=req.description,
        extra=req.extra or None,
    )
    return result.to_dict()


@router.post("/v1/campaign/", tags=["campaign"], summary="キャンペーン作成 — Sprint 60")
def campaign_create(req: _CampaignCreateReq):
    """新規キャンペーンを作成する（コピー生成なし）。"""
    import uuid as _uuid
    from open_mythos.skills.campaign_manager import Campaign as _Campaign
    try:
        obj = _AdObjective(req.objective)
    except ValueError:
        raise HTTPException(422, f"Unknown objective: {req.objective}")
    channels = [_AdChannel(ch) for ch in req.channels] if req.channels else []
    budget = _CampaignBudget(
        total=req.budget_total,
        daily=req.budget_daily,
        currency=req.currency,
    )
    campaign = _Campaign(
        id=str(_uuid.uuid4()),
        name=req.name,
        objective=obj,
        budget=budget,
        channels=channels,
        cep_ids=req.cep_ids or [],
        tags=req.tags or [],
        description=req.description,
    )
    _campaign_store.add(campaign)
    return campaign.to_dict()


@router.get("/v1/campaign/", tags=["campaign"], summary="キャンペーン一覧 — Sprint 60")
def campaign_list():
    """全キャンペーンを返す。"""
    return [c.to_dict() for c in _campaign_store.list_all()]


@router.get("/v1/campaign/{campaign_id}", tags=["campaign"], summary="キャンペーン詳細 — Sprint 60")
def campaign_get(campaign_id: str):
    c = _campaign_store.get(campaign_id)
    if c is None:
        raise HTTPException(404, f"Campaign not found: {campaign_id}")
    return c.to_dict()


@router.delete("/v1/campaign/{campaign_id}", tags=["campaign"], summary="キャンペーン削除 — Sprint 60")
def campaign_delete(campaign_id: str):
    deleted = _campaign_store.delete(campaign_id)
    if not deleted:
        raise HTTPException(404, f"Campaign not found: {campaign_id}")
    return {"deleted": campaign_id}


@router.post("/v1/campaign/{campaign_id}/activate", tags=["campaign"], summary="キャンペーン有効化 — Sprint 60")
def campaign_activate(campaign_id: str):
    c = _campaign_store.get(campaign_id)
    if c is None:
        raise HTTPException(404, f"Campaign not found: {campaign_id}")
    try:
        c.activate()
    except ValueError as e:
        raise HTTPException(422, str(e))
    return c.to_dict()


@router.post("/v1/campaign/{campaign_id}/pause", tags=["campaign"], summary="キャンペーン一時停止 — Sprint 60")
def campaign_pause(campaign_id: str):
    c = _campaign_store.get(campaign_id)
    if c is None:
        raise HTTPException(404, f"Campaign not found: {campaign_id}")
    try:
        c.pause()
    except ValueError as e:
        raise HTTPException(422, str(e))
    return c.to_dict()


@router.post("/v1/campaign/{campaign_id}/complete", tags=["campaign"], summary="キャンペーン完了 — Sprint 60")
def campaign_complete(campaign_id: str):
    c = _campaign_store.get(campaign_id)
    if c is None:
        raise HTTPException(404, f"Campaign not found: {campaign_id}")
    try:
        c.complete()
    except ValueError as e:
        raise HTTPException(422, str(e))
    return c.to_dict()


@router.get("/v1/campaign/{campaign_id}/report/md", tags=["campaign"], summary="キャンペーンレポート Markdown — Sprint 60")
def campaign_report_md(campaign_id: str):
    c = _campaign_store.get(campaign_id)
    if c is None:
        raise HTTPException(404, f"Campaign not found: {campaign_id}")
    md = _campaign_report.campaign_markdown(campaign_id)
    return Response(content=md, media_type="text/markdown")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 64A — A/B テスト + 分析ダッシュボード API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import uuid as _uuid64  # noqa: E402
from open_mythos.skills.ab_test import (  # noqa: E402
    ABTest as _ABTest,
    Variant as _Variant,
    ABTestStore as _ABTestStore,
    ABTestAnalyzer as _ABTestAnalyzer,
    ABTestReportEngine as _ABTestReportEngine,
)
from open_mythos.skills.campaign_analytics import (  # noqa: E402
    CampaignAnalyticsStore as _AnalyticsStore,
    CampaignAnalyticsDashboard as _AnalyticsDashboard,
    AnalyticsReportEngine as _AnalyticsReportEngine,
)

_abtest_store     = _ABTestStore()
_abtest_analyzer  = _ABTestAnalyzer()
_abtest_report    = _ABTestReportEngine(_abtest_analyzer)

_analytics_store     = _AnalyticsStore()
_analytics_dashboard = _AnalyticsDashboard(store=_analytics_store)
_analytics_report    = _AnalyticsReportEngine(_analytics_dashboard)


# ---- A/B テスト ----

class _ABTestCreateReq(BaseModel):
    name:        str
    variants:    list = []   # [{name, content, weight}, ...]
    campaign_id: Optional[str] = None
    description: str = ""


class _RecordStatsReq(BaseModel):
    impressions: int = 0
    clicks:      int = 0
    conversions: int = 0


@router.post("/v1/abtest/", tags=["abtest"], summary="A/B テスト作成 — Sprint 64")
def abtest_create(req: _ABTestCreateReq):
    test = _ABTest(
        id=str(_uuid64.uuid4()),
        name=req.name,
        campaign_id=req.campaign_id,
        description=req.description,
    )
    for v in req.variants:
        test.add_variant(_Variant(
            id=str(_uuid64.uuid4()),
            name=v.get("name", "variant"),
            content=v.get("content", ""),
            weight=float(v.get("weight", 1.0)),
        ))
    _abtest_store.add(test)
    return test.to_dict()


@router.get("/v1/abtest/", tags=["abtest"], summary="A/B テスト一覧 — Sprint 64")
def abtest_list():
    return [t.to_dict() for t in _abtest_store.list_all()]


@router.get("/v1/abtest/{test_id}", tags=["abtest"], summary="A/B テスト詳細 — Sprint 64")
def abtest_get(test_id: str):
    t = _abtest_store.get(test_id)
    if t is None:
        raise HTTPException(404, f"ABTest not found: {test_id}")
    return t.to_dict()


@router.delete("/v1/abtest/{test_id}", tags=["abtest"], summary="A/B テスト削除 — Sprint 64")
def abtest_delete(test_id: str):
    if not _abtest_store.delete(test_id):
        raise HTTPException(404, f"ABTest not found: {test_id}")
    return {"deleted": test_id}


@router.post("/v1/abtest/{test_id}/start", tags=["abtest"], summary="A/B テスト開始 — Sprint 64")
def abtest_start(test_id: str):
    t = _abtest_store.get(test_id)
    if t is None:
        raise HTTPException(404, f"ABTest not found: {test_id}")
    try:
        t.start()
    except ValueError as e:
        raise HTTPException(422, str(e))
    return t.to_dict()


@router.post("/v1/abtest/{test_id}/variant/{variant_id}/record", tags=["abtest"], summary="Variant 実績記録 — Sprint 64")
def abtest_record(test_id: str, variant_id: str, req: _RecordStatsReq):
    t = _abtest_store.get(test_id)
    if t is None:
        raise HTTPException(404, f"ABTest not found: {test_id}")
    v = t.get_variant(variant_id)
    if v is None:
        raise HTTPException(404, f"Variant not found: {variant_id}")
    v.stats.record_impression(req.impressions)
    v.stats.record_click(req.clicks)
    v.stats.record_conversion(req.conversions)
    return v.to_dict()


@router.get("/v1/abtest/{test_id}/report", tags=["abtest"], summary="A/B テストレポート JSON — Sprint 64")
def abtest_report_json(test_id: str):
    t = _abtest_store.get(test_id)
    if t is None:
        raise HTTPException(404, f"ABTest not found: {test_id}")
    return _abtest_report.summary_json(t)


@router.get("/v1/abtest/{test_id}/report/md", tags=["abtest"], summary="A/B テストレポート Markdown — Sprint 64")
def abtest_report_md(test_id: str):
    t = _abtest_store.get(test_id)
    if t is None:
        raise HTTPException(404, f"ABTest not found: {test_id}")
    return Response(content=_abtest_report.markdown(t), media_type="text/markdown")


# ---- 分析ダッシュボード ----

class _AnalyticsRecordReq(BaseModel):
    impressions: int   = 0
    clicks:      int   = 0
    conversions: int   = 0
    spend:       float = 0.0
    revenue:     float = 0.0


@router.post("/v1/analytics/{campaign_id}/record", tags=["analytics"], summary="メトリクス記録 — Sprint 64")
def analytics_record(campaign_id: str, req: _AnalyticsRecordReq):
    point = _analytics_store.record(
        campaign_id,
        impressions=req.impressions,
        clicks=req.clicks,
        conversions=req.conversions,
        spend=req.spend,
        revenue=req.revenue,
    )
    return point.to_dict()


@router.get("/v1/analytics/summary", tags=["analytics"], summary="全キャンペーン分析サマリー — Sprint 64")
def analytics_summary():
    return _analytics_report.summary_json()


@router.get("/v1/analytics/{campaign_id}/kpis", tags=["analytics"], summary="キャンペーン KPI — Sprint 64")
def analytics_kpis(campaign_id: str):
    kpis = _analytics_dashboard.campaign_kpis(campaign_id)
    if kpis is None:
        raise HTTPException(404, f"Analytics not found: {campaign_id}")
    return kpis.to_dict()


@router.get("/v1/analytics/{campaign_id}/report/md", tags=["analytics"], summary="分析レポート Markdown — Sprint 64")
def analytics_report_md(campaign_id: str):
    if _analytics_store.get(campaign_id) is None:
        raise HTTPException(404, f"Analytics not found: {campaign_id}")
    md = _analytics_report.campaign_markdown(campaign_id)
    return Response(content=md, media_type="text/markdown")


# ---- 予算最適化 (Sprint 64C) ----

from open_mythos.skills.budget_optimizer import (  # noqa: E402
    BudgetOptimizer as _BudgetOptimizer,
    AllocationStrategy as _AllocationStrategy,
)

_budget_optimizer = _BudgetOptimizer(store=_analytics_store)


class _OptimizeReq(BaseModel):
    total_budget: float = 0.0
    campaign_ids: list  = []
    strategy:     str   = "roas_weighted"


@router.post("/v1/budget/optimize", tags=["budget"], summary="予算最適化 — Sprint 64")
def budget_optimize(req: _OptimizeReq):
    try:
        strategy = _AllocationStrategy(req.strategy)
    except ValueError:
        raise HTTPException(422, f"Unknown strategy: {req.strategy}")
    try:
        result = _budget_optimizer.optimize(
            total_budget=req.total_budget,
            campaign_ids=req.campaign_ids,
            strategy=strategy,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    return result.to_dict()


@router.post("/v1/budget/recommend-strategy", tags=["budget"], summary="推奨配分戦略 — Sprint 64")
def budget_recommend_strategy(req: _OptimizeReq):
    strategy = _budget_optimizer.recommend_strategy(req.campaign_ids)
    return {"recommended_strategy": strategy.value}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 65 — Fusion マルチモデル融合 (OpenRouter Fusion 移植)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.fusion import (  # noqa: E402
    FusionConfig as _FusionConfig,
    CandidateSpec as _CandidateSpec,
    FusionEngine as _FusionEngine,
    FusionEngineFactory as _FusionEngineFactory,
)

# 既定エンジン（環境変数から構築、API キーがなければヒューリスティック動作）
_fusion_llm = state.llm if hasattr(state, "llm") else None
_fusion_engine: _FusionEngine = _FusionEngineFactory.from_env(llm=_fusion_llm)


class _FusionCandidateReq(BaseModel):
    label:              str
    preferred_provider: Optional[str] = None
    temperature:        float = 0.7
    max_tokens:         int   = 512


class _FusionReq(BaseModel):
    question:           str
    system:             Optional[str] = None
    candidates:         list = []   # [{label, preferred_provider, temperature}, ...]
    judge_provider:     Optional[str] = None
    caller_provider:    Optional[str] = None


@router.post("/v1/fusion/run", tags=["fusion"], summary="Fusion マルチモデル融合 — Sprint 65")
def fusion_run(req: _FusionReq):
    """
    候補モデル群 → 審査モデルが構造化分析 → 呼び出しモデルが最終回答合成。
    candidates 未指定時はデフォルト 3 候補構成を使用する。
    """
    if req.candidates:
        specs = [
            _CandidateSpec(
                label=c.get("label", f"candidate-{i}"),
                preferred_provider=c.get("preferred_provider"),
                temperature=float(c.get("temperature", 0.7)),
                max_tokens=int(c.get("max_tokens", 512)),
            )
            for i, c in enumerate(req.candidates)
        ]
        config = _FusionConfig(
            candidates=specs,
            judge_provider=req.judge_provider,
            caller_provider=req.caller_provider,
        )
        engine = _FusionEngine(config=config, router=_fusion_engine._router)
    else:
        engine = _fusion_engine

    result = engine.run(req.question, system=req.system)
    return result.to_dict()


@router.get("/v1/fusion/status", tags=["fusion"], summary="Fusion エンジン状態 — Sprint 65")
def fusion_status():
    """Fusion エンジンの LLM 接続状態を返す。"""
    return {
        "has_llm":             _fusion_engine.has_llm,
        "default_candidates":  len(_FusionConfig.default().candidates),
    }


@router.post("/v1/fusion/stream", tags=["fusion"], summary="Fusion ストリーミング — Sprint 66")
def fusion_stream(req: _FusionReq):
    """
    Fusion パイプラインを SSE ストリーミングする (Sprint 66A)。
    各段階 (candidates / analysis / delta / done) を event として送出する。
    """
    if req.candidates:
        specs = [
            _CandidateSpec(
                label=c.get("label", f"candidate-{i}"),
                preferred_provider=c.get("preferred_provider"),
                temperature=float(c.get("temperature", 0.7)),
                max_tokens=int(c.get("max_tokens", 512)),
            )
            for i, c in enumerate(req.candidates)
        ]
        config = _FusionConfig(
            candidates=specs,
            judge_provider=req.judge_provider,
            caller_provider=req.caller_provider,
        )
        engine = _FusionEngine(config=config, router=_fusion_engine._router)
    else:
        engine = _fusion_engine

    def _event_stream():
        for event in engine.run_stream(req.question, system=req.system):
            payload = _json_mod.dumps(event["data"], ensure_ascii=False)
            yield f"event: {event['stage']}\ndata: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 66B — A/B → 予算最適化 自動連携オーケストレーター
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.campaign_orchestrator import (  # noqa: E402
    CampaignOrchestrator as _CampaignOrchestrator,
)

_orchestrator = _CampaignOrchestrator(analytics_store=_analytics_store)


class _ReallocateReq(BaseModel):
    test_id:            str
    total_budget:       float
    campaign_ids:       list
    winner_campaign_id: Optional[str] = None


@router.post("/v1/orchestrator/decide-winner/{test_id}", tags=["orchestrator"], summary="A/B 勝者判定 — Sprint 66")
def orchestrator_decide_winner(test_id: str):
    t = _abtest_store.get(test_id)
    if t is None:
        raise HTTPException(404, f"ABTest not found: {test_id}")
    return _orchestrator.decide_winner(t).to_dict()


@router.post("/v1/orchestrator/reallocate", tags=["orchestrator"], summary="勝者ベース予算再配分 — Sprint 66")
def orchestrator_reallocate(req: _ReallocateReq):
    t = _abtest_store.get(req.test_id)
    if t is None:
        raise HTTPException(404, f"ABTest not found: {req.test_id}")
    return _orchestrator.run_workflow(
        t, req.total_budget, req.campaign_ids, req.winner_campaign_id
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 66C — KPI 異常検知アラート
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.anomaly_detector import (  # noqa: E402
    AnomalyDetector as _AnomalyDetector,
    AlertStore as _AlertStore,
    AnomalyReportEngine as _AnomalyReportEngine,
)

_anomaly_detector = _AnomalyDetector()
_alert_store      = _AlertStore()
_anomaly_report   = _AnomalyReportEngine(_alert_store)


class _DetectReq(BaseModel):
    metrics: list = []   # 検知対象指標。空ならデフォルト全指標


@router.post("/v1/anomaly/{campaign_id}/detect", tags=["anomaly"], summary="異常検知実行 — Sprint 66")
def anomaly_detect(campaign_id: str, req: _DetectReq):
    m = _analytics_store.get(campaign_id)
    if m is None:
        raise HTTPException(404, f"Analytics not found: {campaign_id}")
    metric_list = req.metrics or None
    alerts = _anomaly_detector.detect_multi(m, metric_list)
    _alert_store.add_many(alerts)
    return {"detected": len(alerts), "alerts": [a.to_dict() for a in alerts]}


@router.get("/v1/anomaly/alerts", tags=["anomaly"], summary="アラート一覧 — Sprint 66")
def anomaly_alerts():
    return _anomaly_report.summary_json()


@router.get("/v1/anomaly/alerts/report/md", tags=["anomaly"], summary="アラートレポート Markdown — Sprint 66")
def anomaly_alerts_md():
    return Response(content=_anomaly_report.markdown(), media_type="text/markdown")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 67A — 異常検知 → 自動予算停止
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



class _FreezeReq(BaseModel):
    campaign_ids:  list
    total_budget:  float
    use_alert_store: bool = True   # True: 既存 _alert_store を参照, False: 再検知


@router.post("/v1/orchestrator/freeze", tags=["orchestrator"], summary="Critical アラートで予算を自動凍結 — Sprint 67")
def orchestrator_freeze(req: _FreezeReq):
    """
    _alert_store に蓄積された Critical アラートを参照して
    該当キャンペーンの予算を 0 に凍結し、残予算を再配分する。
    use_alert_store=False のときは analytics_store から再検知する。
    """
    alert_store_arg = _alert_store if req.use_alert_store else None
    plan = _orchestrator.freeze_if_critical(
        campaign_ids=req.campaign_ids,
        total_budget=req.total_budget,
        alert_store=alert_store_arg,
    )
    return plan.to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 67B — Fusion 結果キャッシュ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.fusion_cache import (  # noqa: E402
    FusionCache as _FusionCache,
    CachedFusionEngine as _CachedFusionEngine,
)

_fusion_cache        = _FusionCache(ttl=300.0, max_size=128)
_cached_fusion_engine = _CachedFusionEngine(_fusion_engine, cache=_fusion_cache)


class _CachedFusionReq(BaseModel):
    question:  str
    system:    Optional[str] = None
    candidates: list = []
    judge_provider:  Optional[str] = None
    caller_provider: Optional[str] = None


@router.post("/v1/fusion/cached", tags=["fusion"], summary="Fusion キャッシュ付き実行 — Sprint 67")
def fusion_cached_run(req: _CachedFusionReq):
    """
    同一 question+system の 2 回目以降はキャッシュから即返す。
    candidates 指定があるときはキャッシュを使わず毎回生成する（異なる設定のため）。
    """
    if req.candidates:
        # カスタム候補が指定された場合はキャッシュ対象外
        specs = [
            _CandidateSpec(
                label=c.get("label", f"candidate-{i}"),
                preferred_provider=c.get("preferred_provider"),
                temperature=float(c.get("temperature", 0.7)),
                max_tokens=int(c.get("max_tokens", 512)),
            )
            for i, c in enumerate(req.candidates)
        ]
        config = _FusionConfig(
            candidates=specs,
            judge_provider=req.judge_provider,
            caller_provider=req.caller_provider,
        )
        engine = _FusionEngine(config=config, router=_fusion_engine._router)
        result = engine.run(req.question, system=req.system)
    else:
        result = _cached_fusion_engine.run(req.question, system=req.system)

    return result.to_dict()


@router.get("/v1/fusion/cache/stats", tags=["fusion"], summary="Fusion キャッシュ統計 — Sprint 67")
def fusion_cache_stats():
    return _fusion_cache.stats()


@router.delete("/v1/fusion/cache/clear", tags=["fusion"], summary="Fusion キャッシュクリア — Sprint 67")
def fusion_cache_clear():
    n = _fusion_cache.clear()
    return {"cleared": n}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 67C — 広告運用 統合ダッシュボード API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.campaign_analytics import KpiCalculator as _KpiCalculator  # noqa: E402


@router.get("/v1/dashboard/summary", tags=["dashboard"], summary="広告運用サマリー — Sprint 67")
def dashboard_summary():
    """
    KPI・アラート・A/B テストを横断した統合サマリーを 1 エンドポイントで返す。

    レスポンス構造:
      campaigns     : キャンペーン一覧 (ID・KPI 主要指標)
      alert_summary : 異常検知サマリー (件数・重大度別)
      abtest_summary: A/B テスト一覧 (ID・状態)
    """
    # --- キャンペーン KPI ---
    calc = _KpiCalculator()
    campaign_summaries = []
    for cid, metrics in _analytics_store._metrics.items():
        kpis = calc.compute(metrics)
        campaign_summaries.append({
            "campaign_id": cid,
            "impressions": sum(p.impressions for p in metrics.points),
            "clicks":      sum(p.clicks for p in metrics.points),
            "spend":       round(sum(p.spend for p in metrics.points), 4),
            "revenue":     round(sum(p.revenue for p in metrics.points), 4),
            "ctr":         round(kpis.ctr, 4),
            "roas":        round(kpis.roas, 4),
        })

    # --- アラートサマリー ---
    alert_summary = _anomaly_report.summary_json()
    # alerts 全件リストは重いので件数のみ
    alert_summary_light = {k: v for k, v in alert_summary.items() if k != "alerts"}
    alert_summary_light["critical_ids"] = list({
        a["campaign_id"] for a in alert_summary.get("alerts", [])
        if a["severity"] == "critical"
    })

    # --- A/B テストサマリー ---
    abtest_list = []
    for tid, test in _abtest_store._tests.items():
        abtest_list.append({
            "id":     test.id,
            "name":   test.name,
            "status": test.status.value,
            "variant_count": len(test.variants),
        })

    return {
        "campaigns":      campaign_summaries,
        "alert_summary":  alert_summary_light,
        "abtest_summary": abtest_list,
    }


@router.get("/v1/dashboard/campaigns", tags=["dashboard"], summary="キャンペーン一覧 + KPI — Sprint 67")
def dashboard_campaigns():
    """全キャンペーンの KPI を一覧で返す。"""
    calc = _KpiCalculator()
    result = []
    for cid, metrics in _analytics_store._metrics.items():
        kpis = calc.compute(metrics)
        result.append({
            "campaign_id":  cid,
            "data_points":  len(metrics.points),
            "impressions":  sum(p.impressions for p in metrics.points),
            "clicks":       sum(p.clicks for p in metrics.points),
            "conversions":  sum(p.conversions for p in metrics.points),
            "spend":        round(sum(p.spend for p in metrics.points), 4),
            "revenue":      round(sum(p.revenue for p in metrics.points), 4),
            "ctr":          round(kpis.ctr, 4),
            "cvr":          round(kpis.cvr, 4),
            "cpc":          round(kpis.cpc, 4),
            "roas":         round(kpis.roas, 4),
            "roi":          round(kpis.roi, 4),
        })
    return {"campaigns": result, "total": len(result)}


@router.get("/v1/dashboard/alerts/critical", tags=["dashboard"], summary="Critical アラート一覧 — Sprint 67")
def dashboard_critical_alerts():
    """Critical 深刻度のアラートのみを返す。"""
    from open_mythos.skills.anomaly_detector import AlertSeverity as _AS
    critical = _alert_store.list_by_severity(_AS.CRITICAL)
    return {
        "critical_count": len(critical),
        "alerts": [a.to_dict() for a in critical],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 69 — 時系列予測 (TimesFM + マルチモデル)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.time_series import (  # noqa: E402
    TimesFMForecasterFactory as _TSFactory,
    CampaignForecaster       as _CampaignForecaster,
    ForecastStore            as _ForecastStore,
    ForecastReportEngine     as _ForecastReportEngine,
)

# デフォルトは LinearTrend (起動コスト 0)。
# TimesFM を使いたい場合は /v1/forecast/load でモデルをロードする。
_ts_forecaster       = _TSFactory.rule_based()
_ts_campaign_fc      = _CampaignForecaster(_ts_forecaster, _analytics_store)
_forecast_store      = _ForecastStore()
_forecast_report     = _ForecastReportEngine(_forecast_store)


class _ForecastReq(BaseModel):
    metric:  str   = "clicks"
    horizon: int   = 7
    model:   str   = "linear_trend"   # "linear_trend" | "timesfm" | "mock"


class _BatchForecastReq(BaseModel):
    campaign_ids: list
    metric:  str = "clicks"
    horizon: int = 7
    model:   str = "linear_trend"


def _get_forecaster(model_name: str):
    """モデル名からフォーキャスターを返す"""
    if model_name == "timesfm":
        return _TSFactory.from_pretrained()
    elif model_name == "mock":
        return _TSFactory.from_mock()
    else:
        return _TSFactory.rule_based()


# NOTE: 固定パス (/batch, /models) を先に定義し、/{campaign_id} に捕捉されないようにする

@router.get(
    "/v1/forecast/models",
    tags=["forecast"],
    summary="利用可能モデル一覧 — Sprint 69",
)
def forecast_models():
    """利用可能な予測モデルの一覧を返す。"""
    return {"models": _TSFactory.available_models()}


@router.post(
    "/v1/forecast/batch",
    tags=["forecast"],
    summary="バッチ予測 — Sprint 69",
)
def forecast_batch(req: _BatchForecastReq):
    """複数キャンペーンを一括予測する。"""
    fc = _CampaignForecaster(_get_forecaster(req.model), _analytics_store)
    results = fc.forecast_batch(req.campaign_ids, metric=req.metric, horizon=req.horizon)
    for r in results.values():
        _forecast_store.save(r)
    return {
        "metric":    req.metric,
        "horizon":   req.horizon,
        "forecasts": {cid: r.to_dict() for cid, r in results.items()},
    }


@router.post(
    "/v1/forecast/{campaign_id}",
    tags=["forecast"],
    summary="キャンペーン KPI 予測 — Sprint 69",
)
def forecast_campaign(campaign_id: str, req: _ForecastReq):
    """
    指定キャンペーンの KPI 指標を予測する。

    model:
      - linear_trend : 線形トレンド外挿 (高速・外部依存なし)
      - timesfm      : Google TimesFM 2.5 (初回はモデルロードが発生)
      - mock         : テスト用モック
    """
    fc = _CampaignForecaster(_get_forecaster(req.model), _analytics_store)
    try:
        result = fc.forecast_metric(campaign_id, metric=req.metric, horizon=req.horizon)
    except ValueError as e:
        raise HTTPException(404, str(e))
    _forecast_store.save(result)
    return result.to_dict()


@router.post(
    "/v1/forecast/{campaign_id}/all",
    tags=["forecast"],
    summary="全指標一括予測 — Sprint 69",
)
def forecast_campaign_all(campaign_id: str, req: _ForecastReq):
    """指定キャンペーンの全 KPI 指標を一括予測する。"""
    fc = _CampaignForecaster(_get_forecaster(req.model), _analytics_store)
    try:
        results = fc.forecast_all_metrics(campaign_id, horizon=req.horizon)
    except ValueError as e:
        raise HTTPException(404, str(e))
    for r in results.values():
        _forecast_store.save(r)
    return {
        "campaign_id": campaign_id,
        "forecasts": {m: r.to_dict() for m, r in results.items()},
    }


@router.get(
    "/v1/forecast/{campaign_id}/history",
    tags=["forecast"],
    summary="予測履歴 — Sprint 69",
)
def forecast_history(campaign_id: str, metric: Optional[str] = None):
    """キャンペーンの予測履歴を返す。"""
    if metric:
        latest = _forecast_store.latest(campaign_id, metric)
        if latest is None:
            raise HTTPException(404, f"No forecast found for {campaign_id}/{metric}")
        return latest.to_dict()
    results = _forecast_store.list_by_campaign(campaign_id)
    return {"campaign_id": campaign_id, "forecasts": [r.to_dict() for r in results]}


@router.get(
    "/v1/forecast/report/md/{forecast_id}",
    tags=["forecast"],
    summary="予測レポート Markdown — Sprint 69",
)
def forecast_report_md(forecast_id: str):
    result = _forecast_store.get(forecast_id)
    if result is None:
        raise HTTPException(404, f"Forecast not found: {forecast_id}")
    return Response(content=_forecast_report.markdown(result), media_type="text/markdown")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 70A — 予測アラート (ForecastAlert)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.forecast_alert import (  # noqa: E402
    AlertThreshold     as _AlertThreshold70,
    ForecastAlertRule  as _ForecastAlertRule,
    ForecastAlertRuleStore as _ForecastAlertRuleStore,
    ForecastAlertEngine    as _ForecastAlertEngine,
)

_fa_rule_store  = _ForecastAlertRuleStore()
_fa_engine      = _ForecastAlertEngine(forecast_store=_forecast_store, rule_store=_fa_rule_store)


class _ForecastAlertRuleReq(BaseModel):
    campaign_id: str
    metric:      str   = "clicks"
    upper_limit: Optional[float] = None
    lower_limit: Optional[float] = None
    severity:    str   = "warning"


@router.get(
    "/v1/forecast/alert/rules",
    tags=["forecast-alert"],
    summary="予測アラートルール一覧 — Sprint 70A",
)
def forecast_alert_rules_list():
    return {"rules": [r.to_dict() for r in _fa_rule_store.list()]}


@router.post(
    "/v1/forecast/alert/rules",
    tags=["forecast-alert"],
    summary="予測アラートルール追加 — Sprint 70A",
)
def forecast_alert_rules_add(req: _ForecastAlertRuleReq):
    rule_id = str(uuid.uuid4())[:8]
    threshold = _AlertThreshold70(
        metric=req.metric,
        upper_limit=req.upper_limit,
        lower_limit=req.lower_limit,
    )
    rule = _ForecastAlertRule(
        id=rule_id, campaign_id=req.campaign_id,
        threshold=threshold, severity=req.severity,
    )
    _fa_rule_store.add(rule)
    return {"rule_id": rule_id, "rule": rule.to_dict()}


@router.delete(
    "/v1/forecast/alert/rules/{rule_id}",
    tags=["forecast-alert"],
    summary="予測アラートルール削除 — Sprint 70A",
)
def forecast_alert_rules_delete(rule_id: str):
    _fa_rule_store.delete(rule_id)
    return {"deleted": rule_id}


@router.get(
    "/v1/forecast/alert/check/{campaign_id}",
    tags=["forecast-alert"],
    summary="予測アラートチェック — Sprint 70A",
)
def forecast_alert_check(campaign_id: str):
    checks = _fa_engine.check_all(campaign_id)
    return {
        "campaign_id": campaign_id,
        "checks": [c.to_dict() for c in checks],
        "triggered_count": sum(1 for c in checks if c.triggered),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 70B — レポート配信 Webhook (ReportDispatcher)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.report_dispatcher import (  # noqa: E402
    WebhookTarget   as _WebhookTarget,
    WebhookStore    as _WebhookStore70,
    ReportDispatcher as _ReportDispatcher,
)

_webhook_store   = _WebhookStore70()
_report_dispatch = _ReportDispatcher(
    webhook_store=_webhook_store,
    analytics_store=_analytics_store,
)


class _WebhookAddReq(BaseModel):
    name:    str
    url:     str
    type:    str  = "generic"
    enabled: bool = True


class _DispatchReq(BaseModel):
    webhook_id:  str
    report_type: str = "generic"
    campaign_id: Optional[str] = None


class _DispatchAllReq(BaseModel):
    report_type: str = "generic"
    campaign_id: Optional[str] = None


@router.get(
    "/v1/report/webhooks",
    tags=["report-dispatch"],
    summary="Webhook 一覧 — Sprint 70B",
)
def webhooks_list():
    return {"webhooks": [wh.to_dict() for wh in _webhook_store.list()]}


@router.post(
    "/v1/report/webhooks",
    tags=["report-dispatch"],
    summary="Webhook 追加 — Sprint 70B",
)
def webhooks_add(req: _WebhookAddReq):
    webhook_id = str(uuid.uuid4())[:8]
    wh = _WebhookTarget(
        id=webhook_id, name=req.name, url=req.url,
        type=req.type, enabled=req.enabled,
    )
    _webhook_store.add(wh)
    return {"webhook_id": webhook_id, "webhook": wh.to_dict()}


@router.delete(
    "/v1/report/webhooks/{webhook_id}",
    tags=["report-dispatch"],
    summary="Webhook 削除 — Sprint 70B",
)
def webhooks_delete(webhook_id: str):
    _webhook_store.delete(webhook_id)
    return {"deleted": webhook_id}


@router.post(
    "/v1/report/dispatch",
    tags=["report-dispatch"],
    summary="レポート配信 (mock) — Sprint 70B",
)
def report_dispatch(req: _DispatchReq):
    result = _report_dispatch.dispatch_mock(req.webhook_id, req.report_type, req.campaign_id)
    return {"result": result.to_dict()}


@router.post(
    "/v1/report/dispatch/all",
    tags=["report-dispatch"],
    summary="全 Webhook にレポート配信 (mock) — Sprint 70B",
)
def report_dispatch_all(req: _DispatchAllReq):
    results = _report_dispatch.dispatch_all_mock(req.report_type, req.campaign_id)
    return {"results": [r.to_dict() for r in results], "count": len(results)}


@router.get(
    "/v1/report/dispatch/history",
    tags=["report-dispatch"],
    summary="配信履歴 — Sprint 70B",
)
def report_dispatch_history():
    return {"history": [r.to_dict() for r in _report_dispatch.history()]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 70C — 自然言語クエリ (NLQ)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.nlq_agent import (  # noqa: E402
    NLQParser   as _NLQParser,
    NLQExecutor as _NLQExecutor,
)

_nlq_parser   = _NLQParser()
_nlq_executor = _NLQExecutor(
    analytics_store=_analytics_store,
    forecast_store=_forecast_store,
    alert_store=_alert_store,
)


class _NLQReq(BaseModel):
    text: str


@router.post(
    "/v1/nlq/query",
    tags=["nlq"],
    summary="自然言語クエリ実行 — Sprint 70C",
)
def nlq_query(req: _NLQReq):
    if not req.text:
        return {
            "intent": "unknown",
            "query": {"raw": "", "intent": "unknown", "campaign_id": None, "metric": None, "params": {}},
            "result": {"query": {}, "intent": "unknown", "data": None,
                       "message": "空のクエリです。", "success": False},
        }
    query = _nlq_parser.parse(req.text)
    result = _nlq_executor.execute(query)
    return {
        "intent": query.intent.value,
        "query":  query.to_dict(),
        "result": result.to_dict(),
    }



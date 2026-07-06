"""
serve/routers/growing_ai.py — 育つAI・マーケ分析ドメイン API (Sprint 18/35/20〜30/43)

A/B テスト (18) / ROAS Monte Carlo (35) / Debate・KPI・Profiler・ExternalSignal・
ErrorMemory・SelfDistill (P1〜P6) / LongTermMemory・Ensemble・PromptEvolution・
TaskPlanner (P7〜P10) / GrowingAIOrchestrator (30) / HermesOrchestrator (43)。
serve/api.py のモノリスから分割 (認証は app 全体の verify_api_key に委譲)。
"""

from __future__ import annotations

import os
import time

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from serve.auth import verify_api_key
import torch

from serve.state import (
    DEFAULT_LOOPS,
    MAX_LOOPS,
    TASK_LOOPS,
    TaskType,
    state,
    get_mistake_store as _get_mistake_store,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Sprint 18 — A/B テストエンドポイント (18.4)
# ---------------------------------------------------------------------------
#
# 設計:
#   - hash(user_id) % 100 < AB_OPENMYTHOS_PCT → openmythos グループ (直接モデル推論)
#   - それ以外 → existing_ml グループ (スタブ: 決定論的スコア返却)
#   - /v1/ab/stats で集計 + Welch t 検定を返す
#   - 既存 serve/ab_router.py (スタンドアロン A/B サーバ) とは独立

import hashlib  # noqa: E402
import math as _math  # noqa: E402
from collections import defaultdict  # noqa: E402

AB_OPENMYTHOS_PCT: int = int(os.getenv("AB_OPENMYTHOS_PCT", "20"))


class _ABStats:
    def __init__(self):
        self.counts: Dict[str, int] = defaultdict(int)
        self.latencies: Dict[str, List[float]] = defaultdict(list)
        self.scores: Dict[str, List[float]] = defaultdict(list)
        self.correct: Dict[str, int] = defaultdict(int)


_ab_stats = _ABStats()


def _ab_route(user_id: str) -> str:
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    return "openmythos" if h < AB_OPENMYTHOS_PCT else "existing_ml"


def _ab_significance(a: List[float], b: List[float], alpha: float = 0.05) -> dict:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {
            "p_value": 1.0,
            "significant": False,
            "mean_a": sum(a) / na if na else float("nan"),
            "mean_b": sum(b) / nb if nb else float("nan"),
            "n_a": na,
            "n_b": nb,
        }
    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)
    se = _math.sqrt(var_a / na + var_b / nb)
    if se == 0:
        p_value = 0.0 if mean_a != mean_b else 1.0
    else:
        t_stat = (mean_a - mean_b) / se
        num = (var_a / na + var_b / nb) ** 2
        den = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
        df = num / den if den > 0 else 1.0
        z = abs(t_stat) * _math.sqrt(1 + df / (df + t_stat**2 + 1e-9))
        p_one = 0.5 * _math.erfc(z / _math.sqrt(2))
        p_value = min(2 * p_one, 1.0)
    return {
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
        "mean_a": round(mean_a, 6),
        "mean_b": round(mean_b, 6),
        "n_a": na,
        "n_b": nb,
    }


class ABInferRequest(BaseModel):
    user_id: str = Field(..., description="ルーティングハッシュに使用するユーザーID")
    text: str = Field(..., description="推論対象テキスト")
    task: TaskType = Field("general", description="タスク種別")
    loops: int = Field(DEFAULT_LOOPS, ge=1, le=16)
    ground_truth: Optional[int] = Field(None, description="正解ラベル (評価用、省略可)")


class ABInferResponse(BaseModel):
    model_id: str
    ab_group: str
    label: int
    score: float
    latency_ms: float
    traffic_pct: int


@router.post(
    "/v1/ab/infer",
    response_model=ABInferResponse,
    tags=["infer"],
    summary="A/B テスト推論",
    description=(
        "user_id のハッシュで OpenMythos (20%) または既存 ML スタブ (80%) に振り分ける。"
        "`AB_OPENMYTHOS_PCT` 環境変数でトラフィック比率を変更可能。"
    ),
)
def ab_infer(req: ABInferRequest):
    """A/B テスト推論エンドポイント。

    hash(user_id) % 100 < AB_OPENMYTHOS_PCT ならば OpenMythos モデルで推論、
    それ以外は決定論的スタブ (既存MLモデル代替) を返す。
    """
    group = _ab_route(req.user_id)
    loops = min(req.loops, MAX_LOOPS)
    if req.task != "general" and req.loops == DEFAULT_LOOPS:
        loops = TASK_LOOPS.get(req.task, DEFAULT_LOOPS)

    enc = state.tokenizer(
        req.text, return_tensors="pt", truncation=True, max_length=512
    )
    input_ids = enc["input_ids"].to(state.device)

    t0 = time.perf_counter()

    if group == "openmythos":
        with torch.no_grad():
            logits = state.model(input_ids, n_loops=loops)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        score = float(probs.max())
        label = 1 if score >= 0.5 else 0
        model_id = "openmythos-rdt"
    else:
        # Claude API (Opus 4.8) — CLAUDE_API_KEY があれば実 API、なければスタブ
        claude_api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if claude_api_key:
            try:
                import anthropic as _anthropic
                _client = _anthropic.Anthropic(api_key=claude_api_key)
                _msg = _client.messages.create(
                    model="claude-opus-4-8",
                    max_tokens=64,
                    messages=[{"role": "user", "content": req.text[:500]}],
                )
                _response_text = _msg.content[0].text if _msg.content else ""
                # レスポンス長をスコアの代理指標として使用
                score = min(len(_response_text) / 200.0, 1.0)
                label = 1 if score >= 0.5 else 0
                model_id = "claude-opus-4-8"
            except Exception:
                # API エラー時はスタブにフォールバック
                h = int(hashlib.md5(req.user_id.encode()).hexdigest(), 16)
                score = 0.5 + (h % 500) / 1000.0
                label = 1 if score >= 0.5 else 0
                model_id = "claude-api-error-stub"
        else:
            # API キー未設定: 決定論的スタブ
            h = int(hashlib.md5(req.user_id.encode()).hexdigest(), 16)
            score = 0.5 + (h % 500) / 1000.0
            label = 1 if score >= 0.5 else 0
            model_id = "claude-stub-no-key"

    latency_ms = (time.perf_counter() - t0) * 1000

    # 集計
    _ab_stats.counts[group] += 1
    _ab_stats.latencies[group].append(latency_ms)
    _ab_stats.scores[group].append(score)
    if req.ground_truth is not None and label == req.ground_truth:
        _ab_stats.correct[group] += 1

    return ABInferResponse(
        model_id=model_id,
        ab_group=group,
        label=label,
        score=round(score, 4),
        latency_ms=round(latency_ms, 2),
        traffic_pct=(
            AB_OPENMYTHOS_PCT if group == "openmythos" else 100 - AB_OPENMYTHOS_PCT
        ),
    )


# ---------------------------------------------------------------------------
# Sprint 35: ROAS Monte Carlo シミュレーター API
# ---------------------------------------------------------------------------


class RoasSimulateRequest(BaseModel):
    ad_spend: float = Field(..., gt=0, description="広告費 (USD)")
    ctr: float = Field(..., gt=0, description="期待クリック率 (clicks per USD)")
    cvr: float = Field(..., gt=0, le=1.0, description="期待成約率 (0〜1)")
    aov: float = Field(..., gt=0, description="平均注文金額 (USD)")
    n: int = Field(1000, ge=100, le=100000, description="シミュレーション回数")
    noise: float = Field(0.20, ge=0.0, le=1.0, description="ノイズ幅 (デフォルト: ±20%)")
    noise_dist: str = Field("uniform", description="ノイズ分布: 'uniform' or 'normal'")
    seed: Optional[int] = Field(None, description="乱数シード (再現性用)")


@router.post(
    "/v1/roas/simulate",
    tags=["marketing"],
    summary="ROAS モンテカルロシミュレーション",
    description=(
        "広告費・CTR・CVR・AOV に対してモンテカルロ法で ROAS 分布を推定し、"
        "90%/50% 信頼区間・収益確率・期待収益を返す。"
    ),
)
def roas_simulate_endpoint(req: RoasSimulateRequest):
    from open_mythos.tools_marketing import roas_simulate
    return roas_simulate(
        ad_spend=req.ad_spend,
        ctr=req.ctr,
        cvr=req.cvr,
        aov=req.aov,
        n=req.n,
        noise=req.noise,
        seed=req.seed,
        noise_dist=req.noise_dist,
    )


# ---------------------------------------------------------------------------
# Sprint 25: Self Distill Loop endpoints
# ---------------------------------------------------------------------------


class DistillRunRequest(BaseModel):
    prompts: list = Field(..., description="蒸留に使うプロンプトリスト")
    n_rounds: int = Field(3, ge=1, le=10, description="蒸留ラウンド数")
    score_threshold: float = Field(0.6, ge=0.0, le=1.0, description="フィルタスコア閾値")
    early_stop_score: float = Field(0.85, ge=0.0, le=1.0, description="早期終了スコア閾値")


@router.post(
    "/v1/distill/run",
    tags=["distill"],
    summary="自己蒸留ループ実行",
    description="Collect→Filter→SFT→Eval サイクルを n_rounds 実行し蒸留結果を返す。",
)
def distill_run(req: DistillRunRequest, _: str = Depends(verify_api_key)):
    from open_mythos.self_distill import SelfDistillConfig, SelfDistillLoop

    cfg = SelfDistillConfig(
        n_rounds=req.n_rounds,
        score_threshold=req.score_threshold,
        early_stop_score=req.early_stop_score,
    )
    loop = SelfDistillLoop(cfg)
    result = loop.run(prompts=[str(p) for p in req.prompts])

    return {
        "rounds_completed": result.rounds_completed,
        "total_samples": result.total_samples,
        "initial_mean_score": result.initial_mean_score,
        "final_mean_score": result.final_mean_score,
        "mean_score_improvement": result.mean_score_improvement,
        "early_stopped": result.early_stopped,
        "total_latency_ms": result.total_latency_ms,
        "rounds": [
            {
                "round": r.round_num,
                "collected": r.collected,
                "filtered": r.filtered,
                "mean_score": r.mean_score,
            }
            for r in result.round_results
        ],
    }


@router.get(
    "/v1/distill/status",
    tags=["distill"],
    summary="蒸留ステータス",
    description="蒸留ループのステータスを返す (スタブ)。",
)
def distill_status(_: str = Depends(verify_api_key)):
    return {"status": "idle", "message": "蒸留ループは現在待機中です。"}


# ---------------------------------------------------------------------------
# Sprint 24: Error Memory / Mistake Guard endpoints
# ---------------------------------------------------------------------------


class MistakeRecordRequest(BaseModel):
    text: str = Field(..., description="ミステキスト")
    category: Optional[str] = Field(None, description="カテゴリ (省略時は自動分類)")
    severity: str = Field("medium", description="重要度 high/medium/low")
    context: str = Field("", description="発生コンテキスト")


class MistakeCheckRequest(BaseModel):
    text: str = Field(..., description="チェック対象テキスト")




@router.post(
    "/v1/mistakes/record",
    tags=["mistakes"],
    summary="ミス記録",
    description="ミスをストアに記録する。category 省略時は自動分類。",
)
def mistakes_record(req: MistakeRecordRequest, _: str = Depends(verify_api_key)):
    from open_mythos.error_memory import MistakeClassifier

    store = _get_mistake_store()
    category = req.category
    if not category:
        category = MistakeClassifier().classify(req.text)
    record = store.append(req.text, category=category, severity=req.severity, context=req.context)
    return {
        "record_id": record.record_id,
        "category": record.category,
        "severity": record.severity,
        "total_records": store.total,
    }


@router.get(
    "/v1/mistakes/rules",
    tags=["mistakes"],
    summary="防止ルール取得",
    description="蓄積ミスから自動生成した防止ルール一覧を返す。",
)
def mistakes_rules(_: str = Depends(verify_api_key)):
    from open_mythos.error_memory import RuleExtractor

    store = _get_mistake_store()
    rules = RuleExtractor(store).extract()
    return {
        "n_rules": len(rules),
        "rules": [
            {
                "rule_id": r.rule_id,
                "category": r.category,
                "pattern": r.pattern,
                "description": r.description,
                "severity": r.severity,
                "source_count": r.source_count,
            }
            for r in rules
        ],
    }


@router.post(
    "/v1/mistakes/check",
    tags=["mistakes"],
    summary="ミスガード チェック",
    description="テキストをルールDB照合し、ブロック判定を返す。",
)
def mistakes_check(req: MistakeCheckRequest, _: str = Depends(verify_api_key)):
    from open_mythos.error_memory import RuleExtractor, MistakeGuard

    store = _get_mistake_store()
    rules = RuleExtractor(store).extract()
    guard = MistakeGuard(rules=rules, store=store)
    result = guard.check(req.text)

    return {
        "text": req.text,
        "blocked": result.blocked,
        "block_reason": result.block_reason,
        "matched_rule": (
            {
                "rule_id": result.matched_rule.rule_id,
                "category": result.matched_rule.category,
                "pattern": result.matched_rule.pattern,
            }
            if result.matched_rule
            else None
        ),
        "n_similar_records": len(result.similar_records),
        "check_latency_ms": result.check_latency_ms,
    }


@router.get(
    "/v1/mistakes/export",
    tags=["mistakes"],
    summary="ミス記録エクスポート (Sprint 32)",
    description="蓄積したミス記録を JSONL または JSON 形式でエクスポートする。"
                " category で絞り込み可。",
    dependencies=[Depends(verify_api_key)],
)
def mistakes_export(
    format:   str            = "jsonl",
    category: Optional[str] = None,
):
    from fastapi.responses import Response as _Resp
    store   = _get_mistake_store()
    records = store.export_records()
    if category:
        records = [r for r in records if r["category"] == category]

    if format == "json":
        return {"records": records, "total": len(records)}

    # JSONL (default)
    import json as _json
    lines   = [_json.dumps(r, ensure_ascii=False) for r in records]
    content = "\n".join(lines)
    return _Resp(content=content, media_type="text/plain; charset=utf-8")


@router.delete(
    "/v1/mistakes/clear",
    tags=["mistakes"],
    summary="ミス記録全削除 (Sprint 32)",
    description="蓄積した全ミス記録を削除する。",
    dependencies=[Depends(verify_api_key)],
)
def mistakes_clear():
    store = _get_mistake_store()
    store.clear()
    return {"cleared": True, "total": store.total}


# ---------------------------------------------------------------------------
# Sprint 23: External Signal Agent endpoints
# ---------------------------------------------------------------------------


class SignalDetectRequest(BaseModel):
    context: str = Field("", description="分析対象コンテキスト")
    keyword: str = Field("", description="対象キーワード")
    month: Optional[int] = Field(None, ge=1, le=12, description="現在月 (1〜12)")
    kpi_name: str = Field("llmo_score", description="影響推定するKPI名")


class SignalCounterRequest(BaseModel):
    context: str = Field(..., description="最適化対象コンテキスト")
    keyword: str = Field("", description="対象キーワード")
    month: Optional[int] = Field(None, ge=1, le=12, description="現在月")
    kpi_name: str = Field("llmo_score")


@router.post(
    "/v1/signal/detect",
    tags=["signal"],
    summary="外部シグナル検出",
    description="季節・トレンド・競合・市場シグナルを検出し KPI への推定影響を返す。",
)
def signal_detect(req: SignalDetectRequest, _: str = Depends(verify_api_key)):
    from open_mythos.external_signal import SignalDetector, ImpactEstimator

    detector = SignalDetector()
    estimator = ImpactEstimator()
    signals = detector.detect(req.context, keyword=req.keyword, month=req.month)
    impacts = [estimator.estimate(s, req.kpi_name) for s in signals]
    net = sum(i.impact_delta for i in impacts)

    return {
        "keyword": req.keyword,
        "signals": [
            {
                "type": s.signal_type,
                "name": s.name,
                "strength": round(s.strength, 4),
                "direction": s.direction,
                "is_threat": s.is_threat,
            }
            for s in signals
        ],
        "impacts": [
            {
                "kpi_name": i.kpi_name,
                "impact_delta": i.impact_delta,
                "severity": i.severity,
                "confidence": i.confidence,
                "explanation": i.explanation,
            }
            for i in impacts
        ],
        "net_kpi_impact": round(net, 4),
        "n_threats": sum(1 for s in signals if s.is_threat),
        "n_opportunities": sum(1 for s in signals if s.is_opportunity),
    }


@router.post(
    "/v1/signal/counter",
    tags=["signal"],
    summary="外部シグナル対抗アクション",
    description="シグナルを検出し、対応するカウンターアクションを適用した最適化コンテキストを返す。",
)
def signal_counter(req: SignalCounterRequest, _: str = Depends(verify_api_key)):
    from open_mythos.external_signal import ExternalSignalAgent

    agent = ExternalSignalAgent()
    result = agent.run(
        context=req.context,
        keyword=req.keyword,
        month=req.month,
        kpi_name=req.kpi_name,
    )

    return {
        "keyword": result.keyword,
        "n_signals": len(result.signals),
        "threat_count": result.threat_count,
        "opportunity_count": result.opportunity_count,
        "net_kpi_impact": result.net_kpi_impact,
        "counter_actions": [
            {
                "action_id": a.action_id,
                "description": a.description,
                "estimated_kpi_recovery": a.estimated_kpi_recovery,
            }
            for a in result.counter_actions
        ],
        "optimized_context": result.optimized_context,
        "total_latency_ms": result.total_latency_ms,
    }


# ---------------------------------------------------------------------------
# Sprint 22: Profiler Agent endpoints
# ---------------------------------------------------------------------------


class ProfileRunRequest(BaseModel):
    input_text: str = Field(..., description="パイプラインへの入力テキスト")
    stages: Optional[list] = Field(None, description="使用するステージ名リスト (省略時はデフォルト3ステージ)")


class ProfileFixRequest(BaseModel):
    input_text: str = Field(..., description="パイプラインへの入力テキスト")


def _default_stages():
    """デモ用デフォルトステージ (スコア付き)。"""
    from open_mythos.llmo import LLMOScorer
    scorer = LLMOScorer()

    def fetch(text: str):
        return text + " [fetched]", scorer.score(text).llmo_total

    def rank(text: str):
        ranked = text + " [ranked]"
        return ranked, scorer.score(ranked).llmo_total

    def fmt(text: str):
        formatted = f"## 結果\n{text}\n[formatted]"
        return formatted, scorer.score(formatted).llmo_total

    return {"fetch": fetch, "rank": rank, "format": fmt}


@router.post(
    "/v1/profile/run",
    tags=["profiler"],
    summary="パイプラインプロファイル",
    description="各ステージの実行時間・スコアを計測し、ボトルネック候補を返す。",
)
def profile_run(req: ProfileRunRequest, _: str = Depends(verify_api_key)):
    from open_mythos.profiler import PipelineProfiler, BottleneckDetector

    stages = _default_stages()
    profiler = PipelineProfiler(stages)
    result = profiler.run(req.input_text)
    report = BottleneckDetector().detect(result)

    return {
        "total_latency_ms": result.total_latency_ms,
        "stages": {
            name: {
                "latency_ms": m.latency_ms,
                "score": round(m.score, 4) if m.score >= 0 else None,
                "ok": m.ok,
            }
            for name, m in result.stages.items()
        },
        "bottleneck_stage": report.bottleneck_stage,
        "bottleneck_type": report.bottleneck_type,
        "severity": report.severity,
        "diagnosis": report.diagnosis,
        "suggested_fix": report.suggested_fix,
    }


@router.post(
    "/v1/profile/fix",
    tags=["profiler"],
    summary="ボトルネック自動修正",
    description="profile → detect → auto_fix を一括実行し、修正前後のレイテンシ改善率を返す。",
)
def profile_fix(req: ProfileFixRequest, _: str = Depends(verify_api_key)):
    from open_mythos.profiler import ProfilerAgent

    agent = ProfilerAgent(_default_stages())
    fix_result = agent.profile_and_fix(req.input_text)

    return {
        "bottleneck_stage": fix_result.bottleneck_report.bottleneck_stage,
        "bottleneck_type": fix_result.bottleneck_report.bottleneck_type,
        "before_latency_ms": fix_result.before_profile.total_latency_ms,
        "after_latency_ms": fix_result.after_profile.total_latency_ms,
        "latency_improvement_pct": fix_result.latency_improvement_pct,
        "score_improvement": fix_result.score_improvement,
        "fixed": fix_result.fixed,
        "fix_description": fix_result.fix_description,
    }


@router.get(
    "/v1/profile/report",
    tags=["profiler"],
    summary="プロファイル履歴",
    description="直近のプロファイル実行結果サマリーを返す (スタブ)。",
)
def profile_report(_: str = Depends(verify_api_key)):
    return {"message": "プロファイル履歴機能は今後のバージョンで実装予定です。"}


# ---------------------------------------------------------------------------
# Sprint 21: KPI Agent endpoints
# ---------------------------------------------------------------------------


class KPIDefineRequest(BaseModel):
    name: str = Field(..., description="KPI識別名 (例: llmo_score, roas)")
    target: float = Field(..., description="達成目標値")
    context: str = Field("", description="計測対象コンテキスト文字列")
    higher_is_better: bool = Field(True, description="大きいほど良いKPIか")
    unit: str = Field("", description="単位ラベル")
    action_budget: int = Field(3, ge=1, le=6, description="1サイクルのアクション上限")


class KPIMeasureRequest(BaseModel):
    name: str = Field(..., description="KPI識別名")
    target: float = Field(..., description="目標値")
    context: str = Field("", description="計測対象コンテキスト")
    higher_is_better: bool = Field(True)


class KPIImproveRequest(BaseModel):
    name: str = Field(..., description="KPI識別名")
    target: float = Field(..., description="目標値")
    context: str = Field("", description="改善対象コンテキスト")
    n_cycles: int = Field(3, ge=1, le=10, description="改善サイクル数")
    higher_is_better: bool = Field(True)
    action_budget: int = Field(3, ge=1, le=6)
    early_stop: bool = Field(True, description="目標達成時に早期終了するか")


def _llmo_measure_fn(text: str) -> float:
    """LLMO スコアを KPI 計測関数として使用。"""
    from open_mythos.llmo import LLMOScorer
    return LLMOScorer().score(text).llmo_total


@router.post(
    "/v1/kpi/measure",
    tags=["kpi"],
    summary="KPI 計測",
    description="コンテキストに対して KPI 値を計測し KPISnapshot を返す。",
)
def kpi_measure(req: KPIMeasureRequest, _: str = Depends(verify_api_key)):
    from open_mythos.kpi_agent import KPIDefinition, KPIAgent

    kpi = KPIDefinition(
        name=req.name,
        target=req.target,
        measure_fn=_llmo_measure_fn,
        context=req.context,
        higher_is_better=req.higher_is_better,
    )
    agent = KPIAgent(kpi)
    snapshot = agent.measure(req.context, cycle=0)
    gap_report = agent.analyze(snapshot)
    return {
        "kpi_name": snapshot.kpi_name,
        "value": round(snapshot.value, 4),
        "target": req.target,
        "gap": round(gap_report.gap, 4),
        "gap_pct": gap_report.gap_pct,
        "priority": gap_report.priority,
        "diagnosis": gap_report.diagnosis,
        "achieved": gap_report.achieved,
    }


@router.post(
    "/v1/kpi/improve",
    tags=["kpi"],
    summary="KPI 自律改善",
    description=(
        "measure → analyze → plan → execute サイクルを n_cycles 回自律実行し、"
        "KPI を目標値に近づける。"
    ),
)
def kpi_improve(req: KPIImproveRequest, _: str = Depends(verify_api_key)):
    from open_mythos.kpi_agent import KPIDefinition, KPIAgent

    kpi = KPIDefinition(
        name=req.name,
        target=req.target,
        measure_fn=_llmo_measure_fn,
        context=req.context,
        higher_is_better=req.higher_is_better,
        action_budget=req.action_budget,
    )
    agent = KPIAgent(kpi)
    result = agent.improve_loop(n_cycles=req.n_cycles, early_stop=req.early_stop)

    return {
        "kpi_name": result.kpi_name,
        "initial_value": round(result.initial_snapshot.value, 4),
        "final_value": round(result.final_snapshot.value, 4),
        "target": req.target,
        "achieved_target": result.achieved_target,
        "improvement": round(result.improvement, 4),
        "improvement_pct": round(result.improvement_pct, 2),
        "n_cycles_used": result.n_cycles_used,
        "total_latency_ms": result.total_latency_ms,
        "snapshots": [
            {"cycle": s.cycle, "value": round(s.value, 4)}
            for s in result.snapshots
        ],
    }


# ---------------------------------------------------------------------------
# Sprint 20: Debate Orchestrator endpoints
# ---------------------------------------------------------------------------


class DebateRunRequest(BaseModel):
    topic: str = Field(..., description="討議トピック / 質問")
    n_agents: int = Field(3, ge=2, le=8, description="討議エージェント数")
    n_rounds: int = Field(2, ge=1, le=5, description="討議ラウンド数")
    consensus_threshold: float = Field(0.75, ge=0.0, le=1.0, description="早期終了する合意スコア閾値")
    max_new_tokens: int = Field(64, ge=1, le=256, description="1生成あたりの最大トークン数")


@router.post(
    "/v1/debate/run",
    tags=["debate"],
    summary="討議型集合知",
    description=(
        "複数エージェントが Propose → Critique → Refine → Consensus の4フェーズで討議し、"
        "合意テキストと agreement_score を返す。"
    ),
)
def debate_run(req: DebateRunRequest, _: str = Depends(verify_api_key)):
    from open_mythos.debate import DebateConfig, DebateOrchestrator

    cfg = DebateConfig(
        n_agents=req.n_agents,
        n_rounds=req.n_rounds,
        consensus_threshold=req.consensus_threshold,
    )
    with DebateOrchestrator(
        state.model,
        cfg,
        device=str(state.device),
        max_new_tokens=req.max_new_tokens,
    ) as debate:
        result = debate.run(req.topic)

    rounds_summary = [
        {
            "round": r.round_num,
            "agreement_score": round(r.agreement_score, 4),
            "latency_ms": r.latency_ms,
            "n_proposals": len(r.proposals),
        }
        for r in result.rounds
    ]
    return {
        "topic": result.topic,
        "consensus": result.consensus,
        "agreement_score": round(result.agreement_score, 4),
        "confidence": round(result.confidence, 4),
        "n_rounds_used": result.n_rounds_used,
        "early_stopped": result.early_stopped,
        "improved_over_solo": result.improved_over_solo,
        "total_latency_ms": result.total_latency_ms,
        "rounds": rounds_summary,
    }


@router.get(
    "/v1/ab/stats",
    tags=["infer"],
    summary="A/B テスト集計",
    description="OpenMythos / 既存 ML のリクエスト数・平均レイテンシ・平均スコア + Welch t 検定結果を返す。",
)
def ab_stats():
    """A/Bテスト集計結果をリアルタイムで返す。"""
    result: dict = {}
    for group in ["openmythos", "existing_ml"]:
        n = _ab_stats.counts[group]
        lats = _ab_stats.latencies[group]
        scrs = _ab_stats.scores[group]
        corr = _ab_stats.correct[group]
        result[group] = {
            "requests": n,
            "avg_latency_ms": round(sum(lats) / n, 2) if n else None,
            "avg_score": round(sum(scrs) / n, 4) if n else None,
            "accuracy": round(corr / n, 4) if n else None,
            "traffic_pct": (
                AB_OPENMYTHOS_PCT if group == "openmythos" else 100 - AB_OPENMYTHOS_PCT
            ),
        }
    result["significance_test"] = _ab_significance(
        _ab_stats.scores["openmythos"], _ab_stats.scores["existing_ml"]
    )
    return result


# ===========================================================================
# Sprint 26: LongTermMemoryAgent (P7) — /v1/memory/*
# ===========================================================================

from open_mythos.long_term_memory import (
    LongTermMemoryAgent,
)

_memory_agent = LongTermMemoryAgent(score_threshold=0.5, max_episodes=2000)


class MemoryStoreRequest(BaseModel):
    context: str = Field(..., description="発火元クエリ")
    text: str = Field(..., description="記憶本文 (応答またはファクト)")
    score: float = Field(0.8, ge=0.0, le=1.0, description="品質スコア")
    category: str = Field("episode", description="'episode' or 'knowledge'")
    key: Optional[str] = Field(None, description="knowledge category 時のキー")
    tags: list[str] = Field(default_factory=list, description="検索タグ")


class MemoryRetrieveRequest(BaseModel):
    query: str = Field(..., description="検索クエリ")
    top_k: int = Field(5, ge=1, le=20)
    min_relevance: float = Field(0.0, ge=0.0, le=1.0)
    include_knowledge: bool = Field(True)
    tags: Optional[list[str]] = None


@router.post(
    "/v1/memory/store",
    tags=["memory"],
    summary="長期記憶を保存 (P7)",
    description="エピソード記憶またはセマンティック知識を LongTermMemoryAgent に格納する。",
    dependencies=[Depends(verify_api_key)],
)
def memory_store(req: MemoryStoreRequest):
    entry: Optional[object]
    if req.category == "knowledge" and req.key:
        entry = _memory_agent.store_knowledge(req.key, req.text, tags=req.tags, score=req.score)
    else:
        entry = _memory_agent.store_episode(req.context, req.text, score=req.score, tags=req.tags)
    if entry is None:
        return {"stored": False, "reason": "filtered (score < threshold or duplicate)"}
    return {"stored": True, "entry_id": entry.entry_id, "category": entry.category}


@router.post(
    "/v1/memory/retrieve",
    tags=["memory"],
    summary="長期記憶を検索 (P7)",
    description="クエリに関連するエピソード + セマンティック記憶を統合検索する。",
    dependencies=[Depends(verify_api_key)],
)
def memory_retrieve(req: MemoryRetrieveRequest):
    result = _memory_agent.retrieve(
        req.query,
        top_k=req.top_k,
        min_relevance=req.min_relevance,
        include_knowledge=req.include_knowledge,
        tags=req.tags,
    )
    return {
        "query": result.query,
        "total_searched": result.total_searched,
        "entries": [
            {
                "entry_id": e.entry_id,
                "text": e.text,
                "context": e.context,
                "score": e.score,
                "category": e.category,
                "relevance": r,
            }
            for e, r in zip(result.entries, result.relevance_scores)
        ],
        "context_string": result.to_context_string(),
    }


@router.post(
    "/v1/memory/consolidate",
    tags=["memory"],
    summary="記憶を整理・重複除去 (P7)",
    dependencies=[Depends(verify_api_key)],
)
def memory_consolidate():
    result = _memory_agent.consolidate()
    stats = _memory_agent.stats()
    return {"consolidation": result, "stats": stats}


# ===========================================================================
# Sprint 27: EnsembleScorer (P8) — /v1/ensemble/*
# ===========================================================================

from open_mythos.ensemble_scorer import EnsembleScorer as _EnsembleScorer

_ensemble_scorer = _EnsembleScorer(adaptive=True)


class EnsembleScoreRequest(BaseModel):
    text: str = Field(..., description="評価対象テキスト")
    query: Optional[str] = Field(None, description="検索クエリ")
    context: Optional[str] = Field(None, description="追加コンテキスト")


class EnsembleBatchRequest(BaseModel):
    texts: list[str] = Field(..., description="評価するテキストのリスト")
    query: Optional[str] = None


class EnsembleFeedbackRequest(BaseModel):
    text: str
    human_score: float = Field(..., ge=0.0, le=1.0)


@router.post(
    "/v1/ensemble/score",
    tags=["ensemble"],
    summary="アンサンブル品質評価 (P8)",
    description="LLMO + クエリ関連度 + セキュリティ + 構造スコアを重み付きで統合評価する。",
    dependencies=[Depends(verify_api_key)],
)
def ensemble_score(req: EnsembleScoreRequest):
    result = _ensemble_scorer.score(req.text, query=req.query, context=req.context)
    return {
        "ensemble_score": result.ensemble_score,
        "high_confidence": result.high_confidence,
        "variance": result.variance,
        "breakdown": [
            {"scorer": b.scorer_name, "score": b.raw_score, "weight": b.weight,
             "contribution": b.contribution}
            for b in result.breakdown
        ],
    }


@router.post(
    "/v1/ensemble/rank",
    tags=["ensemble"],
    summary="複数テキストをアンサンブルスコアでランキング (P8)",
    dependencies=[Depends(verify_api_key)],
)
def ensemble_rank(req: EnsembleBatchRequest):
    results = _ensemble_scorer.score_batch(req.texts, query=req.query)
    return {
        "ranked": [
            {"text": r.text[:200], "ensemble_score": r.ensemble_score,
             "high_confidence": r.high_confidence}
            for r in results
        ]
    }


@router.post(
    "/v1/ensemble/feedback",
    tags=["ensemble"],
    summary="アンサンブル重みへのフィードバック (P8 adaptive)",
    dependencies=[Depends(verify_api_key)],
)
def ensemble_feedback(req: EnsembleFeedbackRequest):
    _ensemble_scorer.record_feedback(req.text, req.human_score)
    return {"recorded": True, "weights": _ensemble_scorer.weights_summary}


# ===========================================================================
# Sprint 28: PromptEvolution (P9) — /v1/evolve/*
# ===========================================================================

from open_mythos.prompt_evolution import EvolutionConfig, PromptEvolution


class PromptEvolveRequest(BaseModel):
    seed_prompt: str = Field(..., description="進化の出発点となるプロンプト")
    topic_keywords: list[str] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list)
    population_size: int = Field(6, ge=2, le=20)
    n_generations: int = Field(4, ge=1, le=20)
    mutation_rate: float = Field(0.3, ge=0.0, le=1.0)
    crossover_rate: float = Field(0.7, ge=0.0, le=1.0)
    elite_size: int = Field(2, ge=1, le=5)


@router.post(
    "/v1/evolve/run",
    tags=["evolve"],
    summary="遺伝的アルゴリズムでプロンプトを進化 (P9)",
    description="LLMO スコアをフィットネスとして N 世代プロンプトを最適化する。",
    dependencies=[Depends(verify_api_key)],
)
def evolve_run(req: PromptEvolveRequest):
    cfg = EvolutionConfig(
        population_size=req.population_size,
        n_generations=req.n_generations,
        mutation_rate=req.mutation_rate,
        crossover_rate=req.crossover_rate,
        elite_size=req.elite_size,
    )
    evo = PromptEvolution(config=cfg)
    result = evo.evolve(
        req.seed_prompt,
        topic_keywords=req.topic_keywords or None,
        templates=req.templates or None,
    )
    return {
        "best_prompt": result.best_prompt,
        "best_fitness": result.best_gene.fitness,
        "improvement": result.improvement,
        "n_generations_run": result.n_generations_run,
        "converged": result.converged,
        "fitness_history": result.fitness_history,
        "rounds": [
            {
                "generation": r.generation,
                "best_fitness": r.best_fitness,
                "mean_fitness": r.mean_fitness,
                "diversity": r.diversity,
            }
            for r in result.rounds
        ],
    }


# ===========================================================================
# Sprint 29: TaskPlanner (P10) — /v1/plan/*
# ===========================================================================

from open_mythos.task_planner import TaskPlanner as _TaskPlanner


class TaskPlanRequest(BaseModel):
    goal: str = Field(..., description="達成すべきゴール")
    context: dict = Field(default_factory=dict, description="追加コンテキスト")
    n_agents: int = Field(1, ge=1, le=8)
    kpi_target: float = Field(0.7, ge=0.0, le=1.0)
    max_parallel: int = Field(4, ge=1, le=10)


@router.post(
    "/v1/plan/decompose",
    tags=["plan"],
    summary="ゴールをサブタスクに分解 (P10)",
    description="ゴール文字列をルールベースで階層的サブタスクに分解する。",
    dependencies=[Depends(verify_api_key)],
)
def plan_decompose(req: TaskPlanRequest):
    planner = _TaskPlanner(max_parallel=req.max_parallel, kpi_target=req.kpi_target)
    plan = planner.decompose(req.goal, req.context)
    return {
        "plan_id": plan.plan_id,
        "goal": plan.goal,
        "total_tasks": plan.total_tasks,
        "n_waves": plan.n_waves,
        "tasks": [
            {
                "name": t.name, "goal": t.goal, "task_type": t.task_type,
                "priority": t.priority, "depends_on": t.depends_on,
            }
            for t in plan.tasks
        ],
        "waves": [[t.name for t in w] for w in plan.waves],
    }


@router.post(
    "/v1/plan/execute",
    tags=["plan"],
    summary="ゴールを分解・実行・統合 (P10)",
    description="タスクを分解して実行し、結果を統合した最終アウトプットを返す。",
    dependencies=[Depends(verify_api_key)],
)
def plan_execute(req: TaskPlanRequest):
    planner = _TaskPlanner(max_parallel=req.max_parallel, kpi_target=req.kpi_target)
    result = planner.execute(req.goal, context=req.context, n_agents=req.n_agents)
    return {
        "goal": result.plan.goal,
        "synthesized_output": result.synthesized_output,
        "total_score": result.total_score,
        "kpi_achieved": result.kpi_achieved,
        "success_rate": result.success_rate,
        "total_latency_ms": result.total_latency_ms,
        "subtasks": [
            {
                "name": r.task.name,
                "task_type": r.task.task_type,
                "output": r.output[:200],
                "score": r.score,
                "success": r.success,
                "latency_ms": r.latency_ms,
            }
            for r in result.subtask_results
        ],
    }




# Sprint 30: GrowingAIOrchestrator — /v1/grow/run
from open_mythos.growing_ai_orchestrator import (
    GrowingAIOrchestrator as _GrowingAIOrchestrator,
)

# Sprint 43: HermesOrchestrator — /v1/hermes/*
from open_mythos.hermes_orchestrator import (
    HermesOrchestrator as _HermesOrchestrator,
)


class GrowRunRequest(BaseModel):
    goal: str = Field(..., description="達成したい目標・質問・タスク記述")
    hints: list[str] = Field(default_factory=list, description="パターン選択ヒント")
    max_patterns: int = Field(3, ge=1, le=10, description="同時適用パターン上限")
    metadata: dict = Field(default_factory=dict, description="任意付加情報")


@router.post(
    "/v1/grow/run",
    tags=["grow"],
    summary="P1〜P10 統合オーケストレーター実行 (Sprint 30)",
    description="ゴールを受け取り、最適な育つAIパターンを自動選択・実行して統合結果を返す。",
    dependencies=[Depends(verify_api_key)],
)
def grow_run(req: GrowRunRequest):
    orch   = _GrowingAIOrchestrator(max_patterns=req.max_patterns)
    result = orch.run(req.goal, hints=req.hints, metadata=req.metadata)
    return {
        "goal":             result.goal,
        "patterns_used":    [p.value for p in result.patterns_used],
        "final_output":     result.final_output,
        "overall_score":    result.overall_score,
        "total_latency_ms": result.total_latency_ms,
        "results": [
            {
                "pattern":    r.pattern.value,
                "score":      r.score,
                "latency_ms": r.latency_ms,
                "error":      r.error,
            }
            for r in result.results
        ],
    }


# ---------------------------------------------------------------------------
# Sprint 43: HermesOrchestrator — /v1/hermes/*
# Plan → Spawn → Parallel Execute → Verify → Report
# ---------------------------------------------------------------------------


class HermesRunRequest(BaseModel):
    goal: str = Field(..., description="達成したいゴール・タスク記述")
    context: dict = Field(default_factory=dict, description="付加コンテキスト情報")
    max_subtasks: int = Field(4, ge=1, le=8, description="最大サブタスク数")
    max_concurrent: int = Field(3, ge=1, le=8, description="並列エージェント数上限")
    max_new_tokens: int = Field(256, ge=1, le=1024, description="エージェントあたりの生成トークン上限")


class HermesPlanRequest(BaseModel):
    goal: str = Field(..., description="タスク分解したいゴール")
    context: dict = Field(default_factory=dict, description="付加コンテキスト情報")
    max_subtasks: int = Field(4, ge=1, le=8, description="最大サブタスク数")


def _build_hermes_orch(req_max_subtasks: int, req_max_concurrent: int, req_max_new_tokens: int) -> _HermesOrchestrator:
    """HermesOrchestrator インスタンスを構築する (Layer 1 API を自己呼び出し)。
    本番では HERMES_BASE_URL 環境変数でターゲットを指定可能。"""
    base_url = os.getenv("HERMES_BASE_URL", "http://localhost:8000")
    return _HermesOrchestrator(
        base_url=base_url,
        max_subtasks=req_max_subtasks,
        max_concurrent=req_max_concurrent,
        max_new_tokens=req_max_new_tokens,
    )


@router.post(
    "/v1/hermes/run",
    tags=["hermes"],
    summary="Hermes Layer 2 Ultracode フルパイプライン実行 (Sprint 43)",
    description=(
        "Plan → Spawn → Parallel Execute → Verify → Report の 5 フェーズを"
        "asyncio で実行し、統合レポートを返す。"
    ),
    dependencies=[Depends(verify_api_key)],
)
async def hermes_run(req: HermesRunRequest):
    """Hermes Ultracode Mode — フルパイプライン非同期実行"""
    orch = _build_hermes_orch(req.max_subtasks, req.max_concurrent, req.max_new_tokens)
    rpt = await orch.run_async(req.goal, req.context or None)
    return {
        "run_id":       rpt.run_id,
        "goal":         rpt.goal,
        "subtask_count": len(rpt.subtasks),
        "subtasks": [
            {
                "task_id":     st.task_id,
                "name":        st.name,
                "description": st.description,
                "priority":    st.priority,
                "depends_on":  st.depends_on,
            }
            for st in rpt.subtasks
        ],
        "agent_results": [
            {
                "agent_id":   ar.agent_id,
                "task_id":    ar.task_id,
                "task_name":  ar.task_name,
                "success":    ar.success,
                "latency_ms": ar.latency_ms,
                "error":      ar.error,
            }
            for ar in rpt.agent_results
        ],
        "verification_results": [
            {
                "agent_id":        vr.agent_id,
                "task_id":         vr.task_id,
                "task_name":       vr.task_name,
                "passed":          vr.passed,
                "score":           vr.score,
                "issues":          vr.issues,
            }
            for vr in rpt.verification_results
        ],
        "final_output":     rpt.final_output,
        "overall_score":    rpt.overall_score,
        "success_rate":     rpt.success_rate,
        "total_latency_ms": rpt.total_latency_ms,
        "phase_timings":    rpt.phase_timings,
    }


@router.post(
    "/v1/hermes/plan",
    tags=["hermes"],
    summary="Hermes Phase 1 — タスク分解のみ実行 (Sprint 43)",
    description="ゴールをサブタスクリストに分解して返す。実行は行わない。",
    dependencies=[Depends(verify_api_key)],
)
def hermes_plan(req: HermesPlanRequest):
    """Hermes Phase 1 (Plan) — タスク分解のみ"""
    orch = _build_hermes_orch(req.max_subtasks, 1, 256)
    subtasks = orch.plan(req.goal, req.context or None)
    return {
        "goal":         req.goal,
        "subtask_count": len(subtasks),
        "subtasks": [
            {
                "task_id":     st.task_id,
                "name":        st.name,
                "description": st.description,
                "priority":    st.priority,
                "depends_on":  st.depends_on,
            }
            for st in subtasks
        ],
    }



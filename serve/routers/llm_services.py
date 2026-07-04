"""
serve/routers/llm_services.py — LLM サービスドメイン API (Sprint 54〜58)

OpenAI Assistants 互換レイヤー / ストリーミング & SSE / マルチプロバイダー LLM /
LLM 評価フレームワーク / LLMO ダッシュボード・CEP 管理・競合分析。
serve/api.py のモノリスから分割 (認証は app 全体の verify_api_key に委譲)。
"""

from __future__ import annotations

import time

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from serve.auth import verify_api_key
from serve.state import state

router = APIRouter()

# ===========================================================================
# Sprint 54 — OpenAI Assistants API 互換レイヤー
# ===========================================================================

from open_mythos.assistant import (  # noqa: E402
    AssistantRunner as _AssistantRunner,
    AssistantTool as _AssistantTool,
    get_default_store as _get_default_store,
)
from typing import List as _List, Optional as _Optional  # noqa: E402


class _CreateAssistantRequest(BaseModel):
    model: str = "openmythos"
    name: _Optional[str] = None
    description: _Optional[str] = None
    instructions: _Optional[str] = None
    tools: _List[dict] = []
    metadata: dict = {}


class _CreateThreadRequest(BaseModel):
    metadata: dict = {}


class _AddMessageRequest(BaseModel):
    role: str = "user"
    content: str = ""
    metadata: dict = {}


class _CreateRunRequest(BaseModel):
    assistant_id: str
    model: _Optional[str] = None
    instructions: _Optional[str] = None
    tools: _List[dict] = []
    metadata: dict = {}


# --- Assistants ---

@router.post(
    "/v1/assistants",
    tags=["assistants"],
    summary="Create assistant (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def create_assistant(req: _CreateAssistantRequest):
    """OpenAI Assistants API 互換: アシスタントを作成する。"""
    tools = [_AssistantTool(**t) for t in req.tools]
    store = _get_default_store()
    asst = store.create_assistant(
        model=req.model,
        name=req.name,
        description=req.description,
        instructions=req.instructions,
        tools=tools,
        metadata=req.metadata,
    )
    return asst.to_dict()


@router.get(
    "/v1/assistants",
    tags=["assistants"],
    summary="List assistants (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def list_assistants(limit: int = 20):
    """OpenAI Assistants API 互換: アシスタント一覧を取得する。"""
    store = _get_default_store()
    items = store.list_assistants(limit=limit)
    return {"object": "list", "data": [a.to_dict() for a in items]}


@router.get(
    "/v1/assistants/{assistant_id}",
    tags=["assistants"],
    summary="Get assistant (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def get_assistant(assistant_id: str):
    """OpenAI Assistants API 互換: 指定 ID のアシスタントを取得する。"""
    store = _get_default_store()
    asst = store.get_assistant(assistant_id)
    if asst is None:
        from fastapi import HTTPException  # noqa: F811
        raise HTTPException(status_code=404, detail="Assistant not found")
    return asst.to_dict()


@router.delete(
    "/v1/assistants/{assistant_id}",
    tags=["assistants"],
    summary="Delete assistant (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def delete_assistant(assistant_id: str):
    """OpenAI Assistants API 互換: アシスタントを削除する。"""
    store = _get_default_store()
    deleted = store.delete_assistant(assistant_id)
    return {"id": assistant_id, "object": "assistant.deleted", "deleted": deleted}


# --- Threads ---

@router.post(
    "/v1/threads",
    tags=["assistants"],
    summary="Create thread (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def create_thread(req: _CreateThreadRequest):
    """OpenAI Assistants API 互換: スレッドを作成する。"""
    store = _get_default_store()
    thread = store.create_thread(metadata=req.metadata)
    return thread.to_dict()


@router.get(
    "/v1/threads/{thread_id}",
    tags=["assistants"],
    summary="Get thread (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def get_thread(thread_id: str):
    """OpenAI Assistants API 互換: 指定 ID のスレッドを取得する。"""
    store = _get_default_store()
    thread = store.get_thread(thread_id)
    if thread is None:
        from fastapi import HTTPException  # noqa: F811
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread.to_dict()


@router.delete(
    "/v1/threads/{thread_id}",
    tags=["assistants"],
    summary="Delete thread (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def delete_thread(thread_id: str):
    """OpenAI Assistants API 互換: スレッドを削除する。"""
    store = _get_default_store()
    deleted = store.delete_thread(thread_id)
    return {"id": thread_id, "object": "thread.deleted", "deleted": deleted}


# --- Messages ---

@router.post(
    "/v1/threads/{thread_id}/messages",
    tags=["assistants"],
    summary="Add message to thread (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def add_message(thread_id: str, req: _AddMessageRequest):
    """OpenAI Assistants API 互換: スレッドにメッセージを追加する。"""
    store = _get_default_store()
    msg = store.add_message(
        thread_id=thread_id,
        role=req.role,
        content=req.content,
        metadata=req.metadata,
    )
    return msg.to_dict()


@router.get(
    "/v1/threads/{thread_id}/messages",
    tags=["assistants"],
    summary="List messages in thread (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def list_messages(thread_id: str, limit: int = 20):
    """OpenAI Assistants API 互換: スレッド内のメッセージ一覧を取得する。"""
    store = _get_default_store()
    msgs = store.list_messages(thread_id=thread_id, limit=limit)
    return {"object": "list", "data": [m.to_dict() for m in msgs]}


# --- Runs ---

@router.post(
    "/v1/threads/{thread_id}/runs",
    tags=["assistants"],
    summary="Create and execute run (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def create_run(thread_id: str, req: _CreateRunRequest):
    """OpenAI Assistants API 互換: ランを作成して実行する。"""
    store = _get_default_store()
    run = store.create_run(
        thread_id=thread_id,
        assistant_id=req.assistant_id,
        model=req.model,
        instructions=req.instructions,
        metadata=req.metadata,
    )
    runner = _AssistantRunner(store)
    result = runner.execute(run)
    return result.to_dict()


@router.get(
    "/v1/threads/{thread_id}/runs/{run_id}",
    tags=["assistants"],
    summary="Get run (Sprint 54)",
    dependencies=[Depends(verify_api_key)],
)
def get_run(thread_id: str, run_id: str):
    """OpenAI Assistants API 互換: 指定 ID のランを取得する。"""
    store = _get_default_store()
    run = store.get_run(run_id)
    if run is None:
        from fastapi import HTTPException  # noqa: F811
        raise HTTPException(status_code=404, detail="Run not found")
    return run.to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 55 — ストリーミング & SSE エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.streaming import StreamingRunner as _StreamingRunner  # noqa: E402


class _ChatStreamRequest(BaseModel):
    messages: list[dict] = Field(..., description="OpenAI 形式のメッセージ列")
    model:    str         = Field("openmythos", description="モデル名")
    max_tokens: int       = Field(256, ge=1, le=2048)
    stream:   bool        = Field(True, description="常に True（SSE 専用エンドポイント）")


@router.post(
    "/v1/chat/stream",
    tags=["streaming"],
    summary="Chat ストリーミング (SSE) — Sprint 55",
    description=(
        "チャットメッセージを受け取り、Server-Sent Events でトークンを逐次返す。\n\n"
        "フォーマット:\n"
        "```\n"
        "event: delta\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"こんにちは\"},...}]}\n\n"
        "data: [DONE]\n\n"
        "```"
    ),
)
def chat_stream(req: _ChatStreamRequest):
    """OpenAI Chat Completions 互換 SSE ストリーミングエンドポイント。"""
    # ユーザーメッセージを結合してプロンプトを構築
    system_msg = next(
        (m.get("content", "") for m in req.messages if m.get("role") == "system"),
        None,
    )
    user_parts = [
        m.get("content", "")
        for m in req.messages
        if m.get("role") in ("user", "assistant")
    ]
    prompt = " ".join(user_parts).strip() or "こんにちは"

    llm_instance = state.llm if hasattr(state, "llm") else None
    runner = _StreamingRunner(model_name=req.model, llm=llm_instance)

    def _sse_gen():
        yield from runner.run_as_sse(prompt, max_tokens=req.max_tokens, system=system_msg)

    return StreamingResponse(_sse_gen(), media_type="text/event-stream")


class _RunStreamRequest(BaseModel):
    assistant_id: str          = Field(..., description="アシスタント ID")
    model:        Optional[str] = Field(None,  description="モデル上書き（省略可）")
    instructions: Optional[str] = Field(None,  description="指示上書き")
    max_tokens:   int            = Field(256, ge=1, le=2048)


@router.post(
    "/v1/threads/{thread_id}/runs/stream",
    tags=["streaming"],
    summary="Assistants Run ストリーミング (SSE) — Sprint 55",
    description=(
        "Assistants Run をストリーミングで実行する。\n"
        "スレッドの最新ユーザーメッセージをプロンプトとし、"
        "SSE で逐次トークンを返す。完了時に Run レコードを `completed` に更新する。"
    ),
)
def run_stream(thread_id: str, req: _RunStreamRequest):
    """Assistants API 互換: Run の結果を SSE で逐次配信する。"""
    store = _get_default_store()

    # スレッド存在確認
    thread = store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # アシスタント存在確認
    asst = store.get_assistant(req.assistant_id)
    if asst is None:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Run を作成（status=in_progress）
    run = store.create_run(
        thread_id=thread_id,
        assistant_id=req.assistant_id,
        model=req.model or asst.model,
        instructions=req.instructions or asst.instructions,
    )
    run.status = "in_progress"

    # プロンプト: スレッドの最新ユーザーメッセージ
    msgs = store.list_messages(thread_id)
    user_texts = [m.text for m in msgs if m.role == "user"]
    prompt = user_texts[-1] if user_texts else "こんにちは"
    system = run.instructions or ""

    llm_instance = state.llm if hasattr(state, "llm") else None
    runner = _StreamingRunner(model_name=run.model, llm=llm_instance)

    collected: list[str] = []

    def _sse_gen():
        for chunk in runner.run(prompt, max_tokens=req.max_tokens, system=system or None):
            if not chunk.done:
                collected.append(chunk.delta.content)
                import json as _j
                payload = {
                    "object":  "thread.run.step.delta",
                    "run_id":  run.id,
                    "thread_id": thread_id,
                    "delta": {
                        "step_details": {
                            "type": "message_creation",
                            "message_creation": {
                                "content": chunk.delta.content,
                                "index": chunk.delta.index,
                            },
                        }
                    },
                }
                yield f"event: thread.run.step.delta\ndata: {_j.dumps(payload, ensure_ascii=False)}\n\n"
            else:
                # 完了: アシスタントメッセージを保存して Run を completed に
                full_text = "".join(collected)
                store.add_message(
                    thread_id=thread_id,
                    role="assistant",
                    content=full_text,
                    assistant_id=req.assistant_id,
                    run_id=run.id,
                )
                run.status       = "completed"
                run.completed_at = int(time.time())
                run.usage.completion_tokens = len(collected)
                run.usage.total_tokens      = len(collected)

                import json as _j
                done_payload = {
                    "object":   "thread.run",
                    "id":       run.id,
                    "status":   "completed",
                    "thread_id": thread_id,
                }
                yield f"event: thread.run.completed\ndata: {_j.dumps(done_payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(_sse_gen(), media_type="text/event-stream")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 56 — マルチプロバイダー LLM エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.llm_providers import (  # noqa: E402
    MultiProviderRouter as _MultiProviderRouter,
    LLMRequest as _LLMRequest,
    ProviderType as _ProviderType,
)


class _ProviderCompleteRequest(BaseModel):
    prompt:      str             = Field(..., description="プロンプト")
    system:      Optional[str]   = Field(None,  description="システムプロンプト")
    max_tokens:  int             = Field(256, ge=1, le=4096)
    temperature: float           = Field(0.7, ge=0.0, le=2.0)
    preferred_provider: Optional[str] = Field(
        None, description="優先プロバイダー: claude | openai | openmythos"
    )


class _ProviderCompleteResponse(BaseModel):
    text:              str
    provider_used:     str
    model:             str
    latency_ms:        float
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int


@router.post(
    "/v1/llm/complete",
    response_model=_ProviderCompleteResponse,
    tags=["providers"],
    summary="マルチプロバイダー LLM 補完 — Sprint 56",
    description=(
        "Claude / OpenAI / OpenMythos の中から利用可能なプロバイダーを自動選択して補完する。\n"
        "`preferred_provider` で優先プロバイダーを指定できる。"
    ),
)
def llm_complete(req: _ProviderCompleteRequest):
    """マルチプロバイダー LLM 補完エンドポイント。"""
    llm = state.llm if hasattr(state, "llm") else None
    router = _MultiProviderRouter.from_env(llm=llm)

    preferred = None
    if req.preferred_provider:
        try:
            preferred = _ProviderType(req.preferred_provider.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"不明なプロバイダー: {req.preferred_provider}. "
                       "有効値: claude, openai, openmythos",
            )

    llm_req = _LLMRequest(
        prompt=req.prompt,
        system=req.system,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    try:
        resp = router.complete(llm_req, preferred=preferred)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return _ProviderCompleteResponse(
        text=resp.text,
        provider_used=resp.provider_used,
        model=resp.model,
        latency_ms=resp.latency_ms,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        total_tokens=resp.total_tokens,
    )


@router.get(
    "/v1/llm/providers",
    tags=["providers"],
    summary="利用可能プロバイダー一覧 — Sprint 56",
)
def list_providers():
    """設定済みの LLM プロバイダー一覧と稼働状態を返す。"""
    llm = state.llm if hasattr(state, "llm") else None
    router = _MultiProviderRouter.from_env(llm=llm)
    available = router.available_providers()
    return {
        "available":  available,
        "total":      len(available),
        "all_providers": ["claude", "openai", "openmythos"],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 57 — LLM 評価フレームワーク エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.evaluation import (  # noqa: E402
    EvalSample as _EvalSample,
    BenchmarkRunner as _BenchmarkRunner,
    AdEvaluator as _AdEvaluator,
    TextEvaluator as _TextEvaluator,
)


class _EvalSampleIn(BaseModel):
    id:         str
    input:      str
    prediction: str
    reference:  Optional[str] = None


class _BenchmarkRequest(BaseModel):
    samples:         list[_EvalSampleIn]  = Field(..., min_length=1)
    model_name:      str                   = Field("unknown")
    evaluator_type:  str                   = Field(
        "text", description="'text' (汎用) | 'ad' (広告コピー専用)"
    )
    brand_keywords:  Optional[list[str]]   = Field(None, description="ad モード用ブランドキーワード")


class _LeaderboardRequest(BaseModel):
    samples:     list[_EvalSampleIn]
    models:      list[str]            = Field(..., min_length=2)
    predictions: dict[str, list[str]] = Field(
        ..., description="{model_name: [prediction1, ...]}"
    )
    evaluator_type: str = Field("text")
    brand_keywords: Optional[list[str]] = None


@router.post(
    "/v1/eval/benchmark",
    tags=["evaluation"],
    summary="LLM ベンチマーク評価 — Sprint 57",
    description=(
        "サンプルセットに対してテキスト評価または広告コピー評価を実行し、"
        "BenchmarkReport を返す。\n\n"
        "evaluator_type:\n"
        "- `text`: 汎用 BLEU/ROUGE/長さ/多様性\n"
        "- `ad`: 広告LLMO + CTR予測 + ブランド適合"
    ),
)
def run_benchmark(req: _BenchmarkRequest):
    """LLM 出力を自動ベンチマーク評価してレポートを返す。"""
    samples = [
        _EvalSample(
            id=s.id, input=s.input,
            prediction=s.prediction, reference=s.reference,
        )
        for s in req.samples
    ]

    if req.evaluator_type == "ad":
        evaluator = _AdEvaluator(brand_keywords=req.brand_keywords)
    else:
        evaluator = _TextEvaluator()

    runner = _BenchmarkRunner(evaluator=evaluator, model_name=req.model_name)
    report = runner.run(samples)
    return report.to_dict()


@router.post(
    "/v1/eval/benchmark/md",
    tags=["evaluation"],
    summary="ベンチマーク Markdown レポート — Sprint 57",
)
def run_benchmark_md(req: _BenchmarkRequest):
    """ベンチマーク結果を Markdown 形式で返す。"""
    samples = [
        _EvalSample(id=s.id, input=s.input, prediction=s.prediction, reference=s.reference)
        for s in req.samples
    ]
    evaluator = _AdEvaluator(brand_keywords=req.brand_keywords) \
        if req.evaluator_type == "ad" else _TextEvaluator()
    runner = _BenchmarkRunner(evaluator=evaluator, model_name=req.model_name)
    report = runner.run(samples)
    return {"markdown": report.to_markdown(), "avg_overall": report.avg_overall}


@router.post(
    "/v1/eval/leaderboard",
    tags=["evaluation"],
    summary="モデル比較リーダーボード — Sprint 57",
    description="複数モデルの予測を比較してリーダーボードを生成する。",
)
def run_leaderboard(req: _LeaderboardRequest):
    """複数モデルを比較してリーダーボードを返す。"""
    samples = [
        _EvalSample(id=s.id, input=s.input, prediction=s.prediction, reference=s.reference)
        for s in req.samples
    ]
    evaluator = _AdEvaluator(brand_keywords=req.brand_keywords) \
        if req.evaluator_type == "ad" else _TextEvaluator()
    runner = _BenchmarkRunner(evaluator=evaluator)
    board  = runner.compare(samples, req.models, req.predictions)
    return {
        "rankings": board.rankings(),
        "winner":   board.winner(),
        "markdown": board.to_markdown(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 58 — LLMO ダッシュボード・CEP管理・競合分析 エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.llmo_dashboard import (  # noqa: E402
    CepStore as _CepStore,
    CepCategory as _CepCategory,
    LlmoDashboard as _LlmoDashboard,
    LlmoReportEngine as _LlmoReportEngine,
)

# サービス全体で共有するシングルトン (インプロセス)
_cep_store: _CepStore = _CepStore()
_dashboards: dict[str, _LlmoDashboard] = {}


def _get_dashboard(brand: str) -> _LlmoDashboard:
    if brand not in _dashboards:
        _dashboards[brand] = _LlmoDashboard(brand_name=brand)
    return _dashboards[brand]


# ── CEP 管理 ────────────────────────────────────────────────────

class _CepCreateReq(BaseModel):
    scenario:  str
    category:  str           = Field("other", description="problem/comparison/recommend/how_to/purchase/other")
    target:    Optional[str] = None
    keywords:  Optional[list[str]] = None
    priority:  int           = Field(3, ge=1, le=5)
    notes:     str           = ""


@router.post("/v1/cep", tags=["cep"], summary="CEP 登録 — Sprint 58")
def cep_create(req: _CepCreateReq):
    try:
        cat = _CepCategory(req.category)
    except ValueError:
        raise HTTPException(400, f"不明なカテゴリー: {req.category}")
    entry = _cep_store.add(
        scenario=req.scenario, category=cat,
        target=req.target, keywords=req.keywords,
        priority=req.priority, notes=req.notes,
    )
    return entry.to_dict()


@router.get("/v1/cep", tags=["cep"], summary="CEP 一覧 — Sprint 58")
def cep_list(category: Optional[str] = None, max_priority: Optional[int] = None):
    if category:
        try:
            cat = _CepCategory(category)
            entries = _cep_store.by_category(cat)
        except ValueError:
            raise HTTPException(400, f"不明なカテゴリー: {category}")
    elif max_priority:
        entries = _cep_store.by_priority(max_priority)
    else:
        entries = _cep_store.list_all()
    return {"entries": [e.to_dict() for e in entries], "total": len(entries)}


@router.delete("/v1/cep/{cep_id}", tags=["cep"], summary="CEP 削除 — Sprint 58")
def cep_delete(cep_id: str):
    ok = _cep_store.delete(cep_id)
    if not ok:
        raise HTTPException(404, "CEP not found")
    return {"deleted": cep_id}


# ── スナップショット / ダッシュボード ─────────────────────────

class _SnapshotReq(BaseModel):
    brand_name:     str
    prompt:         str
    mention_rate:   float = Field(..., ge=0.0, le=1.0)
    citation_rate:  float = Field(0.0, ge=0.0, le=1.0)
    reference_rate: float = Field(0.0, ge=0.0, le=1.0)
    cep_id:         Optional[str] = None
    notes:          str   = ""


@router.post("/v1/llmo/snapshot", tags=["llmo-dashboard"], summary="LLMO スナップショット追加 — Sprint 58")
def add_snapshot(req: _SnapshotReq):
    db = _get_dashboard(req.brand_name)
    snap = db.add_snapshot(
        prompt=req.prompt,
        mention_rate=req.mention_rate,
        citation_rate=req.citation_rate,
        reference_rate=req.reference_rate,
        cep_id=req.cep_id,
        notes=req.notes,
    )
    return snap.to_dict()


@router.get("/v1/llmo/dashboard/{brand_name}", tags=["llmo-dashboard"], summary="LLMO ダッシュボード — Sprint 58")
def get_dashboard(brand_name: str):
    db = _get_dashboard(brand_name)
    engine = _LlmoReportEngine(db)
    return engine.summary()


@router.get("/v1/llmo/dashboard/{brand_name}/report", tags=["llmo-dashboard"], summary="LLMO Markdown レポート — Sprint 58")
def get_dashboard_report(brand_name: str):
    db = _get_dashboard(brand_name)
    engine = _LlmoReportEngine(db)
    return {"markdown": engine.to_markdown(), "brand": brand_name}


@router.get("/v1/llmo/dashboard/{brand_name}/trend", tags=["llmo-dashboard"], summary="LLMO 時系列トレンド — Sprint 58")
def get_trend(brand_name: str):
    db = _get_dashboard(brand_name)
    return {"brand": brand_name, "trend": db.trend(), "trend_delta": db.trend_delta()}


# ── 競合分析 ────────────────────────────────────────────────────

class _CompetitorReq(BaseModel):
    brand_name:  str
    name:        str
    category:    str  = ""
    url:         Optional[str]       = None
    keywords:    Optional[list[str]] = None


class _CompAnalyzeReq(BaseModel):
    brand_name:    str
    competitor_id: str
    prompt:        str
    our_mention:   float = Field(..., ge=0.0, le=1.0)
    comp_mention:  float = Field(..., ge=0.0, le=1.0)


@router.post("/v1/llmo/competitor", tags=["llmo-dashboard"], summary="競合ブランド登録 — Sprint 58")
def add_competitor(req: _CompetitorReq):
    db   = _get_dashboard(req.brand_name)
    comp = db.add_competitor(
        name=req.name, category=req.category,
        url=req.url, keywords=req.keywords,
    )
    return comp.to_dict()


@router.post("/v1/llmo/competitor/analyze", tags=["llmo-dashboard"], summary="競合比較分析 — Sprint 58")
def analyze_competitor(req: _CompAnalyzeReq):
    db     = _get_dashboard(req.brand_name)
    result = db.analyze_competitor(
        competitor_id=req.competitor_id,
        prompt=req.prompt,
        our_mention=req.our_mention,
        comp_mention=req.comp_mention,
    )
    if result is None:
        raise HTTPException(404, "Competitor not found")
    return result.to_dict()


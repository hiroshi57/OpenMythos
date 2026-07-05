#!/usr/bin/env python3
"""
OpenMythos inference API — FastAPI server.

Endpoints:

    POST /infer          — text classification / identity scoring
    POST /infer/raw      — raw logits for downstream use
    POST /generate       — autoregressive text generation (OpenMythosLLM)
    GET  /generate/stream — streaming text generation (Server-Sent Events)
    POST /agent          — MythosAgent with conversation history
    DELETE /agent/{session_id} — reset agent session history
    GET  /health         — server health + capabilities

The key design: `loops` is a first-class API parameter.
- Low loops  (2–4)  → fast, suitable for real-time identity checks
- High loops (8–16) → deeper reasoning, for fraud / anomaly detection

Start:
    uvicorn serve.api:app --reload

Docker:
    docker build -t openmythos-serve -f serve/Dockerfile .
    docker run -p 8000:8000 openmythos-serve
"""

import json as _json_mod
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Iterator, List, Literal, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from transformers import AutoTokenizer

from open_mythos import __version__ as _OM_VERSION
from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.agents import MythosAgent, OpenMythosLLM
from serve.auth import RateLimitMiddleware, verify_api_key
from serve.dashboard import router as _dashboard_router
from serve.ad_router import router as _ad_router

try:
    from prometheus_client import (
        Counter as _PCounter,
        Histogram as _PHistogram,
        CollectorRegistry as _PRegistry,
        generate_latest as _prom_latest,
        CONTENT_TYPE_LATEST as _PROM_CT,
    )
    _PROM_REGISTRY = _PRegistry()
    _REQ_COUNT = _PCounter(
        "openmythos_requests_total", "Total HTTP requests",
        ["endpoint", "method", "status"], registry=_PROM_REGISTRY,
    )
    _REQ_LATENCY = _PHistogram(
        "openmythos_request_duration_seconds", "Request duration",
        ["endpoint"], buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        registry=_PROM_REGISTRY,
    )
    _PROM_OK = True
except Exception:  # noqa: BLE001
    _PROM_OK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIM = int(os.getenv("MODEL_DIM", "256"))
MODEL_ATTN = os.getenv("MODEL_ATTN", "gqa")
MODEL_CHECKPOINT = os.getenv(
    "MODEL_CHECKPOINT", ""
)  # path to .pt file; empty = random weights
DEVICE = os.getenv("DEVICE", "cpu")
from serve.state import (  # noqa: E402
    DEFAULT_LOOPS,
    MAX_LOOPS,
    TASK_LOOPS,
    TaskType,
    get_mistake_store as _get_mistake_store,
)
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "gpt2")


def _build_config(dim: int, attn: str) -> MythosConfig:
    return MythosConfig(
        vocab_size=50257,
        dim=dim,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=512,
        max_loop_iters=MAX_LOOPS,
        prelude_layers=1,
        coda_layers=1,
        attn_type=attn,
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=dim // 2,
        act_threshold=0.99,
        lora_rank=8,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------


# 共有シングルトン (serve/state.py に分離。各ドメインルーターからも参照される)
from serve.state import state  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device(DEVICE)
    print(f"[startup] device={device}  checkpoint='{MODEL_CHECKPOINT or 'random'}'")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    cfg = _build_config(MODEL_DIM, MODEL_ATTN)
    model = OpenMythos(cfg).to(device)

    if MODEL_CHECKPOINT:
        ckpt = torch.load(MODEL_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt)
        print(f"[startup] loaded checkpoint: {MODEL_CHECKPOINT}")

    model.eval()
    state.model = model
    state.tokenizer = tokenizer
    state.device = device
    state.n_params = sum(p.numel() for p in model.parameters())
    state.llm = OpenMythosLLM(
        model=model,
        device=str(device),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
        temperature=float(os.getenv("TEMPERATURE", "1.0")),
        top_k=int(os.getenv("TOP_K", "50")),
        top_p=float(os.getenv("TOP_P", "0.95")),
    )
    print(f"[startup] ready  params={state.n_params:,}  default_loops={DEFAULT_LOOPS}")
    yield
    print("[shutdown] done")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenMythos API",
    description=(
        "OpenMythos — Recurrent-Depth Transformer による SEO / LLMO / 広告最適化 API。\n\n"
        "**認証**: `Authorization: Bearer <api-key>` ヘッダ必須 (環境変数 `API_KEY` 設定時)。\n\n"
        "**レート制限**: デフォルト 60 rpm (環境変数 `RATE_LIMIT_RPM` で変更可)。"
    ),
    version=_OM_VERSION,
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
    openapi_tags=[
        # ── 基盤 ──────────────────────────────────────────────────
        {"name": "health", "description": "ヘルスチェック・サーバ情報"},
        {"name": "infer", "description": "スコアリング・分類推論"},
        {"name": "generate", "description": "テキスト生成 (SEO / LLMO / 広告コピー)"},
        {"name": "agent", "description": "多ターン対話エージェント"},
        {"name": "chat", "description": "OpenAI 互換 /v1/chat/completions"},
        {"name": "seo", "description": "SEO/LLMO スコアリング・最適化・改善提案"},
        {"name": "thinking", "description": "Extended Thinking (内部思考トレース)"},
        {"name": "tools", "description": "Tool Use / Function Calling"},
        {"name": "rag", "description": "RAG (Retrieval-Augmented Generation)"},
        {"name": "sessions", "description": "会話セッション管理"},
        {"name": "batch", "description": "バッチ推論"},
        # ── 育つ AI P1〜P10 ────────────────────────────────────────
        {"name": "debate", "description": "P1: 討議型集合知 — DebateOrchestrator / ConsensusEngine"},
        {"name": "kpi", "description": "P2: KPI 駆動自己改善 — KPIAgent"},
        {"name": "profiler", "description": "P3: ボトルネック発見・解消 — ProfilerAgent"},
        {"name": "signal", "description": "P4: 外部要因適応 — ExternalSignalAgent"},
        {"name": "mistakes", "description": "P5: ミスから学習 — MistakeGuard / ErrorMemory"},
        {"name": "distill", "description": "P6: 継続的自己蒸留 — SelfDistillLoop"},
        {"name": "memory", "description": "P7: 長期記憶統合 — LongTermMemoryAgent (FAISS ANN)"},
        {"name": "ensemble", "description": "P8: アンサンブル品質評価 — EnsembleScorer"},
        {"name": "evolve", "description": "P9: 適応型プロンプト進化 — PromptEvolution (GA)"},
        {"name": "plan", "description": "P10: 自律タスク計画 — TaskPlanner"},
        # ── 統合・ガード ───────────────────────────────────────────
        {"name": "grow", "description": "統合オーケストレーター — GrowingAIOrchestrator (P1〜P10 連携)"},
        {"name": "guard", "description": "MistakeGuardMiddleware — 全 API ミス透過チェック"},
        {"name": "hermes", "description": "Layer 2 Ultracode Orchestrator — Plan→Spawn→Parallel→Verify→Report (Sprint 43)"},
    ],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sprint 39: ショーケースダッシュボード + Prometheus
app.include_router(_dashboard_router)
app.include_router(_ad_router)


@app.get(
    "/metrics",
    tags=["health"],
    summary="Prometheus メトリクス",
    description="Prometheus スクレイプ用テキスト形式のメトリクスを返す。",
    include_in_schema=True,
)
def prometheus_metrics():
    if not _PROM_OK:
        return Response("# prometheus_client not installed\n", media_type="text/plain")
    return Response(
        content=_prom_latest(_PROM_REGISTRY),
        media_type=_PROM_CT,
    )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------



class InferRequest(BaseModel):
    text: str = Field(..., description="Input text to process")
    loops: int = Field(
        DEFAULT_LOOPS,
        ge=1,
        le=MAX_LOOPS,
        description="Recurrent loop depth. Higher = deeper reasoning, slower.",
    )
    task: TaskType = Field(
        "general",
        description=(
            "Task type. Controls recommended loop depth:\n"
            "  ad_performance  — 広告効果予測: fast (loops 2–4)\n"
            "  content_quality — SEO品質スコア: balanced (loops 6–12)\n"
            "  persona_segment — ペルソナ分類: balanced (loops 4–8)\n"
            "  market_research — 市場調査要約: balanced (loops 4–8)\n"
            "  identity_verify — 本人確認: real-time (loops 2–4)\n"
            "  fraud_detect    — 詐欺検知: high accuracy (loops 8–16)\n"
            "  seo_content     — SEO記事生成: balanced (loops 6–8)\n"
            "  llmo_optimize   — LLMO最適化: deep (loops 8–12)\n"
            "  ad_copy         — 広告コピー生成: fast (loops 2–4)\n"
            "  persona_message — ペルソナ別メッセージ: balanced (loops 4–6)\n"
            "  market_summary  — 市場調査サマリー: balanced (loops 4–8)\n"
            "  general         — use loops as specified"
        ),
    )
    max_new_tokens: Optional[int] = Field(
        None,
        description="If set, run autoregressive generation instead of scoring.",
    )


# ---------------------------------------------------------------------------
# Generation / Agent schemas
# ---------------------------------------------------------------------------

# タスク別システムプロンプト
_TASK_SYSTEM_PROMPTS: dict[str, str] = {
    "seo_content": (
        "あなたは日本語SEOコンテンツの専門ライターです。"
        "検索意図を満たし、E-E-A-T（経験・専門性・権威性・信頼性）を意識した"
        "高品質な記事・見出し・メタディスクリプションを生成します。"
        "キーワードを自然に組み込み、読者にとって価値ある情報を提供してください。"
    ),
    "llmo_optimize": (
        "あなたはLLMO（Large Language Model Optimization）の専門家です。"
        "ChatGPT・Claude・Geminiなどの生成AIに検索・引用されやすいコンテンツを作成します。"
        "具体的な数値・固有名詞・明確な定義・構造化された情報を盛り込み、"
        "AIが回答として引用しやすい形式で出力してください。"
    ),
    "ad_copy": (
        "あなたは日本市場向け広告コピーライターです。"
        "ターゲットの心理を動かすキャッチコピー・広告文・CTAを生成します。"
        "PREP法・PASONAの法則を活用し、クリック率・コンバージョン率を最大化する"
        "compelling なコピーを作成してください。"
    ),
    "persona_message": (
        "あなたはペルソナ分析とメッセージング設計の専門家です。"
        "指定されたペルソナの価値観・悩み・行動パターンを踏まえ、"
        "そのペルソナに最も響くメッセージ・訴求軸・トーンで文章を生成します。"
    ),
    "market_summary": (
        "あなたは市場調査・競合分析の専門アナリストです。"
        "提供された情報を整理し、エグゼクティブサマリー・市場規模・トレンド・"
        "競合状況・示唆を構造化してまとめます。経営判断に使えるレポートを生成してください。"
    ),
    "general": "あなたはOpenMythosを使った日本語アシスタントです。",
}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="生成プロンプト")
    task: TaskType = Field(
        "general", description="タスク種別（システムプロンプトを自動設定）"
    )
    max_new_tokens: int = Field(256, ge=1, le=1024, description="最大生成トークン数")
    temperature: float = Field(1.0, ge=0.01, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=500)
    n_loops: Optional[int] = Field(None, description="ループ深度オーバーライド")
    system_prompt: Optional[str] = Field(
        None, description="カスタムシステムプロンプト（指定時はtaskより優先）"
    )


class GenerateResponse(BaseModel):
    text: str
    task: str
    prompt_len: int
    generated_tokens: int
    latency_ms: float


class AgentRequest(BaseModel):
    task_input: str = Field(..., description="エージェントへの入力テキスト")
    session_id: Optional[str] = Field(
        None, description="会話セッションID（省略時は新規作成）"
    )
    task: TaskType = Field("general", description="タスク種別")
    system_prompt: Optional[str] = Field(None, description="カスタムシステムプロンプト")
    max_new_tokens: int = Field(256, ge=1, le=1024)
    n_loops: Optional[int] = Field(None)


class AgentResponse(BaseModel):
    session_id: str
    response: str
    task: str
    turn: int
    latency_ms: float


class InferResponse(BaseModel):
    score: float = Field(
        description="Confidence score in [0, 1] (higher = more confident positive)"
    )
    label: int = Field(
        description="Predicted label: 1 = positive / verified, 0 = negative / flagged"
    )
    loops_used: int
    latency_ms: float
    model_params: int


class RawInferResponse(BaseModel):
    logits: list[list[float]] = Field(description="Raw logits (seq_len, vocab_size)")
    loops_used: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Recommended loops per task (advisory — caller can override)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    tags=["health"],
    summary="サーバヘルスチェック",
    description="モデルパラメータ数・デバイス・対応タスク一覧を返す。認証・レート制限スキップ。",
)
def health():
    return {
        "status": "ok",
        "params": state.n_params,
        "device": str(state.device),
        "default_loops": DEFAULT_LOOPS,
        "supported_tasks": list(TASK_LOOPS.keys()),
        "task_default_loops": TASK_LOOPS,
        "active_sessions": len(state.agents),
        "endpoints": {
            "POST /infer": "スコアリング・分類",
            "POST /infer/raw": "rawロジット取得",
            "POST /generate": "テキスト生成（SEO/LLMO/広告コピー等）",
            "GET  /generate/stream": "ストリーミング生成（SSE）",
            "POST /agent": "多ターン対話エージェント",
            "DELETE /agent/{id}": "セッションリセット",
        },
    }


@app.post(
    "/infer",
    response_model=InferResponse,
    tags=["infer"],
    summary="スコアリング・タスク分類",
    description="テキストをモデルに通してスコアを返す。`task` パラメータで推奨ループ数が自動選択される。",
)
def infer(req: InferRequest):
    # For task-specific requests, use recommended loops if caller didn't override
    loops = req.loops
    if req.task != "general" and req.loops == DEFAULT_LOOPS:
        loops = TASK_LOOPS[req.task]
    if loops > MAX_LOOPS:
        raise HTTPException(400, f"loops {loops} exceeds server MAX_LOOPS={MAX_LOOPS}")

    enc = state.tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = enc["input_ids"].to(state.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = state.model(input_ids, n_loops=loops)  # (1, T, vocab)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Use mean-pooled last-token probability as a scalar confidence score.
    # In production this head would be replaced by a fine-tuned binary classifier.
    last_logit = logits[0, -1, :]  # (vocab,)
    probs = torch.softmax(last_logit, dim=-1)
    score = float(probs.max())  # max-prob as confidence
    label = 1 if score >= 0.5 else 0

    return InferResponse(
        score=score,
        label=label,
        loops_used=loops,
        latency_ms=round(latency_ms, 2),
        model_params=state.n_params,
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["generate"],
    summary="テキスト生成",
    description="SEO記事・広告コピー・LLMOコンテンツ等を生成する。`task` でシステムプロンプトが自動選択される。",
)
def generate(req: GenerateRequest):
    """テキスト生成エンドポイント。SEO記事・広告コピー・LLMOコンテンツ等を生成。"""
    sys_prompt = req.system_prompt or _TASK_SYSTEM_PROMPTS.get(
        req.task, _TASK_SYSTEM_PROMPTS["general"]
    )
    full_prompt = f"{sys_prompt}\n\n{req.prompt}" if sys_prompt else req.prompt

    # 入力トークン数を計測
    enc = state.tokenizer(
        full_prompt, return_tensors="pt", truncation=True, max_length=512
    )
    prompt_len = enc["input_ids"].shape[1]

    t0 = time.perf_counter()
    state.llm.max_new_tokens = req.max_new_tokens
    state.llm.temperature = req.temperature
    state.llm.top_p = req.top_p
    state.llm.top_k = req.top_k

    text = state.llm.run(full_prompt)
    latency_ms = (time.perf_counter() - t0) * 1000

    # 生成テキストのトークン数を簡易計測
    gen_enc = state.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1024
    )
    generated_tokens = gen_enc["input_ids"].shape[1]

    return GenerateResponse(
        text=text,
        task=req.task,
        prompt_len=prompt_len,
        generated_tokens=generated_tokens,
        latency_ms=round(latency_ms, 2),
    )


@app.get(
    "/generate/stream",
    tags=["generate"],
    summary="ストリーミングテキスト生成 (SSE)",
    description="Server-Sent Events でトークンを逐次返す。リアルタイム表示用。",
)
def generate_stream(
    prompt: str,
    task: TaskType = "general",
    max_new_tokens: int = 256,
    n_loops: Optional[int] = None,
):
    """Server-Sent Events によるストリーミング生成。"""
    sys_prompt = _TASK_SYSTEM_PROMPTS.get(task, _TASK_SYSTEM_PROMPTS["general"])
    full_prompt = f"{sys_prompt}\n\n{prompt}"
    state.llm.max_new_tokens = max_new_tokens

    def _event_stream() -> Iterator[str]:
        for token in state.llm.stream(full_prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.post(
    "/agent",
    response_model=AgentResponse,
    tags=["agent"],
    summary="多ターン対話エージェント",
    description="`session_id` で会話履歴を維持。SEO相談・広告コピー反復改善等に使用。",
)
def agent_run(req: AgentRequest):
    """MythosAgent エンドポイント。session_id で会話履歴を維持した多ターン対話。

    ユースケース例:
      - SEO相談チャット: task=seo_content で記事構成をやり取り
      - マーケティング戦略ブレスト: task=market_summary で複数ターン議論
      - 広告コピー反復改善: task=ad_copy でフィードバックを受けてコピーを磨く
    """
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in state.agents:
        sys_prompt = req.system_prompt or _TASK_SYSTEM_PROMPTS.get(
            req.task, _TASK_SYSTEM_PROMPTS["general"]
        )
        agent = MythosAgent(
            model=state.model,
            device=str(state.device),
            max_new_tokens=req.max_new_tokens,
            system_prompt=sys_prompt,
        )
        state.agents[session_id] = agent
    else:
        agent = state.agents[session_id]
        # max_new_tokens はリクエストごとに更新可能
        agent.max_new_tokens = req.max_new_tokens

    turn = len(agent._history) + 1

    t0 = time.perf_counter()
    response = agent.run(req.task_input)
    latency_ms = (time.perf_counter() - t0) * 1000

    return AgentResponse(
        session_id=session_id,
        response=response,
        task=req.task,
        turn=turn,
        latency_ms=round(latency_ms, 2),
    )


@app.delete("/agent/{session_id}")
def agent_reset(session_id: str):
    """指定セッションの会話履歴をリセット。"""
    if session_id not in state.agents:
        raise HTTPException(404, f"session '{session_id}' not found")
    state.agents[session_id].reset()
    return {
        "ok": True,
        "session_id": session_id,
        "message": "会話履歴をリセットしました",
    }


@app.post(
    "/infer/raw",
    response_model=RawInferResponse,
    tags=["infer"],
    summary="Raw ロジット取得",
    description="全トークン位置のロジットを返す。ファインチューニング・蒸留用。",
)
def infer_raw(req: InferRequest):
    """Return raw logits for all positions — useful for downstream fine-tuning."""
    loops = min(req.loops, MAX_LOOPS)

    enc = state.tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = enc["input_ids"].to(state.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = state.model(input_ids, n_loops=loops)
    latency_ms = (time.perf_counter() - t0) * 1000

    return RawInferResponse(
        logits=logits[0].tolist(),
        loops_used=loops,
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# OpenAI 互換 /v1/chat/completions
# ---------------------------------------------------------------------------

# ── Sprint 40: サンプリング共通ヘルパー ────────────────────────────────────


def _apply_top_p(logits: "torch.Tensor", top_p: float) -> "torch.Tensor":
    """Nucleus (top-p) フィルタリングを適用してマスク済み logits を返す。"""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[mask] = float("-inf")
    return torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)


def _apply_sampling_penalties(
    logits: "torch.Tensor",
    generated: List[int],
    presence_penalty: float,
    frequency_penalty: float,
) -> "torch.Tensor":
    """presence_penalty / frequency_penalty を logits に適用する。"""
    if presence_penalty == 0.0 and frequency_penalty == 0.0:
        return logits
    logits = logits.clone()
    token_counts: dict[int, int] = {}
    for tok in generated:
        token_counts[tok] = token_counts.get(tok, 0) + 1
    for tok, count in token_counts.items():
        if 0 <= tok < logits.shape[0]:
            logits[tok] -= presence_penalty
            logits[tok] -= frequency_penalty * count
    return logits


def _collect_logprobs(
    logits: "torch.Tensor", chosen_token: int, top_k: int
) -> Optional[dict]:
    """chosen トークンの log prob と top_k logprobs を返す。top_k=0 なら None。"""
    if top_k == 0:
        return None
    log_probs = torch.log_softmax(logits, dim=-1)
    chosen_lp = float(log_probs[chosen_token].item())
    if top_k > 0:
        vals, idxs = torch.topk(log_probs, min(top_k, log_probs.shape[0]))
        top = [
            {"token": int(i.item()), "logprob": float(v.item())}
            for v, i in zip(vals, idxs)
        ]
    else:
        top = []
    return {"token": chosen_token, "logprob": chosen_lp, "top_logprobs": top}


def _check_stop(text: str, stop: Optional[List[str]]) -> Optional[str]:
    """text の末尾にマッチする stop シーケンスを返す。なければ None。"""
    if not stop:
        return None
    for s in stop:
        if s and text.endswith(s):
            return s
    return None


def _truncate_at_stop(text: str, stop: Optional[List[str]]) -> str:
    """text 内で最初に出現する stop シーケンスの手前でカットする。"""
    if not stop:
        return text
    best_pos = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1 and idx < best_pos:
            best_pos = idx
    return text[:best_pos]


# ── Sprint 41: Function Calling ヘルパー ────────────────────────────────────

import re as _re
_TOOL_CALL_RE = _re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    _re.DOTALL,
)

_TOOL_SYSTEM_TEMPLATE = """\
You have access to the following tools. To call a tool, output ONLY the tag below \
on its own line (JSON must be valid):
<tool_call>{{"name": "<tool_name>", "arguments": {{<json_arguments>}}}}</tool_call>

Available tools:
{tool_schema}

If no tool call is needed, respond normally.
"""


def _build_tools_system_block(tools: List[dict]) -> str:
    """tools[] 定義を system prompt 用テキストに変換する。"""
    import json as _j
    return _TOOL_SYSTEM_TEMPLATE.format(
        tool_schema=_j.dumps(tools, ensure_ascii=False, indent=2)
    )


def _parse_tool_calls_from_text(text: str) -> Optional[List[dict]]:
    """テキストから <tool_call>...</tool_call> を抽出して OpenAI 形式で返す。

    見つからなければ None を返す。
    """
    import json as _j
    matches = _TOOL_CALL_RE.findall(text)
    if not matches:
        return None
    results: List[dict] = []
    for i, raw in enumerate(matches):
        try:
            parsed = _j.loads(raw)
        except _j.JSONDecodeError:
            continue
        name = parsed.get("name", "")
        args = parsed.get("arguments", {})
        results.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": _j.dumps(args, ensure_ascii=False),
            },
        })
    return results if results else None


def _inject_tools_into_prompt(
    base_prompt: str,
    tools: Optional[List[dict]],
    tool_choice: Optional[str],
) -> str:
    """base_prompt の先頭に tool system block を注入する。

    tool_choice == "none" の場合は注入しない。
    """
    if not tools or tool_choice == "none":
        return base_prompt
    block = _build_tools_system_block(tools)
    return block + "\n\n" + base_prompt


class ChatMessage(BaseModel):
    # Sprint 41: "tool" ロール追加
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = ""
    # Sprint 41: Function Calling 用フィールド
    tool_call_id: Optional[str] = Field(None, description="tool ロール使用時の呼び出し ID")
    tool_calls: Optional[List[dict]] = Field(None, description="assistant が要求する tool 呼び出し")


class ChatRequest(BaseModel):
    model: str = Field(
        "openmythos", description="Model identifier (ignored; uses loaded model)"
    )
    messages: list[ChatMessage]
    max_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    loops: int = Field(DEFAULT_LOOPS, ge=1, le=16)
    stream: bool = Field(False, description="Return Server-Sent Events stream")
    task: TaskType = Field("general")
    # Sprint 40: 拡張フィールド
    stop: Optional[List[str]] = Field(None, description="Stop sequences (up to 4)")
    n: int = Field(1, ge=1, le=4, description="Number of completions to generate")
    logprobs: bool = Field(False, description="Include token log probabilities")
    top_logprobs: int = Field(0, ge=0, le=5, description="Top-K logprobs per token (0=disabled)")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    # Sprint 41: Function Calling
    tools: Optional[List[dict]] = Field(None, description="OpenAI 互換 tools[] 定義")
    tool_choice: Optional[str] = Field(
        None,
        description="'auto' | 'none' | 'required' — ツール選択方針",
    )


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[dict] = None
    # Sprint 41: tool_calls は message.tool_calls と同じ内容を最上位でも公開
    tool_calls: Optional[List[dict]] = None


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
    system_fingerprint: str = "fp_openmythos"


def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    """Convert chat messages to a flat prompt string.

    Sprint 41: tool / assistant(tool_calls) ロールに対応。
    """
    import json as _j
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"[System]: {m.content}")
        elif m.role == "user":
            parts.append(f"[User]: {m.content}")
        elif m.role == "assistant":
            if m.tool_calls:
                # tool 呼び出しを要求した assistant ターン
                for tc in m.tool_calls:
                    fn = tc.get("function", {})
                    parts.append(
                        "[Assistant]: <tool_call>"
                        + _j.dumps(
                            {"name": fn.get("name", ""), "arguments": fn.get("arguments", {})},
                            ensure_ascii=False,
                        )
                        + "</tool_call>"
                    )
            else:
                parts.append(f"[Assistant]: {m.content}")
        elif m.role == "tool":
            parts.append(f"[Tool({m.tool_call_id or ''})]: {m.content}")
    parts.append("[Assistant]:")
    return "\n".join(parts)


@app.post(
    "/v1/chat/completions",
    tags=["chat"],
    summary="OpenAI 互換チャット推論",
    description=(
        "OpenAI `/v1/chat/completions` 互換。`stream=true` で SSE ストリーミング対応。\n\n"
        "**Sprint 40 拡張**: `stop` / `n` / `logprobs` / `presence_penalty` / `frequency_penalty`"
    ),
)
def chat_completions(req: ChatRequest):
    """OpenAI 互換チャット推論エンドポイント。

    ``stream=false`` (デフォルト) は ``ChatResponse`` JSON を返す。
    ``stream=true`` は Server-Sent Events (SSE) でトークンを逐次送出する。
    """
    loops = min(req.loops, MAX_LOOPS)
    if req.task != "general":
        loops = TASK_LOOPS.get(req.task, loops)

    prompt = _build_chat_prompt(req.messages)
    # Sprint 41: tools が指定されていれば system block を先頭に注入
    prompt = _inject_tools_into_prompt(prompt, req.tools, req.tool_choice)
    enc = state.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = enc["input_ids"].to(state.device)
    prompt_tokens = input_ids.shape[1]

    # ── SSE ストリーミング ────────────────────────────────────────────────
    if req.stream:
        def _event_stream():
            import json as _json

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            generated: list[int] = []
            decoded_so_far = ""
            cur_ids = input_ids
            finish_reason = "length"  # default; overridden on EOS/stop

            for _ in range(req.max_tokens):
                with torch.no_grad():
                    logits = state.model(cur_ids, n_loops=loops)
                next_logits = logits[0, -1, :] / max(req.temperature, 1e-6)
                next_logits = _apply_sampling_penalties(
                    next_logits, generated,
                    req.presence_penalty, req.frequency_penalty,
                )
                next_logits = _apply_top_p(next_logits, req.top_p)
                probs = torch.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())
                generated.append(next_token)
                token_text = state.tokenizer.decode(
                    [next_token], skip_special_tokens=True
                )
                decoded_so_far += token_text
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[next_token]], device=state.device)], dim=1
                )

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": "openmythos",
                    "choices": [
                        {
                            "delta": {"content": token_text},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {_json.dumps(chunk)}\n\n"

                if next_token == state.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                if _check_stop(decoded_so_far, req.stop):
                    finish_reason = "stop"
                    break

            # Sprint 41: tool_calls が検出された場合 finish_reason を上書き
            tool_calls_parsed = _parse_tool_calls_from_text(decoded_so_far)
            if tool_calls_parsed:
                finish_reason = "tool_calls"
                # tool_calls delta chunk を送出
                tc_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": "openmythos",
                    "choices": [{
                        "delta": {"tool_calls": tool_calls_parsed},
                        "index": 0,
                        "finish_reason": None,
                    }],
                }
                yield f"data: {_json.dumps(tc_chunk)}\n\n"

            # final chunk with finish_reason + usage
            done_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": "openmythos",
                "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(generated),
                    "total_tokens": prompt_tokens + len(generated),
                },
            }
            yield f"data: {_json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # ── 非ストリーミング: n 候補一括生成 ────────────────────────────────
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    choices: list[ChatChoice] = []
    total_completion_tokens = 0

    for choice_idx in range(req.n):
        generated_ids: list[int] = []
        lp_list: list[dict] = []
        cur_ids = input_ids
        finish_reason = "length"

        with torch.no_grad():
            for _ in range(req.max_tokens):
                logits = state.model(cur_ids, n_loops=loops)
                next_logits = logits[0, -1, :] / max(req.temperature, 1e-6)
                next_logits = _apply_sampling_penalties(
                    next_logits, generated_ids,
                    req.presence_penalty, req.frequency_penalty,
                )
                next_logits = _apply_top_p(next_logits, req.top_p)
                probs = torch.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())

                if req.logprobs or req.top_logprobs > 0:
                    lp = _collect_logprobs(next_logits, next_token, req.top_logprobs)
                    if lp:
                        lp_list.append(lp)

                generated_ids.append(next_token)
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[next_token]], device=state.device)], dim=1
                )

                if next_token == state.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break

        completion_text = state.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        completion_text = _truncate_at_stop(completion_text, req.stop)
        # re-evaluate finish_reason after stop truncation
        if finish_reason == "length" and req.stop:
            raw = state.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if raw != completion_text:
                finish_reason = "stop"

        # Sprint 41: tool_calls 検出
        tool_calls_result = _parse_tool_calls_from_text(completion_text)
        if tool_calls_result:
            finish_reason = "tool_calls"
            # tool call が含まれる場合 content は None 相当 (空文字)
            completion_text = ""

        total_completion_tokens += len(generated_ids)
        choices.append(
            ChatChoice(
                index=choice_idx,
                message=ChatMessage(
                    role="assistant",
                    content=completion_text,
                    tool_calls=tool_calls_result,
                ),
                finish_reason=finish_reason,
                logprobs={"tokens": lp_list} if lp_list else None,
                tool_calls=tool_calls_result,
            )
        )

    return ChatResponse(
        id=completion_id,
        created=int(time.time()),
        model="openmythos",
        choices=choices,
        usage=ChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=prompt_tokens + total_completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Sprint 42: /v1/embeddings + /v1/semantic-search (OpenAI 互換)
# ---------------------------------------------------------------------------

import base64 as _base64
import struct as _struct


class EmbeddingRequest(BaseModel):
    """OpenAI /v1/embeddings 互換リクエスト。"""
    model: str = Field("openmythos", description="Model identifier")
    input: "str | List[str]" = Field(..., description="テキスト or テキストリスト")
    encoding_format: Literal["float", "base64"] = Field(
        "float", description="'float' → list[float] / 'base64' → base64 エンコード済みバイト列"
    )
    n_loops: int = Field(1, ge=1, le=4, description="ループ深度 (1=高速)")
    dimensions: Optional[int] = Field(
        None, ge=1, description="返す次元数 (None = モデル次元全体)"
    )


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: "List[float] | str"  # float モードはリスト、base64 モードは文字列


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


def _encode_text(text: str, n_loops: int, dimensions: Optional[int]) -> List[float]:
    """テキストを埋め込みベクトルに変換して mean-pool + L2 正規化して返す。"""
    enc = state.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
    )
    input_ids = enc["input_ids"].to(state.device)
    with torch.no_grad():
        hidden = state.model.encode(input_ids, n_loops=n_loops)  # (1, T, dim)
    # 最終トークンではなく全トークンの mean pooling
    vec = hidden[0].mean(dim=0)  # (dim,)
    # L2 正規化
    norm = vec.norm(p=2)
    if norm > 0:
        vec = vec / norm
    # 次元数を制限
    if dimensions is not None:
        vec = vec[:dimensions]
    return vec.cpu().tolist()


def _vec_to_base64(vec: List[float]) -> str:
    """float32 ベクトルを base64 エンコードする。"""
    packed = _struct.pack(f"{len(vec)}f", *vec)
    return _base64.b64encode(packed).decode("ascii")


@app.post(
    "/v1/embeddings",
    tags=["embeddings"],
    summary="テキスト埋め込み (OpenAI 互換)",
    description=(
        "OpenAI `/v1/embeddings` 互換。テキストを dense vector に変換する。\n\n"
        "返却ベクトルは L2 正規化済み。`encoding_format=base64` で float32 バイナリを Base64 エンコードして返す。"
    ),
)
def create_embeddings(req: EmbeddingRequest):
    """テキスト埋め込みを返す。"""
    texts: List[str] = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        raise HTTPException(status_code=422, detail="input must be non-empty")

    data: List[EmbeddingData] = []
    total_tokens = 0

    for i, text in enumerate(texts):
        # token 数を推定
        enc = state.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        n_tok = enc["input_ids"].shape[1]
        total_tokens += n_tok

        vec = _encode_text(text, req.n_loops, req.dimensions)
        if req.encoding_format == "base64":
            embedding_val: "List[float] | str" = _vec_to_base64(vec)
        else:
            embedding_val = vec

        data.append(EmbeddingData(index=i, embedding=embedding_val))

    return EmbeddingResponse(
        data=data,
        model="openmythos",
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )


# ── セマンティック検索 ────────────────────────────────────────────────────

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="検索クエリ")
    documents: List[str] = Field(..., min_length=1, description="検索対象ドキュメントリスト")
    top_k: int = Field(3, ge=1, le=50, description="上位 K 件を返す")
    n_loops: int = Field(1, ge=1, le=4)


class SemanticSearchResult(BaseModel):
    index: int
    document: str
    score: float  # コサイン類似度 [-1.0, 1.0]


class SemanticSearchResponse(BaseModel):
    query: str
    results: List[SemanticSearchResult]
    total_documents: int


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """L2 正規化済みベクトル同士のコサイン類似度 (= ドット積)。"""
    if len(a) != len(b) or not a:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


@app.post(
    "/v1/semantic-search",
    tags=["embeddings"],
    summary="セマンティック検索",
    description=(
        "クエリと複数ドキュメントを埋め込みに変換し、コサイン類似度でランキングして返す。\n\n"
        "内部で `/v1/embeddings` と同じ埋め込み計算を使用。"
    ),
)
def semantic_search(req: SemanticSearchRequest):
    """クエリに最も近いドキュメントを top_k 件返す。"""
    query_vec = _encode_text(req.query, req.n_loops, None)
    results = []
    for i, doc in enumerate(req.documents):
        doc_vec = _encode_text(doc, req.n_loops, None)
        score = _cosine_similarity(query_vec, doc_vec)
        results.append(SemanticSearchResult(index=i, document=doc, score=score))

    results.sort(key=lambda r: r.score, reverse=True)
    return SemanticSearchResponse(
        query=req.query,
        results=results[: req.top_k],
        total_documents=len(req.documents),
    )


# ---------------------------------------------------------------------------
# Sprint 40: /v1/completions — テキスト補完 (OpenAI 互換)
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    model: str = Field("openmythos", description="Model identifier")
    prompt: str = Field(..., description="Text prompt to complete")
    max_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    loops: int = Field(DEFAULT_LOOPS, ge=1, le=16)
    stream: bool = Field(False, description="Return Server-Sent Events stream")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    n: int = Field(1, ge=1, le=4, description="Number of completions")
    logprobs: bool = Field(False)
    top_logprobs: int = Field(0, ge=0, le=5)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    echo: bool = Field(False, description="Echo prompt in the returned text")


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str
    logprobs: Optional[dict] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: ChatUsage
    system_fingerprint: str = "fp_openmythos"


@app.post(
    "/v1/completions",
    tags=["chat"],
    summary="OpenAI 互換テキスト補完",
    description=(
        "OpenAI `/v1/completions` 互換。`stream=true` で SSE 対応。\n\n"
        "チャット形式 (`messages`) ではなく平文 `prompt` を受け付ける。"
    ),
)
def text_completions(req: CompletionRequest):
    """OpenAI 互換テキスト補完エンドポイント。"""
    loops = min(req.loops, MAX_LOOPS)
    enc = state.tokenizer(
        req.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = enc["input_ids"].to(state.device)
    prompt_tokens = input_ids.shape[1]

    # ── SSE ストリーミング ────────────────────────────────────────────────
    if req.stream:
        def _event_stream():
            import json as _json

            cid = f"cmpl-{uuid.uuid4().hex[:8]}"
            generated: list[int] = []
            decoded_so_far = req.prompt if req.echo else ""
            cur_ids = input_ids
            finish_reason = "length"

            for _ in range(req.max_tokens):
                with torch.no_grad():
                    logits = state.model(cur_ids, n_loops=loops)
                next_logits = logits[0, -1, :] / max(req.temperature, 1e-6)
                next_logits = _apply_sampling_penalties(
                    next_logits, generated,
                    req.presence_penalty, req.frequency_penalty,
                )
                next_logits = _apply_top_p(next_logits, req.top_p)
                probs = torch.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())
                generated.append(next_token)
                token_text = state.tokenizer.decode(
                    [next_token], skip_special_tokens=True
                )
                decoded_so_far += token_text
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[next_token]], device=state.device)], dim=1
                )

                chunk = {
                    "id": cid,
                    "object": "text_completion",
                    "model": "openmythos",
                    "choices": [
                        {"text": token_text, "index": 0, "finish_reason": None}
                    ],
                }
                yield f"data: {_json.dumps(chunk)}\n\n"

                if next_token == state.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                if _check_stop(decoded_so_far, req.stop):
                    finish_reason = "stop"
                    break

            done_chunk = {
                "id": cid,
                "object": "text_completion",
                "model": "openmythos",
                "choices": [{"text": "", "index": 0, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(generated),
                    "total_tokens": prompt_tokens + len(generated),
                },
            }
            yield f"data: {_json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # ── 非ストリーミング ──────────────────────────────────────────────────
    cid = f"cmpl-{uuid.uuid4().hex[:8]}"
    choices: list[CompletionChoice] = []
    total_completion_tokens = 0

    for choice_idx in range(req.n):
        generated_ids: list[int] = []
        lp_list: list[dict] = []
        cur_ids = input_ids
        finish_reason = "length"

        with torch.no_grad():
            for _ in range(req.max_tokens):
                logits = state.model(cur_ids, n_loops=loops)
                next_logits = logits[0, -1, :] / max(req.temperature, 1e-6)
                next_logits = _apply_sampling_penalties(
                    next_logits, generated_ids,
                    req.presence_penalty, req.frequency_penalty,
                )
                next_logits = _apply_top_p(next_logits, req.top_p)
                probs = torch.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())

                if req.logprobs or req.top_logprobs > 0:
                    lp = _collect_logprobs(next_logits, next_token, req.top_logprobs)
                    if lp:
                        lp_list.append(lp)

                generated_ids.append(next_token)
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[next_token]], device=state.device)], dim=1
                )
                if next_token == state.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break

        text = state.tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = _truncate_at_stop(text, req.stop)
        if finish_reason == "length" and req.stop:
            raw = state.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if raw != text:
                finish_reason = "stop"
        if req.echo:
            text = req.prompt + text

        total_completion_tokens += len(generated_ids)
        choices.append(
            CompletionChoice(
                index=choice_idx,
                text=text,
                finish_reason=finish_reason,
                logprobs={"tokens": lp_list} if lp_list else None,
            )
        )

    return CompletionResponse(
        id=cid,
        created=int(time.time()),
        model="openmythos",
        choices=choices,
        usage=ChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=prompt_tokens + total_completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# バッチ推論 /v1/batch
# ---------------------------------------------------------------------------


class BatchItem(BaseModel):
    text: str
    task: TaskType = "general"
    loops: int = Field(DEFAULT_LOOPS, ge=1, le=16)


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=64)


class BatchResponseItem(BaseModel):
    index: int
    score: float
    label: int
    loops_used: int
    latency_ms: float
    task: str


class BatchResponse(BaseModel):
    results: list[BatchResponseItem]
    total_latency_ms: float
    n_items: int


@app.post(
    "/v1/batch",
    response_model=BatchResponse,
    tags=["batch"],
    summary="バッチ推論",
    description="複数テキストを一括で推論する。各アイテム独立・ループ数個別指定可。",
)
def batch_infer(req: BatchRequest):
    """複数テキストを一括推論する。

    各アイテムは独立に推論され、ループ数はアイテムごとに指定可能。
    最大 64 アイテムまで対応。
    """
    t_start = time.perf_counter()
    results: list[BatchResponseItem] = []

    for i, item in enumerate(req.items):
        loops = min(item.loops, MAX_LOOPS)
        if item.task != "general" and item.loops == DEFAULT_LOOPS:
            loops = TASK_LOOPS.get(item.task, DEFAULT_LOOPS)

        enc = state.tokenizer(
            item.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(state.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = state.model(input_ids, n_loops=loops)
        latency_ms = (time.perf_counter() - t0) * 1000

        last_logit = logits[0, -1, :]
        probs = torch.softmax(last_logit, dim=-1)
        score = float(probs.max())
        label = 1 if score >= 0.5 else 0

        results.append(
            BatchResponseItem(
                index=i,
                score=round(score, 4),
                label=label,
                loops_used=loops,
                latency_ms=round(latency_ms, 2),
                task=item.task,
            )
        )

    total_ms = (time.perf_counter() - t_start) * 1000
    return BatchResponse(
        results=results,
        total_latency_ms=round(total_ms, 2),
        n_items=len(results),
    )


# ---------------------------------------------------------------------------
# SEO / LLMO エンドポイント
# ---------------------------------------------------------------------------

from open_mythos.llmo import LLMOOptimizer, LLMOScorer  # noqa: E402

_llmo_scorer = LLMOScorer()
_llmo_optimizer = LLMOOptimizer(scorer=_llmo_scorer)


class SEOScoreRequest(BaseModel):
    text: str = Field(..., description="スコアリング対象テキスト")


class SEOScoreResponse(BaseModel):
    entity_density: float
    answer_directness: float
    citability: float
    llmo_total: float
    entities: list[str]
    word_count: int
    latency_ms: float


class SEOGenerateRequest(BaseModel):
    prompt: str = Field(..., description="生成プロンプト")
    style: Literal["answer_first", "faq", "entity_rich"] = Field(
        "answer_first",
        description="コンテンツスタイル",
    )
    max_new_tokens: int = Field(128, ge=1, le=512)
    loops: int = Field(DEFAULT_LOOPS, ge=1, le=16)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class SEOGenerateResponse(BaseModel):
    text: str
    style: str
    llmo_total: float
    entity_density: float
    answer_directness: float
    citability: float
    entities: list[str]
    word_count: int
    loops_used: int
    latency_ms: float


@app.post(
    "/v1/seo/score",
    response_model=SEOScoreResponse,
    tags=["seo"],
    summary="SEO / LLMO スコアリング",
    description="entity_density / answer_directness / citability の 3 軸で SEO品質を評価する。",
)
def seo_score(req: SEOScoreRequest):
    """テキストの LLMO / SEO スコアを計算する。

    entity_density / answer_directness / citability の 3 軸と
    加重平均の llmo_total (0–1) を返す。
    """
    t0 = time.perf_counter()
    result = _llmo_scorer.score(req.text)
    latency_ms = (time.perf_counter() - t0) * 1000

    return SEOScoreResponse(
        entity_density=result.entity_density,
        answer_directness=result.answer_directness,
        citability=result.citability,
        llmo_total=result.llmo_total,
        entities=result.entities,
        word_count=result.word_count,
        latency_ms=round(latency_ms, 2),
    )


def _seo_style_prefix(style: str) -> str:
    prefixes = {
        "answer_first": (
            "[System]: Always start with a direct answer. Then supporting details.\n"
        ),
        "faq": (
            "[System]: Format your response as Q&A pairs. Each answer: 2-4 sentences.\n"
        ),
        "entity_rich": (
            "[System]: Include specific numbers, dates, proper nouns, and tech terms.\n"
        ),
    }
    return prefixes.get(style, prefixes["answer_first"])


@app.post(
    "/v1/seo/generate",
    response_model=SEOGenerateResponse,
    tags=["seo"],
    summary="SEO / LLMO コンテンツ生成",
    description="style (qa / listicle / entity_rich) を指定してコンテンツを生成し LLMO スコアを付与して返す。",
)
def seo_generate(req: SEOGenerateRequest):
    """SEO / LLMO 最適化コンテンツを生成し、スコアを付与して返す。

    prompt に対して style に応じたシステムプレフィックスを付与して生成し、
    生成結果を即座に LLMO スコアリングする。
    """
    loops = min(req.loops, MAX_LOOPS)
    prefix = _seo_style_prefix(req.style)
    full_prompt = f"{prefix}[User]: {req.prompt}\n[Assistant]:"

    enc = state.tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = enc["input_ids"].to(state.device)

    t0 = time.perf_counter()
    generated_ids: list[int] = []
    cur_ids = input_ids

    with torch.no_grad():
        for _ in range(req.max_new_tokens):
            logits = state.model(cur_ids, n_loops=loops)
            next_logits = logits[0, -1, :] / max(req.temperature, 1e-6)
            if req.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > req.top_p
                sorted_logits[mask] = float("-inf")
                next_logits = torch.full_like(next_logits, float("-inf")).scatter(
                    0, sorted_idx, sorted_logits
                )
            probs = torch.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probs, 1).item())
            generated_ids.append(next_token)
            cur_ids = torch.cat(
                [cur_ids, torch.tensor([[next_token]], device=state.device)], dim=1
            )
            if next_token == state.tokenizer.eos_token_id:
                break

    latency_ms = (time.perf_counter() - t0) * 1000
    generated_text = state.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # LLMO スコアリング
    llmo = _llmo_scorer.score(generated_text)

    return SEOGenerateResponse(
        text=generated_text,
        style=req.style,
        llmo_total=llmo.llmo_total,
        entity_density=llmo.entity_density,
        answer_directness=llmo.answer_directness,
        citability=llmo.citability,
        entities=llmo.entities,
        word_count=llmo.word_count,
        loops_used=loops,
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# Sprint 19: LLMO 強化エンドポイント
# ---------------------------------------------------------------------------


class LLMOSuggestRequest(BaseModel):
    text: str = Field(..., description="改善提案対象テキスト")
    query: str = Field("", description="検索クエリ (指定時は query_relevance も分析)")
    max_suggestions: int = Field(5, ge=1, le=10, description="最大提案数")


class ImprovementOut(BaseModel):
    category: str
    priority: str
    description: str
    example: str = ""
    expected_delta: float


class LLMOSuggestResponse(BaseModel):
    text_preview: str
    llmo_total: float
    entity_density: float
    answer_directness: float
    citability: float
    query_relevance: float
    intent_type: str
    suggestions: list[ImprovementOut]
    latency_ms: float


@app.post(
    "/v1/llmo/suggest",
    response_model=LLMOSuggestResponse,
    tags=["seo"],
    summary="LLMO 改善提案",
    description=(
        "テキストを分析し、LLMO スコアを向上させる具体的な改善提案を返す。"
        "`query` を指定すると query_relevance とクエリ意図型も分析する。"
    ),
)
def llmo_suggest(req: LLMOSuggestRequest):
    """LLMO 改善提案エンドポイント。

    entity / directness / citability / structure / length / query の各軸で
    priority 順に具体的な改善提案を返す。
    """
    t0 = time.perf_counter()
    score = (
        _llmo_scorer.score_with_query(req.text, req.query)
        if req.query
        else _llmo_scorer.score(req.text)
    )
    suggestions = _llmo_scorer.suggest_improvements(
        req.text, query=req.query, max_suggestions=req.max_suggestions
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    return LLMOSuggestResponse(
        text_preview=req.text[:100] + ("..." if len(req.text) > 100 else ""),
        llmo_total=score.llmo_total,
        entity_density=score.entity_density,
        answer_directness=score.answer_directness,
        citability=score.citability,
        query_relevance=score.query_relevance,
        intent_type=score.intent_type,
        suggestions=[
            ImprovementOut(
                category=s.category,
                priority=s.priority,
                description=s.description,
                example=s.example,
                expected_delta=s.expected_delta,
            )
            for s in suggestions
        ],
        latency_ms=round(latency_ms, 2),
    )


class LLMOOptimizeRequest(BaseModel):
    text: str = Field(..., description="最適化対象テキスト")
    query: str = Field("", description="検索クエリ (指定時は query 関連最適化も実行)")
    target_score: float = Field(
        0.75, ge=0.1, le=1.0, description="目標 llmo_total スコア"
    )
    max_iterations: int = Field(3, ge=1, le=5, description="最大最適化イテレーション数")


class LLMOOptimizeResponse(BaseModel):
    original_text: str
    optimized_text: str
    original_llmo_total: float
    optimized_llmo_total: float
    improvement_pct: float
    changes_applied: list[str]
    iterations: int
    target_achieved: bool
    latency_ms: float


@app.post(
    "/v1/llmo/optimize",
    response_model=LLMOOptimizeResponse,
    tags=["seo"],
    summary="LLMO テキスト自動最適化",
    description=(
        "テキストをルールベースで自動最適化し LLMO スコアを向上させる。"
        "`target_score` 到達まで最大 `max_iterations` 回変換を繰り返す。"
        "外部モデル不要 (pure Python)。"
    ),
)
def llmo_optimize(req: LLMOOptimizeRequest):
    """LLMO テキスト自動最適化エンドポイント。

    entity_density / answer_directness / citability / query_relevance の各軸を
    ルールベース変換で改善し、最適化前後のテキストとスコアを返す。
    """
    t0 = time.perf_counter()
    result = _llmo_optimizer.optimize(
        req.text,
        query=req.query,
        target_score=req.target_score,
        max_iterations=req.max_iterations,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    return LLMOOptimizeResponse(
        original_text=result.original_text,
        optimized_text=result.optimized_text,
        original_llmo_total=result.original_score.llmo_total,
        optimized_llmo_total=result.optimized_score.llmo_total,
        improvement_pct=result.improvement_pct,
        changes_applied=result.changes_applied,
        iterations=result.iterations,
        target_achieved=result.optimized_score.llmo_total >= req.target_score,
        latency_ms=round(latency_ms, 2),
    )


class LLMOQueryScoreRequest(BaseModel):
    text: str = Field(..., description="スコアリング対象テキスト")
    query: str = Field(..., description="検索クエリ")


class LLMOQueryScoreResponse(BaseModel):
    entity_density: float
    answer_directness: float
    citability: float
    llmo_total: float
    query_relevance: float
    intent_type: str
    entities: list[str]
    word_count: int
    latency_ms: float


@app.post(
    "/v1/llmo/score",
    response_model=LLMOQueryScoreResponse,
    tags=["seo"],
    summary="クエリ対応 LLMO スコアリング",
    description=(
        "クエリを考慮した LLMO スコアを計算する。"
        "3 軸スコアに加え `query_relevance` (TF-IDF コサイン類似度) と "
        "`intent_type` (informational / navigational / transactional / commercial) を返す。"
    ),
)
def llmo_query_score(req: LLMOQueryScoreRequest):
    """クエリ対応 LLMO スコアリングエンドポイント。"""
    t0 = time.perf_counter()
    score = _llmo_scorer.score_with_query(req.text, req.query)
    latency_ms = (time.perf_counter() - t0) * 1000

    return LLMOQueryScoreResponse(
        entity_density=score.entity_density,
        answer_directness=score.answer_directness,
        citability=score.citability,
        llmo_total=score.llmo_total,
        query_relevance=score.query_relevance,
        intent_type=score.intent_type,
        entities=score.entities[:10],
        word_count=score.word_count,
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# Extended Thinking エンドポイント
# ---------------------------------------------------------------------------


class ThinkingRequest(BaseModel):
    prompt: str = Field(..., description="思考対象プロンプト")
    think_loops: int = Field(8, ge=1, le=16, description="思考フェーズのループ数")
    answer_loops: int = Field(4, ge=1, le=16, description="回答フェーズのループ数")
    max_new_tokens: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.9, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    include_loop_states: bool = Field(
        False, description="ループごとの内部状態メタ情報を返すか"
    )


class ThinkingResponse(BaseModel):
    thinking: str = Field(description="思考トレース (<thinking>...</thinking>)")
    answer: str = Field(description="最終回答テキスト")
    loops_used: int
    think_loops: int
    answer_loops: int
    loop_states: list[dict] = Field(default_factory=list)
    latency_ms: float


@app.post(
    "/v1/thinking",
    response_model=ThinkingResponse,
    tags=["thinking"],
    summary="Extended Thinking (思考トレース付き生成)",
    description="内部ループの状態変化を `<thinking>` ブロックとして外部公開する。Opus 4.8 対抗機能。",
)
def extended_thinking(req: ThinkingRequest):
    """Extended Thinking — 思考トレース付きで回答を生成する。

    ClaudeMythos の Extended Thinking 相当機能。
    think_loops 回のループで深い推論を行い、内部状態変化を
    <thinking>...</thinking> ブロックとして外部公開する。
    その後 answer_loops 回のループで最終回答を生成する。
    """
    from open_mythos.thinking import ThinkingEngine

    engine = ThinkingEngine(state.model, device=str(state.device))

    result = engine.generate_with_thinking(
        prompt=req.prompt,
        think_loops=req.think_loops,
        answer_loops=req.answer_loops,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        vocab_size=state.tokenizer.vocab_size,
    )

    return ThinkingResponse(
        thinking=result.thinking,
        answer=result.answer,
        loops_used=result.loops_used,
        think_loops=result.think_loops,
        answer_loops=result.answer_loops,
        loop_states=result.loop_states if req.include_loop_states else [],
        latency_ms=result.latency_ms,
    )


# ---------------------------------------------------------------------------
# Tool Use / Function Calling エンドポイント
# ---------------------------------------------------------------------------

from open_mythos.tools import (  # noqa: E402
    ToolRegistry as _ToolRegistry,
    ToolCall,
    execute_tool_calls,
)

# デフォルトのマーケ特化ツールレジストリ (起動時1回構築)
_default_tool_registry = _ToolRegistry.default()


class ToolCallRequest(BaseModel):
    name: str = Field(..., description="呼び出すツール名")
    arguments: dict = Field(default_factory=dict, description="ツール引数")
    call_id: str = Field("", description="呼び出しID (オプション)")


class ToolCallResponse(BaseModel):
    tool_name: str
    content: dict = Field(default_factory=dict)
    success: bool
    error: str = ""
    latency_ms: float


class ToolsBatchRequest(BaseModel):
    calls: list[ToolCallRequest] = Field(..., min_length=1, max_length=16)


class ToolsBatchResponse(BaseModel):
    results: list[ToolCallResponse]
    total_latency_ms: float


class ToolsListResponse(BaseModel):
    tools: list[dict]
    n_tools: int


@app.get(
    "/v1/tools",
    response_model=ToolsListResponse,
    tags=["tools"],
    summary="利用可能なツール一覧",
    description="登録済みツールを OpenAI 互換 function schema 形式で返す。",
)
def list_tools():
    """利用可能なツール一覧を OpenAI 互換 schema で返す。"""
    return ToolsListResponse(
        tools=_default_tool_registry.to_openai_tools(),
        n_tools=len(_default_tool_registry),
    )


@app.post(
    "/v1/tools/call",
    response_model=ToolCallResponse,
    tags=["tools"],
    summary="ツール呼び出し",
    description="search_competitor / calculate_roi / fetch_trend / score_content 等を実行する。",
)
def call_tool(req: ToolCallRequest):
    """単一のツールを呼び出す。

    マーケ特化ツール: search_competitor / calculate_roi / fetch_trend / score_content
    """
    tc = ToolCall(name=req.name, arguments=req.arguments, call_id=req.call_id)
    result = _default_tool_registry.call(tc)

    content = {}
    if result.content is not None:
        if isinstance(result.content, dict):
            content = result.content
        else:
            content = {"result": result.content}

    return ToolCallResponse(
        tool_name=result.tool_name,
        content=content,
        success=result.success,
        error=result.error,
        latency_ms=result.latency_ms,
    )


@app.post(
    "/v1/tools/batch",
    response_model=ToolsBatchResponse,
    tags=["tools"],
    summary="ツール一括呼び出し",
    description="最大16件のツール呼び出しを一括実行する。",
)
def call_tools_batch(req: ToolsBatchRequest):
    """複数ツールを一括呼び出しする (最大16件)。"""
    t_start = time.perf_counter()
    tool_calls = [
        ToolCall(name=r.name, arguments=r.arguments, call_id=r.call_id)
        for r in req.calls
    ]
    results = execute_tool_calls(tool_calls, _default_tool_registry)
    total_ms = (time.perf_counter() - t_start) * 1000

    responses = []
    for r in results:
        content = {}
        if r.content is not None:
            content = (
                r.content if isinstance(r.content, dict) else {"result": r.content}
            )
        responses.append(
            ToolCallResponse(
                tool_name=r.tool_name,
                content=content,
                success=r.success,
                error=r.error,
                latency_ms=r.latency_ms,
            )
        )

    return ToolsBatchResponse(
        results=responses,
        total_latency_ms=round(total_ms, 2),
    )


# ---------------------------------------------------------------------------
# RAG エンドポイント
# ---------------------------------------------------------------------------

from open_mythos.rag import RAGPipeline as _RAGPipeline  # noqa: E402

# グローバル RAG パイプライン (起動後にドキュメントを追加して使う)
_rag_pipeline: Optional[_RAGPipeline] = None


def _get_rag() -> _RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = _RAGPipeline(state.model, device=str(state.device))
    return _rag_pipeline


class RAGIndexRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=256, description="インデックスするテキスト群"
    )
    doc_ids: list[str] = Field(
        default_factory=list, description="ドキュメントID (省略可)"
    )
    metadatas: list[dict] = Field(
        default_factory=list, description="メタデータ (省略可)"
    )


class RAGIndexResponse(BaseModel):
    added: int
    total_docs: int


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="検索クエリ")
    top_k: int = Field(3, ge=1, le=10, description="取得するドキュメント数")
    max_new_tokens: int = Field(128, ge=1, le=512)
    loops: int = Field(DEFAULT_LOOPS, ge=1, le=16)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    generate: bool = Field(True, description="False の場合は検索のみ (生成なし)")


class RAGDocResult(BaseModel):
    doc_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    query: str
    answer: str = ""
    retrieved_docs: list[RAGDocResult]
    n_docs_in_store: int
    latency_ms: float


@app.post(
    "/v1/rag/index",
    response_model=RAGIndexResponse,
    tags=["rag"],
    summary="RAG インデックス追加",
    description="ドキュメントをベクトルストアに追加する。numpy ベース (FAISS オプション対応)。",
)
def rag_index(req: RAGIndexRequest):
    """ドキュメントをRAGインデックスに追加する。"""
    rag = _get_rag()
    n = rag.add_documents(
        texts=req.texts,
        doc_ids=req.doc_ids or None,
        metadatas=req.metadatas or None,
    )
    return RAGIndexResponse(added=n, total_docs=rag.n_docs())


@app.post(
    "/v1/rag",
    response_model=RAGQueryResponse,
    tags=["rag"],
    summary="RAG 検索 + 生成",
    description="クエリに関連するドキュメントを検索し、コンテキストとして生成を行う。",
)
def rag_query(req: RAGQueryRequest):
    """RAG検索 + 生成を実行する。

    `generate=False` の場合は検索結果のみ返す。
    `generate=True` (デフォルト) の場合は検索結果をコンテキストに組み込んで生成する。
    """
    rag = _get_rag()
    loops = min(req.loops, MAX_LOOPS)

    if req.generate:
        result = rag.generate_with_context(
            query=req.query,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            loops=loops,
        )
        return RAGQueryResponse(
            query=req.query,
            answer=result.answer,
            retrieved_docs=[
                RAGDocResult(
                    doc_id=d.doc_id,
                    text=d.text,
                    score=round(d.score, 4),
                    metadata=d.metadata,
                )
                for d in result.retrieved_docs
            ],
            n_docs_in_store=result.n_docs_in_store,
            latency_ms=result.latency_ms,
        )
    else:
        # 検索のみ
        t0 = time.perf_counter()
        docs = rag.retrieve(req.query, top_k=req.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        return RAGQueryResponse(
            query=req.query,
            answer="",
            retrieved_docs=[
                RAGDocResult(
                    doc_id=d.doc_id,
                    text=d.text,
                    score=round(d.score, 4),
                    metadata=d.metadata,
                )
                for d in docs
            ],
            n_docs_in_store=rag.n_docs(),
            latency_ms=round(latency_ms, 2),
        )


# ---------------------------------------------------------------------------
# ReAct エージェント エンドポイント
# ---------------------------------------------------------------------------

from open_mythos.react import ReActAgent as _ReActAgent  # noqa: E402

# デフォルトエージェント (ツールレジストリ付き)
_default_agent: Optional[_ReActAgent] = None


def _get_agent() -> _ReActAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = _ReActAgent(
            model=state.model,
            registry=_default_tool_registry,
            device=str(state.device),
        )
    return _default_agent


class AgentRunRequest(BaseModel):
    task: str = Field(..., description="エージェントに解決させるタスク")
    system_prompt: str = Field("", description="カスタムシステムプロンプト (省略可)")
    max_iterations: int = Field(6, ge=1, le=12, description="最大イテレーション数")
    max_new_tokens: int = Field(
        128, ge=1, le=512, description="各ステップの最大生成トークン数"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    loops: int = Field(DEFAULT_LOOPS, ge=1, le=16)


class AgentStepResponse(BaseModel):
    step_type: str
    content: str
    iteration: int
    latency_ms: float
    tool_name: str = ""
    tool_success: bool = True


class AgentRunResponse(BaseModel):
    task: str
    final_answer: str
    steps: list[AgentStepResponse]
    iterations_used: int
    n_tool_calls: int
    stopped_reason: str
    total_latency_ms: float


@app.post(
    "/v1/agent/run",
    response_model=AgentRunResponse,
    tags=["agent"],
    summary="ReAct エージェントループ実行",
    description="Think→Act→Observe サイクルで複数ステップのタスクを自律実行する。Tool Use 連携対応。",
)
def react_agent_run(req: AgentRunRequest):
    """ReAct エージェントループでタスクを解決する。

    Tool Use (Sprint 11) を活用して複数ステップのタスクを自律実行する。
    """
    agent = _get_agent()
    agent.max_iterations = req.max_iterations
    agent.max_new_tokens = req.max_new_tokens
    agent.temperature = req.temperature
    agent.loops = min(req.loops, MAX_LOOPS)

    result = agent.run(task=req.task, system_prompt=req.system_prompt)

    step_responses = []
    for step in result.steps:
        tool_name = step.tool_call.name if step.tool_call else ""
        tool_success = step.tool_result.success if step.tool_result else True
        step_responses.append(
            AgentStepResponse(
                step_type=step.step_type,
                content=step.content[:500],  # 最大500文字
                iteration=step.iteration,
                latency_ms=step.latency_ms,
                tool_name=tool_name,
                tool_success=tool_success,
            )
        )

    return AgentRunResponse(
        task=result.task,
        final_answer=result.final_answer,
        steps=step_responses,
        iterations_used=result.iterations_used,
        n_tool_calls=result.n_tool_calls,
        stopped_reason=result.stopped_reason,
        total_latency_ms=result.total_latency_ms,
    )


# ---------------------------------------------------------------------------
# Sessions / Conversation Memory エンドポイント
# ---------------------------------------------------------------------------

from open_mythos.conversation import SessionStore as _SessionStore  # noqa: E402

_session_store = _SessionStore(max_sessions=500, max_turns=20, max_chars=4000)


class SessionCreateRequest(BaseModel):
    session_id: str = Field("", description="セッション ID (省略時は自動生成)")
    system_msg: str = Field("", description="システムメッセージ")


class SessionCreateResponse(BaseModel):
    session_id: str
    created: bool


class TurnAddRequest(BaseModel):
    role: str = Field(..., description="'user' または 'assistant'")
    content: str = Field(..., description="ターンの内容")


class SessionStatsResponse(BaseModel):
    session_id: str
    n_turns: int
    total_chars: int
    has_summary: bool
    summary_turns: int


class SessionContextResponse(BaseModel):
    session_id: str
    context: str
    n_turns: int


@app.post(
    "/v1/sessions",
    response_model=SessionCreateResponse,
    tags=["sessions"],
    summary="会話セッション作成",
    description="新しい ConversationMemory セッションを作成する。`session_id` 省略で UUID 自動生成。",
)
def create_session(req: SessionCreateRequest):
    """新しい会話セッションを作成する。"""
    sid = req.session_id if req.session_id else None
    already_exists = sid is not None and _session_store.get(sid) is not None
    new_sid, _ = _session_store.get_or_create(session_id=sid, system_msg=req.system_msg)
    return SessionCreateResponse(session_id=new_sid, created=not already_exists)


@app.get(
    "/v1/sessions/{session_id}",
    response_model=SessionStatsResponse,
    tags=["sessions"],
    summary="セッション情報取得",
)
def get_session(session_id: str):
    """セッションの統計情報を取得する。"""
    mem = _session_store.get(session_id)
    if mem is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    s = mem.stats()
    return SessionStatsResponse(
        session_id=session_id,
        n_turns=s["n_turns"],
        total_chars=s["total_chars"],
        has_summary=s["has_summary"],
        summary_turns=s["summary_turns"],
    )


@app.delete(
    "/v1/sessions/{session_id}",
    tags=["sessions"],
    summary="セッション削除",
)
def delete_session(session_id: str):
    """セッションを削除する。"""
    deleted = _session_store.delete(session_id)
    return {"deleted": deleted, "session_id": session_id}


@app.post(
    "/v1/sessions/{session_id}/turns",
    response_model=SessionStatsResponse,
    tags=["sessions"],
    summary="ターン追加",
    description="セッションに user / assistant ターンを追加する。自動圧縮対応。",
)
def add_turn(session_id: str, req: TurnAddRequest):
    """セッションにターンを追加する。"""
    mem = _session_store.get(session_id)
    if mem is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    mem.add_turn(req.role, req.content)
    s = mem.stats()
    return SessionStatsResponse(
        session_id=session_id,
        n_turns=s["n_turns"],
        total_chars=s["total_chars"],
        has_summary=s["has_summary"],
        summary_turns=s["summary_turns"],
    )


@app.get(
    "/v1/sessions/{session_id}/context",
    response_model=SessionContextResponse,
    tags=["sessions"],
    summary="セッションコンテキスト取得",
    description="モデルへの入力文字列形式でセッション履歴を返す。",
)
def get_session_context(session_id: str):
    """セッションのコンテキスト文字列を取得する。"""
    mem = _session_store.get(session_id)
    if mem is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return SessionContextResponse(
        session_id=session_id,
        context=mem.to_context_string(),
        n_turns=mem.n_turns,
    )


# ===========================================================================
# Sprint 34: MistakeGuardMiddleware — 全エンドポイント透過チェック
# ===========================================================================

from open_mythos.error_memory import (  # noqa: E402
    GuardMiddlewareConfig as _GuardMiddlewareConfig,
    MistakeGuardMiddleware as _MistakeGuardMiddleware,
)

# 環境変数でガード有効/無効を制御 (デフォルト: 有効)
_GUARD_ENABLED:  bool = os.environ.get("MISTAKE_GUARD_ENABLED", "true").lower() != "false"
_GUARD_SEV:      str  = os.environ.get("MISTAKE_GUARD_SEVERITY", "medium")
_GUARD_REFRESH:  int  = int(os.environ.get("MISTAKE_GUARD_REFRESH_INTERVAL", "100"))

_guard_config = _GuardMiddlewareConfig(
    enabled=_GUARD_ENABLED,
    auto_record_blocked=True,
    severity_threshold=_GUARD_SEV,
    refresh_interval=_GUARD_REFRESH,
)

# グローバルミドルウェアインスタンス (mistake_store と共有)
_guard_middleware: Optional[_MistakeGuardMiddleware] = None


def _get_guard_middleware() -> _MistakeGuardMiddleware:
    """MistakeGuardMiddleware シングルトンを返す (mistake_store と共有)。"""
    global _guard_middleware
    if _guard_middleware is None:
        _guard_middleware = _MistakeGuardMiddleware(
            store=_get_mistake_store(),
            config=_guard_config,
        )
    return _guard_middleware


class _MistakeGuardHTTPMiddleware(BaseHTTPMiddleware):
    """
    FastAPI HTTP ミドルウェア — 全 POST / PUT リクエストのボディを透過チェックする。

    ブロック時: 422 + JSON {"detail": ..., "block_reason": ...} を返す。
    ヘッダー:
        X-Guard-Blocked: "true" / "false"
        X-Guard-Rule-Id: matched rule_id (ブロック時のみ)
    """

    async def dispatch(self, request: Request, call_next):
        # /health はスキップ (startup 前に呼ばれる可能性あり)
        if request.url.path == "/health":
            return await call_next(request)

        try:
            gm = _get_guard_middleware()
        except Exception:
            return await call_next(request)

        if not gm.is_enabled:
            return await call_next(request)

        # POST / PUT のリクエストボディをチェック
        if request.method in ("POST", "PUT"):
            try:
                body_bytes = await request.body()
                if body_bytes:
                    body_text = body_bytes.decode("utf-8", errors="replace")
                    guard_result = gm.process(body_text)
                    if guard_result.blocked:
                        return Response(
                            content=_json_mod.dumps(
                                {
                                    "detail": "Request blocked by MistakeGuard",
                                    "block_reason": guard_result.block_reason,
                                },
                                ensure_ascii=False,
                            ),
                            status_code=422,
                            media_type="application/json",
                            headers={
                                "X-Guard-Blocked": "true",
                                "X-Guard-Rule-Id": (
                                    guard_result.matched_rule.rule_id
                                    if guard_result.matched_rule else ""
                                ),
                            },
                        )
            except Exception:
                pass  # ボディ読み取り失敗時はスルー

        response = await call_next(request)
        response.headers["X-Guard-Blocked"] = "false"
        return response


# ミドルウェアを登録 (CORSMiddleware より前に追加)
app.add_middleware(_MistakeGuardHTTPMiddleware)


# ---------------------------------------------------------------------------
# Guard エンドポイント
# ---------------------------------------------------------------------------


@app.get(
    "/v1/guard/stats",
    tags=["guard"],
    summary="MistakeGuard 統計 (Sprint 34)",
    description="ガードミドルウェアの統計 (総リクエスト数・ブロック数・ブロック率・アクティブルール数) を返す。",
    dependencies=[Depends(verify_api_key)],
)
def guard_stats():
    """MistakeGuardMiddleware の統計情報を返す。"""
    gm = _get_guard_middleware()
    return gm.stats()


@app.post(
    "/v1/guard/refresh",
    tags=["guard"],
    summary="MistakeGuard ルール再抽出 (Sprint 34)",
    description="ErrorMemoryStore の最新データからルールを再抽出する。",
    dependencies=[Depends(verify_api_key)],
)
def guard_refresh():
    """ガードルールを手動で再抽出する。"""
    gm = _get_guard_middleware()
    n_rules = gm.refresh()
    return {"refreshed": True, "rule_count": n_rules}


# ---------------------------------------------------------------------------
# Sprint 36.5: Core Web Vitals シミュレーター
# ---------------------------------------------------------------------------


class CWVSimulateRequest(BaseModel):
    content: str = Field(..., description="ページ本文テキスト")
    n_images: int = Field(0, ge=0, description="ページ内の画像数")
    server_response_ms: float = Field(200.0, ge=0.0, description="サーバー初期応答時間 (ms)")


@app.post(
    "/v1/cwv/simulate",
    tags=["seo"],
    summary="Core Web Vitals シミュレーション (Sprint 36)",
    description=(
        "コンテンツ長・画像数・サーバー応答時間から LCP / CLS / FID を疑似推定する。"
        "実ブラウザ計測の代替ではなく、コンテンツ改善の方向性確認に使用する。"
    ),
)
def cwv_simulate(req: CWVSimulateRequest):
    from open_mythos.structured import CoreWebVitalsSimulator
    sim = CoreWebVitalsSimulator()
    result = sim.simulate(
        content=req.content,
        n_images=req.n_images,
        server_response_ms=req.server_response_ms,
    )
    return result.to_dict()



# ---------------------------------------------------------------------------
# Sprint 71〜77 — 都市地図ドメイン (serve/routers/map.py に分割)
# ---------------------------------------------------------------------------

from serve.routers.map import router as _map_router  # noqa: E402
from serve.routers.growing_ai import router as _growing_ai_router  # noqa: E402
from serve.routers.skills_integrations import router as _skills_router  # noqa: E402
from serve.routers.llm_services import router as _llm_services_router  # noqa: E402
from serve.routers.ads import router as _ads_router  # noqa: E402
from serve.routers.security import router as _security_router  # noqa: E402

app.include_router(_map_router)
app.include_router(_growing_ai_router)
app.include_router(_skills_router)
app.include_router(_llm_services_router)
app.include_router(_ads_router)
app.include_router(_security_router)

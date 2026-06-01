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

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Iterator, Literal, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.agents import MythosAgent, OpenMythosLLM
from serve.auth import RateLimitMiddleware, verify_api_key

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIM = int(os.getenv("MODEL_DIM", "256"))
MODEL_ATTN = os.getenv("MODEL_ATTN", "gqa")
MODEL_CHECKPOINT = os.getenv(
    "MODEL_CHECKPOINT", ""
)  # path to .pt file; empty = random weights
DEVICE = os.getenv("DEVICE", "cpu")
DEFAULT_LOOPS = int(os.getenv("DEFAULT_LOOPS", "4"))
MAX_LOOPS = int(os.getenv("MAX_LOOPS", "16"))
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


class _State:
    model: OpenMythos
    tokenizer: AutoTokenizer
    device: torch.device
    n_params: int
    llm: OpenMythosLLM
    # session_id -> MythosAgent（会話履歴管理）
    agents: dict[str, MythosAgent]


state = _State()
state.agents = {}


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
    version="0.22.0",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
    openapi_tags=[
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
    ],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

TaskType = Literal[
    "ad_performance",  # 広告クリエイティブ効果予測
    "content_quality",  # SEO / LLMO コンテンツ品質スコアリング
    "persona_segment",  # ユーザーペルソナ分類
    "market_research",  # 市場調査レポート要約
    "identity_verify",  # 本人確認（リアルタイム）
    "fraud_detect",  # 詐欺検知（高精度）
    "seo_content",  # SEO記事・メタタグ生成
    "llmo_optimize",  # LLM検索最適化（LLMO）コンテンツ生成
    "ad_copy",  # 広告コピー生成（マーケティング）
    "persona_message",  # ペルソナ別メッセージ生成
    "market_summary",  # 市場調査サマリー生成
    "general",  # 汎用
]


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

TASK_LOOPS: dict[str, int] = {
    "ad_performance": 2,  # リアルタイム入稿審査: 速度優先
    "content_quality": 6,  # SEO品質スコア: 精度と速度のバランス
    "persona_segment": 4,  # ペルソナ分類: 中程度
    "market_research": 4,  # 市場調査要約: 中程度
    "identity_verify": 4,  # 本人確認: リアルタイム
    "fraud_detect": 12,  # 詐欺検知: 精度最優先
    "seo_content": 6,  # SEO記事生成: 品質重視
    "llmo_optimize": 8,  # LLMO最適化: 深い推論で構造化
    "ad_copy": 2,  # 広告コピー: 速度優先
    "persona_message": 4,  # ペルソナ別メッセージ: 中程度
    "market_summary": 6,  # 市場調査サマリー: 品質重視
    "general": 4,  # DEFAULT_LOOPS
}


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


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


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


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


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


def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    """Convert chat messages to a flat prompt string."""
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"[System]: {m.content}")
        elif m.role == "user":
            parts.append(f"[User]: {m.content}")
        elif m.role == "assistant":
            parts.append(f"[Assistant]: {m.content}")
    parts.append("[Assistant]:")
    return "\n".join(parts)


@app.post(
    "/v1/chat/completions",
    tags=["chat"],
    summary="OpenAI 互換チャット推論",
    description="OpenAI `/v1/chat/completions` 互換。`stream=true` で SSE ストリーミング対応。",
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
    enc = state.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = enc["input_ids"].to(state.device)
    prompt_tokens = input_ids.shape[1]

    if req.stream:
        # --- SSE ストリーミング ---
        def _event_stream():
            import json as _json

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            generated: list[int] = []
            cur_ids = input_ids

            for _ in range(req.max_tokens):
                with torch.no_grad():
                    logits = state.model(cur_ids, n_loops=loops)
                next_logits = logits[0, -1, :] / max(req.temperature, 1e-6)
                if req.top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > req.top_p
                    sorted_logits[mask] = float("-inf")
                    next_logits = torch.full_like(next_logits, float("-inf")).scatter(
                        0, sorted_idx, sorted_logits
                    )
                probs = torch.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())
                generated.append(next_token)
                token_text = state.tokenizer.decode(
                    [next_token], skip_special_tokens=True
                )
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[next_token]], device=state.device)], dim=1
                )

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
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
                    break

            done_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            }
            yield f"data: {_json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # --- 非ストリーミング: 一括生成 ---
    generated_ids: list[int] = []
    cur_ids = input_ids

    with torch.no_grad():
        for _ in range(req.max_tokens):
            logits = state.model(cur_ids, n_loops=loops)
            next_logits = logits[0, -1, :] / max(req.temperature, 1e-6)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probs, 1).item())
            generated_ids.append(next_token)
            cur_ids = torch.cat(
                [cur_ids, torch.tensor([[next_token]], device=state.device)], dim=1
            )
            if next_token == state.tokenizer.eos_token_id:
                break

    completion_text = state.tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_tokens = len(generated_ids)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model="openmythos",
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=completion_text),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
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
from typing import Dict, List  # noqa: E402

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


@app.post(
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
        # 既存 ML スタブ: user_id ハッシュから決定論的スコアを生成
        h = int(hashlib.md5(req.user_id.encode()).hexdigest(), 16)
        score = 0.5 + (h % 500) / 1000.0  # 0.5–1.0 の決定論的スコア
        label = 1 if score >= 0.5 else 0
        model_id = "existing-ml-stub"

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


@app.post(
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


@app.post(
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


@app.post(
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


@app.get(
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

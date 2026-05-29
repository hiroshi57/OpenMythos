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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.agents import MythosAgent, OpenMythosLLM

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
    title="OpenMythos Inference API",
    description="Recurrent-Depth Transformer inference with variable loop depth.",
    version="0.1.0",
    lifespan=lifespan,
)

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
    "fraud_detect",     # 詐欺検知（高精度）
    "seo_content",      # SEO記事・メタタグ生成
    "llmo_optimize",    # LLM検索最適化（LLMO）コンテンツ生成
    "ad_copy",          # 広告コピー生成（マーケティング）
    "persona_message",  # ペルソナ別メッセージ生成
    "market_summary",   # 市場調査サマリー生成
    "general",          # 汎用
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
    task: TaskType = Field("general", description="タスク種別（システムプロンプトを自動設定）")
    max_new_tokens: int = Field(256, ge=1, le=1024, description="最大生成トークン数")
    temperature: float = Field(1.0, ge=0.01, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=500)
    n_loops: Optional[int] = Field(None, description="ループ深度オーバーライド")
    system_prompt: Optional[str] = Field(None, description="カスタムシステムプロンプト（指定時はtaskより優先）")


class GenerateResponse(BaseModel):
    text: str
    task: str
    prompt_len: int
    generated_tokens: int
    latency_ms: float


class AgentRequest(BaseModel):
    task_input: str = Field(..., description="エージェントへの入力テキスト")
    session_id: Optional[str] = Field(None, description="会話セッションID（省略時は新規作成）")
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
    "ad_performance":  2,   # リアルタイム入稿審査: 速度優先
    "content_quality": 6,   # SEO品質スコア: 精度と速度のバランス
    "persona_segment": 4,   # ペルソナ分類: 中程度
    "market_research": 4,   # 市場調査要約: 中程度
    "identity_verify": 4,   # 本人確認: リアルタイム
    "fraud_detect":    12,  # 詐欺検知: 精度最優先
    "seo_content":     6,   # SEO記事生成: 品質重視
    "llmo_optimize":   8,   # LLMO最適化: 深い推論で構造化
    "ad_copy":         2,   # 広告コピー: 速度優先
    "persona_message": 4,   # ペルソナ別メッセージ: 中程度
    "market_summary":  6,   # 市場調査サマリー: 品質重視
    "general":         4,   # DEFAULT_LOOPS
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
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
            "POST /infer":           "スコアリング・分類",
            "POST /infer/raw":       "rawロジット取得",
            "POST /generate":        "テキスト生成（SEO/LLMO/広告コピー等）",
            "GET  /generate/stream": "ストリーミング生成（SSE）",
            "POST /agent":           "多ターン対話エージェント",
            "DELETE /agent/{id}":    "セッションリセット",
        },
    }


@app.post("/infer", response_model=InferResponse)
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


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """テキスト生成エンドポイント。SEO記事・広告コピー・LLMOコンテンツ等を生成。"""
    sys_prompt = req.system_prompt or _TASK_SYSTEM_PROMPTS.get(req.task, _TASK_SYSTEM_PROMPTS["general"])
    full_prompt = f"{sys_prompt}\n\n{req.prompt}" if sys_prompt else req.prompt

    n_loops = req.n_loops or TASK_LOOPS.get(req.task, DEFAULT_LOOPS)

    # 入力トークン数を計測
    enc = state.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_len = enc["input_ids"].shape[1]

    t0 = time.perf_counter()
    state.llm.max_new_tokens = req.max_new_tokens
    state.llm.temperature = req.temperature
    state.llm.top_p = req.top_p
    state.llm.top_k = req.top_k

    text = state.llm.run(full_prompt)
    latency_ms = (time.perf_counter() - t0) * 1000

    # 生成テキストのトークン数を簡易計測
    gen_enc = state.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    generated_tokens = gen_enc["input_ids"].shape[1]

    return GenerateResponse(
        text=text,
        task=req.task,
        prompt_len=prompt_len,
        generated_tokens=generated_tokens,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/generate/stream")
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


@app.post("/agent", response_model=AgentResponse)
def agent_run(req: AgentRequest):
    """MythosAgent エンドポイント。session_id で会話履歴を維持した多ターン対話。

    ユースケース例:
      - SEO相談チャット: task=seo_content で記事構成をやり取り
      - マーケティング戦略ブレスト: task=market_summary で複数ターン議論
      - 広告コピー反復改善: task=ad_copy でフィードバックを受けてコピーを磨く
    """
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in state.agents:
        sys_prompt = req.system_prompt or _TASK_SYSTEM_PROMPTS.get(req.task, _TASK_SYSTEM_PROMPTS["general"])
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
    return {"ok": True, "session_id": session_id, "message": "会話履歴をリセットしました"}


@app.post("/infer/raw", response_model=RawInferResponse)
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


@app.post("/v1/chat/completions")
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
                    next_logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)
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


@app.post("/v1/batch", response_model=BatchResponse)
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

from open_mythos.llmo import LLMOScorer  # noqa: E402

_llmo_scorer = LLMOScorer()


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


@app.post("/v1/seo/score", response_model=SEOScoreResponse)
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


@app.post("/v1/seo/generate", response_model=SEOGenerateResponse)
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
    prompt_tokens = input_ids.shape[1]

    t0 = time.perf_counter()
    generated_ids: list[int] = []
    cur_ids = input_ids

    with torch.no_grad():
        for _ in range(req.max_new_tokens):
            logits = state.model(cur_ids, n_loops=loops)
            next_logits = logits[0, -1, :] / max(req.temperature, 1e-6)
            if req.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > req.top_p
                sorted_logits[mask] = float("-inf")
                next_logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)
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


@app.post("/v1/thinking", response_model=ThinkingResponse)
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

from open_mythos.tools import ToolRegistry as _ToolRegistry, ToolCall, execute_tool_calls  # noqa: E402

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


@app.get("/v1/tools", response_model=ToolsListResponse)
def list_tools():
    """利用可能なツール一覧を OpenAI 互換 schema で返す。"""
    return ToolsListResponse(
        tools=_default_tool_registry.to_openai_tools(),
        n_tools=len(_default_tool_registry),
    )


@app.post("/v1/tools/call", response_model=ToolCallResponse)
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


@app.post("/v1/tools/batch", response_model=ToolsBatchResponse)
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
            content = r.content if isinstance(r.content, dict) else {"result": r.content}
        responses.append(ToolCallResponse(
            tool_name=r.tool_name,
            content=content,
            success=r.success,
            error=r.error,
            latency_ms=r.latency_ms,
        ))

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
    texts: list[str] = Field(..., min_length=1, max_length=256, description="インデックスするテキスト群")
    doc_ids: list[str] = Field(default_factory=list, description="ドキュメントID (省略可)")
    metadatas: list[dict] = Field(default_factory=list, description="メタデータ (省略可)")


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


@app.post("/v1/rag/index", response_model=RAGIndexResponse)
def rag_index(req: RAGIndexRequest):
    """ドキュメントをRAGインデックスに追加する。"""
    rag = _get_rag()
    n = rag.add_documents(
        texts=req.texts,
        doc_ids=req.doc_ids or None,
        metadatas=req.metadatas or None,
    )
    return RAGIndexResponse(added=n, total_docs=rag.n_docs())


@app.post("/v1/rag", response_model=RAGQueryResponse)
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

from open_mythos.react import ReActAgent as _ReActAgent, AgentStep as _AgentStep  # noqa: E402

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
    max_new_tokens: int = Field(128, ge=1, le=512, description="各ステップの最大生成トークン数")
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


@app.post("/v1/agent/run", response_model=AgentRunResponse)
def agent_run(req: AgentRunRequest):
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
        step_responses.append(AgentStepResponse(
            step_type=step.step_type,
            content=step.content[:500],  # 最大500文字
            iteration=step.iteration,
            latency_ms=step.latency_ms,
            tool_name=tool_name,
            tool_success=tool_success,
        ))

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


@app.post("/v1/sessions", response_model=SessionCreateResponse)
def create_session(req: SessionCreateRequest):
    """新しい会話セッションを作成する。"""
    sid = req.session_id if req.session_id else None
    already_exists = sid is not None and _session_store.get(sid) is not None
    new_sid, _ = _session_store.get_or_create(session_id=sid, system_msg=req.system_msg)
    return SessionCreateResponse(session_id=new_sid, created=not already_exists)


@app.get("/v1/sessions/{session_id}", response_model=SessionStatsResponse)
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


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str):
    """セッションを削除する。"""
    deleted = _session_store.delete(session_id)
    return {"deleted": deleted, "session_id": session_id}


@app.post("/v1/sessions/{session_id}/turns", response_model=SessionStatsResponse)
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


@app.get("/v1/sessions/{session_id}/context", response_model=SessionContextResponse)
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

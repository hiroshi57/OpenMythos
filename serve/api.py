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
import torch.nn.functional as F
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

MODEL_DIM        = int(os.getenv("MODEL_DIM", "256"))
MODEL_ATTN       = os.getenv("MODEL_ATTN", "gqa")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "")   # path to .pt file; empty = random weights
DEVICE           = os.getenv("DEVICE", "cpu")
DEFAULT_LOOPS    = int(os.getenv("DEFAULT_LOOPS", "4"))
MAX_LOOPS        = int(os.getenv("MAX_LOOPS", "16"))
TOKENIZER_NAME   = os.getenv("TOKENIZER_NAME", "gpt2")


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
    "ad_performance",   # 広告クリエイティブ効果予測
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
    score: float = Field(description="Confidence score in [0, 1] (higher = more confident positive)")
    label: int   = Field(description="Predicted label: 1 = positive / verified, 0 = negative / flagged")
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
    "general":         4,  # DEFAULT_LOOPS
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
    last_logit = logits[0, -1, :]            # (vocab,)
    probs = torch.softmax(last_logit, dim=-1)
    score = float(probs.max())               # max-prob as confidence
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

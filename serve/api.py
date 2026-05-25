#!/usr/bin/env python3
"""
OpenMythos inference API — FastAPI server.

Exposes two endpoints:

    POST /infer          — text classification / identity scoring
    POST /infer/raw      — raw logits for downstream use

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
from typing import Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from open_mythos.main import MythosConfig, OpenMythos

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


state = _State()


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
    "fraud_detect",  # 詐欺検知（高精度）
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
            "  general         — use loops as specified"
        ),
    )
    max_new_tokens: Optional[int] = Field(
        None,
        description="If set, run autoregressive generation instead of scoring.",
    )


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
    "general": DEFAULT_LOOPS,
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

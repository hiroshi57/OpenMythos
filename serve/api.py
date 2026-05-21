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
from contextlib import asynccontextmanager
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from open_mythos.main import MythosConfig, OpenMythos


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
    "ad_performance",   # 広告クリエイティブ効果予測
    "content_quality",  # SEO / LLMO コンテンツ品質スコアリング
    "persona_segment",  # ユーザーペルソナ分類
    "market_research",  # 市場調査レポート要約
    "identity_verify",  # 本人確認（リアルタイム）
    "fraud_detect",     # 詐欺検知（高精度）
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
            "  general         — use loops as specified"
        ),
    )
    max_new_tokens: Optional[int] = Field(
        None,
        description="If set, run autoregressive generation instead of scoring.",
    )


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
    "general":         DEFAULT_LOOPS,
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

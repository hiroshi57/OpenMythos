#!/usr/bin/env python3
"""
A/B テストルーター — 既存MLモデル vs OpenMythos RDT

Phase 4-2: 本番トラフィックの一部を OpenMythos に流し、
既存MLモデルとの精度・レイテンシを比較する。

設計:
    - トラフィックを hash(user_id) % 100 で分割
    - 各リクエストの結果・レイテンシ・モデルIDをログに記録
    - /ab/stats エンドポイントでリアルタイム比較表示
    - 既存MLモデルは HTTP エンドポイントとして呼び出す（差し替え可能）

使い方:
    uvicorn serve.ab_router:app --port 8001
"""

import hashlib
import json
import time
import urllib.request
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# OpenMythos API（serve/api.py）の URL
OPENMYTHOS_URL  = "http://localhost:8000"
# 既存MLモデルの推論エンドポイント（差し替えてください）
EXISTING_ML_URL = "http://localhost:9000"

# OpenMythosに流すトラフィック比率（0〜100）
OPENMYTHOS_TRAFFIC_PCT = 20  # 20%を新モデルへ


# ---------------------------------------------------------------------------
# インメモリ集計（本番はRedis等に差し替え）
# ---------------------------------------------------------------------------

class _Stats:
    def __init__(self):
        self.counts:    dict[str, int]   = defaultdict(int)
        self.latencies: dict[str, list]  = defaultdict(list)
        self.scores:    dict[str, list]  = defaultdict(list)
        self.correct:   dict[str, int]   = defaultdict(int)

stats = _Stats()


# ---------------------------------------------------------------------------
# リクエスト / レスポンス
# ---------------------------------------------------------------------------

class ABRequest(BaseModel):
    user_id: str = Field(..., description="ユーザーID（ルーティングのハッシュに使用）")
    text: str
    task: Literal["identity_verify", "fraud_detect", "content_quality",
                  "ad_performance", "general"] = "general"
    loops: int = Field(4, ge=1, le=16)
    ground_truth: int | None = Field(None, description="正解ラベル（評価用。省略可）")


class ABResponse(BaseModel):
    model_id: str
    label: int
    score: float
    latency_ms: float
    ab_group: str  # "openmythos" or "existing_ml"


# ---------------------------------------------------------------------------
# ルーティング
# ---------------------------------------------------------------------------

def _route(user_id: str) -> str:
    """user_id のハッシュ値でA/Bグループを決定（再現性あり）。"""
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    return "openmythos" if h < OPENMYTHOS_TRAFFIC_PCT else "existing_ml"


def _call_openmythos(text: str, task: str, loops: int) -> tuple[int, float, float]:
    """OpenMythos API を呼び出して (label, score, latency_ms) を返す。"""
    payload = json.dumps({"text": text, "task": task, "loops": loops}).encode()
    req = urllib.request.Request(
        f"{OPENMYTHOS_URL}/infer",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as res:
        data = json.loads(res.read())
    latency = (time.perf_counter() - t0) * 1000
    return data["label"], data["score"], latency


def _call_existing_ml(text: str, task: str) -> tuple[int, float, float]:
    """
    既存MLモデルを呼び出す。
    実際のエンドポイント仕様に合わせてここを修正してください。
    デフォルトはモック（常に label=1, score=0.75 を返す）。
    """
    t0 = time.perf_counter()
    try:
        payload = json.dumps({"text": text, "task": task}).encode()
        req = urllib.request.Request(
            f"{EXISTING_ML_URL}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read())
        latency = (time.perf_counter() - t0) * 1000
        return data.get("label", 1), data.get("score", 0.75), latency
    except Exception:
        # 既存MLが落ちていてもルーターは止まらない
        latency = (time.perf_counter() - t0) * 1000
        return 1, 0.75, latency


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[ab_router] OpenMythos traffic: {OPENMYTHOS_TRAFFIC_PCT}%")
    yield

app = FastAPI(title="OpenMythos A/B Router", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.post("/infer", response_model=ABResponse)
def infer(req: ABRequest):
    group = _route(req.user_id)

    if group == "openmythos":
        label, score, latency = _call_openmythos(req.text, req.task, req.loops)
        model_id = "openmythos-rdt"
    else:
        label, score, latency = _call_existing_ml(req.text, req.task)
        model_id = "existing-ml"

    # 集計
    stats.counts[group]    += 1
    stats.latencies[group].append(latency)
    stats.scores[group].append(score)
    if req.ground_truth is not None:
        if label == req.ground_truth:
            stats.correct[group] += 1

    return ABResponse(
        model_id=model_id,
        label=label,
        score=score,
        latency_ms=round(latency, 2),
        ab_group=group,
    )


@app.get("/ab/stats")
def ab_stats():
    """A/Bテスト集計をリアルタイムで返す。"""
    result = {}
    for group in ["openmythos", "existing_ml"]:
        n = stats.counts[group]
        lats = stats.latencies[group]
        scrs = stats.scores[group]
        corr = stats.correct[group]
        result[group] = {
            "requests":       n,
            "avg_latency_ms": round(sum(lats) / n, 2) if n else None,
            "avg_score":      round(sum(scrs) / n, 4) if n else None,
            "accuracy":       round(corr / n, 4)      if n else None,
            "traffic_pct":    OPENMYTHOS_TRAFFIC_PCT if group == "openmythos"
                              else 100 - OPENMYTHOS_TRAFFIC_PCT,
        }
    return result


@app.get("/health")
def health():
    return {"status": "ok", "openmythos_pct": OPENMYTHOS_TRAFFIC_PCT}

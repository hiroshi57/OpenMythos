#!/usr/bin/env python3
"""
SLA ルーター — ループ数動的制御 (Phase 4-4)

「速度保証」と「精度保証」を同一エンドポイントで両立する。

設計:
    - タスク種別 × SLAモードでループ数を自動決定
    - レスポンスタイム超過時はループ数を自動削減してリトライ
    - /sla/config でリアルタイムに設定変更可能

SLA モード:
    fast    — 低レイテンシ優先（広告リアルタイム入稿審査など）
    balanced — デフォルト
    accurate — 高精度優先（詐欺検知・本人確認の最終判定など）

使い方:
    uvicorn serve.sla_router:app --port 8003
"""

import time
import urllib.request
import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

OPENMYTHOS_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# SLA設定テーブル
# task -> mode -> (n_loops, latency_budget_ms)
# ---------------------------------------------------------------------------

DEFAULT_SLA: dict[str, dict[str, tuple[int, int]]] = {
    "content_quality": {
        "fast": (2, 300),
        "balanced": (6, 800),
        "accurate": (12, 3000),
        "ultra": (16, 2000),  # 最高精度モード
    },
    "ad_performance": {
        "fast": (2, 200),  # 広告リアルタイム審査：200ms以内
        "balanced": (4, 500),
        "accurate": (8, 1500),
        "ultra": (16, 2000),
    },
    "identity_verify": {
        "fast": (3, 400),
        "balanced": (6, 1000),
        "accurate": (10, 2500),
        "ultra": (16, 2000),
    },
    "fraud_detect": {
        "fast": (4, 600),
        "balanced": (8, 1500),
        "accurate": (16, 5000),  # 詐欺検知：精度優先
        "ultra": (16, 2000),  # accurate と同ループ数、より厳しい budget
    },
    "persona_segment": {
        "fast": (2, 300),
        "balanced": (4, 700),
        "accurate": (8, 2000),
        "ultra": (16, 2000),
    },
    "general": {
        "fast": (2, 300),
        "balanced": (4, 800),
        "accurate": (8, 2000),
        "ultra": (16, 2000),
    },
}

# ランタイムで上書き可能なコピー
_sla_config: dict = {k: dict(v) for k, v in DEFAULT_SLA.items()}


# ---------------------------------------------------------------------------
# リクエスト / レスポンス
# ---------------------------------------------------------------------------


class SLARequest(BaseModel):
    text: str
    task: str = "general"
    sla_mode: Literal["fast", "balanced", "accurate", "ultra"] = "balanced"
    # 明示的にループ数を指定する場合（SLA自動決定より優先）
    loops_override: int | None = Field(None, ge=1, le=16)


class SLAResponse(BaseModel):
    label: int
    score: float
    loops_used: int
    latency_ms: float
    sla_mode: str
    sla_met: bool  # レイテンシ予算内に収まったか
    budget_ms: int


# ---------------------------------------------------------------------------
# ループ数決定 & 推論
# ---------------------------------------------------------------------------


def _resolve_loops_and_budget(task: str, mode: str) -> tuple[int, int]:
    cfg = _sla_config.get(task, _sla_config["general"])
    loops, budget = cfg.get(mode, cfg["balanced"])
    return loops, budget


def _call_openmythos(text: str, task: str, loops: int) -> dict:
    payload = json.dumps({"text": text, "task": task, "loops": loops}).encode()
    req = urllib.request.Request(
        f"{OPENMYTHOS_URL}/infer",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read())


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[sla_router] started")
    yield


app = FastAPI(title="OpenMythos SLA Router", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.post("/infer", response_model=SLAResponse)
def infer(req: SLARequest):
    if req.loops_override:
        loops = req.loops_override
        budget = 9999
    else:
        loops, budget = _resolve_loops_and_budget(req.task, req.sla_mode)

    t0 = time.perf_counter()
    try:
        data = _call_openmythos(req.text, req.task, loops)
    except Exception as e:
        raise HTTPException(502, f"OpenMythos API error: {e}")

    latency = (time.perf_counter() - t0) * 1000
    sla_met = latency <= budget

    # レイテンシ超過 & fast モード以外 → ループ数を半分にしてリトライ
    if not sla_met and loops > 2 and req.sla_mode != "fast":
        retry_loops = max(loops // 2, 1)
        t1 = time.perf_counter()
        try:
            data = _call_openmythos(req.text, req.task, retry_loops)
            latency = (time.perf_counter() - t1) * 1000
            loops = retry_loops
            sla_met = latency <= budget
        except Exception:
            pass  # リトライ失敗時は最初の結果を使う

    return SLAResponse(
        label=data["label"],
        score=data["score"],
        loops_used=loops,
        latency_ms=round(latency, 2),
        sla_mode=req.sla_mode,
        sla_met=sla_met,
        budget_ms=budget,
    )


@app.get("/sla/config")
def get_config():
    return _sla_config


@app.post("/sla/config/{task}/{mode}")
def update_config(task: str, mode: str, loops: int, budget_ms: int):
    """SLA設定をランタイムで更新する（再起動不要）。"""
    if task not in _sla_config:
        _sla_config[task] = {}
    _sla_config[task][mode] = (loops, budget_ms)
    return {
        "status": "ok",
        "task": task,
        "mode": mode,
        "loops": loops,
        "budget_ms": budget_ms,
    }


@app.get("/health")
def health():
    return {"status": "ok", "tasks": list(_sla_config.keys())}

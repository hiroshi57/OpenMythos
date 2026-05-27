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
import math
import time
import urllib.request
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# OpenMythos API（serve/api.py）の URL
OPENMYTHOS_URL = "http://localhost:8000"
# 既存MLモデルの推論エンドポイント（差し替えてください）
EXISTING_ML_URL = "http://localhost:9000"

# OpenMythosに流すトラフィック比率（0〜100）
OPENMYTHOS_TRAFFIC_PCT = 20  # 20%を新モデルへ


# ---------------------------------------------------------------------------
# インメモリ集計（本番はRedis等に差し替え）
# ---------------------------------------------------------------------------


class _Stats:
    def __init__(self):
        self.counts: dict[str, int] = defaultdict(int)
        self.latencies: dict[str, list] = defaultdict(list)
        self.scores: dict[str, list] = defaultdict(list)
        self.correct: dict[str, int] = defaultdict(int)


stats = _Stats()


# ---------------------------------------------------------------------------
# リクエスト / レスポンス
# ---------------------------------------------------------------------------


class ABRequest(BaseModel):
    user_id: str = Field(..., description="ユーザーID（ルーティングのハッシュに使用）")
    text: str
    task: Literal[
        "identity_verify",
        "fraud_detect",
        "content_quality",
        "ad_performance",
        "general",
    ] = "general"
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


def _significance_test(a: list[float], b: list[float], alpha: float = 0.05) -> dict:
    """
    Welch の t 検定で 2 グループのスコア平均を比較する（scipy 不要）。

    Returns:
        {
            "p_value": float,       # 両側 p 値（近似）
            "significant": bool,    # p_value < alpha かどうか
            "mean_a": float,
            "mean_b": float,
            "n_a": int,
            "n_b": int,
        }
    """
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

    se = math.sqrt(var_a / na + var_b / nb)
    if se == 0:
        # 両グループとも分散ゼロ: 平均が同じなら p=1, 異なれば完全分離で p≈0
        p_value = 0.0 if mean_a != mean_b else 1.0
    else:
        t_stat = (mean_a - mean_b) / se
        # Welch–Satterthwaite 自由度
        num = (var_a / na + var_b / nb) ** 2
        den = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
        df = num / den if den > 0 else 1.0
        # t 分布の両側 p 値近似: 大 df では標準正規に近づく
        z = abs(t_stat) * math.sqrt(1 + df / (df + t_stat**2 + 1e-9))
        # 標準正規の上側確率（近似）
        p_one = 0.5 * math.erfc(z / math.sqrt(2))
        p_value = min(2 * p_one, 1.0)

    return {
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
        "mean_a": round(mean_a, 6),
        "mean_b": round(mean_b, 6),
        "n_a": na,
        "n_b": nb,
    }


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
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


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
    stats.counts[group] += 1
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
    """A/Bテスト集計をリアルタイムで返す（統計的有意性検定付き）。"""
    result = {}
    for group in ["openmythos", "existing_ml"]:
        n = stats.counts[group]
        lats = stats.latencies[group]
        scrs = stats.scores[group]
        corr = stats.correct[group]
        result[group] = {
            "requests": n,
            "avg_latency_ms": round(sum(lats) / n, 2) if n else None,
            "avg_score": round(sum(scrs) / n, 4) if n else None,
            "accuracy": round(corr / n, 4) if n else None,
            "traffic_pct": (
                OPENMYTHOS_TRAFFIC_PCT
                if group == "openmythos"
                else 100 - OPENMYTHOS_TRAFFIC_PCT
            ),
        }

    # スコアの統計的有意性検定（両グループにデータが 2 件以上ある場合）
    sig = _significance_test(stats.scores["openmythos"], stats.scores["existing_ml"])
    result["significance_test"] = sig

    return result


@app.get("/health")
def health():
    return {"status": "ok", "openmythos_pct": OPENMYTHOS_TRAFFIC_PCT}

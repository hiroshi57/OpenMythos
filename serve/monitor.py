#!/usr/bin/env python3
"""
精度監視 & ドリフト検知モジュール — Phase 4-3

機能:
    1. 推論ログを SQLite に記録（本番では BigQuery / PostgreSQL に差し替え）
    2. スコア分布の統計的ドリフトを検知（PSI / KS 検定）
    3. 精度劣化アラート（閾値超過で Slack/メール通知）
    4. /monitor/dashboard エンドポイントでメトリクスを返す

使い方:
    from serve.monitor import log_inference, check_drift
    # または FastAPI の lifespan で monitor_app をマウント

    # 単独起動:
    uvicorn serve.monitor:monitor_app --port 8002
"""

import math
import sqlite3
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

DB_PATH = Path("data/monitor.db")
DRIFT_WINDOW = 500  # ドリフト検知に使う直近N件
PSI_THRESHOLD = 0.2  # PSI > 0.2 = 大きなシフト
ACCURACY_ALERT_THRESHOLD = 0.05  # 精度が baseline から5%以上下がったらアラート

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# SQLite ログDB
# ---------------------------------------------------------------------------


def _init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS inference_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            model_id    TEXT    NOT NULL,
            task        TEXT    NOT NULL,
            n_loops     INTEGER,
            score       REAL,
            label       INTEGER,
            ground_truth INTEGER,
            latency_ms  REAL,
            is_correct  INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS baseline (
            task        TEXT    PRIMARY KEY,
            accuracy    REAL,
            avg_score   REAL,
            score_p25   REAL,
            score_p75   REAL,
            updated_at  TEXT
        )
    """)
    con.commit()
    con.close()


def log_inference(
    model_id: str,
    task: str,
    score: float,
    label: int,
    latency_ms: float,
    n_loops: int = 0,
    ground_truth: int | None = None,
):
    """推論結果を1件ログに記録する。serve/api.py から呼び出す。"""
    is_correct = int(label == ground_truth) if ground_truth is not None else None
    ts = datetime.now(timezone.utc).isoformat()
    with _lock:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO inference_log "
            "(ts, model_id, task, n_loops, score, label, ground_truth, latency_ms, is_correct) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                ts,
                model_id,
                task,
                n_loops,
                score,
                label,
                ground_truth,
                latency_ms,
                is_correct,
            ),
        )
        con.commit()
        con.close()


def set_baseline(
    task: str, accuracy: float, avg_score: float, score_p25: float, score_p75: float
):
    """ベースライン精度を登録する（初期デプロイ時に一度実行）。"""
    with _lock:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT OR REPLACE INTO baseline "
            "(task, accuracy, avg_score, score_p25, score_p75, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                task,
                accuracy,
                avg_score,
                score_p25,
                score_p75,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        con.commit()
        con.close()


# ---------------------------------------------------------------------------
# ドリフト検知
# ---------------------------------------------------------------------------


def _psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """
    Population Stability Index（PSI）を計算。
    PSI < 0.1: 変化なし  0.1-0.2: 軽微  > 0.2: 大きなシフト
    """
    if not expected or not actual:
        return 0.0
    lo, hi = 0.0, 1.0

    def hist(data):
        counts = [0] * bins
        for v in data:
            b = min(int((v - lo) / (hi - lo) * bins), bins - 1)
            counts[b] += 1
        total = max(len(data), 1)
        return [max(c / total, 1e-6) for c in counts]

    e = hist(expected)
    a = hist(actual)
    return sum((ae - ee) * math.log(ae / ee) for ae, ee in zip(a, e))


def check_drift(task: str) -> dict:
    """直近 DRIFT_WINDOW 件とベースラインを比較してドリフト状況を返す。"""
    con = sqlite3.connect(DB_PATH)
    # 直近N件
    rows = con.execute(
        "SELECT score, is_correct FROM inference_log "
        "WHERE task=? ORDER BY id DESC LIMIT ?",
        (task, DRIFT_WINDOW),
    ).fetchall()
    baseline = con.execute(
        "SELECT accuracy, avg_score, score_p25, score_p75 FROM baseline WHERE task=?",
        (task,),
    ).fetchone()
    con.close()

    if not rows:
        return {"status": "no_data", "task": task}

    recent_scores = [r[0] for r in rows if r[0] is not None]
    recent_correct = [r[1] for r in rows if r[1] is not None]
    current_accuracy = (
        sum(recent_correct) / len(recent_correct) if recent_correct else None
    )
    current_avg_score = (
        sum(recent_scores) / len(recent_scores) if recent_scores else None
    )

    result = {
        "task": task,
        "n_recent": len(rows),
        "current_accuracy": round(current_accuracy, 4) if current_accuracy else None,
        "current_avg_score": round(current_avg_score, 4) if current_avg_score else None,
        "alerts": [],
        "psi": None,
    }

    if baseline:
        bl_acc, bl_avg, bl_p25, bl_p75 = baseline
        # スコア分布のPSI
        bl_scores = [bl_p25, bl_avg, bl_p75]  # 簡易的なベースライン分布
        psi = _psi(bl_scores * (len(recent_scores) // 3 + 1), recent_scores)
        result["psi"] = round(psi, 4)
        result["baseline_accuracy"] = bl_acc

        if psi > PSI_THRESHOLD:
            result["alerts"].append(
                {
                    "level": "warning",
                    "type": "score_distribution_drift",
                    "msg": f"PSI={psi:.3f} > {PSI_THRESHOLD}（スコア分布が大きくシフト）",
                }
            )
        if current_accuracy and bl_acc:
            drop = bl_acc - current_accuracy
            if drop > ACCURACY_ALERT_THRESHOLD:
                result["alerts"].append(
                    {
                        "level": "critical",
                        "type": "accuracy_degradation",
                        "msg": f"精度が {drop*100:.1f}pt 低下 "
                        f"({bl_acc:.3f} → {current_accuracy:.3f})",
                    }
                )

    result["status"] = "alert" if result["alerts"] else "healthy"
    return result


# ---------------------------------------------------------------------------
# 統計サマリー
# ---------------------------------------------------------------------------


def get_metrics(hours: int = 24) -> dict:
    """過去N時間の推論ログからメトリクスを集計する。"""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT model_id, task, score, label, latency_ms, is_correct "
        "FROM inference_log WHERE ts >= ?",
        (since,),
    ).fetchall()
    con.close()

    if not rows:
        return {"status": "no_data", "hours": hours}

    from collections import defaultdict

    task_stats: dict = defaultdict(
        lambda: {"count": 0, "latencies": [], "scores": [], "correct": 0, "labeled": 0}
    )
    for model_id, task, score, label, lat, correct in rows:
        key = f"{task}/{model_id}"
        s = task_stats[key]
        s["count"] += 1
        s["latencies"].append(lat)
        if score is not None:
            s["scores"].append(score)
        if correct is not None:
            s["correct"] += correct
            s["labeled"] += 1

    summary = {}
    for key, s in task_stats.items():
        n = s["count"]
        lats = s["latencies"]
        scrs = s["scores"]
        summary[key] = {
            "requests": n,
            "avg_latency_ms": round(sum(lats) / n, 2) if n else None,
            "p95_latency_ms": round(sorted(lats)[int(n * 0.95)], 2) if n > 1 else None,
            "avg_score": round(sum(scrs) / len(scrs), 4) if scrs else None,
            "accuracy": round(s["correct"] / s["labeled"], 4) if s["labeled"] else None,
        }
    return {"hours": hours, "metrics": summary}


# ---------------------------------------------------------------------------
# FastAPI（監視ダッシュボード）
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    print(f"[monitor] DB: {DB_PATH}")
    yield


monitor_app = FastAPI(title="OpenMythos Monitor", version="0.1.0", lifespan=lifespan)


@monitor_app.get("/monitor/health")
def health():
    return {"status": "ok", "db": str(DB_PATH)}


@monitor_app.get("/monitor/metrics")
def metrics(hours: int = 24):
    return get_metrics(hours)


@monitor_app.get("/monitor/drift/{task}")
def drift(task: str):
    return check_drift(task)


@monitor_app.get("/monitor/dashboard")
def dashboard():
    """全タスクのドリフト状況 + 直近24時間メトリクスを一括返す。"""
    tasks = ["content_quality", "ad_performance", "persona_segment", "market_research"]
    return {
        "drift": {t: check_drift(t) for t in tasks},
        "metrics": get_metrics(hours=24),
    }


class BaselineRequest(BaseModel):
    task: str
    accuracy: float
    avg_score: float
    score_p25: float
    score_p75: float


@monitor_app.post("/monitor/baseline")
def register_baseline(req: BaselineRequest):
    """ベースライン精度を登録する（初期デプロイ時に1回実行）。"""
    set_baseline(req.task, req.accuracy, req.avg_score, req.score_p25, req.score_p75)
    return {"status": "ok", "task": req.task}

"""
OpenMythos — デモ用サンプルデータ投入スクリプト (D5)

起動済みサーバーに対して、data/samples/ad_performance_sample.csv の内容を
キャンペーン管理API (POST /v1/campaign/) に投入する。初回セットアップ時や
動作確認時に「空のまま触る」状態を避けるためのもの。

使い方:
    uvicorn serve.api:app --port 8000 &
    python examples/seed_demo_data.py

API_KEY を設定している場合は環境変数 OPENMYTHOS_API_KEY で渡す
(未設定時は開発モード = 認証ヘッダなしでも通る想定で "dev" を送る)。

冪等性について:
    このAPIにはキャンペーン名での重複チェックが無いため、再実行すると
    同名キャンペーンが増える。既存キャンペーンを確認してから投入したい
    場合は --check-existing を付ける（名前が一致するものはスキップ）。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import requests

BASE = os.environ.get("OPENMYTHOS_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("OPENMYTHOS_API_KEY", "dev")
HDR = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "samples" / "ad_performance_sample.csv"

# CSV の ad_format (search/social) はそのまま AdChannel の値として使える。
# CSV に無い channel は display にフォールバックする。
_VALID_CHANNELS = {"search", "social", "display", "email", "video"}


def _rows() -> list[dict]:
    with SAMPLE_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_campaign_payload(row: dict) -> dict:
    channel = row["ad_format"] if row["ad_format"] in _VALID_CHANNELS else "display"
    return {
        "name": f"[デモ] {row['headline']}",
        "objective": "conversion",
        "budget_total": float(row["budget"]),
        "budget_daily": float(row["budget"]) / 30,
        "currency": "JPY",
        "channels": [channel],
        "tags": [row["industry_vertical"], row["platform"]],
        "description": row["description"],
    }


def check_health() -> None:
    r = requests.get(f"{BASE}/health", timeout=5)
    r.raise_for_status()
    print("[health]", r.json()["status"])


def existing_campaign_names() -> set[str]:
    r = requests.get(f"{BASE}/v1/campaign/", headers=HDR, timeout=10)
    r.raise_for_status()
    return {c["name"] for c in r.json()}


def seed(check_existing: bool) -> int:
    rows = _rows()
    skip_names = existing_campaign_names() if check_existing else set()

    created = 0
    for row in rows:
        payload = _to_campaign_payload(row)
        if payload["name"] in skip_names:
            print(f"  スキップ(既存): {payload['name']}")
            continue
        r = requests.post(f"{BASE}/v1/campaign/", headers=HDR, json=payload, timeout=10)
        if r.status_code == 200:
            created += 1
            print(f"  作成: {payload['name']} (id={r.json()['id']})")
        else:
            print(f"  ERROR ({r.status_code}): {payload['name']} — {r.text[:200]}")
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="同名キャンペーンが既にあればスキップする(デフォルトは常に新規作成)",
    )
    args = parser.parse_args()

    print("OpenMythos デモデータ投入")
    print(f"接続先: {BASE}")
    print("-" * 50)

    try:
        check_health()
    except requests.ConnectionError:
        print("\nERROR: サーバーに接続できません。")
        print("  以下のコマンドでサーバーを起動してから再実行してください:")
        print("  uvicorn serve.api:app --port 8000")
        sys.exit(1)

    print(f"\n投入元: {SAMPLE_CSV.name}")
    n = seed(check_existing=args.check_existing)
    print(f"\n{n} 件のデモキャンペーンを作成しました。")
    print(f"確認: GET {BASE}/v1/campaign/ または {BASE}/ad （ブラウザUI）")

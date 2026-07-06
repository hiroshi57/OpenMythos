"""
serve/state.py — アプリ全体で共有するモデル状態シングルトン

serve/api.py の lifespan で起動時に初期化され、
各ドメインルーター (serve/routers/*) からも参照される。
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import torch
    from transformers import AutoTokenizer

    from open_mythos.agents import MythosAgent, OpenMythosLLM
    from open_mythos.main import OpenMythos


class AppState:
    model: "OpenMythos"
    tokenizer: "AutoTokenizer"
    device: "torch.device"
    n_params: int
    llm: "OpenMythosLLM"
    # session_id -> MythosAgent（会話履歴管理）
    agents: dict[str, "MythosAgent"]


state = AppState()
state.agents = {}


# ─── 推論ループ設定 (api.py と各ルーターで共有) ────────────────────────────

DEFAULT_LOOPS = int(os.getenv("DEFAULT_LOOPS", "4"))
MAX_LOOPS = int(os.getenv("MAX_LOOPS", "16"))

# タスク種別ごとの推奨ループ数
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


# ─── タスク種別 (api.py と各ルーターで共有) ────────────────────────────

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


# ─── ErrorMemory ストア シングルトン (Sprint 24/34 で共有) ──────────────

_mistake_store: Optional[object] = None


def get_mistake_store():
    global _mistake_store
    if _mistake_store is None:
        from open_mythos.error_memory import ErrorMemoryStore
        backend = os.environ.get("MISTAKES_BACKEND", "memory")    # "memory" | "sqlite"
        db_path = os.environ.get("MISTAKES_DB_PATH",  "mistakes.db")
        _mistake_store = ErrorMemoryStore(backend=backend, db_path=db_path)
    return _mistake_store

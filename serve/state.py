"""
serve/state.py — アプリ全体で共有するモデル状態シングルトン

serve/api.py の lifespan で起動時に初期化され、
各ドメインルーター (serve/routers/*) からも参照される。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

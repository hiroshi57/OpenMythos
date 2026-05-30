"""
OpenMythos Conversation Memory — 長期対話状態管理。

ユーザーとモデルの対話履歴を管理し、コンテキストウィンドウに収まるよう
古いターンを自動的に要約・圧縮する。

設計:
    Turn               -- 1ターン (role + content)
    ConversationMemory -- rolling window + summary compression
    SessionStore       -- セッション ID ベースのメモリ管理

使い方::

    from open_mythos.conversation import ConversationMemory, SessionStore

    memory = ConversationMemory(max_turns=10, max_chars=2000)
    memory.add_turn("user", "LLMOとは何ですか？")
    memory.add_turn("assistant", "LLMO は Large Language Model Optimization の略です。")

    # コンテキスト文字列として取得
    context = memory.to_context_string()
    print(context)  # Human: LLMOとは...\\nAssistant: LLMO は...

    # セッション管理
    store = SessionStore()
    session_id = store.create_session()
    store.get(session_id).add_turn("user", "こんにちは")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """対話の1ターン。"""

    role: str
    """ターンの役割: "user" / "assistant" / "system" / "tool" """

    content: str
    """ターンの内容。"""

    created_at: float = field(default_factory=time.time)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Turn":
        return cls(
            role=data["role"],
            content=data["content"],
            turn_id=data.get("turn_id", uuid.uuid4().hex[:8]),
            created_at=data.get("created_at", time.time()),
        )

    @property
    def char_len(self) -> int:
        return len(self.content)


@dataclass
class MemorySummary:
    """圧縮された対話履歴サマリー。"""

    text: str
    """サマリーテキスト。"""

    turns_summarized: int
    """要約したターン数。"""

    original_chars: int
    """要約前の総文字数。"""

    created_at: float = field(default_factory=time.time)

    @property
    def compression_ratio(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return 1.0 - len(self.text) / self.original_chars


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class ConversationMemory:
    """
    対話メモリ管理クラス。

    最大ターン数・最大文字数を超えたとき、古いターンを要約して
    コンテキストウィンドウ内に収める。

    Args:
        max_turns   -- メモリに保持する最大ターン数 (デフォルト: 20)
        max_chars   -- メモリの最大文字数 (デフォルト: 4000)
        system_msg  -- システムメッセージ (空の場合は使用しない)
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_chars: int = 4000,
        system_msg: str = "",
    ) -> None:
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.system_msg = system_msg
        self._turns: list[Turn] = []
        self._summary: Optional[MemorySummary] = None
        self._session_id: str = uuid.uuid4().hex

    # ------------------------------------------------------------------
    # ターン管理
    # ------------------------------------------------------------------

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> Turn:
        """
        ターンを追加する。

        追加後、制限を超えた場合は自動的に古いターンを圧縮する。

        Args:
            role     -- "user" / "assistant" / "system" / "tool"
            content  -- ターンの内容
            metadata -- 任意のメタデータ

        Returns:
            追加した Turn
        """
        turn = Turn(role=role, content=content, metadata=metadata or {})
        self._turns.append(turn)
        self._maybe_compress()
        return turn

    def add_user(self, content: str) -> Turn:
        """ユーザーターンを追加する。"""
        return self.add_turn("user", content)

    def add_assistant(self, content: str) -> Turn:
        """アシスタントターンを追加する。"""
        return self.add_turn("assistant", content)

    def clear(self) -> None:
        """メモリを全消去する。"""
        self._turns.clear()
        self._summary = None

    def pop_last(self) -> Optional[Turn]:
        """最後のターンを削除して返す。"""
        return self._turns.pop() if self._turns else None

    # ------------------------------------------------------------------
    # コンテキスト生成
    # ------------------------------------------------------------------

    def to_context_string(
        self,
        user_prefix: str = "Human",
        assistant_prefix: str = "Assistant",
        include_summary: bool = True,
    ) -> str:
        """
        対話履歴をプロンプト用文字列に変換する。

        Args:
            user_prefix      -- ユーザーのプレフィックス
            assistant_prefix -- アシスタントのプレフィックス
            include_summary  -- True の場合、圧縮サマリーを先頭に含める

        Returns:
            対話履歴テキスト
        """
        parts: list[str] = []

        if self.system_msg:
            parts.append(f"System: {self.system_msg}")

        if include_summary and self._summary:
            parts.append(f"[Earlier conversation summary]:\n{self._summary.text}")

        for turn in self._turns:
            if turn.role == "user":
                parts.append(f"{user_prefix}: {turn.content}")
            elif turn.role == "assistant":
                parts.append(f"{assistant_prefix}: {turn.content}")
            elif turn.role == "system":
                parts.append(f"System: {turn.content}")
            elif turn.role == "tool":
                parts.append(f"Tool: {turn.content}")

        return "\n".join(parts)

    def to_messages(self) -> list[dict]:
        """OpenAI messages 形式のリストを返す。"""
        messages: list[dict] = []

        if self.system_msg:
            messages.append({"role": "system", "content": self.system_msg})

        if self._summary:
            messages.append({
                "role": "system",
                "content": f"[Earlier conversation]:\n{self._summary.text}",
            })

        for turn in self._turns:
            messages.append(turn.to_dict())

        return messages

    # ------------------------------------------------------------------
    # 統計・プロパティ
    # ------------------------------------------------------------------

    @property
    def n_turns(self) -> int:
        return len(self._turns)

    @property
    def total_chars(self) -> int:
        return sum(t.char_len for t in self._turns)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def has_summary(self) -> bool:
        return self._summary is not None

    @property
    def last_turn(self) -> Optional[Turn]:
        return self._turns[-1] if self._turns else None

    @property
    def last_user_turn(self) -> Optional[Turn]:
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn
        return None

    @property
    def last_assistant_turn(self) -> Optional[Turn]:
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn
        return None

    def stats(self) -> dict:
        return {
            "session_id": self._session_id,
            "n_turns": self.n_turns,
            "total_chars": self.total_chars,
            "has_summary": self.has_summary,
            "summary_turns": self._summary.turns_summarized if self._summary else 0,
            "max_turns": self.max_turns,
            "max_chars": self.max_chars,
        }

    # ------------------------------------------------------------------
    # 圧縮
    # ------------------------------------------------------------------

    def _maybe_compress(self) -> None:
        """制限超過時に古いターンを要約して圧縮する。"""
        needs_compress = (
            len(self._turns) > self.max_turns
            or self.total_chars > self.max_chars
        )
        if not needs_compress:
            return

        # 現在のターン数の半分を保持 (最低1ターンは残す)
        current_n = len(self._turns)
        keep_n = max(1, min(current_n - 1, max(2, self.max_turns // 2)))
        to_compress = self._turns[:-keep_n]
        self._turns = self._turns[-keep_n:]

        if not to_compress:
            return

        # テキストベースのサマリー生成 (ルールベース)
        summary_lines = []
        for turn in to_compress:
            prefix = {"user": "U", "assistant": "A", "system": "S", "tool": "T"}.get(turn.role, "?")
            # 内容を最大80文字に要約
            short = turn.content[:80].replace("\n", " ")
            if len(turn.content) > 80:
                short += "..."
            summary_lines.append(f"[{prefix}] {short}")

        summary_text = "\n".join(summary_lines)
        original_chars = sum(t.char_len for t in to_compress)

        # 既存サマリーがあれば連結
        if self._summary:
            summary_text = self._summary.text + "\n" + summary_text
            original_chars += self._summary.original_chars
            turns_count = self._summary.turns_summarized + len(to_compress)
        else:
            turns_count = len(to_compress)

        self._summary = MemorySummary(
            text=summary_text,
            turns_summarized=turns_count,
            original_chars=original_chars,
        )

    def compress_now(self, keep_n: Optional[int] = None) -> Optional[MemorySummary]:
        """
        手動で圧縮を実行する。

        Args:
            keep_n -- 保持するターン数 (None の場合は max_turns // 2)

        Returns:
            作成したサマリー (圧縮が不要な場合は None)
        """
        if len(self._turns) <= 1:
            return None

        _keep = keep_n if keep_n is not None else max(1, self.max_turns // 2)
        # _maybe_compress と同じロジック、ターン数に関係なく実行
        keep = min(_keep, len(self._turns) - 1)
        to_compress = self._turns[:-keep] if keep > 0 else self._turns[:]
        self._turns = self._turns[-keep:] if keep > 0 else []

        if not to_compress:
            return None

        lines = []
        for turn in to_compress:
            prefix = {"user": "U", "assistant": "A", "system": "S", "tool": "T"}.get(turn.role, "?")
            short = turn.content[:80].replace("\n", " ")
            if len(turn.content) > 80:
                short += "..."
            lines.append(f"[{prefix}] {short}")

        summary_text = "\n".join(lines)
        original_chars = sum(t.char_len for t in to_compress)

        if self._summary:
            summary_text = self._summary.text + "\n" + summary_text
            original_chars += self._summary.original_chars
            turns_count = self._summary.turns_summarized + len(to_compress)
        else:
            turns_count = len(to_compress)

        self._summary = MemorySummary(
            text=summary_text,
            turns_summarized=turns_count,
            original_chars=original_chars,
        )
        return self._summary


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """
    セッション ID ベースのメモリ管理。

    複数ユーザー/スレッドのセッションを分離して管理する。

    Args:
        max_sessions  -- 最大同時セッション数 (超過時は最古を削除)
        max_turns     -- 各セッションの最大ターン数
        max_chars     -- 各セッションの最大文字数
        ttl_seconds   -- セッションの TTL (秒, 0 で無制限)
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        max_turns: int = 20,
        max_chars: int = 4000,
        ttl_seconds: float = 0.0,
    ) -> None:
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, ConversationMemory] = {}
        self._created_at: dict[str, float] = {}

    # ------------------------------------------------------------------
    # セッション CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: Optional[str] = None,
        system_msg: str = "",
    ) -> str:
        """
        新しいセッションを作成してセッション ID を返す。

        Args:
            session_id -- 任意のセッション ID (None なら自動生成)
            system_msg -- システムメッセージ

        Returns:
            セッション ID
        """
        if session_id is None:
            session_id = uuid.uuid4().hex

        # LRU eviction
        if len(self._sessions) >= self.max_sessions:
            oldest = min(self._created_at, key=lambda k: self._created_at[k])
            self._sessions.pop(oldest, None)
            self._created_at.pop(oldest, None)

        self._sessions[session_id] = ConversationMemory(
            max_turns=self.max_turns,
            max_chars=self.max_chars,
            system_msg=system_msg,
        )
        self._created_at[session_id] = time.time()
        return session_id

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        """セッションを取得する。存在しない場合は None を返す。"""
        self._evict_expired()
        return self._sessions.get(session_id)

    def get_or_create(
        self,
        session_id: Optional[str] = None,
        system_msg: str = "",
    ) -> tuple[str, ConversationMemory]:
        """
        セッションを取得、または作成する。

        Returns:
            (session_id, memory)
        """
        if session_id and session_id in self._sessions:
            return session_id, self._sessions[session_id]
        sid = self.create_session(session_id=session_id, system_msg=system_msg)
        return sid, self._sessions[sid]

    def delete(self, session_id: str) -> bool:
        """セッションを削除する。"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._created_at.pop(session_id, None)
            return True
        return False

    def list_sessions(self) -> list[str]:
        """全セッション ID のリストを返す。"""
        self._evict_expired()
        return list(self._sessions.keys())

    def __len__(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # TTL 管理
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        expired = [
            sid for sid, t in self._created_at.items()
            if now - t > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._created_at.pop(sid, None)

    def stats(self) -> dict:
        self._evict_expired()
        return {
            "n_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "ttl_seconds": self.ttl_seconds,
        }

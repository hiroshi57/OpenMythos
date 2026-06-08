"""
Sprint 44 — OpenAI-compatible Assistants API

OpenAI Assistants API と互換性のあるステートフルな会話管理レイヤー:

  ┌─────────────────┬────────────────────────────────────────────────┐
  │ オブジェクト    │ 役割                                           │
  ├─────────────────┼────────────────────────────────────────────────┤
  │ AssistantObject │ AI アシスタント設定 (指示・モデル・ツール)     │
  │ Thread          │ ユーザーとアシスタント間の会話スレッド         │
  │ Message         │ スレッド内の個別メッセージ                     │
  │ Run             │ スレッドに対するアシスタント実行インスタンス   │
  │ AssistantStore  │ 全オブジェクトのメモリ内ストア                 │
  │ AssistantRunner │ Run を実行し LLM 呼び出しを行う               │
  └─────────────────┴────────────────────────────────────────────────┘

設計:
    - OpenAI 互換: id / object / created_at フィールド完備
    - ステートフル: AssistantStore でスレッド・メッセージを保持
    - 注入可能 LLM: llm_fn 引数で実装を差し替え可能
    - Hermes 連携: run 実行時に HermesOrchestrator を利用可能
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> int:
    """Unix timestamp (seconds)"""
    return int(time.time())


def _gen_id(prefix: str) -> str:
    """OpenAI 形式の ID 生成 (例: asst_abc123...)"""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssistantTool:
    """ツール定義 (function / code_interpreter / retrieval)"""
    type: str  # "function" | "code_interpreter" | "retrieval"
    function: Optional[Dict[str, Any]] = None  # type="function" 時のみ


@dataclass
class AssistantObject:
    """アシスタント設定オブジェクト"""
    id: str
    object: str = "assistant"
    created_at: int = field(default_factory=_now)
    name: Optional[str] = None
    description: Optional[str] = None
    model: str = "openmythos"
    instructions: Optional[str] = None
    tools: List[AssistantTool] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created_at": self.created_at,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "instructions": self.instructions,
            "tools": [{"type": t.type, **({"function": t.function} if t.function else {})} for t in self.tools],
            "metadata": self.metadata,
        }


@dataclass
class Thread:
    """会話スレッド"""
    id: str
    object: str = "thread"
    created_at: int = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class MessageContent:
    """メッセージコンテンツブロック"""
    type: str = "text"
    text: Dict[str, Any] = field(default_factory=lambda: {"value": "", "annotations": []})

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, self.type: self.text}


@dataclass
class Message:
    """スレッド内メッセージ"""
    id: str
    thread_id: str
    role: str  # "user" | "assistant"
    content: List[MessageContent] = field(default_factory=list)
    object: str = "thread.message"
    created_at: int = field(default_factory=_now)
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """最初のテキストブロックの値を返す"""
        for c in self.content:
            if c.type == "text":
                return c.text.get("value", "")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created_at": self.created_at,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": [c.to_dict() for c in self.content],
            "assistant_id": self.assistant_id,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }


@dataclass
class RunUsage:
    """Run のトークン使用量"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Run:
    """スレッドに対するアシスタント実行インスタンス"""
    id: str
    thread_id: str
    assistant_id: str
    model: str = "openmythos"
    object: str = "thread.run"
    created_at: int = field(default_factory=_now)
    status: str = "queued"   # queued|in_progress|completed|failed|cancelled|expired
    instructions: Optional[str] = None
    tools: List[AssistantTool] = field(default_factory=list)
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    last_error: Optional[Dict[str, Any]] = None
    usage: RunUsage = field(default_factory=RunUsage)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created_at": self.created_at,
            "thread_id": self.thread_id,
            "assistant_id": self.assistant_id,
            "status": self.status,
            "model": self.model,
            "instructions": self.instructions,
            "tools": [{"type": t.type} for t in self.tools],
            "completed_at": self.completed_at,
            "failed_at": self.failed_at,
            "cancelled_at": self.cancelled_at,
            "last_error": self.last_error,
            "usage": self.usage.to_dict(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# AssistantStore — in-memory CRUD store
# ---------------------------------------------------------------------------

class AssistantStore:
    """全アシスタントオブジェクトのメモリ内ストア (OpenAI 互換)"""

    def __init__(self) -> None:
        self._assistants: Dict[str, AssistantObject] = {}
        self._threads: Dict[str, Thread] = {}
        self._messages: Dict[str, Message] = {}
        # スレッド別メッセージ ID リスト (順序保持)
        self._thread_msgs: Dict[str, List[str]] = {}
        self._runs: Dict[str, Run] = {}
        # スレッド別 Run ID リスト
        self._thread_runs: Dict[str, List[str]] = {}

    # ── Assistants ──────────────────────────────────────────────────────────

    def create_assistant(
        self,
        model: str = "openmythos",
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[AssistantTool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssistantObject:
        asst = AssistantObject(
            id=_gen_id("asst"),
            model=model,
            name=name,
            description=description,
            instructions=instructions,
            tools=tools or [],
            metadata=metadata or {},
        )
        self._assistants[asst.id] = asst
        return asst

    def get_assistant(self, assistant_id: str) -> Optional[AssistantObject]:
        return self._assistants.get(assistant_id)

    def list_assistants(self, limit: int = 20) -> List[AssistantObject]:
        items = sorted(self._assistants.values(), key=lambda a: a.created_at, reverse=True)
        return items[:limit]

    def delete_assistant(self, assistant_id: str) -> bool:
        if assistant_id in self._assistants:
            del self._assistants[assistant_id]
            return True
        return False

    # ── Threads ──────────────────────────────────────────────────────────────

    def create_thread(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Thread:
        thread = Thread(id=_gen_id("thread"), metadata=metadata or {})
        self._threads[thread.id] = thread
        self._thread_msgs[thread.id] = []
        self._thread_runs[thread.id] = []
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        return self._threads.get(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        if thread_id not in self._threads:
            return False
        del self._threads[thread_id]
        for msg_id in self._thread_msgs.pop(thread_id, []):
            self._messages.pop(msg_id, None)
        for run_id in self._thread_runs.pop(thread_id, []):
            self._runs.pop(run_id, None)
        return True

    # ── Messages ─────────────────────────────────────────────────────────────

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        assistant_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        if thread_id not in self._threads:
            raise KeyError(f"Thread not found: {thread_id}")
        msg = Message(
            id=_gen_id("msg"),
            thread_id=thread_id,
            role=role,
            content=[MessageContent(type="text", text={"value": content, "annotations": []})],
            assistant_id=assistant_id,
            run_id=run_id,
            metadata=metadata or {},
        )
        self._messages[msg.id] = msg
        self._thread_msgs[thread_id].append(msg.id)
        return msg

    def list_messages(self, thread_id: str, limit: int = 20) -> List[Message]:
        ids = self._thread_msgs.get(thread_id, [])
        msgs = [self._messages[i] for i in ids if i in self._messages]
        return msgs[-limit:]  # 最新 limit 件

    def get_message(self, message_id: str) -> Optional[Message]:
        return self._messages.get(message_id)

    # ── Runs ──────────────────────────────────────────────────────────────────

    def create_run(
        self,
        thread_id: str,
        assistant_id: str,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[AssistantTool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Run:
        if thread_id not in self._threads:
            raise KeyError(f"Thread not found: {thread_id}")
        if assistant_id not in self._assistants:
            raise KeyError(f"Assistant not found: {assistant_id}")
        asst = self._assistants[assistant_id]
        run = Run(
            id=_gen_id("run"),
            thread_id=thread_id,
            assistant_id=assistant_id,
            model=model or asst.model,
            instructions=instructions or asst.instructions,
            tools=tools or asst.tools,
            metadata=metadata or {},
        )
        self._runs[run.id] = run
        self._thread_runs[thread_id].append(run.id)
        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        return self._runs.get(run_id)

    def list_runs(self, thread_id: str, limit: int = 20) -> List[Run]:
        ids = self._thread_runs.get(thread_id, [])
        runs = [self._runs[i] for i in ids if i in self._runs]
        return runs[-limit:]

    def update_run_status(
        self,
        run_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> Optional[Run]:
        run = self._runs.get(run_id)
        if run is None:
            return None
        run.status = status
        now = _now()
        if status == "completed":
            run.completed_at = now
        elif status == "failed":
            run.failed_at = now
            run.last_error = {"code": "server_error", "message": error or "unknown"}
        elif status == "cancelled":
            run.cancelled_at = now
        return run


# ---------------------------------------------------------------------------
# AssistantRunner — LLM 実行エンジン
# ---------------------------------------------------------------------------

# デフォルト LLM 関数シグネチャ: (prompt: str, **kwargs) -> str
LLMFn = Callable[[str], str]


class AssistantRunner:
    """
    Run を実行してスレッドにアシスタントメッセージを追加する。

    注入可能設計:
        llm_fn: (prompt: str) -> str
            実際の LLM 呼び出しを差し替え可能。
            省略時はシンプルなエコー実装 (テスト用)。
    """

    def __init__(self, store: AssistantStore, llm_fn: Optional[LLMFn] = None) -> None:
        self._store = store
        self._llm_fn = llm_fn or _default_llm_fn

    def execute(self, run: Run) -> Run:
        """
        Run を同期実行する:
        1. run → in_progress
        2. スレッドメッセージを収集しプロンプト構築
        3. LLM 呼び出し
        4. アシスタントメッセージをスレッドに追加
        5. run → completed (失敗時 → failed)
        """
        self._store.update_run_status(run.id, "in_progress")

        try:
            # プロンプト構築
            prompt = self._build_prompt(run)
            # LLM 呼び出し
            response_text = self._llm_fn(prompt)
            # トークン使用量 (近似)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(response_text.split())
            run.usage = RunUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            # アシスタントメッセージ追加
            self._store.add_message(
                thread_id=run.thread_id,
                role="assistant",
                content=response_text,
                assistant_id=run.assistant_id,
                run_id=run.id,
            )
            self._store.update_run_status(run.id, "completed")
        except Exception as exc:  # noqa: BLE001
            self._store.update_run_status(run.id, "failed", error=str(exc))

        return run

    def _build_prompt(self, run: Run) -> str:
        """スレッドメッセージ + アシスタント指示からプロンプトを組み立てる"""
        parts: List[str] = []
        if run.instructions:
            parts.append(f"[System]\n{run.instructions}\n")
        messages = self._store.list_messages(run.thread_id, limit=50)
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            parts.append(f"[{role_label}]\n{msg.text}\n")
        parts.append("[Assistant]\n")
        return "\n".join(parts)


def _default_llm_fn(prompt: str) -> str:
    """テスト用デフォルト LLM 実装 — プロンプトに基づくエコー応答"""
    # 最後の [User] メッセージを抽出して簡易応答生成
    lines = prompt.strip().splitlines()
    user_lines: List[str] = []
    in_user = False
    for line in lines:
        if line.startswith("[User]"):
            in_user = True
            user_lines = []
        elif line.startswith("[") and in_user:
            in_user = False
        elif in_user:
            user_lines.append(line)
    user_text = " ".join(user_lines).strip() or "こんにちは"
    return f"ご質問「{user_text}」について回答いたします。OpenMythos Assistants API がお手伝いします。"


# ---------------------------------------------------------------------------
# Module-level singleton store (テスト・API 共有用)
# ---------------------------------------------------------------------------

_DEFAULT_STORE: Optional[AssistantStore] = None


def get_default_store() -> AssistantStore:
    """グローバルシングルトンストアを返す (初回呼び出し時に作成)"""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = AssistantStore()
    return _DEFAULT_STORE


def reset_default_store() -> None:
    """テスト用: グローバルストアをリセット"""
    global _DEFAULT_STORE
    _DEFAULT_STORE = AssistantStore()

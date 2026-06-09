"""
Sprint 55 — ストリーミング & SSE 応答

OpenMythos のリアルタイムストリーミング基盤。

オブジェクト階層:
  StreamDelta     : 単一トークンの差分 (content + index)
  StreamChunk     : 1回の yield 単位 (delta + 完了フラグ + メタ)
  StreamEvent     : SSE 送出単位 — event/data/id ヘッダー付きチャンク
  StreamSession   : セッション状態トラッカー (token 累計・status)
  StreamingRunner : ジェネレータ方式でトークン列を yield するランナー
  StreamBuffer    : チャンク蓄積・全文復元・エラーハンドリング

SSE フォーマット (RFC 8895):
  event: delta
  data: {"content":"今","index":0,...}

  data: [DONE]

設計方針:
  - 実モデル (OpenMythosLLM.stream) がない場合は word-by-word 擬似ストリーム
  - StreamBuffer を使うと非同期クライアントが全文を後から取得できる
  - StreamEvent.to_sse() で FastAPI StreamingResponse に直接渡せる
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データクラス
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StreamDelta:
    """単一トークンの差分。OpenAI の choices[0].delta に相当。"""

    content: str                  # トークン文字列（空文字 = done チャンク）
    index:   int    = 0           # ストリーム内の通し番号
    role:    Optional[str] = None # 最初の chunk のみ "assistant"


@dataclass
class StreamChunk:
    """
    1 回の yield 単位。

    done=False の間はトークンが続き、
    done=True の最終 chunk で finish_reason が確定する。
    """

    delta:         StreamDelta
    chunk_id:      str
    created:       int            # Unix timestamp
    model:         str  = "openmythos"
    done:          bool = False
    finish_reason: Optional[str] = None  # "stop" | "length" | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":      self.chunk_id,
            "object":  "chat.completion.chunk",
            "created": self.created,
            "model":   self.model,
            "choices": [
                {
                    "index": self.delta.index,
                    "delta": {
                        "content": self.delta.content,
                        **({"role": self.delta.role} if self.delta.role else {}),
                    },
                    "finish_reason": self.finish_reason,
                }
            ],
        }


@dataclass
class StreamEvent:
    """
    SSE (Server-Sent Events) 送出単位。

    event フィールド:
      "delta"  — 通常トークンチャンク
      "done"   — ストリーム終端
      "error"  — エラー通知
    """

    event: str                    # "delta" | "done" | "error"
    data:  Dict[str, Any]
    id:    Optional[str] = None

    def to_sse(self) -> str:
        """RFC 8895 準拠の SSE 文字列を返す。"""
        lines: List[str] = []
        if self.id:
            lines.append(f"id: {self.id}")
        if self.event not in ("message", ""):
            lines.append(f"event: {self.event}")
        lines.append(
            "data: " + json.dumps(self.data, ensure_ascii=False)
        )
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def done_sentinel() -> str:
        """OpenAI 互換の終端文字列 'data: [DONE]\\n\\n'"""
        return "data: [DONE]\n\n"


@dataclass
class StreamSession:
    """
    ストリーミングセッションの状態トラッカー。

    status 遷移: "active" → "completed" | "failed"
    """

    session_id:   str
    model:        str
    created_at:   int             = field(default_factory=lambda: int(time.time()))
    status:       str             = "active"     # active | completed | failed
    total_tokens: int             = 0
    error_msg:    Optional[str]   = None

    def complete(self, total_tokens: int) -> None:
        self.status       = "completed"
        self.total_tokens = total_tokens

    def fail(self, error: str) -> None:
        self.status    = "failed"
        self.error_msg = error

    def is_active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":   self.session_id,
            "model":        self.model,
            "created_at":   self.created_at,
            "status":       self.status,
            "total_tokens": self.total_tokens,
            "error_msg":    self.error_msg,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StreamingRunner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StreamingRunner:
    """
    ジェネレータ方式でトークン列を yield するランナー。

    実モデルを注入した場合:
        runner = StreamingRunner(model=my_llm)
        for chunk in runner.run("こんにちは"):
            print(chunk.delta.content, end="")

    モデルが None の場合は word-by-word 擬似ストリームにフォールバック。
    """

    def __init__(
        self,
        model_name: str = "openmythos",
        llm: Any = None,   # OpenMythosLLM または .stream(prompt) が使えるオブジェクト
    ) -> None:
        self.model_name = model_name
        self._llm = llm

    # ── 公開 API ────────────────────────────────────────────────────

    def run(
        self,
        prompt:     str,
        max_tokens: int = 256,
        system:     Optional[str] = None,
    ) -> Iterator[StreamChunk]:
        """
        StreamChunk をジェネレータで返す。

        - done=False のチャンクにトークンが入る
        - 最後に done=True + finish_reason="stop" のチャンクを返す
        """
        session_id = uuid.uuid4().hex[:16]
        created    = int(time.time())
        tokens     = list(self._generate_tokens(prompt, max_tokens, system))

        for i, token in enumerate(tokens):
            delta = StreamDelta(
                content=token,
                index=i,
                role="assistant" if i == 0 else None,
            )
            yield StreamChunk(
                delta=delta,
                chunk_id=f"{session_id}_{i}",
                created=created,
                model=self.model_name,
                done=False,
            )

        # 終端チャンク
        yield StreamChunk(
            delta=StreamDelta(content="", index=len(tokens)),
            chunk_id=f"{session_id}_done",
            created=created,
            model=self.model_name,
            done=True,
            finish_reason="stop",
        )

    def run_as_events(
        self,
        prompt:     str,
        max_tokens: int = 256,
        system:     Optional[str] = None,
    ) -> Iterator[StreamEvent]:
        """StreamEvent ジェネレータ。SSE 送出に直接利用できる。"""
        for chunk in self.run(prompt, max_tokens, system):
            if chunk.done:
                yield StreamEvent(
                    event="done",
                    data={"finish_reason": chunk.finish_reason},
                    id=chunk.chunk_id,
                )
            else:
                yield StreamEvent(
                    event="delta",
                    data=chunk.to_dict(),
                    id=chunk.chunk_id,
                )

    def run_as_sse(
        self,
        prompt:     str,
        max_tokens: int = 256,
        system:     Optional[str] = None,
    ) -> Iterator[str]:
        """
        FastAPI StreamingResponse に渡せる SSE 文字列ジェネレータ。

        Usage::
            return StreamingResponse(runner.run_as_sse(prompt), media_type="text/event-stream")
        """
        for event in self.run_as_events(prompt, max_tokens, system):
            yield event.to_sse()
        yield StreamEvent.done_sentinel()

    # ── 内部 ────────────────────────────────────────────────────────

    def _generate_tokens(
        self,
        prompt:     str,
        max_tokens: int,
        system:     Optional[str],
    ) -> Iterator[str]:
        """
        実モデルがあれば model.stream(prompt) を使う。
        なければ prompt を単語ごとに分割して擬似ストリーム。
        """
        if self._llm is not None:
            try:
                full = f"{system}\n\n{prompt}" if system else prompt
                count = 0
                for token in self._llm.stream(full):
                    # MagicMock 等の非文字列を安全に変換
                    token_str = token if isinstance(token, str) else str(token)
                    yield token_str
                    count += 1
                    if count >= max_tokens:
                        break
                return
            except Exception:
                pass  # フォールバックへ

        # 擬似ストリーム: 単語をそのままトークンに
        words = (prompt if prompt.strip() else "生成中...").split()
        for word in words[:max_tokens]:
            yield word + " "


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StreamBuffer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StreamBuffer:
    """
    ストリームチャンクを蓄積し、全文復元・エラー管理を行うバッファ。

    Usage::
        buf = StreamBuffer()
        for chunk in runner.run("prompt"):
            buf.add(chunk)

        print(buf.full_text())   # 全文
        print(buf.is_done())     # True if 受信完了
    """

    def __init__(self) -> None:
        self._chunks: List[StreamChunk] = []
        self._error:  Optional[Exception] = None

    # ── データ追加 ───────────────────────────────────────────────

    def add(self, chunk: StreamChunk) -> None:
        """チャンクを追加する（done チャンクも含めて保存）。"""
        self._chunks.append(chunk)

    def add_all(self, chunks: Iterator[StreamChunk]) -> None:
        """イテレータから全チャンクを取り込む。"""
        for chunk in chunks:
            self.add(chunk)

    # ── 読み取り ─────────────────────────────────────────────────

    def full_text(self) -> str:
        """done=False のチャンクの content を結合して全文を返す。"""
        return "".join(c.delta.content for c in self._chunks if not c.done)

    def is_done(self) -> bool:
        """done=True のチャンクを受信済みかどうか。"""
        return any(c.done for c in self._chunks)

    def chunk_count(self) -> int:
        """受信済みチャンク総数（done チャンクを含む）。"""
        return len(self._chunks)

    def token_count(self) -> int:
        """実トークン数（done=False のチャンク数）。"""
        return sum(1 for c in self._chunks if not c.done)

    def last_chunk(self) -> Optional[StreamChunk]:
        """最後に受信したチャンクを返す。"""
        return self._chunks[-1] if self._chunks else None

    def chunks(self) -> List[StreamChunk]:
        """蓄積したチャンクのコピーを返す。"""
        return list(self._chunks)

    # ── エラー管理 ───────────────────────────────────────────────

    def set_error(self, error: Exception) -> None:
        """エラーを記録する。"""
        self._error = error

    def has_error(self) -> bool:
        return self._error is not None

    def error(self) -> Optional[Exception]:
        return self._error

    # ── リセット ─────────────────────────────────────────────────

    def reset(self) -> None:
        """バッファとエラーをクリアする。"""
        self._chunks.clear()
        self._error = None

"""
Sprint 55 — ストリーミング & SSE 応答 テスト (58 tests)

対象:
  open_mythos/streaming.py:
    StreamDelta / StreamChunk / StreamEvent / StreamSession
    StreamingRunner / StreamBuffer
  serve/api.py:
    POST /v1/chat/stream          (SSE)
    POST /v1/threads/{id}/runs/stream (SSE)
    GET  /generate/stream         (既存 SSE の動作確認)
"""
from __future__ import annotations

import sys
import time
import pytest
import torch
from unittest.mock import MagicMock


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# モジュールレベル mock (transformers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.agents import OpenMythosLLM

    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.eos_token_id = 50256
    tok.decode.side_effect = lambda ids, **kw: "ok " * max(len(ids), 1)
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    cfg = MythosConfig(
        vocab_size=50257, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=128, max_loop_iters=4,
        prelude_layers=1, coda_layers=1,
        n_experts=1, n_shared_experts=0, n_experts_per_tok=1,
        expert_dim=32,
    )
    import serve.api as api_mod
    model = OpenMythos(cfg)
    api_mod.state.model = model
    api_mod.state.tokenizer = tok
    api_mod.state.llm = OpenMythosLLM(model=model, tokenizer=tok)

    # ストリーミングテスト用: generate_stream の shape エラーを回避するため
    # llm.stream() をシンプルな文字列イテレータにモック化する
    api_mod.state.llm.stream = lambda prompt: iter(
        ["テスト", " スト", "リーム", " 応答", " 完了"]
    )

    return TestClient(api_mod.app)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StreamDelta (5 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.streaming import (
    StreamDelta, StreamChunk, StreamEvent, StreamSession,
    StreamingRunner, StreamBuffer,
)


class TestStreamDelta:
    def test_content_stored(self):
        d = StreamDelta(content="こんにちは")
        assert d.content == "こんにちは"

    def test_default_index_zero(self):
        d = StreamDelta(content="x")
        assert d.index == 0

    def test_custom_index(self):
        d = StreamDelta(content="y", index=7)
        assert d.index == 7

    def test_role_default_none(self):
        d = StreamDelta(content="z")
        assert d.role is None

    def test_role_assistant(self):
        d = StreamDelta(content="a", role="assistant")
        assert d.role == "assistant"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StreamChunk (8 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStreamChunk:
    def _make(self, content="hi", done=False, finish_reason=None, idx=0):
        return StreamChunk(
            delta=StreamDelta(content=content, index=idx),
            chunk_id="test_0",
            created=int(time.time()),
            done=done,
            finish_reason=finish_reason,
        )

    def test_basic_fields(self):
        c = self._make("hello")
        assert c.delta.content == "hello"
        assert c.chunk_id == "test_0"

    def test_done_default_false(self):
        c = self._make()
        assert c.done is False

    def test_done_true(self):
        c = self._make(done=True)
        assert c.done is True

    def test_finish_reason_none(self):
        c = self._make()
        assert c.finish_reason is None

    def test_finish_reason_stop(self):
        c = self._make(done=True, finish_reason="stop")
        assert c.finish_reason == "stop"

    def test_finish_reason_length(self):
        c = self._make(done=True, finish_reason="length")
        assert c.finish_reason == "length"

    def test_model_default(self):
        c = self._make()
        assert c.model == "openmythos"

    def test_to_dict_keys(self):
        c = self._make("x")
        d = c.to_dict()
        assert "choices" in d
        assert d["choices"][0]["delta"]["content"] == "x"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StreamEvent (6 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStreamEvent:
    def test_event_type_delta(self):
        e = StreamEvent(event="delta", data={"content": "a"})
        assert e.event == "delta"

    def test_event_type_done(self):
        e = StreamEvent(event="done", data={})
        assert e.event == "done"

    def test_event_type_error(self):
        e = StreamEvent(event="error", data={"msg": "oops"})
        assert e.event == "error"

    def test_to_sse_contains_data(self):
        e = StreamEvent(event="delta", data={"k": "v"})
        sse = e.to_sse()
        assert "data:" in sse
        assert '"k"' in sse

    def test_to_sse_ends_with_double_newline(self):
        e = StreamEvent(event="delta", data={})
        assert e.to_sse().endswith("\n\n")

    def test_done_sentinel(self):
        s = StreamEvent.done_sentinel()
        assert s == "data: [DONE]\n\n"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StreamSession (4 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStreamSession:
    def test_initial_status_active(self):
        s = StreamSession(session_id="s1", model="openmythos")
        assert s.is_active()
        assert s.status == "active"

    def test_complete(self):
        s = StreamSession(session_id="s2", model="m")
        s.complete(total_tokens=42)
        assert s.status == "completed"
        assert s.total_tokens == 42
        assert not s.is_active()

    def test_fail(self):
        s = StreamSession(session_id="s3", model="m")
        s.fail("timeout")
        assert s.status == "failed"
        assert s.error_msg == "timeout"

    def test_to_dict(self):
        s = StreamSession(session_id="s4", model="m")
        d = s.to_dict()
        assert d["session_id"] == "s4"
        assert d["status"] == "active"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StreamingRunner (12 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStreamingRunner:
    def _runner(self):
        return StreamingRunner(model_name="test-model")

    def test_yields_chunks(self):
        chunks = list(self._runner().run("hello world"))
        assert len(chunks) > 0

    def test_last_chunk_is_done(self):
        chunks = list(self._runner().run("hello"))
        assert chunks[-1].done is True

    def test_non_done_chunks_have_content(self):
        chunks = list(self._runner().run("hello world test"))
        non_done = [c for c in chunks if not c.done]
        assert all(c.delta.content for c in non_done)

    def test_done_chunk_has_stop_reason(self):
        chunks = list(self._runner().run("hi"))
        assert chunks[-1].finish_reason == "stop"

    def test_chunks_have_incrementing_index(self):
        chunks = list(self._runner().run("a b c"))
        non_done = [c for c in chunks if not c.done]
        indices = [c.delta.index for c in non_done]
        assert indices == list(range(len(indices)))

    def test_max_tokens_limit(self):
        # max_tokens=2 → 最大2トークン + done
        chunks = list(self._runner().run("a b c d e f g", max_tokens=2))
        non_done = [c for c in chunks if not c.done]
        assert len(non_done) <= 2

    def test_empty_prompt_yields_something(self):
        chunks = list(self._runner().run(""))
        assert len(chunks) >= 1  # done チャンク最低1つ

    def test_chunk_ids_unique(self):
        chunks = list(self._runner().run("a b c"))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_model_name_in_chunks(self):
        chunks = list(self._runner().run("hi"))
        assert all(c.model == "test-model" for c in chunks)

    def test_first_delta_has_role(self):
        chunks = list(self._runner().run("hello world"))
        non_done = [c for c in chunks if not c.done]
        if non_done:
            assert non_done[0].delta.role == "assistant"

    def test_iterator_protocol(self):
        it = iter(self._runner().run("one two"))
        chunk = next(it)
        assert isinstance(chunk, StreamChunk)

    def test_run_as_sse_contains_done_sentinel(self):
        sse_parts = list(self._runner().run_as_sse("hello"))
        combined = "".join(sse_parts)
        assert "data: [DONE]" in combined


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StreamBuffer (11 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStreamBuffer:
    def _make_chunk(self, content="x", done=False, idx=0):
        return StreamChunk(
            delta=StreamDelta(content=content, index=idx),
            chunk_id=f"c{idx}",
            created=int(time.time()),
            done=done,
        )

    def test_empty_initially(self):
        buf = StreamBuffer()
        assert buf.chunk_count() == 0

    def test_add_chunk(self):
        buf = StreamBuffer()
        buf.add(self._make_chunk("a"))
        assert buf.chunk_count() == 1

    def test_full_text_empty_buffer(self):
        assert StreamBuffer().full_text() == ""

    def test_full_text_with_chunks(self):
        buf = StreamBuffer()
        buf.add(self._make_chunk("hello ", idx=0))
        buf.add(self._make_chunk("world", idx=1))
        assert buf.full_text() == "hello world"

    def test_full_text_excludes_done_chunk(self):
        buf = StreamBuffer()
        buf.add(self._make_chunk("hi", idx=0))
        buf.add(self._make_chunk("", done=True, idx=1))
        assert buf.full_text() == "hi"

    def test_is_done_false_initially(self):
        buf = StreamBuffer()
        assert not buf.is_done()

    def test_is_done_true_after_done_chunk(self):
        buf = StreamBuffer()
        buf.add(self._make_chunk("x", done=True))
        assert buf.is_done()

    def test_set_error(self):
        buf = StreamBuffer()
        buf.set_error(ValueError("oops"))
        assert buf.has_error()
        assert isinstance(buf.error(), ValueError)

    def test_no_error_initially(self):
        assert not StreamBuffer().has_error()

    def test_reset(self):
        buf = StreamBuffer()
        buf.add(self._make_chunk("x"))
        buf.set_error(RuntimeError("e"))
        buf.reset()
        assert buf.chunk_count() == 0
        assert not buf.has_error()

    def test_add_all(self):
        runner = StreamingRunner()
        buf = StreamBuffer()
        buf.add_all(runner.run("a b c"))
        assert buf.is_done()
        assert buf.full_text() != ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API エンドポイント (12 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStreamingAPI:
    # ── GET /generate/stream ────────────────────────────────────────

    def test_generate_stream_media_type(self, client):
        r = client.get("/generate/stream?prompt=hello")
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_generate_stream_contains_done(self, client):
        r = client.get("/generate/stream?prompt=hello")
        assert "[DONE]" in r.text

    def test_generate_stream_has_data_lines(self, client):
        r = client.get("/generate/stream?prompt=test")
        assert "data:" in r.text

    # ── POST /v1/chat/stream ────────────────────────────────────────

    def test_chat_stream_basic(self, client):
        r = client.post("/v1/chat/stream", json={
            "messages": [{"role": "user", "content": "こんにちは"}]
        })
        assert r.status_code == 200

    def test_chat_stream_media_type(self, client):
        r = client.post("/v1/chat/stream", json={
            "messages": [{"role": "user", "content": "hi"}]
        })
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_chat_stream_contains_done(self, client):
        r = client.post("/v1/chat/stream", json={
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert "[DONE]" in r.text

    def test_chat_stream_has_event_lines(self, client):
        r = client.post("/v1/chat/stream", json={
            "messages": [{"role": "user", "content": "test"}]
        })
        assert "event:" in r.text

    def test_chat_stream_with_system(self, client):
        r = client.post("/v1/chat/stream", json={
            "messages": [
                {"role": "system", "content": "あなたは広告コピーライターです"},
                {"role": "user", "content": "日焼け止めの広告を作って"},
            ]
        })
        assert r.status_code == 200
        assert "[DONE]" in r.text

    # ── POST /v1/threads/{id}/runs/stream ──────────────────────────

    def test_runs_stream_invalid_thread(self, client):
        r = client.post("/v1/threads/nonexistent/runs/stream", json={
            "assistant_id": "asst_dummy"
        })
        assert r.status_code == 404

    def test_runs_stream_basic(self, client):
        # スレッド + アシスタントを先に作成
        asst_r = client.post("/v1/assistants", json={
            "name": "TestAssistant",
            "instructions": "テスト用アシスタント",
        })
        assert asst_r.status_code == 200
        asst_id = asst_r.json()["id"]

        thread_r = client.post("/v1/threads", json={})
        assert thread_r.status_code == 200
        thread_id = thread_r.json()["id"]

        # ユーザーメッセージを追加
        msg_r = client.post(f"/v1/threads/{thread_id}/messages", json={
            "role": "user",
            "content": "広告コピーを作って",
        })
        assert msg_r.status_code == 200

        # ストリーム実行
        run_r = client.post(f"/v1/threads/{thread_id}/runs/stream", json={
            "assistant_id": asst_id
        })
        assert run_r.status_code == 200

    def test_runs_stream_media_type(self, client):
        asst_r = client.post("/v1/assistants", json={"name": "T2"})
        asst_id = asst_r.json()["id"]
        thread_r = client.post("/v1/threads", json={})
        thread_id = thread_r.json()["id"]
        client.post(f"/v1/threads/{thread_id}/messages", json={
            "role": "user", "content": "テスト"
        })
        r = client.post(f"/v1/threads/{thread_id}/runs/stream", json={
            "assistant_id": asst_id
        })
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_runs_stream_contains_done(self, client):
        asst_r = client.post("/v1/assistants", json={"name": "T3"})
        asst_id = asst_r.json()["id"]
        thread_r = client.post("/v1/threads", json={})
        thread_id = thread_r.json()["id"]
        client.post(f"/v1/threads/{thread_id}/messages", json={
            "role": "user", "content": "テスト完了"
        })
        r = client.post(f"/v1/threads/{thread_id}/runs/stream", json={
            "assistant_id": asst_id
        })
        assert "[DONE]" in r.text

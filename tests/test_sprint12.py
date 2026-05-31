"""
Sprint 12 テスト

Track A — ReAct エージェントループ
  12.1.1  open_mythos/react.py — AgentStep / AgentResult / ReActAgent / format_agent_trace
  12.1.2  serve/api.py — /v1/agent/run

Track B — Prompt KV Prefix Cache
  12.2.1  open_mythos/prefix_cache.py — PrefixCacheEntry / PromptPrefixCache / CachedGenResult

Track C — Conversation Memory / Session API
  12.3.1  open_mythos/conversation.py — Turn / ConversationMemory / SessionStore
  12.3.2  serve/api.py — /v1/sessions/*
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch


# ===========================================================================
# serve/api.py mock (transformers)
# ===========================================================================


def _make_tok_mock() -> MagicMock:
    tok = MagicMock()
    tok.side_effect = lambda text, **kw: {
        "input_ids": torch.zeros(1, max(1, len(str(text).split())), dtype=torch.long)
    }
    tok.decode = MagicMock(return_value="generated text")
    tok.eos_token_id = 0
    tok.vocab_size = 50257
    return tok


@pytest.fixture(scope="module", autouse=True)
def mock_transformers_sprint12():
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=_make_tok_mock())
    orig = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    yield
    if orig is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = orig


# ===========================================================================
# ヘルパー
# ===========================================================================


def _tiny_cfg():
    from open_mythos.main import MythosConfig
    return MythosConfig(
        vocab_size=512, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=128, max_loop_iters=4, prelude_layers=1, coda_layers=1,
        n_experts=4, n_shared_experts=1, n_experts_per_tok=1, expert_dim=32, lora_rank=4,
    )


def _tiny_model():
    from open_mythos.main import OpenMythos
    return OpenMythos(_tiny_cfg()).eval()


# ===========================================================================
# Track A — 12.1.1  AgentStep / AgentResult
# ===========================================================================


class TestAgentStep:
    def test_defaults(self):
        from open_mythos.react import AgentStep
        step = AgentStep(step_type="thought", content="Let me think...")
        assert step.step_type == "thought"
        assert step.tool_call is None
        assert step.tool_result is None
        assert step.iteration == 0

    def test_step_types(self):
        from open_mythos.react import AgentStep
        for stype in ("thought", "action", "observation", "answer"):
            step = AgentStep(step_type=stype, content="test")
            assert step.step_type == stype


class TestAgentResult:
    def _make_result(self):
        from open_mythos.react import AgentResult, AgentStep
        steps = [
            AgentStep(step_type="thought", content="thinking...", iteration=0),
            AgentStep(step_type="action", content="calling tool", iteration=0),
            AgentStep(step_type="observation", content="result", iteration=0),
            AgentStep(step_type="answer", content="final", iteration=0),
        ]
        return AgentResult(
            task="test task",
            final_answer="final answer",
            steps=steps,
            iterations_used=1,
            total_latency_ms=42.0,
            stopped_reason="completed",
        )

    def test_thought_steps(self):
        r = self._make_result()
        assert len(r.thought_steps) == 1

    def test_action_steps(self):
        r = self._make_result()
        assert len(r.action_steps) == 1

    def test_n_tool_calls(self):
        r = self._make_result()
        assert r.n_tool_calls == 1

    def test_observation_steps(self):
        r = self._make_result()
        assert len(r.observation_steps) == 1


class TestReActAgent:
    def test_instantiation(self):
        from open_mythos.react import ReActAgent
        from open_mythos.tools import ToolRegistry
        model = _tiny_model()
        reg = ToolRegistry()
        agent = ReActAgent(model, reg, max_iterations=3)
        assert agent.max_iterations == 3

    def test_run_no_tools_returns_result(self):
        """ツールなしレジストリでもクラッシュせず AgentResult を返す。"""
        from open_mythos.react import ReActAgent
        from open_mythos.tools import ToolRegistry
        model = _tiny_model()
        reg = ToolRegistry()
        agent = ReActAgent(model, reg, max_iterations=2, max_new_tokens=5, loops=1)
        result = agent.run(task="テストタスク")
        assert result.task == "テストタスク"
        assert isinstance(result.final_answer, str)
        assert result.iterations_used >= 1
        assert result.stopped_reason in ("completed", "max_iterations", "no_tools_needed")

    def test_run_with_marketing_tools(self):
        """マーケ特化ツールを持つ registry でエージェントが動作する。"""
        from open_mythos.react import ReActAgent
        from open_mythos.tools import ToolRegistry
        model = _tiny_model()
        reg = ToolRegistry.default()
        agent = ReActAgent(model, reg, max_iterations=2, max_new_tokens=8, loops=1)
        result = agent.run(task="競合他社の広告費を調べて")
        assert result is not None
        assert isinstance(result.steps, list)

    def test_result_has_steps(self):
        from open_mythos.react import ReActAgent
        from open_mythos.tools import ToolRegistry
        model = _tiny_model()
        reg = ToolRegistry()
        agent = ReActAgent(model, reg, max_iterations=1, max_new_tokens=3, loops=1)
        result = agent.run(task="test")
        assert len(result.steps) >= 1

    def test_max_iterations_respected(self):
        from open_mythos.react import ReActAgent
        from open_mythos.tools import ToolRegistry

        model = _tiny_model()
        reg = ToolRegistry()
        max_iter = 2
        agent = ReActAgent(model, reg, max_iterations=max_iter, max_new_tokens=3, loops=1)
        result = agent.run(task="loop test")
        assert result.iterations_used <= max_iter

    def test_latency_recorded(self):
        from open_mythos.react import ReActAgent
        from open_mythos.tools import ToolRegistry
        model = _tiny_model()
        agent = ReActAgent(model, ToolRegistry(), max_iterations=1, max_new_tokens=3, loops=1)
        result = agent.run(task="latency test")
        assert result.total_latency_ms >= 0.0

    def test_max_iterations_zero_no_crash(self):
        """max_iterations=0 で UnboundLocalError が出ないことを確認（バグ修正テスト）。"""
        from open_mythos.react import ReActAgent
        from open_mythos.tools import ToolRegistry
        model = _tiny_model()
        agent = ReActAgent(model, ToolRegistry(), max_iterations=0, max_new_tokens=3, loops=1)
        result = agent.run(task="zero iterations test")
        assert result.iterations_used == 0
        assert isinstance(result.final_answer, str)


class TestFormatAgentTrace:
    def test_returns_string(self):
        from open_mythos.react import ReActAgent, format_agent_trace
        from open_mythos.tools import ToolRegistry
        model = _tiny_model()
        agent = ReActAgent(model, ToolRegistry(), max_iterations=1, max_new_tokens=3, loops=1)
        result = agent.run(task="trace test")
        trace = format_agent_trace(result)
        assert isinstance(trace, str)
        assert "Task:" in trace
        assert "Final Answer:" in trace

    def test_max_content_len_respected(self):
        from open_mythos.react import AgentResult, AgentStep, format_agent_trace
        steps = [AgentStep(step_type="thought", content="x" * 300, iteration=0)]
        result = AgentResult(
            task="t", final_answer="a", steps=steps,
            iterations_used=1, total_latency_ms=1.0,
        )
        trace = format_agent_trace(result, max_content_len=50)
        # 行内容が max_content_len + "..." に収まっているか
        for line in trace.splitlines():
            # trace line の content 部分だけ確認 (行全体ではなく)
            if "THOUGHT:" in line.upper():
                assert len(line) < 300


# ===========================================================================
# Track A — 12.1.2  serve/api.py — /v1/agent/run
# ===========================================================================


class TestAPIAgentEndpoints:
    def test_agent_run_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/agent/run" in routes

    def test_agent_run_request_model(self):
        from serve.api import AgentRunRequest
        req = AgentRunRequest(task="test task", max_iterations=3)
        assert req.task == "test task"
        assert req.max_iterations == 3

    def test_agent_run_response_model(self):
        from serve.api import AgentRunResponse, AgentStepResponse
        resp = AgentRunResponse(
            task="t",
            final_answer="a",
            steps=[AgentStepResponse(step_type="thought", content="x", iteration=0, latency_ms=1.0)],
            iterations_used=1,
            n_tool_calls=0,
            stopped_reason="completed",
            total_latency_ms=10.0,
        )
        assert resp.n_tool_calls == 0


# ===========================================================================
# Track B — 12.2.1  PromptPrefixCache
# ===========================================================================


class TestPrefixCacheEntry:
    def test_defaults(self):
        from open_mythos.prefix_cache import PrefixCacheEntry
        entry = PrefixCacheEntry(
            prefix_text="hello",
            prefix_ids=[104, 101, 108, 108, 111],
            cached_logits=torch.randn(512),
            n_loops=4,
        )
        assert entry.prefix_len == 5
        assert entry.hits == 0
        assert len(entry.cache_key) == 16

    def test_cache_key_deterministic(self):
        from open_mythos.prefix_cache import PrefixCacheEntry
        e1 = PrefixCacheEntry("abc", [97, 98, 99], torch.zeros(512), 4)
        e2 = PrefixCacheEntry("abc", [97, 98, 99], torch.zeros(512), 4)
        assert e1.cache_key == e2.cache_key


class TestPromptPrefixCache:
    def test_cache_and_get(self):
        from open_mythos.prefix_cache import PromptPrefixCache
        model = _tiny_model()
        cache = PromptPrefixCache(model, max_entries=10)
        prefix = "システムプロンプト: あなたはAIです。"
        entry = cache.cache_prefix(prefix, n_loops=1)
        assert entry is not None
        assert len(cache) == 1

        # 2回目はキャッシュヒット
        entry2 = cache.get(prefix)
        assert entry2 is not None
        assert entry2.hits == 1

    def test_lru_eviction(self):
        from open_mythos.prefix_cache import PromptPrefixCache
        model = _tiny_model()
        cache = PromptPrefixCache(model, max_entries=3)
        for i in range(4):
            cache.cache_prefix(f"prefix_{i}", n_loops=1)
        assert len(cache) == 3

    def test_hit_rate(self):
        from open_mythos.prefix_cache import PromptPrefixCache
        model = _tiny_model()
        cache = PromptPrefixCache(model, max_entries=10)
        cache.cache_prefix("hit_test", n_loops=1)
        cache.get("hit_test")  # hit
        cache.get("miss_test")  # miss
        assert 0.0 <= cache.hit_rate <= 1.0

    def test_stats(self):
        from open_mythos.prefix_cache import PromptPrefixCache
        model = _tiny_model()
        cache = PromptPrefixCache(model, max_entries=5)
        stats = cache.stats
        assert "n_entries" in stats
        assert "hit_rate" in stats

    def test_clear(self):
        from open_mythos.prefix_cache import PromptPrefixCache
        model = _tiny_model()
        cache = PromptPrefixCache(model, max_entries=5)
        cache.cache_prefix("to_clear", n_loops=1)
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

    def test_evict_specific(self):
        from open_mythos.prefix_cache import PromptPrefixCache
        model = _tiny_model()
        cache = PromptPrefixCache(model, max_entries=5)
        cache.cache_prefix("evict_me", n_loops=1)
        result = cache.evict("evict_me")
        assert result is True
        assert len(cache) == 0

    def test_generate_with_cache_miss(self):
        from open_mythos.prefix_cache import PromptPrefixCache
        model = _tiny_model()
        cache = PromptPrefixCache(model, max_entries=5)
        result = cache.generate_with_cache("テスト", max_new_tokens=3, loops=1)
        assert isinstance(result.text, str)
        assert result.cache_hit is False
        assert result.prefill_skipped_tokens == 0

    def test_generate_with_cache_hit(self):
        from open_mythos.prefix_cache import PromptPrefixCache
        model = _tiny_model()
        cache = PromptPrefixCache(model, max_entries=5)
        prefix = "共通プレフィックス"
        cache.cache_prefix(prefix, n_loops=1)
        result = cache.generate_with_cache(prefix, max_new_tokens=3, loops=1)
        assert result.cache_hit is True

    def test_generate_with_cache_partial_prefix(self):
        from open_mythos.prefix_cache import PromptPrefixCache
        model = _tiny_model()
        cache = PromptPrefixCache(model, max_entries=5)
        prompt = "システムプロンプト: AIです。ユーザー: 質問"
        result = cache.generate_with_cache(
            prompt, max_new_tokens=3, loops=1, cache_prefix_len=12
        )
        assert isinstance(result.text, str)
        assert result.prefix_len == 12


class TestCachedGenResult:
    def test_fields(self):
        from open_mythos.prefix_cache import CachedGenResult
        r = CachedGenResult(
            text="generated",
            prompt_used="prompt",
            cache_hit=True,
            cache_key="abc123",
            prefix_len=10,
            latency_ms=5.0,
            prefill_skipped_tokens=10,
        )
        assert r.cache_hit is True
        assert r.prefill_skipped_tokens == 10


# ===========================================================================
# Track C — 12.3.1  ConversationMemory / SessionStore
# ===========================================================================


class TestTurn:
    def test_defaults(self):
        from open_mythos.conversation import Turn
        t = Turn(role="user", content="hello")
        assert t.role == "user"
        assert t.char_len == 5
        assert len(t.turn_id) == 8

    def test_to_dict(self):
        from open_mythos.conversation import Turn
        t = Turn(role="assistant", content="response")
        d = t.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "response"

    def test_from_dict(self):
        from open_mythos.conversation import Turn
        d = {"role": "user", "content": "hi", "turn_id": "abc12345", "created_at": 1.0}
        t = Turn.from_dict(d)
        assert t.role == "user"
        assert t.turn_id == "abc12345"


class TestConversationMemory:
    def test_add_turn(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory()
        t = mem.add_turn("user", "こんにちは")
        assert mem.n_turns == 1
        assert t.role == "user"

    def test_add_user_assistant(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_user("質問です")
        mem.add_assistant("回答です")
        assert mem.n_turns == 2
        assert mem.last_user_turn.content == "質問です"
        assert mem.last_assistant_turn.content == "回答です"

    def test_to_context_string(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory(system_msg="AIです")
        mem.add_user("質問")
        mem.add_assistant("回答")
        ctx = mem.to_context_string()
        assert "System: AIです" in ctx
        assert "Human: 質問" in ctx
        assert "Assistant: 回答" in ctx

    def test_to_messages_openai_format(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_user("test")
        msgs = mem.to_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_auto_compress_on_max_turns(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory(max_turns=4)
        for i in range(6):
            mem.add_user(f"メッセージ {i}")
        # max_turns を超えたので圧縮が発生しているはず
        assert mem.n_turns <= 4
        assert mem.has_summary

    def test_auto_compress_on_max_chars(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory(max_chars=100)
        for i in range(5):
            mem.add_user("a" * 30)  # 30文字 × 5 = 150 > 100
        assert mem.has_summary

    def test_compress_now(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory()
        for i in range(6):
            mem.add_user(f"msg {i}")
        summary = mem.compress_now(keep_n=2)
        assert summary is not None
        assert summary.turns_summarized >= 4
        assert mem.n_turns == 2

    def test_pop_last(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_user("to pop")
        t = mem.pop_last()
        assert t.content == "to pop"
        assert mem.n_turns == 0

    def test_clear(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_user("x")
        mem.clear()
        assert mem.n_turns == 0
        assert not mem.has_summary

    def test_stats(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_user("test")
        s = mem.stats()
        assert "session_id" in s
        assert s["n_turns"] == 1

    def test_summary_includes_earlier_content(self):
        from open_mythos.conversation import ConversationMemory
        mem = ConversationMemory(max_turns=2)
        mem.add_user("古いメッセージ")
        mem.add_user("最新メッセージ1")
        mem.add_user("最新メッセージ2")
        ctx = mem.to_context_string(include_summary=True)
        assert "[Earlier conversation summary]" in ctx or "古いメッセージ" in ctx


class TestMemorySummary:
    def test_compression_ratio(self):
        from open_mythos.conversation import MemorySummary
        s = MemorySummary(text="short", turns_summarized=5, original_chars=100)
        assert 0.0 <= s.compression_ratio <= 1.0
        assert s.compression_ratio == pytest.approx(1.0 - 5 / 100)


class TestSessionStore:
    def test_create_and_get(self):
        from open_mythos.conversation import SessionStore
        store = SessionStore()
        sid = store.create_session(system_msg="テストシステム")
        mem = store.get(sid)
        assert mem is not None
        assert mem.system_msg == "テストシステム"

    def test_create_with_explicit_id(self):
        from open_mythos.conversation import SessionStore
        store = SessionStore()
        sid = store.create_session(session_id="my-session-001")
        assert sid == "my-session-001"

    def test_get_nonexistent_returns_none(self):
        from open_mythos.conversation import SessionStore
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_get_or_create(self):
        from open_mythos.conversation import SessionStore
        store = SessionStore()
        sid, mem = store.get_or_create(system_msg="hello")
        assert mem is not None
        # 同じ id で再取得
        sid2, mem2 = store.get_or_create(session_id=sid)
        assert sid == sid2
        assert mem is mem2

    def test_delete(self):
        from open_mythos.conversation import SessionStore
        store = SessionStore()
        sid = store.create_session()
        assert store.delete(sid) is True
        assert store.get(sid) is None

    def test_delete_nonexistent_returns_false(self):
        from open_mythos.conversation import SessionStore
        store = SessionStore()
        assert store.delete("ghost") is False

    def test_max_sessions_lru_eviction(self):
        from open_mythos.conversation import SessionStore
        store = SessionStore(max_sessions=3)
        for i in range(4):
            store.create_session(session_id=f"session-{i}")
        assert len(store) == 3

    def test_list_sessions(self):
        from open_mythos.conversation import SessionStore
        store = SessionStore()
        store.create_session(session_id="s1")
        store.create_session(session_id="s2")
        sessions = store.list_sessions()
        assert "s1" in sessions
        assert "s2" in sessions

    def test_stats(self):
        from open_mythos.conversation import SessionStore
        store = SessionStore(max_sessions=10)
        store.create_session()
        s = store.stats()
        assert s["n_sessions"] == 1
        assert s["max_sessions"] == 10

    def test_ttl_eviction(self):
        import time
        from open_mythos.conversation import SessionStore
        store = SessionStore(ttl_seconds=0.05)  # 50ms
        sid = store.create_session()
        time.sleep(0.1)  # TTL を超過させる
        store._evict_expired()
        assert store.get(sid) is None


# ===========================================================================
# Track C — 12.3.2  serve/api.py — /v1/sessions
# ===========================================================================


class TestAPISessionsEndpoints:
    def test_sessions_post_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/sessions" in routes

    def test_sessions_get_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/sessions/{session_id}" in routes

    def test_sessions_delete_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/sessions/{session_id}" in routes

    def test_sessions_turns_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/sessions/{session_id}/turns" in routes

    def test_sessions_context_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/sessions/{session_id}/context" in routes

    def test_session_create_request_model(self):
        from serve.api import SessionCreateRequest
        req = SessionCreateRequest(session_id="test-session", system_msg="AI です")
        assert req.session_id == "test-session"

    def test_session_stats_response_model(self):
        from serve.api import SessionStatsResponse
        r = SessionStatsResponse(
            session_id="s1", n_turns=3, total_chars=100,
            has_summary=False, summary_turns=0,
        )
        assert r.n_turns == 3

    def test_turn_add_request_model(self):
        from serve.api import TurnAddRequest
        req = TurnAddRequest(role="user", content="こんにちは")
        assert req.role == "user"


# ===========================================================================
# 統合テスト: __init__.py からの import
# ===========================================================================


class TestSprint12Imports:
    def test_react_importable(self):
        from open_mythos import ReActAgent
        assert ReActAgent is not None

    def test_prefix_cache_importable(self):
        from open_mythos import PromptPrefixCache
        assert PromptPrefixCache is not None

    def test_conversation_importable(self):
        from open_mythos import ConversationMemory, SessionStore
        assert ConversationMemory is not None
        assert SessionStore is not None

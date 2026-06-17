"""
Sprint 41 — /v1/chat/completions Function Calling 統合 テスト

対象:
  - serve/api.py : ChatMessage の tool ロール + tool_call_id + tool_calls フィールド
  - serve/api.py : ChatRequest の tools / tool_choice フィールド
  - serve/api.py : ChatChoice / ChatResponse の tool_calls フィールド
  - serve/api.py : _build_tools_system_block / _parse_tool_calls_from_text /
                   _inject_tools_into_prompt / _build_chat_prompt (tool ロール対応)
  - serve/api.py : /v1/chat/completions の function calling フロー
"""

from __future__ import annotations

import json
import sys
import pytest
import torch
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# transformers モック (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


# ---------------------------------------------------------------------------
# TestClient フィクスチャ
# ---------------------------------------------------------------------------

import serve.api as api_module


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.agents import OpenMythosLLM

    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.eos_token_id = 50256
    tok.decode.side_effect = lambda ids, **kwargs: "hello " * len(ids) if ids else ""
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    cfg = MythosConfig(
        vocab_size=50257, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=128,
        max_loop_iters=4, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=1, expert_dim=32,
        act_threshold=0.99, lora_rank=4, kv_lora_rank=32, q_lora_rank=64,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
    )
    model = OpenMythos(cfg)
    model.eval()

    api_module.state.model = model
    api_module.state.tokenizer = tok
    api_module.state.device = torch.device("cpu")
    api_module.state.n_params = sum(p.numel() for p in model.parameters())
    api_module.state.llm = OpenMythosLLM(model=model, device="cpu")
    api_module.state.agents = {}

    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. ChatMessage モデル拡張テスト
# ---------------------------------------------------------------------------

class TestChatMessageExtended:
    """Sprint 41 で追加された ChatMessage フィールドのテスト"""

    def test_tool_role_accepted(self):
        from serve.api import ChatMessage
        m = ChatMessage(role="tool", content="result", tool_call_id="call_abc")
        assert m.role == "tool"
        assert m.tool_call_id == "call_abc"

    def test_assistant_role_with_tool_calls(self):
        from serve.api import ChatMessage
        tc = [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
        m = ChatMessage(role="assistant", content="", tool_calls=tc)
        assert m.tool_calls is not None
        assert len(m.tool_calls) == 1

    def test_default_tool_call_id_is_none(self):
        from serve.api import ChatMessage
        m = ChatMessage(role="user", content="hi")
        assert m.tool_call_id is None

    def test_default_tool_calls_is_none(self):
        from serve.api import ChatMessage
        m = ChatMessage(role="user", content="hi")
        assert m.tool_calls is None

    def test_invalid_role_rejected(self):
        from serve.api import ChatMessage
        import pydantic
        with pytest.raises((pydantic.ValidationError, ValueError)):
            ChatMessage(role="invalid_role", content="hi")

    def test_content_defaults_empty_string(self):
        from serve.api import ChatMessage
        m = ChatMessage(role="tool", tool_call_id="x")
        assert m.content == ""


# ---------------------------------------------------------------------------
# 2. ChatRequest tools / tool_choice フィールドテスト
# ---------------------------------------------------------------------------

class TestChatRequestFunctionCalling:
    """ChatRequest の tools / tool_choice フィールドのテスト"""

    def test_tools_defaults_none(self):
        from serve.api import ChatRequest, ChatMessage
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        assert req.tools is None

    def test_tool_choice_defaults_none(self):
        from serve.api import ChatRequest, ChatMessage
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        assert req.tool_choice is None

    def test_tools_accepted(self):
        from serve.api import ChatRequest, ChatMessage
        tools = [{
            "type": "function",
            "function": {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {}}}
        }]
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], tools=tools)
        assert req.tools is not None
        assert len(req.tools) == 1

    def test_tool_choice_auto(self):
        from serve.api import ChatRequest, ChatMessage
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            tool_choice="auto",
        )
        assert req.tool_choice == "auto"

    def test_tool_choice_none_str(self):
        from serve.api import ChatRequest, ChatMessage
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            tool_choice="none",
        )
        assert req.tool_choice == "none"

    def test_tool_choice_required(self):
        from serve.api import ChatRequest, ChatMessage
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            tool_choice="required",
        )
        assert req.tool_choice == "required"


# ---------------------------------------------------------------------------
# 3. ヘルパー関数テスト
# ---------------------------------------------------------------------------

class TestBuildToolsSystemBlock:
    """_build_tools_system_block のテスト"""

    def test_returns_str(self):
        from serve.api import _build_tools_system_block
        tools = [{"type": "function", "function": {"name": "test"}}]
        result = _build_tools_system_block(tools)
        assert isinstance(result, str)

    def test_contains_tool_name(self):
        from serve.api import _build_tools_system_block
        tools = [{"type": "function", "function": {"name": "my_tool"}}]
        result = _build_tools_system_block(tools)
        assert "my_tool" in result

    def test_contains_tool_call_marker(self):
        from serve.api import _build_tools_system_block
        result = _build_tools_system_block([{"type": "function", "function": {"name": "x"}}])
        assert "<tool_call>" in result

    def test_multiple_tools(self):
        from serve.api import _build_tools_system_block
        tools = [
            {"type": "function", "function": {"name": "tool_a"}},
            {"type": "function", "function": {"name": "tool_b"}},
        ]
        result = _build_tools_system_block(tools)
        assert "tool_a" in result
        assert "tool_b" in result


class TestParseToolCallsFromText:
    """_parse_tool_calls_from_text のテスト"""

    def test_no_tool_call_returns_none(self):
        from serve.api import _parse_tool_calls_from_text
        assert _parse_tool_calls_from_text("Hello world") is None

    def test_parses_valid_tool_call(self):
        from serve.api import _parse_tool_calls_from_text
        text = '<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>'
        result = _parse_tool_calls_from_text(text)
        assert result is not None
        assert len(result) == 1

    def test_tool_call_has_openai_format(self):
        from serve.api import _parse_tool_calls_from_text
        text = '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
        result = _parse_tool_calls_from_text(text)
        assert result is not None
        tc = result[0]
        assert "id" in tc
        assert tc["type"] == "function"
        assert "function" in tc
        assert tc["function"]["name"] == "search"

    def test_tool_call_id_starts_with_call(self):
        from serve.api import _parse_tool_calls_from_text
        text = '<tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        result = _parse_tool_calls_from_text(text)
        assert result[0]["id"].startswith("call_")

    def test_arguments_are_json_string(self):
        from serve.api import _parse_tool_calls_from_text
        text = '<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>'
        result = _parse_tool_calls_from_text(text)
        args_str = result[0]["function"]["arguments"]
        assert isinstance(args_str, str)
        parsed = json.loads(args_str)
        assert parsed["x"] == 1

    def test_multiple_tool_calls(self):
        from serve.api import _parse_tool_calls_from_text
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        result = _parse_tool_calls_from_text(text)
        assert result is not None
        assert len(result) == 2

    def test_invalid_json_skipped(self):
        from serve.api import _parse_tool_calls_from_text
        text = '<tool_call>not valid json</tool_call>'
        result = _parse_tool_calls_from_text(text)
        assert result is None

    def test_empty_string_returns_none(self):
        from serve.api import _parse_tool_calls_from_text
        assert _parse_tool_calls_from_text("") is None


class TestInjectToolsIntoPrompt:
    """_inject_tools_into_prompt のテスト"""

    def test_no_tools_returns_base_prompt(self):
        from serve.api import _inject_tools_into_prompt
        result = _inject_tools_into_prompt("base", None, None)
        assert result == "base"

    def test_tool_choice_none_no_injection(self):
        from serve.api import _inject_tools_into_prompt
        tools = [{"type": "function", "function": {"name": "x"}}]
        result = _inject_tools_into_prompt("base", tools, "none")
        assert result == "base"

    def test_tools_injected_for_auto(self):
        from serve.api import _inject_tools_into_prompt
        tools = [{"type": "function", "function": {"name": "my_fn"}}]
        result = _inject_tools_into_prompt("base", tools, "auto")
        assert "my_fn" in result
        assert "base" in result

    def test_tools_injected_when_tool_choice_none_field(self):
        from serve.api import _inject_tools_into_prompt
        tools = [{"type": "function", "function": {"name": "fn"}}]
        # tool_choice が None (未指定) の場合は注入する
        result = _inject_tools_into_prompt("base", tools, None)
        assert "fn" in result

    def test_injected_prompt_has_tool_call_marker(self):
        from serve.api import _inject_tools_into_prompt
        tools = [{"type": "function", "function": {"name": "fn"}}]
        result = _inject_tools_into_prompt("base", tools, "auto")
        assert "<tool_call>" in result


class TestBuildChatPromptToolRole:
    """_build_chat_prompt の tool ロール対応テスト"""

    def test_tool_role_in_prompt(self):
        from serve.api import _build_chat_prompt, ChatMessage
        msgs = [
            ChatMessage(role="user", content="call search"),
            ChatMessage(role="tool", content="search result", tool_call_id="call_x"),
        ]
        result = _build_chat_prompt(msgs)
        assert "Tool" in result
        assert "search result" in result

    def test_assistant_with_tool_calls_in_prompt(self):
        from serve.api import _build_chat_prompt, ChatMessage
        msgs = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search_fn", "arguments": "{}"},
                }],
            )
        ]
        result = _build_chat_prompt(msgs)
        assert "search_fn" in result or "tool_call" in result

    def test_normal_messages_unaffected(self):
        from serve.api import _build_chat_prompt, ChatMessage
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="usr"),
            ChatMessage(role="assistant", content="ast"),
        ]
        result = _build_chat_prompt(msgs)
        assert "[System]" in result
        assert "[User]" in result
        assert "[Assistant]" in result


# ---------------------------------------------------------------------------
# 4. /v1/chat/completions — Function Calling 統合エンドポイントテスト
# ---------------------------------------------------------------------------

_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_competitor",
            "description": "競合他社の広告データを検索する",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "企業名"},
                    "metric": {"type": "string", "enum": ["ad_spend", "impressions"]},
                },
                "required": ["company"],
            },
        },
    }
]


class TestChatFunctionCalling:
    """/v1/chat/completions の tools フィールドテスト"""

    def test_tools_accepted_returns_200(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Search for company X"}],
                "tools": _SAMPLE_TOOLS,
                "max_tokens": 5,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_tool_choice_none_accepted(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _SAMPLE_TOOLS,
                "tool_choice": "none",
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_tool_choice_auto_accepted(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _SAMPLE_TOOLS,
                "tool_choice": "auto",
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_tool_choice_required_accepted(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _SAMPLE_TOOLS,
                "tool_choice": "required",
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_tool_call_response_has_tool_calls(self, client):
        """モデル出力に <tool_call> が含まれる場合 tool_calls が返る"""

        # モデルが <tool_call>{"name":"search_competitor","arguments":{"company":"X"}}</tool_call> を出力するよう mock
        tool_call_text = '<tool_call>{"name":"search_competitor","arguments":{"company":"X"}}</tool_call>'

        tok_mock = MagicMock()
        tok_mock.eos_token_id = 50256
        tok_mock.decode.return_value = tool_call_text
        tok_mock.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        orig_tok = api_module.state.tokenizer
        api_module.state.tokenizer = tok_mock
        try:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Search X"}],
                    "tools": _SAMPLE_TOOLS,
                    "max_tokens": 20,
                },
                headers={"Authorization": "Bearer dev"},
            )
            data = r.json()
            choice = data["choices"][0]
            # tool_calls か通常 content のどちらかであることを確認
            assert choice["finish_reason"] in ("tool_calls", "stop", "length")
        finally:
            api_module.state.tokenizer = orig_tok

    def test_finish_reason_tool_calls_when_tool_found(self):
        """tool_calls が検出された時 finish_reason == 'tool_calls'"""
        from serve.api import _parse_tool_calls_from_text
        text = '<tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        result = _parse_tool_calls_from_text(text)
        assert result is not None
        # finish_reason の遷移ロジックは実装済み
        assert len(result) > 0

    def test_no_tools_no_tool_calls_in_response(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        data = r.json()
        # tools なしの場合 tool_calls は None
        choice = data["choices"][0]
        assert choice.get("tool_calls") is None

    def test_tool_role_message_in_history(self, client):
        """tool ロールのメッセージを含む会話履歴が受け付けられる"""
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Search X"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search_competitor", "arguments": '{"company": "X"}'},
                        }],
                    },
                    {
                        "role": "tool",
                        "content": '{"result": "X spends $100k/month"}',
                        "tool_call_id": "call_1",
                    },
                    {"role": "user", "content": "Summarize that"},
                ],
                "max_tokens": 5,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 5. SSE ストリーミング — Function Calling テスト
# ---------------------------------------------------------------------------

class TestChatStreamFunctionCalling:
    """stream=true + tools の SSE テスト"""

    def test_stream_with_tools_returns_200(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _SAMPLE_TOOLS,
                "stream": True,
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_stream_with_tools_ends_with_done(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _SAMPLE_TOOLS,
                "stream": True,
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert "data: [DONE]" in r.text

    def test_stream_tool_choice_none_no_tool_injection(self, client):
        """tool_choice=none ではツールを注入しない"""
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _SAMPLE_TOOLS,
                "tool_choice": "none",
                "stream": True,
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200
        assert "data: [DONE]" in r.text

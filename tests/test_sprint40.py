"""
Sprint 40 — OpenAI 互換 Streaming 強化 テスト

対象:
  - serve/api.py : /v1/chat/completions の SSE + stop + n + logprobs + penalties
  - serve/api.py : /v1/completions (新規テキスト補完エンドポイント)
  - ヘルパー関数 : _apply_top_p, _apply_sampling_penalties, _collect_logprobs,
                   _check_stop, _truncate_at_stop
"""

from __future__ import annotations

import json
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    # decode: return short string per token id
    tok.decode.side_effect = lambda ids, **kwargs: "hi " * len(ids) if ids else ""
    tok.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]])
    }

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
# 1. ヘルパー関数単体テスト
# ---------------------------------------------------------------------------

class TestApplyTopP:
    """_apply_top_p のユニットテスト"""

    def test_top_p_1_unchanged(self):
        from serve.api import _apply_top_p
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = _apply_top_p(logits, 1.0)
        assert torch.allclose(result, logits)

    def test_top_p_suppresses_low_prob(self):
        from serve.api import _apply_top_p
        logits = torch.tensor([10.0, -10.0, -10.0])
        result = _apply_top_p(logits, 0.5)
        # 高確率のトークンは残る
        assert result[0] > float("-inf")

    def test_top_p_returns_tensor(self):
        from serve.api import _apply_top_p
        logits = torch.randn(100)
        result = _apply_top_p(logits, 0.9)
        assert isinstance(result, torch.Tensor)
        assert result.shape == logits.shape


class TestApplySamplingPenalties:
    """_apply_sampling_penalties のユニットテスト"""

    def test_zero_penalties_unchanged(self):
        from serve.api import _apply_sampling_penalties
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = _apply_sampling_penalties(logits, [0, 1], 0.0, 0.0)
        assert torch.allclose(result, logits)

    def test_presence_penalty_reduces_seen_token(self):
        from serve.api import _apply_sampling_penalties
        logits = torch.zeros(10)
        result = _apply_sampling_penalties(logits, [3], 1.0, 0.0)
        assert result[3] < logits[3]

    def test_frequency_penalty_scales_with_count(self):
        from serve.api import _apply_sampling_penalties
        logits = torch.zeros(10)
        result_once = _apply_sampling_penalties(logits.clone(), [5], 0.0, 1.0)
        result_twice = _apply_sampling_penalties(logits.clone(), [5, 5], 0.0, 1.0)
        assert result_twice[5] < result_once[5]

    def test_unseen_tokens_unaffected(self):
        from serve.api import _apply_sampling_penalties
        logits = torch.zeros(10)
        result = _apply_sampling_penalties(logits, [0], 1.0, 1.0)
        assert result[5] == pytest.approx(0.0)

    def test_returns_tensor(self):
        from serve.api import _apply_sampling_penalties
        logits = torch.randn(50257)
        result = _apply_sampling_penalties(logits, list(range(10)), 0.5, 0.5)
        assert isinstance(result, torch.Tensor)


class TestCollectLogprobs:
    """_collect_logprobs のユニットテスト"""

    def test_top_k_zero_returns_none(self):
        from serve.api import _collect_logprobs
        logits = torch.randn(100)
        result = _collect_logprobs(logits, 5, 0)
        assert result is None

    def test_returns_dict_with_keys(self):
        from serve.api import _collect_logprobs
        logits = torch.randn(100)
        result = _collect_logprobs(logits, 5, 3)
        assert result is not None
        assert "token" in result
        assert "logprob" in result
        assert "top_logprobs" in result

    def test_chosen_token_in_result(self):
        from serve.api import _collect_logprobs
        logits = torch.randn(100)
        result = _collect_logprobs(logits, 42, 5)
        assert result["token"] == 42

    def test_logprob_is_negative(self):
        from serve.api import _collect_logprobs
        logits = torch.randn(100)
        result = _collect_logprobs(logits, 0, 3)
        assert result["logprob"] <= 0

    def test_top_logprobs_count(self):
        from serve.api import _collect_logprobs
        logits = torch.randn(100)
        result = _collect_logprobs(logits, 0, 5)
        assert len(result["top_logprobs"]) <= 5


class TestCheckStop:
    """_check_stop のユニットテスト"""

    def test_no_stop_returns_none(self):
        from serve.api import _check_stop
        assert _check_stop("hello world", None) is None

    def test_empty_stop_returns_none(self):
        from serve.api import _check_stop
        assert _check_stop("hello", []) is None

    def test_detects_suffix_match(self):
        from serve.api import _check_stop
        result = _check_stop("hello\n", ["\n"])
        assert result == "\n"

    def test_no_match_returns_none(self):
        from serve.api import _check_stop
        assert _check_stop("hello world", ["END", "STOP"]) is None

    def test_multiple_stops_first_match(self):
        from serve.api import _check_stop
        result = _check_stop("done!", ["!", "done"])
        assert result == "!"


class TestTruncateAtStop:
    """_truncate_at_stop のユニットテスト"""

    def test_no_stop_unchanged(self):
        from serve.api import _truncate_at_stop
        assert _truncate_at_stop("hello world", None) == "hello world"

    def test_truncates_at_stop(self):
        from serve.api import _truncate_at_stop
        result = _truncate_at_stop("hello\nworld", ["\n"])
        assert result == "hello"

    def test_truncates_at_earliest_stop(self):
        from serve.api import _truncate_at_stop
        result = _truncate_at_stop("ab cd ef", ["cd", "ab"])
        assert result == ""

    def test_no_match_unchanged(self):
        from serve.api import _truncate_at_stop
        result = _truncate_at_stop("hello", ["STOP"])
        assert result == "hello"


# ---------------------------------------------------------------------------
# 2. /v1/chat/completions — 非ストリーミング拡張
# ---------------------------------------------------------------------------

class TestChatCompletionsExtended:
    """Sprint 40 拡張フィールドの非ストリーミングテスト"""

    def test_basic_returns_200(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_system_fingerprint_present(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer dev"},
        )
        data = r.json()
        assert "system_fingerprint" in data
        assert data["system_fingerprint"] == "fp_openmythos"

    def test_n_equals_2_returns_two_choices(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "n": 2,
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        data = r.json()
        assert len(data["choices"]) == 2
        assert data["choices"][0]["index"] == 0
        assert data["choices"][1]["index"] == 1

    def test_n_choices_all_have_finish_reason(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}], "n": 3, "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        data = r.json()
        for choice in data["choices"]:
            assert choice["finish_reason"] in ("stop", "length")

    def test_logprobs_field_when_requested(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
                "top_logprobs": 3,
                "max_tokens": 2,
            },
            headers={"Authorization": "Bearer dev"},
        )
        data = r.json()
        choice = data["choices"][0]
        # logprobs が None でないことを確認
        assert choice.get("logprobs") is not None

    def test_logprobs_false_returns_none(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "logprobs": False, "top_logprobs": 0},
            headers={"Authorization": "Bearer dev"},
        )
        data = r.json()
        choice = data["choices"][0]
        assert choice.get("logprobs") is None

    def test_stop_field_accepted(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["\n", "END"],
                "max_tokens": 5,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_presence_penalty_accepted(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "presence_penalty": 0.5,
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_frequency_penalty_accepted(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "frequency_penalty": 0.5,
                "max_tokens": 3,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_usage_total_tokens_correct(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            headers={"Authorization": "Bearer dev"},
        )
        data = r.json()
        u = data["usage"]
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]

    def test_n_4_max(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "n": 4, "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200
        assert len(r.json()["choices"]) == 4

    def test_n_over_limit_rejected(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "n": 5},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 3. /v1/chat/completions — SSE ストリーミングテスト
# ---------------------------------------------------------------------------

class TestChatCompletionsStream:
    """stream=true の SSE レスポンスを検証"""

    def _get_chunks(self, client, payload: dict) -> list[dict]:
        payload.setdefault("stream", True)
        r = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200
        chunks = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data:") and line != "data: [DONE]":
                chunks.append(json.loads(line[5:].strip()))
        return chunks

    def test_returns_200(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 3},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_content_type_is_event_stream(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_ends_with_done(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 3},
            headers={"Authorization": "Bearer dev"},
        )
        assert "data: [DONE]" in r.text

    def test_chunks_have_id(self, client):
        chunks = self._get_chunks(client, {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3
        })
        for c in chunks:
            assert "id" in c
            assert c["id"].startswith("chatcmpl-")

    def test_chunks_have_object_field(self, client):
        chunks = self._get_chunks(client, {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3
        })
        for c in chunks:
            assert c.get("object") == "chat.completion.chunk"

    def test_chunks_have_model_field(self, client):
        chunks = self._get_chunks(client, {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3
        })
        for c in chunks:
            assert "model" in c

    def test_last_chunk_has_finish_reason(self, client):
        chunks = self._get_chunks(client, {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4
        })
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] in ("stop", "length")

    def test_last_chunk_has_usage(self, client):
        chunks = self._get_chunks(client, {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3
        })
        last = chunks[-1]
        assert "usage" in last
        u = last["usage"]
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]

    def test_intermediate_chunks_have_delta_content(self, client):
        chunks = self._get_chunks(client, {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3
        })
        # 最後以外のチャンクは delta.content を持つ
        for c in chunks[:-1]:
            assert "delta" in c["choices"][0]

    def test_all_chunks_same_id(self, client):
        chunks = self._get_chunks(client, {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5
        })
        ids = {c["id"] for c in chunks}
        assert len(ids) == 1

    def test_stream_with_stop(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stop": ["\n"],
                "max_tokens": 10,
            },
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200
        assert "data: [DONE]" in r.text


# ---------------------------------------------------------------------------
# 4. /v1/completions テスト
# ---------------------------------------------------------------------------

class TestTextCompletions:
    """/v1/completions エンドポイントの基本検証"""

    def test_returns_200(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_object_is_text_completion(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.json()["object"] == "text_completion"

    def test_has_id_starting_with_cmpl(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.json()["id"].startswith("cmpl-")

    def test_choices_have_text_field(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 3},
            headers={"Authorization": "Bearer dev"},
        )
        choices = r.json()["choices"]
        assert len(choices) >= 1
        assert "text" in choices[0]

    def test_finish_reason_in_choices(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] in ("stop", "length")

    def test_usage_fields_present(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 3},
            headers={"Authorization": "Bearer dev"},
        )
        u = r.json()["usage"]
        assert "prompt_tokens" in u
        assert "completion_tokens" in u
        assert "total_tokens" in u

    def test_system_fingerprint_present(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.json()["system_fingerprint"] == "fp_openmythos"

    def test_n_2_returns_two_choices(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "n": 2, "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        assert len(r.json()["choices"]) == 2

    def test_echo_includes_prompt(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "echo": True, "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        text = r.json()["choices"][0]["text"]
        assert text.startswith("hello")

    def test_stop_accepted(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stop": ["\n"], "max_tokens": 5},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_logprobs_returned_when_requested(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "logprobs": True, "top_logprobs": 3, "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        choice = r.json()["choices"][0]
        assert choice.get("logprobs") is not None

    def test_presence_penalty_accepted(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "presence_penalty": 0.5, "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_frequency_penalty_accepted(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "frequency_penalty": 0.5, "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200


class TestTextCompletionsStream:
    """/v1/completions stream=true の SSE 検証"""

    def test_stream_returns_200(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 3},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_stream_content_type(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 2},
            headers={"Authorization": "Bearer dev"},
        )
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_ends_with_done(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 3},
            headers={"Authorization": "Bearer dev"},
        )
        assert "data: [DONE]" in r.text

    def test_stream_chunks_have_text_field(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 3},
            headers={"Authorization": "Bearer dev"},
        )
        chunks = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data:") and line != "data: [DONE]":
                chunks.append(json.loads(line[5:].strip()))
        for c in chunks[:-1]:
            assert "text" in c["choices"][0]

    def test_stream_last_chunk_has_finish_reason(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 3},
            headers={"Authorization": "Bearer dev"},
        )
        chunks = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data:") and line != "data: [DONE]":
                chunks.append(json.loads(line[5:].strip()))
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] in ("stop", "length")

    def test_stream_last_chunk_has_usage(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 3},
            headers={"Authorization": "Bearer dev"},
        )
        chunks = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data:") and line != "data: [DONE]":
                chunks.append(json.loads(line[5:].strip()))
        last = chunks[-1]
        assert "usage" in last

    def test_stream_with_stop_sequence(self, client):
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "stop": ["\n"], "max_tokens": 10},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200
        assert "data: [DONE]" in r.text

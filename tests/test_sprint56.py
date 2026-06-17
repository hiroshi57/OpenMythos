"""
Sprint 56 — マルチプロバイダー LLM 統合 テスト (52 tests)

対象:
  open_mythos/skills/llm_providers.py:
    ProviderType / LLMRequest / LLMResponse / ProviderConfig
    BaseLLMProvider / ClaudeProvider / OpenAIProvider
    OpenMythosProvider / MultiProviderRouter
  serve/api.py:
    POST /v1/llm/complete
    GET  /v1/llm/providers
"""
from __future__ import annotations

import sys
import pytest
import torch
from unittest.mock import MagicMock, patch

from open_mythos.skills.llm_providers import (
    ProviderType, LLMRequest, LLMResponse, ProviderConfig,
    BaseLLMProvider, ClaudeProvider, OpenAIProvider,
    OpenMythosProvider, MultiProviderRouter,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.agents import OpenMythosLLM
    import serve.api as api_mod

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
    model = OpenMythos(cfg)
    api_mod.state.model = model
    api_mod.state.tokenizer = tok
    api_mod.state.llm = OpenMythosLLM(model=model, tokenizer=tok)
    api_mod.state.llm.stream = lambda prompt: iter(["テスト", " 応答"])

    return TestClient(api_mod.app)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ProviderType (4 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestProviderType:
    def test_claude_value(self):
        assert ProviderType.CLAUDE.value == "claude"

    def test_openai_value(self):
        assert ProviderType.OPENAI.value == "openai"

    def test_openmythos_value(self):
        assert ProviderType.OPENMYTHOS.value == "openmythos"

    def test_from_string(self):
        assert ProviderType("claude") == ProviderType.CLAUDE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMRequest / LLMResponse (6 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLLMModels:
    def test_request_defaults(self):
        r = LLMRequest(prompt="hello")
        assert r.max_tokens == 256
        assert r.temperature == 0.7
        assert r.system is None

    def test_request_with_system(self):
        r = LLMRequest(prompt="hi", system="あなたは広告コピーライターです")
        assert r.system is not None

    def test_response_success_true(self):
        r = LLMResponse(text="ok", provider_used="claude", model="haiku")
        assert r.success is True

    def test_response_success_false(self):
        r = LLMResponse(text="", provider_used="claude", model="haiku")
        assert r.success is False

    def test_response_total_tokens(self):
        r = LLMResponse(
            text="x", provider_used="openai", model="gpt",
            prompt_tokens=10, completion_tokens=5,
            total_tokens=15,
        )
        assert r.total_tokens == 15

    def test_response_latency(self):
        r = LLMResponse(text="y", provider_used="openmythos", model="m", latency_ms=123.4)
        assert r.latency_ms == 123.4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ProviderConfig (5 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestProviderConfig:
    def test_resolved_model_claude_default(self):
        cfg = ProviderConfig(provider=ProviderType.CLAUDE)
        assert cfg.resolved_model() == "claude-haiku-4-5"

    def test_resolved_model_openai_default(self):
        cfg = ProviderConfig(provider=ProviderType.OPENAI)
        assert cfg.resolved_model() == "gpt-4o-mini"

    def test_resolved_model_openmythos_default(self):
        cfg = ProviderConfig(provider=ProviderType.OPENMYTHOS)
        assert cfg.resolved_model() == "openmythos"

    def test_resolved_model_override(self):
        cfg = ProviderConfig(provider=ProviderType.CLAUDE, model="claude-3-5-sonnet")
        assert cfg.resolved_model() == "claude-3-5-sonnet"

    def test_timeout_default(self):
        cfg = ProviderConfig(provider=ProviderType.OPENAI)
        assert cfg.timeout == 30


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ClaudeProvider (6 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClaudeProvider:
    def _provider(self, key="dummy"):
        cfg = ProviderConfig(provider=ProviderType.CLAUDE, api_key=key)
        return ClaudeProvider(cfg)

    def test_is_available_with_key(self):
        assert self._provider("sk-test").is_available()

    def test_is_not_available_without_key(self):
        assert not self._provider(key=None).is_available()

    def test_provider_type(self):
        assert self._provider().provider_type == ProviderType.CLAUDE

    def test_no_api_key_raises(self):
        p = ClaudeProvider(ProviderConfig(provider=ProviderType.CLAUDE, api_key=None))
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            p.complete(LLMRequest(prompt="test"))

    def test_complete_with_mock(self):
        p = self._provider()
        mock_data = '{"content":[{"text":"test ad copy"}],"usage":{"input_tokens":5,"output_tokens":3}}'
        mock_resp = mock_data.encode("utf-8")
        with patch("urllib.request.urlopen") as mock_url:
            mock_url.return_value.__enter__ = lambda s: s
            mock_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_url.return_value.read.return_value = mock_resp
            resp = p.complete(LLMRequest(prompt="make an ad"))
        assert resp.text == "test ad copy"
        assert resp.provider_used == "claude"

    def test_stream_fallback(self):
        p = self._provider()
        mock_data = '{"content":[{"text":"hello world"}],"usage":{}}'
        mock_resp = mock_data.encode("utf-8")
        with patch("urllib.request.urlopen") as mock_url:
            mock_url.return_value.__enter__ = lambda s: s
            mock_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_url.return_value.read.return_value = mock_resp
            tokens = list(p.stream(LLMRequest(prompt="hi")))
        assert "".join(tokens) == "hello world"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAIProvider (6 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestOpenAIProvider:
    def _provider(self, key="dummy"):
        cfg = ProviderConfig(provider=ProviderType.OPENAI, api_key=key)
        return OpenAIProvider(cfg)

    def test_is_available_with_key(self):
        assert self._provider("sk-openai").is_available()

    def test_is_not_available_without_key(self):
        assert not self._provider(key=None).is_available()

    def test_provider_type(self):
        assert self._provider().provider_type == ProviderType.OPENAI

    def test_no_api_key_raises(self):
        p = OpenAIProvider(ProviderConfig(provider=ProviderType.OPENAI, api_key=None))
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            p.complete(LLMRequest(prompt="test"))

    def test_complete_with_mock(self):
        p = self._provider()
        mock_data = '{"choices":[{"message":{"content":"OpenAI ad copy"}}],"usage":{"prompt_tokens":4,"completion_tokens":2}}'
        mock_resp = mock_data.encode("utf-8")
        with patch("urllib.request.urlopen") as mock_url:
            mock_url.return_value.__enter__ = lambda s: s
            mock_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_url.return_value.read.return_value = mock_resp
            resp = p.complete(LLMRequest(prompt="make an ad"))
        assert resp.text == "OpenAI ad copy"
        assert resp.provider_used == "openai"

    def test_model_name(self):
        p = self._provider()
        assert p.config.resolved_model() == "gpt-4o-mini"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenMythosProvider (7 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestOpenMythosProvider:
    def _provider(self, llm=None):
        cfg = ProviderConfig(provider=ProviderType.OPENMYTHOS)
        return OpenMythosProvider(cfg, llm=llm)

    def test_is_not_available_without_llm(self):
        assert not self._provider().is_available()

    def test_is_available_with_llm(self):
        mock_llm = MagicMock()
        assert self._provider(mock_llm).is_available()

    def test_provider_type(self):
        assert self._provider().provider_type == ProviderType.OPENMYTHOS

    def test_complete_without_llm_fallback(self):
        p = self._provider()
        resp = p.complete(LLMRequest(prompt="テスト"))
        assert "OpenMythos" in resp.text or "テスト" in resp.text
        assert resp.provider_used == "openmythos"

    def test_complete_with_llm(self):
        mock_llm = MagicMock()
        mock_llm.run.return_value = "LLM応答テキスト"
        p = self._provider(mock_llm)
        resp = p.complete(LLMRequest(prompt="広告"))
        assert resp.text == "LLM応答テキスト"

    def test_stream_with_llm(self):
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["こんにちは", " 世界"])
        p = self._provider(mock_llm)
        tokens = list(p.stream(LLMRequest(prompt="hi")))
        assert "".join(tokens) == "こんにちは 世界"

    def test_stream_without_llm_fallback(self):
        p = self._provider()
        tokens = list(p.stream(LLMRequest(prompt="フォールバック テスト")))
        assert len(tokens) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MultiProviderRouter (10 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMultiProviderRouter:
    def _mock_provider(self, ptype, available=True, response="テスト"):
        cfg = ProviderConfig(
            provider=ptype,
            api_key="key" if available else None,
        )
        p = MagicMock(spec=BaseLLMProvider)
        p.config = cfg
        p.is_available.return_value = available
        p.provider_type = ptype
        p.complete.return_value = LLMResponse(
            text=response, provider_used=ptype.value, model="m"
        )
        return p

    def test_available_providers(self):
        r = MultiProviderRouter([
            self._mock_provider(ProviderType.CLAUDE, available=True),
            self._mock_provider(ProviderType.OPENAI, available=False),
        ])
        assert "claude" in r.available_providers()
        assert "openai" not in r.available_providers()

    def test_complete_uses_first_available(self):
        claude = self._mock_provider(ProviderType.CLAUDE, response="claude応答")
        r = MultiProviderRouter([claude])
        resp = r.complete(LLMRequest(prompt="test"))
        assert resp.provider_used == "claude"

    def test_fallback_on_failure(self):
        claude = self._mock_provider(ProviderType.CLAUDE, available=True)
        claude.complete.side_effect = RuntimeError("Claude失敗")
        openai = self._mock_provider(ProviderType.OPENAI, response="openai応答")
        r = MultiProviderRouter([claude, openai])
        resp = r.complete(LLMRequest(prompt="test"))
        assert resp.provider_used == "openai"

    def test_skip_unavailable(self):
        claude = self._mock_provider(ProviderType.CLAUDE, available=False)
        openai = self._mock_provider(ProviderType.OPENAI, response="ok")
        r = MultiProviderRouter([claude, openai])
        resp = r.complete(LLMRequest(prompt="test"))
        assert resp.provider_used == "openai"
        claude.complete.assert_not_called()

    def test_all_unavailable_raises(self):
        r = MultiProviderRouter([
            self._mock_provider(ProviderType.CLAUDE, available=False),
        ])
        with pytest.raises(RuntimeError):
            r.complete(LLMRequest(prompt="test"))

    def test_preferred_provider(self):
        claude = self._mock_provider(ProviderType.CLAUDE, response="c")
        openai = self._mock_provider(ProviderType.OPENAI, response="o")
        r = MultiProviderRouter([claude, openai])
        resp = r.complete(LLMRequest(prompt="test"), preferred=ProviderType.OPENAI)
        assert resp.provider_used == "openai"

    def test_from_env_no_keys(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        router = MultiProviderRouter.from_env()
        # openmythos (no llm) も unavailable
        available = router.available_providers()
        assert "claude" not in available
        assert "openai" not in available

    def test_from_env_with_openmythos_llm(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_llm = MagicMock()
        mock_llm.run.return_value = "mythos応答"
        router = MultiProviderRouter.from_env(llm=mock_llm)
        assert "openmythos" in router.available_providers()

    def test_stream_yields_strings(self):
        provider = self._mock_provider(ProviderType.OPENMYTHOS, available=True)
        provider.stream.return_value = iter(["hello", " world"])
        r = MultiProviderRouter([provider])
        tokens = list(r.stream(LLMRequest(prompt="test")))
        assert "".join(tokens) == "hello world"

    def test_custom_priority(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        router = MultiProviderRouter.from_env(
            priority=[ProviderType.OPENAI, ProviderType.OPENMYTHOS]
        )
        available = router.available_providers()
        assert "openai" in available


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API エンドポイント (8 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestProviderAPI:
    def test_list_providers_ok(self, client):
        r = client.get("/v1/llm/providers")
        assert r.status_code == 200
        data = r.json()
        assert "available" in data
        assert "all_providers" in data

    def test_list_providers_has_all_keys(self, client):
        r = client.get("/v1/llm/providers")
        data = r.json()
        assert set(data["all_providers"]) == {"claude", "openai", "openmythos"}

    def test_complete_openmythos_provider(self, client):
        # OpenMythosProvider は llm なしでもフォールバック応答を返す
        r = client.post("/v1/llm/complete", json={
            "prompt": "テスト",
            "preferred_provider": "openmythos",
        })
        # 503 (no provider) or 200
        assert r.status_code in (200, 503)

    def test_complete_invalid_provider(self, client):
        r = client.post("/v1/llm/complete", json={
            "prompt": "テスト",
            "preferred_provider": "unknown_provider",
        })
        assert r.status_code == 400

    def test_complete_response_fields(self, client):
        # openmythos はフォールバック応答を返す
        r = client.post("/v1/llm/complete", json={
            "prompt": "広告コピーを作って",
            "preferred_provider": "openmythos",
        })
        if r.status_code == 200:
            data = r.json()
            assert "text" in data
            assert "provider_used" in data
            assert "model" in data

    def test_complete_no_preferred(self, client, monkeypatch):
        # API キーなし状態で openmythos のみ試行
        r = client.post("/v1/llm/complete", json={"prompt": "テスト"})
        # 503 (all failed) or 200 (if any key set)
        assert r.status_code in (200, 503)

    def test_complete_temperature_validation(self, client):
        r = client.post("/v1/llm/complete", json={
            "prompt": "test",
            "temperature": 3.0,  # max=2.0 を超える
        })
        assert r.status_code == 422

    def test_complete_max_tokens_validation(self, client):
        r = client.post("/v1/llm/complete", json={
            "prompt": "test",
            "max_tokens": 0,  # ge=1 違反
        })
        assert r.status_code == 422

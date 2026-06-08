"""
Sprint 46 — 推論バックエンド統合 テスト

対象:
  - open_mythos/skills/inference_backends.py:
      AttentionConfig / AttentionBenchmark / FlashAttentionOptimizer
      GuidanceTemplate / GuidanceResult / GuidanceGenerator
      TRTConfig / TRTGenerationResult / TRTLLMBackend
      TranscriptionResult / WhisperTranscriber
  - serve/api.py:
      POST /v1/attention/benchmark
      POST /v1/guidance/generate
      POST /v1/guidance/grammar/regex
      POST /v1/guidance/grammar/choice
      POST /v1/trt/generate
      GET  /v1/trt/build-engine
      POST /v1/whisper/transcribe
      POST /v1/whisper/detect-language
"""
from __future__ import annotations

import sys
import pytest
import torch
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# transformers モック
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
    tok.decode.side_effect = lambda ids, **kwargs: "ok " * max(len(ids), 1)
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    cfg = MythosConfig(
        vocab_size=50257, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=128,
        max_loop_iters=2, prelude_layers=1, coda_layers=1, attn_type="gqa",
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


_HDR = {"Authorization": "Bearer dev"}

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------

from open_mythos.skills.inference_backends import (
    AttentionConfig, AttentionBenchmark, FlashAttentionOptimizer,
    GuidanceTemplate, GuidanceResult, GuidanceGenerator,
    TRTConfig, TRTGenerationResult, TRTLLMBackend,
    TranscriptionResult, WhisperTranscriber,
)


# ---------------------------------------------------------------------------
# Section A: AttentionConfig / FlashAttentionOptimizer
# ---------------------------------------------------------------------------

class TestAttentionConfig:
    def test_defaults(self):
        cfg = AttentionConfig()
        assert cfg.backend == "auto"
        assert cfg.causal is True
        assert cfg.dtype == "float16"

    def test_custom(self):
        cfg = AttentionConfig(backend="sdpa", causal=False, dtype="bfloat16")
        assert cfg.backend == "sdpa"
        assert cfg.causal is False


class TestFlashAttentionOptimizer:
    def test_active_backend_is_string(self):
        opt = FlashAttentionOptimizer(AttentionConfig())
        assert isinstance(opt.active_backend, str)
        assert opt.active_backend in ("flash", "sdpa", "eager")

    def test_forward_returns_tensor(self):
        cfg = AttentionConfig(backend="eager")
        opt = FlashAttentionOptimizer(cfg)
        q = torch.randn(1, 8, 4, 16)
        k, v = q.clone(), q.clone()
        out = opt.forward(q, k, v)
        assert out.shape == q.shape

    def test_benchmark_returns_result(self):
        opt = FlashAttentionOptimizer(AttentionConfig(backend="eager"))
        bench = opt.benchmark(seq_len=32, batch_size=1)
        assert isinstance(bench, AttentionBenchmark)

    def test_benchmark_fields(self):
        opt = FlashAttentionOptimizer(AttentionConfig(backend="eager"))
        bench = opt.benchmark(seq_len=16, batch_size=1)
        assert bench.seq_len == 16
        assert bench.batch_size == 1
        assert bench.latency_ms >= 0.0
        assert bench.memory_mb >= 0.0

    def test_benchmark_speedup_eager(self):
        opt = FlashAttentionOptimizer(AttentionConfig(backend="eager"))
        bench = opt.benchmark(seq_len=16)
        assert bench.speedup_vs_eager == 1.0

    def test_explicit_sdpa_backend(self):
        cfg = AttentionConfig(backend="sdpa")
        opt = FlashAttentionOptimizer(cfg)
        assert opt.active_backend == "sdpa"


class TestAttentionBenchmark:
    def test_creation(self):
        b = AttentionBenchmark(backend="sdpa", seq_len=512, batch_size=2,
                                latency_ms=1.5, memory_mb=10.0)
        assert b.backend == "sdpa"
        assert b.seq_len == 512


# ---------------------------------------------------------------------------
# Section B: GuidanceGenerator
# ---------------------------------------------------------------------------

class TestGuidanceTemplate:
    def test_creation(self):
        t = GuidanceTemplate(template="Hello {{name}}", variables={"name": "World"})
        assert "{{name}}" in t.template
        assert t.variables["name"] == "World"

    def test_defaults(self):
        t = GuidanceTemplate(template="test")
        assert t.max_tokens == 256
        assert t.stop_sequences == []


class TestGuidanceGenerator:
    def test_is_native_bool(self):
        gen = GuidanceGenerator()
        assert isinstance(gen.is_native, bool)

    def test_generate_fallback(self):
        gen = GuidanceGenerator()
        tpl = GuidanceTemplate(
            template="Hello {{name}}, you are {{age}} years old.",
            variables={"name": "Alice", "age": "30"},
        )
        result = gen.generate(tpl)
        assert isinstance(result, GuidanceResult)
        assert result.success is True

    def test_generate_variable_substitution(self):
        gen = GuidanceGenerator()
        tpl = GuidanceTemplate(
            template="Dear {{title}}",
            variables={"title": "Doctor"},
        )
        result = gen.generate(tpl)
        assert "Doctor" in result.text

    def test_generate_tokens_used_positive(self):
        gen = GuidanceGenerator()
        tpl = GuidanceTemplate(template="one two three", variables={})
        result = gen.generate(tpl)
        assert result.tokens_used >= 0

    def test_build_regex_grammar(self):
        gen = GuidanceGenerator()
        grammar = gen.build_regex_grammar(r"\d{4}")
        assert r"\d{4}" in grammar

    def test_build_choice_grammar(self):
        gen = GuidanceGenerator()
        grammar = gen.build_choice_grammar(["yes", "no", "maybe"])
        assert "yes" in grammar
        assert "no" in grammar


class TestGuidanceResult:
    def test_creation(self):
        r = GuidanceResult(text="hello", variables={}, tokens_used=1)
        assert r.text == "hello"
        assert r.success is True


# ---------------------------------------------------------------------------
# Section C: TRTLLMBackend
# ---------------------------------------------------------------------------

class TestTRTConfig:
    def test_defaults(self):
        cfg = TRTConfig()
        assert cfg.max_batch_size == 8
        assert cfg.dtype == "float16"
        assert cfg.tp_size == 1

    def test_custom(self):
        cfg = TRTConfig(max_batch_size=4, dtype="bfloat16", tp_size=2)
        assert cfg.max_batch_size == 4
        assert cfg.tp_size == 2


class TestTRTLLMBackend:
    def test_is_native_bool(self):
        backend = TRTLLMBackend(TRTConfig())
        assert isinstance(backend.is_native, bool)

    def test_generate_returns_result(self):
        backend = TRTLLMBackend(TRTConfig())
        result = backend.generate([[1, 2, 3]], max_new_tokens=4)
        assert isinstance(result, TRTGenerationResult)

    def test_generate_output_count(self):
        backend = TRTLLMBackend(TRTConfig())
        result = backend.generate([[1, 2], [3, 4]], max_new_tokens=4)
        assert len(result.texts) == 2

    def test_generate_latency_nonneg(self):
        backend = TRTLLMBackend(TRTConfig())
        result = backend.generate([[1]], max_new_tokens=4)
        assert result.latency_ms >= 0.0

    def test_build_engine_command_contains_dirs(self):
        cfg = TRTConfig(engine_dir="./engines")
        backend = TRTLLMBackend(cfg)
        cmd = backend.build_engine("./hf_model")
        assert "./hf_model" in cmd
        assert "./engines" in cmd


class TestTRTGenerationResult:
    def test_creation(self):
        r = TRTGenerationResult(output_ids=[[1, 2]], texts=["hello"], latency_ms=5.0)
        assert len(r.output_ids) == 1
        assert r.texts[0] == "hello"


# ---------------------------------------------------------------------------
# Section D: WhisperTranscriber
# ---------------------------------------------------------------------------

class TestWhisperTranscriber:
    def test_is_native_bool(self):
        w = WhisperTranscriber()
        assert isinstance(w.is_native, bool)

    def test_transcribe_returns_result(self):
        w = WhisperTranscriber()
        result = w.transcribe("mock_audio.wav")
        assert isinstance(result, TranscriptionResult)

    def test_transcribe_text_nonempty(self):
        w = WhisperTranscriber()
        result = w.transcribe("test.wav")
        assert len(result.text) > 0

    def test_transcribe_has_segments(self):
        w = WhisperTranscriber()
        result = w.transcribe("test.wav")
        assert isinstance(result.segments, list)

    def test_detect_language_returns_dict(self):
        w = WhisperTranscriber()
        probs = w.detect_language("test.wav")
        assert isinstance(probs, dict)
        assert len(probs) > 0

    def test_detect_language_values_floats(self):
        w = WhisperTranscriber()
        probs = w.detect_language("test.wav")
        for lang, prob in probs.items():
            assert isinstance(lang, str)
            assert isinstance(prob, float)


class TestTranscriptionResult:
    def test_creation(self):
        r = TranscriptionResult(text="hello", language="en", segments=[])
        assert r.text == "hello"
        assert r.language == "en"

    def test_duration_default(self):
        r = TranscriptionResult(text="x", language="ja", segments=[])
        assert r.duration_s == 0.0


# ---------------------------------------------------------------------------
# Section E: API エンドポイント
# ---------------------------------------------------------------------------

class TestAttentionBenchmarkEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/attention/benchmark",
                        json={"seq_len": 32, "batch_size": 1},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_backend(self, client):
        r = client.post("/v1/attention/benchmark",
                        json={"seq_len": 16},
                        headers=_HDR)
        assert "backend" in r.json()

    def test_has_latency_ms(self, client):
        r = client.post("/v1/attention/benchmark",
                        json={"seq_len": 16},
                        headers=_HDR)
        assert "latency_ms" in r.json()
        assert isinstance(r.json()["latency_ms"], float)

    def test_has_speedup(self, client):
        r = client.post("/v1/attention/benchmark",
                        json={"seq_len": 16},
                        headers=_HDR)
        assert "speedup_vs_eager" in r.json()


class TestGuidanceGenerateEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/guidance/generate",
                        json={"template": "Hello {{name}}", "variables": {"name": "World"}},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_text(self, client):
        r = client.post("/v1/guidance/generate",
                        json={"template": "Dear {{x}}", "variables": {"x": "User"}},
                        headers=_HDR)
        assert "text" in r.json()

    def test_success_flag(self, client):
        r = client.post("/v1/guidance/generate",
                        json={"template": "Hi", "variables": {}},
                        headers=_HDR)
        assert r.json()["success"] is True

    def test_variable_substituted(self, client):
        r = client.post("/v1/guidance/generate",
                        json={"template": "Hello {{name}}", "variables": {"name": "Alice"}},
                        headers=_HDR)
        assert "Alice" in r.json()["text"]


class TestGuidanceGrammarEndpoints:
    def test_regex_returns_200(self, client):
        r = client.post("/v1/guidance/grammar/regex",
                        json={"pattern": r"\d{4}"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_regex_has_grammar(self, client):
        r = client.post("/v1/guidance/grammar/regex",
                        json={"pattern": r"\w+"},
                        headers=_HDR)
        assert "grammar" in r.json()

    def test_choice_returns_200(self, client):
        r = client.post("/v1/guidance/grammar/choice",
                        json={"choices": ["yes", "no"]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_choice_has_grammar(self, client):
        r = client.post("/v1/guidance/grammar/choice",
                        json={"choices": ["a", "b", "c"]},
                        headers=_HDR)
        assert "grammar" in r.json()


class TestTRTGenerateEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/trt/generate",
                        json={"input_ids": [[1, 2, 3]], "max_new_tokens": 4},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_texts(self, client):
        r = client.post("/v1/trt/generate",
                        json={"input_ids": [[1, 2]], "max_new_tokens": 4},
                        headers=_HDR)
        assert isinstance(r.json()["texts"], list)

    def test_batch_count_matches(self, client):
        r = client.post("/v1/trt/generate",
                        json={"input_ids": [[1], [2], [3]], "max_new_tokens": 2},
                        headers=_HDR)
        assert len(r.json()["texts"]) == 3

    def test_has_latency(self, client):
        r = client.post("/v1/trt/generate",
                        json={"input_ids": [[1]], "max_new_tokens": 2},
                        headers=_HDR)
        assert "latency_ms" in r.json()


class TestWhisperEndpoints:
    def test_transcribe_returns_200(self, client):
        r = client.post("/v1/whisper/transcribe",
                        json={"audio_path": "test.wav"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_transcribe_has_text(self, client):
        r = client.post("/v1/whisper/transcribe",
                        json={"audio_path": "test.wav"},
                        headers=_HDR)
        assert "text" in r.json()
        assert len(r.json()["text"]) > 0

    def test_transcribe_has_segments(self, client):
        r = client.post("/v1/whisper/transcribe",
                        json={"audio_path": "test.wav"},
                        headers=_HDR)
        assert isinstance(r.json()["segments"], list)

    def test_detect_language_returns_200(self, client):
        r = client.post("/v1/whisper/detect-language",
                        json={"audio_path": "test.wav"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_detect_language_has_probabilities(self, client):
        r = client.post("/v1/whisper/detect-language",
                        json={"audio_path": "test.wav"},
                        headers=_HDR)
        assert "probabilities" in r.json()
        assert isinstance(r.json()["probabilities"], dict)

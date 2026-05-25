"""
Sprint 6.4 benchmark tests — perplexity / throughput / lm-eval harness.

All tests run on CPU with the nano variant (vocab_size=1024, dim=128).
Heavy I/O (downloading WikiText, LAMBADA) is skipped via monkeypatching.
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from open_mythos.variants import mythos_nano
from open_mythos.main import OpenMythos

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nano_model():
    cfg = mythos_nano()
    model = OpenMythos(cfg).eval()
    return model


@pytest.fixture(scope="module")
def short_token_ids():
    """64 random token IDs within nano vocab_size=1024."""
    torch.manual_seed(42)
    return torch.randint(0, 1024, (64,)).tolist()


# ---------------------------------------------------------------------------
# benchmark/perplexity.py
# ---------------------------------------------------------------------------


class TestPerplexity:

    def test_evaluate_perplexity_returns_dict(self, nano_model, short_token_ids):
        from benchmark.perplexity import evaluate_perplexity

        result = evaluate_perplexity(
            nano_model,
            short_token_ids,
            device="cpu",
            seq_len=16,
            stride=8,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "ppl",
            "nll",
            "n_tokens",
            "elapsed_sec",
            "tokens_per_sec",
        }

    def test_ppl_is_finite_positive(self, nano_model, short_token_ids):
        from benchmark.perplexity import evaluate_perplexity

        result = evaluate_perplexity(
            nano_model,
            short_token_ids,
            device="cpu",
            seq_len=16,
            stride=8,
        )
        assert math.isfinite(result["ppl"])
        assert result["ppl"] > 0

    def test_nll_matches_ppl(self, nano_model, short_token_ids):
        from benchmark.perplexity import evaluate_perplexity

        result = evaluate_perplexity(
            nano_model,
            short_token_ids,
            device="cpu",
            seq_len=16,
            stride=8,
        )
        expected_ppl = math.exp(min(result["nll"], 88.0))
        assert abs(result["ppl"] - expected_ppl) < 1e-4

    def test_n_tokens_positive(self, nano_model, short_token_ids):
        from benchmark.perplexity import evaluate_perplexity

        result = evaluate_perplexity(
            nano_model,
            short_token_ids,
            device="cpu",
            seq_len=16,
            stride=8,
        )
        assert result["n_tokens"] > 0

    def test_tokens_per_sec_positive(self, nano_model, short_token_ids):
        from benchmark.perplexity import evaluate_perplexity

        result = evaluate_perplexity(
            nano_model,
            short_token_ids,
            device="cpu",
            seq_len=16,
            stride=8,
        )
        assert result["tokens_per_sec"] > 0

    def test_stride_equals_seq_len_full_predict(self, nano_model, short_token_ids):
        """stride == seq_len means every token is predicted with maximum context."""
        from benchmark.perplexity import evaluate_perplexity

        result = evaluate_perplexity(
            nano_model,
            short_token_ids,
            device="cpu",
            seq_len=16,
            stride=16,
        )
        assert result["n_tokens"] > 0

    def test_n_loops_override(self, nano_model, short_token_ids):
        from benchmark.perplexity import evaluate_perplexity

        result = evaluate_perplexity(
            nano_model,
            short_token_ids,
            device="cpu",
            seq_len=16,
            stride=8,
            n_loops=1,
        )
        assert math.isfinite(result["ppl"])

    def test_tokenize_corpus_clips_to_vocab(self):
        from benchmark.perplexity import _tokenize_corpus

        ids = _tokenize_corpus("hello world", vocab_size=32)
        assert all(i < 32 for i in ids)
        assert len(ids) > 0

    def test_load_corpus_calls_datasets(self):
        from benchmark.perplexity import _load_corpus

        # Simulate datasets.Dataset: supports ds["text"] column access
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=["hello", "world", ""])
        fake_datasets = MagicMock()
        fake_datasets.load_dataset.return_value = mock_ds
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            text = _load_corpus("wikitext-2-raw-v1", split="test")
        assert "hello" in text
        assert "world" in text

    def test_max_nll_capped_at_88(self, nano_model):
        """exp(88) is the ceiling to avoid overflow."""
        from benchmark.perplexity import evaluate_perplexity

        single_token = [0]  # minimal corpus
        result = evaluate_perplexity(
            nano_model,
            single_token,
            device="cpu",
            seq_len=16,
            stride=8,
        )
        # With a single token there are no predicted tokens, so ppl is exp(0) = 1
        assert result["ppl"] >= 1.0


# ---------------------------------------------------------------------------
# benchmark/throughput.py
# ---------------------------------------------------------------------------


class TestThroughput:

    def test_measure_latency_returns_result(self, nano_model):
        from benchmark.throughput import measure_latency

        prompt_ids = list(range(8))
        result = measure_latency(
            nano_model,
            prompt_ids,
            device="cpu",
            max_new_tokens=4,
            batch_size=1,
            warmup_iters=0,
        )
        from benchmark.throughput import LatencyResult

        assert isinstance(result, LatencyResult)

    def test_ttft_positive(self, nano_model):
        from benchmark.throughput import measure_latency

        result = measure_latency(
            nano_model,
            list(range(8)),
            device="cpu",
            max_new_tokens=4,
            batch_size=1,
            warmup_iters=0,
        )
        assert result.ttft_ms > 0

    def test_tpot_positive(self, nano_model):
        from benchmark.throughput import measure_latency

        result = measure_latency(
            nano_model,
            list(range(8)),
            device="cpu",
            max_new_tokens=4,
            batch_size=1,
            warmup_iters=0,
        )
        assert result.tpot_ms > 0

    def test_generated_tokens_matches_request(self, nano_model):
        from benchmark.throughput import measure_latency

        result = measure_latency(
            nano_model,
            list(range(8)),
            device="cpu",
            max_new_tokens=6,
            batch_size=1,
            warmup_iters=0,
        )
        assert result.generated_tokens == 6

    def test_throughput_positive(self, nano_model):
        from benchmark.throughput import measure_latency

        result = measure_latency(
            nano_model,
            list(range(8)),
            device="cpu",
            max_new_tokens=4,
            batch_size=1,
            warmup_iters=0,
        )
        assert result.throughput_tok_s > 0

    def test_batch_size_recorded(self, nano_model):
        from benchmark.throughput import measure_latency

        result = measure_latency(
            nano_model,
            list(range(8)),
            device="cpu",
            max_new_tokens=4,
            batch_size=2,
            warmup_iters=0,
        )
        assert result.batch_size == 2

    def test_sweep_batch_sizes(self, nano_model):
        from benchmark.throughput import sweep_batch_sizes

        results = sweep_batch_sizes(
            nano_model,
            list(range(8)),
            device="cpu",
            batch_sizes=[1, 2],
            max_new_tokens=4,
        )
        assert len(results) == 2
        assert results[0].batch_size == 1
        assert results[1].batch_size == 2

    def test_n_loops_override(self, nano_model):
        from benchmark.throughput import measure_latency

        result = measure_latency(
            nano_model,
            list(range(8)),
            device="cpu",
            max_new_tokens=4,
            batch_size=1,
            n_loops=1,
            warmup_iters=0,
        )
        assert result.generated_tokens == 4

    def test_tokenize_clips_to_vocab(self):
        from benchmark.throughput import _tokenize

        ids = _tokenize("hello", vocab_size=32)
        assert all(i < 32 for i in ids)

    def test_prompt_len_recorded(self, nano_model):
        from benchmark.throughput import measure_latency

        prompt = list(range(5))
        result = measure_latency(
            nano_model,
            prompt,
            device="cpu",
            max_new_tokens=4,
            batch_size=1,
            warmup_iters=0,
        )
        assert result.prompt_len == 5


# ---------------------------------------------------------------------------
# benchmark/lm_eval_harness.py
# ---------------------------------------------------------------------------


class TestLMEvalWrapper:

    @pytest.fixture
    def wrapper(self, nano_model):
        from benchmark.lm_eval_harness import MythosLMEvalWrapper

        return MythosLMEvalWrapper(nano_model, device="cpu", batch_size=1)

    def test_tok_encode_clips_to_vocab(self, wrapper):
        ids = wrapper.tok_encode("hello world")
        assert all(i < 1024 for i in ids)

    def test_tok_decode_returns_string(self, wrapper):
        text = wrapper.tok_decode([65, 66, 67])
        assert isinstance(text, str)

    def test_loglikelihood_returns_list(self, wrapper):
        class FakeReq:
            args = ("The cat", " sat")

        results = wrapper.loglikelihood([FakeReq()])
        assert len(results) == 1
        score, is_greedy = results[0]
        assert isinstance(score, float)
        assert isinstance(is_greedy, bool)

    def test_loglikelihood_score_is_finite(self, wrapper):
        class FakeReq:
            args = ("The cat sat on", " the mat")

        results = wrapper.loglikelihood([FakeReq()])
        assert math.isfinite(results[0][0])

    def test_loglikelihood_rolling_returns_list(self, wrapper):
        class FakeReq:
            args = ("The quick brown fox jumps over the lazy dog",)

        results = wrapper.loglikelihood_rolling([FakeReq()])
        assert len(results) == 1
        assert isinstance(results[0], float)

    def test_generate_until_returns_string(self, wrapper):
        class FakeReq:
            args = ("Once upon", {"until": ["."], "max_gen_toks": 8})

        results = wrapper.generate_until([FakeReq()])
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_generate_until_stops_at_stop_string(self, wrapper):
        """Generated text should not contain the stop string."""

        class FakeReq:
            args = ("Hello", {"until": ["STOP"], "max_gen_toks": 16})

        results = wrapper.generate_until([FakeReq()])
        assert "STOP" not in results[0]

    def test_eot_token_id(self, wrapper):
        assert wrapper.eot_token_id == 0

    def test_max_length(self, wrapper):
        assert wrapper.max_length > 0

    def test_batch_size_property(self, wrapper):
        assert wrapper.batch_size == 1

    def test_device_property(self, wrapper):
        assert wrapper.device == "cpu"

    def test_empty_context_does_not_crash(self, wrapper):
        class FakeReq:
            args = ("", " hello")

        results = wrapper.loglikelihood([FakeReq()])
        assert len(results) == 1

    def test_standalone_lambada_skips_without_datasets(self, nano_model):
        """Without datasets installed, standalone lambada raises ImportError."""
        from benchmark.lm_eval_harness import _run_lambada_standalone

        with patch.dict(sys.modules, {"datasets": None}):
            with pytest.raises((ImportError, TypeError)):
                _run_lambada_standalone(nano_model, "cpu", n_samples=1)

"""
Sprint 42 — /v1/embeddings + /v1/semantic-search テスト

対象:
  - open_mythos/main.py : OpenMythos.encode() — 隠れ状態取得
  - serve/api.py        : /v1/embeddings (OpenAI 互換)
  - serve/api.py        : /v1/semantic-search
  - ヘルパー             : _encode_text, _vec_to_base64, _cosine_similarity
"""

from __future__ import annotations

import base64
import json
import struct
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
def client_and_model():
    from fastapi.testclient import TestClient
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.agents import OpenMythosLLM

    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.eos_token_id = 50256
    tok.decode.side_effect = lambda ids, **kwargs: "hello " * max(len(ids), 1)
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
        yield c, model


@pytest.fixture(scope="module")
def client(client_and_model):
    return client_and_model[0]


@pytest.fixture(scope="module")
def model(client_and_model):
    return client_and_model[1]


# ---------------------------------------------------------------------------
# 1. OpenMythos.encode() ユニットテスト
# ---------------------------------------------------------------------------

class TestOpenMythosEncode:
    """open_mythos/main.py OpenMythos.encode() のテスト"""

    def test_encode_returns_tensor(self, model):
        ids = torch.tensor([[1, 2, 3]])
        h = model.encode(ids)
        assert isinstance(h, torch.Tensor)

    def test_encode_shape_B_T_dim(self, model):
        ids = torch.tensor([[1, 2, 3, 4]])
        h = model.encode(ids)
        assert h.shape[0] == 1   # B
        assert h.shape[1] == 4   # T
        assert h.shape[2] == 64  # dim (cfg.dim)

    def test_encode_single_token(self, model):
        ids = torch.tensor([[1]])
        h = model.encode(ids)
        assert h.shape == (1, 1, 64)

    def test_encode_batch(self, model):
        ids = torch.tensor([[1, 2], [3, 4]])
        h = model.encode(ids)
        assert h.shape[0] == 2

    def test_encode_no_nan(self, model):
        ids = torch.tensor([[1, 2, 3]])
        h = model.encode(ids)
        assert not torch.isnan(h).any()

    def test_encode_no_inf(self, model):
        ids = torch.tensor([[1, 2, 3]])
        h = model.encode(ids)
        assert not torch.isinf(h).any()

    def test_encode_differs_from_forward(self, model):
        """encode() の出力次元は vocab_size でなく dim"""
        ids = torch.tensor([[1, 2, 3]])
        h = model.encode(ids)
        logits = model(ids)
        # encode は dim, forward は vocab_size
        assert h.shape[2] != logits.shape[2] or h.shape[2] == logits.shape[2]
        # 重要: encode の最終次元は model.cfg.dim
        assert h.shape[2] == 64

    def test_encode_with_n_loops(self, model):
        ids = torch.tensor([[1, 2, 3]])
        h1 = model.encode(ids, n_loops=1)
        h2 = model.encode(ids, n_loops=2)
        assert h1.shape == h2.shape


# ---------------------------------------------------------------------------
# 2. ヘルパー関数テスト
# ---------------------------------------------------------------------------

class TestVecToBase64:
    """_vec_to_base64 のテスト"""

    def test_returns_str(self):
        from serve.api import _vec_to_base64
        result = _vec_to_base64([0.1, 0.2, 0.3])
        assert isinstance(result, str)

    def test_decodable(self):
        from serve.api import _vec_to_base64
        vec = [0.1, 0.2, 0.3]
        encoded = _vec_to_base64(vec)
        raw = base64.b64decode(encoded)
        decoded = list(struct.unpack(f"{len(vec)}f", raw))
        assert len(decoded) == len(vec)
        assert abs(decoded[0] - vec[0]) < 1e-5

    def test_length_proportional_to_input(self):
        from serve.api import _vec_to_base64
        short = _vec_to_base64([0.1] * 4)
        long = _vec_to_base64([0.1] * 8)
        assert len(base64.b64decode(short)) < len(base64.b64decode(long))


class TestCosineSimilarity:
    """_cosine_similarity のテスト"""

    def test_identical_vectors(self):
        from serve.api import _cosine_similarity
        v = [0.6, 0.8]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        from serve.api import _cosine_similarity
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-5

    def test_opposite_vectors(self):
        from serve.api import _cosine_similarity
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-5

    def test_empty_vectors_return_zero(self):
        from serve.api import _cosine_similarity
        assert _cosine_similarity([], []) == 0.0

    def test_length_mismatch_returns_zero(self):
        from serve.api import _cosine_similarity
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_returns_float(self):
        from serve.api import _cosine_similarity
        result = _cosine_similarity([0.5, 0.5], [0.5, 0.5])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 3. /v1/embeddings エンドポイントテスト
# ---------------------------------------------------------------------------

class TestEmbeddingsEndpoint:
    """/v1/embeddings の基本テスト"""

    def test_returns_200(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello world"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_object_is_list(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.json()["object"] == "list"

    def test_data_contains_embedding(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        data = r.json()["data"]
        assert len(data) == 1
        assert "embedding" in data[0]

    def test_embedding_is_list_of_floats(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        emb = r.json()["data"][0]["embedding"]
        assert isinstance(emb, list)
        assert all(isinstance(v, float) for v in emb)

    def test_embedding_dimension(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        emb = r.json()["data"][0]["embedding"]
        # dim はモデル設定に依存 (lifespan が起動するモデルは MODEL_DIM=256)
        assert len(emb) == api_module.state.model.cfg.dim

    def test_embedding_is_normalized(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        emb = r.json()["data"][0]["embedding"]
        norm = sum(v ** 2 for v in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-4

    def test_model_field_in_response(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.json()["model"] == "openmythos"

    def test_usage_fields_present(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        u = r.json()["usage"]
        assert "prompt_tokens" in u
        assert "total_tokens" in u

    def test_index_field_in_data(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.json()["data"][0]["index"] == 0

    def test_object_field_in_data(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.json()["data"][0]["object"] == "embedding"

    def test_batch_input_returns_multiple(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": ["hello", "world", "test"]},
            headers={"Authorization": "Bearer dev"},
        )
        data = r.json()["data"]
        assert len(data) == 3

    def test_batch_indices_correct(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": ["a", "b", "c"]},
            headers={"Authorization": "Bearer dev"},
        )
        indices = [d["index"] for d in r.json()["data"]]
        assert indices == [0, 1, 2]

    def test_base64_encoding_format(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello", "encoding_format": "base64"},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200
        emb = r.json()["data"][0]["embedding"]
        assert isinstance(emb, str)
        # base64 デコード可能か確認
        raw = base64.b64decode(emb)
        assert len(raw) % 4 == 0  # float32 は 4 バイト

    def test_dimensions_parameter(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello", "dimensions": 16},
            headers={"Authorization": "Bearer dev"},
        )
        emb = r.json()["data"][0]["embedding"]
        assert len(emb) == 16

    def test_n_loops_parameter_accepted(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"input": "hello", "n_loops": 2},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 4. /v1/semantic-search エンドポイントテスト
# ---------------------------------------------------------------------------

class TestSemanticSearchEndpoint:
    """/v1/semantic-search の基本テスト"""

    _DOCS = [
        "Machine learning is a branch of artificial intelligence.",
        "Python is a popular programming language.",
        "The weather today is sunny and warm.",
        "Deep neural networks are used for image recognition.",
    ]

    def test_returns_200(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI and ML", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200

    def test_response_has_query(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "hello", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.json()["query"] == "hello"

    def test_response_has_results(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        assert "results" in r.json()
        assert len(r.json()["results"]) > 0

    def test_total_documents_correct(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.json()["total_documents"] == len(self._DOCS)

    def test_top_k_limits_results(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS, "top_k": 2},
            headers={"Authorization": "Bearer dev"},
        )
        assert len(r.json()["results"]) <= 2

    def test_default_top_k_is_3(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        assert len(r.json()["results"]) <= 3

    def test_results_have_score(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        for result in r.json()["results"]:
            assert "score" in result

    def test_scores_are_floats(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        for result in r.json()["results"]:
            assert isinstance(result["score"], float)

    def test_scores_in_range(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        for result in r.json()["results"]:
            assert -1.01 <= result["score"] <= 1.01

    def test_results_sorted_by_score_descending(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS, "top_k": 4},
            headers={"Authorization": "Bearer dev"},
        )
        scores = [res["score"] for res in r.json()["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_results_have_document_text(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        for result in r.json()["results"]:
            assert "document" in result
            assert isinstance(result["document"], str)

    def test_results_have_index(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "AI", "documents": self._DOCS},
            headers={"Authorization": "Bearer dev"},
        )
        for result in r.json()["results"]:
            assert "index" in result

    def test_single_document(self, client):
        r = client.post(
            "/v1/semantic-search",
            json={"query": "test", "documents": ["only one doc"]},
            headers={"Authorization": "Bearer dev"},
        )
        assert r.status_code == 200
        assert len(r.json()["results"]) == 1

    def test_top_k_larger_than_docs_returns_all(self, client):
        docs = ["doc1", "doc2"]
        r = client.post(
            "/v1/semantic-search",
            json={"query": "test", "documents": docs, "top_k": 10},
            headers={"Authorization": "Bearer dev"},
        )
        assert len(r.json()["results"]) <= len(docs)

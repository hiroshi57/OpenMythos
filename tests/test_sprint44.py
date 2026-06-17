"""
Sprint 44 — Vector DB 統合 + Instructor 構造化出力 テスト

対象:
  - open_mythos/skills/vector_store.py: VectorDocument / VectorStoreConfig / ChromaStore / QdrantStore
                                         PineconeStore / FaissStore / VectorStoreFactory
  - open_mythos/skills/instructor_extract.py: ExtractionSchema / ExtractionResult / InstructorExtractor
  - serve/api.py: /v1/vector-store/* + /v1/extract + /v1/extract/prompt
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


# ---------------------------------------------------------------------------
# Section A: VectorDocument
# ---------------------------------------------------------------------------

from open_mythos.skills.vector_store import (
    VectorDocument, VectorStoreConfig, VectorStoreBackend,
    ChromaStore, QdrantStore, PineconeStore, FaissStore, VectorStoreFactory,
)


class TestVectorDocument:
    def test_creation(self):
        doc = VectorDocument(id="d1", vector=[0.1, 0.2, 0.3], text="hello")
        assert doc.id == "d1"
        assert len(doc.vector) == 3
        assert doc.text == "hello"

    def test_auto_id(self):
        doc = VectorDocument(id="", vector=[1.0])
        assert len(doc.id) > 0

    def test_dim(self):
        doc = VectorDocument(id="x", vector=[1.0, 2.0, 3.0, 4.0])
        assert doc.dim == 4

    def test_metadata_default(self):
        doc = VectorDocument(id="m", vector=[0.5])
        assert doc.metadata == {}

    def test_metadata_stored(self):
        doc = VectorDocument(id="m2", vector=[0.5], metadata={"src": "wiki"})
        assert doc.metadata["src"] == "wiki"


# ---------------------------------------------------------------------------
# Section B: VectorStoreConfig
# ---------------------------------------------------------------------------

class TestVectorStoreConfig:
    def test_defaults(self):
        cfg = VectorStoreConfig()
        assert cfg.collection == "default"
        assert cfg.dim == 768
        assert cfg.metric == "cosine"
        assert cfg.top_k == 5

    def test_custom(self):
        cfg = VectorStoreConfig(collection="my_col", dim=64, metric="l2", top_k=10)
        assert cfg.collection == "my_col"
        assert cfg.dim == 64
        assert cfg.metric == "l2"
        assert cfg.top_k == 10


# ---------------------------------------------------------------------------
# Section C: ChromaStore (in-memory fallback)
# ---------------------------------------------------------------------------

class TestChromaStore:
    def _make(self) -> ChromaStore:
        cfg = VectorStoreConfig(collection="test_chroma", dim=4)
        store = ChromaStore(cfg)
        store._native = False   # force in-memory fallback
        return store

    def test_upsert_returns_count(self):
        store = self._make()
        docs = [VectorDocument(id=f"c{i}", vector=[float(i)]*4) for i in range(3)]
        assert store.upsert(docs) == 3

    def test_query_returns_results(self):
        store = self._make()
        docs = [VectorDocument(id="a", vector=[1.0, 0.0, 0.0, 0.0], text="apple"),
                VectorDocument(id="b", vector=[0.0, 1.0, 0.0, 0.0], text="banana")]
        store.upsert(docs)
        results = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0].id == "a"

    def test_delete_removes_doc(self):
        store = self._make()
        store.upsert([VectorDocument(id="d1", vector=[1.0]*4)])
        assert store.delete(["d1"]) == 1
        assert store.count() == 0

    def test_count_zero_initial(self):
        store = self._make()
        assert store.count() == 0

    def test_count_after_upsert(self):
        store = self._make()
        store.upsert([VectorDocument(id="x", vector=[0.1]*4)])
        assert store.count() == 1

    def test_filter_by_metadata(self):
        store = self._make()
        store.upsert([
            VectorDocument(id="a", vector=[1.0, 0.0, 0.0, 0.0], metadata={"tag": "A"}),
            VectorDocument(id="b", vector=[1.0, 0.0, 0.0, 0.0], metadata={"tag": "B"}),
        ])
        results = store.query([1.0, 0.0, 0.0, 0.0], filter={"tag": "A"})
        assert all(r[0].metadata.get("tag") == "A" for r in results)

    def test_clear(self):
        store = self._make()
        store.upsert([VectorDocument(id="z", vector=[0.0]*4)])
        store.clear()
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Section D: QdrantStore (in-memory fallback)
# ---------------------------------------------------------------------------

class TestQdrantStore:
    def _make(self) -> QdrantStore:
        cfg = VectorStoreConfig(collection="test_qdrant", dim=4)
        store = QdrantStore(cfg)
        store._native = False
        return store

    def test_upsert_count(self):
        store = self._make()
        docs = [VectorDocument(id=f"q{i}", vector=[0.1 * i] * 4) for i in range(5)]
        assert store.upsert(docs) == 5
        assert store.count() == 5

    def test_query_top_k(self):
        store = self._make()
        docs = [VectorDocument(id=f"q{i}", vector=[1.0 if j == i else 0.0 for j in range(4)])
                for i in range(4)]
        store.upsert(docs)
        results = store.query([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_delete(self):
        store = self._make()
        store.upsert([VectorDocument(id="del1", vector=[0.0]*4)])
        store.delete(["del1"])
        assert store.count() == 0

    def test_query_scores_are_floats(self):
        store = self._make()
        store.upsert([VectorDocument(id="s1", vector=[1.0, 0.0, 0.0, 0.0])])
        results = store.query([1.0, 0.0, 0.0, 0.0])
        for _, score in results:
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Section E: PineconeStore (in-memory fallback)
# ---------------------------------------------------------------------------

class TestPineconeStore:
    def _make(self) -> PineconeStore:
        cfg = VectorStoreConfig(collection="test_pinecone", dim=4)
        store = PineconeStore(cfg)
        store._native = False
        return store

    def test_upsert_and_count(self):
        store = self._make()
        store.upsert([VectorDocument(id="p1", vector=[0.5]*4)])
        assert store.count() == 1

    def test_query_returns_hits(self):
        store = self._make()
        store.upsert([VectorDocument(id="p2", vector=[1.0, 0.0, 0.0, 0.0])])
        results = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) >= 1

    def test_delete_and_count(self):
        store = self._make()
        store.upsert([VectorDocument(id="p3", vector=[0.2]*4)])
        store.delete(["p3"])
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Section F: FaissStore (in-memory fallback)
# ---------------------------------------------------------------------------

class TestFaissStore:
    def _make(self) -> FaissStore:
        cfg = VectorStoreConfig(collection="test_faiss", dim=4)
        store = FaissStore(cfg)
        store._native = False
        return store

    def test_upsert(self):
        store = self._make()
        docs = [VectorDocument(id=f"f{i}", vector=[float(i)]*4) for i in range(3)]
        assert store.upsert(docs) == 3

    def test_query_sorted(self):
        store = self._make()
        store.upsert([
            VectorDocument(id="near", vector=[1.0, 0.0, 0.0, 0.0]),
            VectorDocument(id="far",  vector=[0.0, 0.0, 0.0, 1.0]),
        ])
        results = store.query([1.0, 0.0, 0.0, 0.0])
        assert results[0][0].id == "near"

    def test_index_types(self):
        cfg = VectorStoreConfig(collection="hnsw_test", dim=4)
        for itype in ["flat", "hnsw"]:
            store = FaissStore(cfg, index_type=itype)
            store._native = False
            store.upsert([VectorDocument(id="x", vector=[1.0]*4)])
            assert store.count() == 1


# ---------------------------------------------------------------------------
# Section G: VectorStoreFactory
# ---------------------------------------------------------------------------

class TestVectorStoreFactory:
    def test_create_faiss(self):
        store = VectorStoreFactory.create(VectorStoreBackend.FAISS, dim=4)
        assert isinstance(store, FaissStore)

    def test_create_chroma(self):
        store = VectorStoreFactory.create(VectorStoreBackend.CHROMA, dim=4)
        store._native = False
        assert isinstance(store, ChromaStore)

    def test_create_qdrant(self):
        store = VectorStoreFactory.create(VectorStoreBackend.QDRANT, dim=4)
        store._native = False
        assert isinstance(store, QdrantStore)

    def test_create_pinecone(self):
        store = VectorStoreFactory.create(VectorStoreBackend.PINECONE, dim=4)
        store._native = False
        assert isinstance(store, PineconeStore)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            VectorStoreFactory.create("invalid_backend")  # type: ignore


# ---------------------------------------------------------------------------
# Section H: InstructorExtractor
# ---------------------------------------------------------------------------

from open_mythos.skills.instructor_extract import ExtractionSchema, InstructorExtractor


class TestExtractionSchema:
    def test_creation(self):
        schema = ExtractionSchema(name="Person", fields={"name": "str", "age": "int"})
        assert schema.name == "Person"
        assert "name" in schema.fields

    def test_to_json_schema(self):
        schema = ExtractionSchema(name="Entity", fields={"title": "str", "count": "int"})
        js = schema.to_json_schema()
        assert js["type"] == "object"
        assert "title" in js["properties"]
        assert "count" in js["properties"]

    def test_required_fields(self):
        schema = ExtractionSchema(name="R", fields={"x": "str"}, required=["x"])
        js = schema.to_json_schema()
        assert "x" in js["required"]

    def test_json_type_mapping(self):
        schema = ExtractionSchema(name="T", fields={
            "s": "str", "i": "int", "f": "float", "b": "bool"
        })
        js = schema.to_json_schema()
        assert js["properties"]["s"]["type"] == "string"
        assert js["properties"]["i"]["type"] == "integer"
        assert js["properties"]["f"]["type"] == "number"
        assert js["properties"]["b"]["type"] == "boolean"


class TestInstructorExtractor:
    def test_extract_json_from_text(self):
        extractor = InstructorExtractor()
        schema = ExtractionSchema(name="Person", fields={"name": "str", "age": "int"})
        text = 'Here is the data: {"name": "Alice", "age": 30}'
        result = extractor.extract(text, schema)
        assert result.success
        assert result.data["name"] == "Alice"

    def test_extract_json_from_code_block(self):
        extractor = InstructorExtractor()
        schema = ExtractionSchema(name="Item", fields={"id": "int", "label": "str"})
        text = '```json\n{"id": 42, "label": "test"}\n```'
        result = extractor.extract(text, schema)
        assert result.success
        assert result.data["id"] == 42

    def test_extract_failure_returns_error(self):
        extractor = InstructorExtractor(max_retries=0)
        schema = ExtractionSchema(name="X", fields={"val": "str"})
        result = extractor.extract("no json here at all", schema)
        assert not result.success
        assert result.error != ""

    def test_extract_batch(self):
        extractor = InstructorExtractor()
        schema = ExtractionSchema(name="N", fields={"n": "int"})
        texts = ['{"n": 1}', '{"n": 2}', '{"n": 3}']
        results = extractor.extract_batch(texts, schema)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_build_prompt_contains_schema(self):
        extractor = InstructorExtractor()
        schema = ExtractionSchema(name="Sch", fields={"key": "str"})
        prompt = extractor.build_prompt(schema)
        assert "Sch" in prompt
        assert "key" in prompt
        assert "JSON" in prompt

    def test_retries_counted(self):
        extractor = InstructorExtractor(max_retries=2)
        schema = ExtractionSchema(name="X", fields={"v": "str"})
        result = extractor.extract("nothing", schema)
        assert result.retries > 0

    def test_type_conversion_int(self):
        extractor = InstructorExtractor()
        schema = ExtractionSchema(name="N", fields={"count": "int"})
        result = extractor.extract('{"count": "5"}', schema)
        assert result.success
        assert result.data["count"] == 5

    def test_result_schema_name(self):
        extractor = InstructorExtractor()
        schema = ExtractionSchema(name="MySchema", fields={"x": "str"})
        result = extractor.extract('{"x": "val"}', schema)
        assert result.schema_name == "MySchema"

    def test_is_native_bool(self):
        extractor = InstructorExtractor()
        assert isinstance(extractor.is_native, bool)


# ---------------------------------------------------------------------------
# Section I: API /v1/vector-store/*
# ---------------------------------------------------------------------------

_VECS = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
_DOCS = [{"id": f"doc{i}", "vector": v, "text": f"text {i}", "metadata": {"idx": i}}
         for i, v in enumerate(_VECS)]
_HDR = {"Authorization": "Bearer dev"}


class TestVectorStoreUpsertEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/vector-store/upsert",
                        json={"backend": "faiss", "collection": "sprint44", "documents": _DOCS},
                        headers=_HDR)
        assert r.status_code == 200

    def test_upserted_count(self, client):
        r = client.post("/v1/vector-store/upsert",
                        json={"backend": "faiss", "collection": "sprint44b", "documents": _DOCS[:2]},
                        headers=_HDR)
        assert r.json()["upserted"] == 2

    def test_backend_in_response(self, client):
        r = client.post("/v1/vector-store/upsert",
                        json={"backend": "faiss", "collection": "s44c", "documents": _DOCS[:1]},
                        headers=_HDR)
        assert r.json()["backend"] == "faiss"

    def test_collection_in_response(self, client):
        r = client.post("/v1/vector-store/upsert",
                        json={"backend": "faiss", "collection": "mycol", "documents": _DOCS[:1]},
                        headers=_HDR)
        assert r.json()["collection"] == "mycol"


class TestVectorStoreQueryEndpoint:
    def test_returns_200(self, client):
        # upsert first
        client.post("/v1/vector-store/upsert",
                    json={"backend": "faiss", "collection": "qcol", "documents": _DOCS},
                    headers=_HDR)
        r = client.post("/v1/vector-store/query",
                        json={"backend": "faiss", "collection": "qcol",
                              "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 2},
                        headers=_HDR)
        assert r.status_code == 200

    def test_hits_list(self, client):
        client.post("/v1/vector-store/upsert",
                    json={"backend": "faiss", "collection": "qcol2", "documents": _DOCS},
                    headers=_HDR)
        r = client.post("/v1/vector-store/query",
                        json={"backend": "faiss", "collection": "qcol2",
                              "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 2},
                        headers=_HDR)
        assert isinstance(r.json()["hits"], list)

    def test_hit_has_score(self, client):
        client.post("/v1/vector-store/upsert",
                    json={"backend": "faiss", "collection": "qcol3", "documents": _DOCS},
                    headers=_HDR)
        r = client.post("/v1/vector-store/query",
                        json={"backend": "faiss", "collection": "qcol3",
                              "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1},
                        headers=_HDR)
        hits = r.json()["hits"]
        if hits:
            assert "score" in hits[0]
            assert isinstance(hits[0]["score"], float)

    def test_top_k_respected(self, client):
        client.post("/v1/vector-store/upsert",
                    json={"backend": "faiss", "collection": "qcol4", "documents": _DOCS},
                    headers=_HDR)
        r = client.post("/v1/vector-store/query",
                        json={"backend": "faiss", "collection": "qcol4",
                              "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1},
                        headers=_HDR)
        assert len(r.json()["hits"]) <= 1


class TestVectorStoreDeleteEndpoint:
    def test_delete_returns_200(self, client):
        client.post("/v1/vector-store/upsert",
                    json={"backend": "faiss", "collection": "dcol", "documents": _DOCS[:2]},
                    headers=_HDR)
        r = client.post("/v1/vector-store/delete",
                        json={"backend": "faiss", "collection": "dcol", "ids": ["doc0"]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_deleted_count(self, client):
        client.post("/v1/vector-store/upsert",
                    json={"backend": "faiss", "collection": "dcol2", "documents": _DOCS[:2]},
                    headers=_HDR)
        r = client.post("/v1/vector-store/delete",
                        json={"backend": "faiss", "collection": "dcol2", "ids": ["doc0", "doc1"]},
                        headers=_HDR)
        assert r.json()["deleted"] == 2


class TestVectorStoreCountEndpoint:
    def test_count_returns_200(self, client):
        r = client.get("/v1/vector-store/count", params={"backend": "faiss", "collection": "cnt1"},
                       headers=_HDR)
        assert r.status_code == 200

    def test_count_is_int(self, client):
        r = client.get("/v1/vector-store/count", params={"backend": "faiss", "collection": "cnt2"},
                       headers=_HDR)
        assert isinstance(r.json()["count"], int)


# ---------------------------------------------------------------------------
# Section J: API /v1/extract
# ---------------------------------------------------------------------------

class TestExtractEndpoint:
    _FIELDS = [{"name": "title", "type": "str"}, {"name": "year", "type": "int"}]

    def test_returns_200(self, client):
        r = client.post("/v1/extract",
                        json={"text": '{"title": "OpenMythos", "year": 2026}',
                              "schema_name": "Paper",
                              "fields": self._FIELDS},
                        headers=_HDR)
        assert r.status_code == 200

    def test_extraction_success(self, client):
        r = client.post("/v1/extract",
                        json={"text": '{"title": "AI", "year": 2025}',
                              "schema_name": "Paper",
                              "fields": self._FIELDS},
                        headers=_HDR)
        assert r.json()["success"] is True

    def test_data_has_fields(self, client):
        r = client.post("/v1/extract",
                        json={"text": '{"title": "Test", "year": 2024}',
                              "schema_name": "Paper",
                              "fields": self._FIELDS},
                        headers=_HDR)
        data = r.json()["data"]
        assert "title" in data or "year" in data

    def test_failure_on_no_json(self, client):
        r = client.post("/v1/extract",
                        json={"text": "no json in this text at all",
                              "schema_name": "X",
                              "fields": [{"name": "v", "type": "str"}],
                              "max_retries": 0},
                        headers=_HDR)
        assert r.status_code == 200
        assert r.json()["success"] is False

    def test_schema_name_in_response(self, client):
        r = client.post("/v1/extract",
                        json={"text": '{"v": "hi"}',
                              "schema_name": "MySchema",
                              "fields": [{"name": "v", "type": "str"}]},
                        headers=_HDR)
        assert r.json()["schema_name"] == "MySchema"


class TestExtractPromptEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/extract/prompt",
                        json={"text": "",
                              "schema_name": "Entity",
                              "fields": [{"name": "name", "type": "str"}]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_prompt_nonempty(self, client):
        r = client.post("/v1/extract/prompt",
                        json={"text": "",
                              "schema_name": "Entity",
                              "fields": [{"name": "name", "type": "str"}]},
                        headers=_HDR)
        assert len(r.json()["prompt"]) > 20

    def test_schema_in_response(self, client):
        r = client.post("/v1/extract/prompt",
                        json={"text": "",
                              "schema_name": "S",
                              "fields": [{"name": "x", "type": "int"}]},
                        headers=_HDR)
        assert "schema" in r.json()

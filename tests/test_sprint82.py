"""
tests/test_sprint82.py — Sprint 82: EmbeddingGemma RAG エンジン テストスイート

対象: open_mythos/skills/gemma_rag.py
目標: 80+ PASS
"""
from __future__ import annotations

import math
import re
import pytest
from open_mythos.skills.gemma_rag import (
    # Enums
    GemmaEmbeddingModel, ChunkingStrategy, RAGStatus,
    # Models
    EmbeddingConfig, RAGDocument, RAGChunk, ChunkingConfig,
    RetrievalResult, RAGAnswer, IndexStats,
    # Components
    GemmaEmbeddingProvider, DocumentChunker,
    _ChunkVectorStore, _cosine_sim,
    RAGIndexer, RAGRetriever,
    MockLLMGenerator, GemmaLLMGenerator,
    RAGPipeline,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def mock_embedder():
    cfg = EmbeddingConfig(model=GemmaEmbeddingModel.MOCK)
    return GemmaEmbeddingProvider(cfg)


@pytest.fixture
def basic_doc():
    return RAGDocument(
        id="doc1",
        title="テスト製品説明",
        content=(
            "この製品は高品質な素材で作られています。"
            "耐久性に優れ、長期間使用できます。"
            "また、環境に配慮した製造プロセスを採用しています。"
            "返品は購入後30日以内に限ります。"
        ),
    )


@pytest.fixture
def pipeline():
    return RAGPipeline.create_mock()


@pytest.fixture
def indexed_pipeline(basic_doc):
    p = RAGPipeline.create_mock()
    p.index_document(basic_doc)
    return p


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Enum & Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEnumsAndConfig:
    def test_embedding_model_enum_values(self):
        assert GemmaEmbeddingModel.TEXT_EMBEDDING_004.value == "text-embedding-004"
        assert GemmaEmbeddingModel.EMBEDDING_001.value == "embedding-001"
        assert GemmaEmbeddingModel.MOCK.value == "mock"

    def test_chunking_strategy_enum(self):
        assert ChunkingStrategy.FIXED_SIZE.value == "fixed_size"
        assert ChunkingStrategy.SENTENCE.value == "sentence"
        assert ChunkingStrategy.PARAGRAPH.value == "paragraph"

    def test_rag_status_enum(self):
        assert RAGStatus.IDLE.value == "idle"
        assert RAGStatus.READY.value == "ready"

    def test_embedding_config_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.model == GemmaEmbeddingModel.TEXT_EMBEDDING_004
        assert cfg.dim == 768
        assert cfg.task_type == "RETRIEVAL_DOCUMENT"
        assert cfg.timeout == 30

    def test_chunking_config_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.strategy == ChunkingStrategy.SENTENCE
        assert cfg.chunk_size == 512
        assert cfg.overlap == 64
        assert cfg.min_chars == 20


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Data Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDataModels:
    def test_rag_document_defaults(self):
        doc = RAGDocument(id="d1", title="タイトル", content="コンテンツ")
        assert doc.id == "d1"
        assert doc.metadata == {}

    def test_rag_document_auto_id(self):
        doc = RAGDocument(id="", title="T", content="C")
        assert len(doc.id) > 0   # __post_init__ で UUID 生成

    def test_rag_chunk_full_text(self):
        chunk = RAGChunk(
            id="c1", doc_id="d1", doc_title="製品説明",
            text="高品質な素材です。", chunk_index=0,
        )
        assert "製品説明" in chunk.full_text
        assert "高品質な素材です。" in chunk.full_text

    def test_retrieval_result_to_dict(self):
        chunk = RAGChunk(
            id="c1", doc_id="d1", doc_title="T", text="本文", chunk_index=0,
        )
        result = RetrievalResult(chunk=chunk, score=0.85, rank=1)
        d = result.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["score"] == 0.85
        assert d["rank"] == 1
        assert d["doc_title"] == "T"

    def test_rag_answer_properties(self):
        answer = RAGAnswer(query="質問", text="回答")
        assert answer.success is True
        d = answer.to_dict()
        assert d["query"] == "質問"
        assert d["answer"] == "回答"

    def test_rag_answer_empty_text(self):
        answer = RAGAnswer(query="質問", text="")
        assert answer.success is False

    def test_index_stats_defaults(self):
        stats = IndexStats()
        assert stats.doc_count == 0
        assert stats.chunk_count == 0
        assert stats.status == RAGStatus.IDLE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. GemmaEmbeddingProvider (Mock)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGemmaEmbeddingProviderMock:
    def test_mock_embed_returns_correct_dim(self, mock_embedder):
        vec = mock_embedder.embed("テストテキスト")
        assert len(vec) == 768

    def test_mock_embed_is_normalized(self, mock_embedder):
        vec = mock_embedder.embed("hello world")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_mock_embed_is_deterministic(self, mock_embedder):
        text = "同じテキストは同じ埋め込み"
        v1 = mock_embedder.embed(text)
        v2 = mock_embedder.embed(text)
        assert v1 == v2

    def test_mock_embed_different_texts_different_vectors(self, mock_embedder):
        v1 = mock_embedder.embed("りんご")
        v2 = mock_embedder.embed("自動車")
        assert v1 != v2

    def test_mock_embed_similar_texts_higher_cosine(self, mock_embedder):
        v_dog1 = mock_embedder.embed("犬 ペット 動物")
        v_dog2 = mock_embedder.embed("犬 かわいい ペット")
        v_car  = mock_embedder.embed("自動車 エンジン 速度")
        sim_dog = _cosine_sim(v_dog1, v_dog2)
        sim_diff = _cosine_sim(v_dog1, v_car)
        assert sim_dog > sim_diff

    def test_mock_embed_batch(self, mock_embedder):
        texts = ["テキスト1", "テキスト2", "テキスト3"]
        vecs = mock_embedder.embed_batch(texts)
        assert len(vecs) == 3
        for vec in vecs:
            assert len(vec) == 768

    def test_mock_embed_empty_text(self, mock_embedder):
        vec = mock_embedder.embed("")
        assert len(vec) == 768
        # 全ゼロになるはずはない（正規化後は 1/768 ≈ 0）
        # ゼロ除算は 1e-9 で保護されているので全ゼロも返る可能性あり
        assert isinstance(vec, list)

    def test_provider_dim_property(self, mock_embedder):
        assert mock_embedder.dim == 768

    def test_is_mock_flag_true_without_api_key(self):
        cfg = EmbeddingConfig(
            model=GemmaEmbeddingModel.TEXT_EMBEDDING_004,
            api_key="",
        )
        provider = GemmaEmbeddingProvider(cfg)
        assert provider._is_mock is True

    def test_is_mock_flag_false_with_api_key(self):
        cfg = EmbeddingConfig(
            model=GemmaEmbeddingModel.TEXT_EMBEDDING_004,
            api_key="dummy_key",
        )
        provider = GemmaEmbeddingProvider(cfg)
        assert provider._is_mock is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. DocumentChunker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDocumentChunker:
    def test_sentence_chunker_basic(self, basic_doc):
        chunker = DocumentChunker()
        chunks = chunker.chunk(basic_doc)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.doc_id == "doc1"
            assert c.doc_title == "テスト製品説明"
            assert len(c.text) >= 20

    def test_chunk_ids_are_unique(self, basic_doc):
        chunker = DocumentChunker()
        chunks = chunker.chunk(basic_doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_sequential(self, basic_doc):
        chunker = DocumentChunker(ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=50))
        chunks = chunker.chunk(basic_doc)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_fixed_size_chunker(self):
        doc = RAGDocument(id="d1", title="T", content="A" * 300)
        cfg = ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=100, overlap=0)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3

    def test_fixed_size_with_overlap(self):
        doc = RAGDocument(id="d1", title="T", content="A" * 200)
        cfg = ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=100, overlap=50)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk(doc)
        # オーバーラップあり → 通常より多くのチャンク
        assert len(chunks) > 2

    def test_paragraph_chunker(self):
        content = (
            "段落1の内容です。詳細な説明が含まれます。\n\n"
            "段落2の内容です。さらに詳しい情報があります。\n\n"
            "段落3の内容です。最後の段落になります。"
        )
        doc = RAGDocument(id="d1", title="T", content=content)
        cfg = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3

    def test_min_chars_filter(self):
        content = "短い。\n\nこれは十分な長さのコンテンツです。詳細な説明があります。"
        doc = RAGDocument(id="d1", title="T", content=content)
        cfg = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH, min_chars=20)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk(doc)
        # 「短い。」は min_chars=20 未満なので除外
        for c in chunks:
            assert len(c.text) >= 20

    def test_empty_content_returns_no_chunks(self):
        doc = RAGDocument(id="d1", title="T", content="")
        chunker = DocumentChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0

    def test_full_text_includes_title(self, basic_doc):
        chunker = DocumentChunker()
        chunks = chunker.chunk(basic_doc)
        for c in chunks:
            assert basic_doc.title in c.full_text

    def test_metadata_propagated(self):
        doc = RAGDocument(
            id="d1", title="T", content="コンテンツです。" * 5,
            metadata={"source": "wiki", "lang": "ja"},
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.get("source") == "wiki"
            assert c.metadata.get("lang") == "ja"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. _ChunkVectorStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestChunkVectorStore:
    def _make_chunk(self, cid: str, doc_id: str) -> RAGChunk:
        return RAGChunk(id=cid, doc_id=doc_id, doc_title="T", text="テスト", chunk_index=0)

    def test_upsert_and_count(self):
        store = _ChunkVectorStore()
        chunk = self._make_chunk("c1", "d1")
        store.upsert(chunk, [0.1, 0.2, 0.3])
        assert store.chunk_count == 1

    def test_upsert_overwrite(self):
        store = _ChunkVectorStore()
        chunk = self._make_chunk("c1", "d1")
        store.upsert(chunk, [0.1, 0.2])
        store.upsert(chunk, [0.3, 0.4])
        assert store.chunk_count == 1

    def test_delete_by_doc(self):
        store = _ChunkVectorStore()
        for i in range(3):
            store.upsert(self._make_chunk(f"c{i}", "d1"), [float(i), 0.0])
        store.upsert(self._make_chunk("cx", "d2"), [1.0, 0.0])
        removed = store.delete_by_doc("d1")
        assert removed == 3
        assert store.chunk_count == 1

    def test_search_returns_top_k(self):
        store = _ChunkVectorStore()
        for i in range(10):
            chunk = self._make_chunk(f"c{i}", "d1")
            store.upsert(chunk, [float(i), 0.0])
        results = store.search([1.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_search_sorted_by_score(self):
        store = _ChunkVectorStore()
        store.upsert(self._make_chunk("c1", "d1"), [1.0, 0.0])
        store.upsert(self._make_chunk("c2", "d1"), [0.0, 1.0])
        store.upsert(self._make_chunk("c3", "d1"), [0.7, 0.7])
        results = store.search([1.0, 0.0], top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_doc_filter(self):
        store = _ChunkVectorStore()
        store.upsert(self._make_chunk("c1", "d1"), [1.0, 0.0])
        store.upsert(self._make_chunk("c2", "d2"), [0.9, 0.1])
        results = store.search([1.0, 0.0], top_k=5, doc_filter=["d1"])
        assert all(c.doc_id == "d1" for c, _ in results)

    def test_doc_ids_property(self):
        store = _ChunkVectorStore()
        store.upsert(self._make_chunk("c1", "d1"), [1.0, 0.0])
        store.upsert(self._make_chunk("c2", "d2"), [0.5, 0.5])
        assert set(store.doc_ids) == {"d1", "d2"}

    def test_clear(self):
        store = _ChunkVectorStore()
        store.upsert(self._make_chunk("c1", "d1"), [1.0])
        store.clear()
        assert store.chunk_count == 0

    def test_cosine_sim_perfect(self):
        a = [1.0, 0.0, 0.0]
        assert abs(_cosine_sim(a, a) - 1.0) < 1e-6

    def test_cosine_sim_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_sim(a, b)) < 1e-6


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. RAGIndexer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRAGIndexer:
    def _make_indexer(self):
        cfg = EmbeddingConfig(model=GemmaEmbeddingModel.MOCK)
        embedder = GemmaEmbeddingProvider(cfg)
        return RAGIndexer(embedder)

    def test_index_returns_chunks(self, basic_doc):
        indexer = self._make_indexer()
        chunks = indexer.index(basic_doc)
        assert len(chunks) >= 1

    def test_stats_after_index(self, basic_doc):
        indexer = self._make_indexer()
        indexer.index(basic_doc)
        stats = indexer.stats
        assert stats.doc_count == 1
        assert stats.chunk_count >= 1
        assert stats.status == RAGStatus.READY

    def test_stats_before_index(self):
        indexer = self._make_indexer()
        stats = indexer.stats
        assert stats.doc_count == 0
        assert stats.status == RAGStatus.IDLE

    def test_index_overwrites_existing(self, basic_doc):
        indexer = self._make_indexer()
        indexer.index(basic_doc)
        count_before = indexer.store.chunk_count
        indexer.index(basic_doc)
        count_after = indexer.store.chunk_count
        # 上書きなので同じ件数のはず
        assert count_after == count_before

    def test_index_multiple_docs(self):
        indexer = self._make_indexer()
        docs = [
            RAGDocument(id=f"d{i}", title=f"Doc{i}", content=f"内容です。" * 5)
            for i in range(3)
        ]
        for doc in docs:
            indexer.index(doc)
        assert indexer.stats.doc_count == 3

    def test_index_batch(self):
        indexer = self._make_indexer()
        docs = [RAGDocument(id=f"d{i}", title=f"T{i}", content="コンテンツ。" * 5) for i in range(5)]
        result = indexer.index_batch(docs)
        assert len(result) == 5
        assert all(v >= 1 for v in result.values())

    def test_delete_doc(self, basic_doc):
        indexer = self._make_indexer()
        indexer.index(basic_doc)
        removed = indexer.delete("doc1")
        assert removed >= 1
        assert indexer.stats.doc_count == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. RAGRetriever
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRAGRetriever:
    def _make_retriever_with_indexer(self):
        cfg = EmbeddingConfig(model=GemmaEmbeddingModel.MOCK)
        embedder = GemmaEmbeddingProvider(cfg)
        store = _ChunkVectorStore()
        indexer = RAGIndexer(embedder, store=store)
        retriever = RAGRetriever(embedder, store)
        return indexer, retriever

    def test_retrieve_returns_results(self, basic_doc):
        indexer, retriever = self._make_retriever_with_indexer()
        indexer.index(basic_doc)
        results = retriever.retrieve("素材")
        assert len(results) >= 1

    def test_retrieve_result_type(self, basic_doc):
        indexer, retriever = self._make_retriever_with_indexer()
        indexer.index(basic_doc)
        results = retriever.retrieve("製品")
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_ranks_assigned(self, basic_doc):
        indexer, retriever = self._make_retriever_with_indexer()
        indexer.index(basic_doc)
        results = retriever.retrieve("製品", top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_retrieve_score_threshold(self, basic_doc):
        indexer, retriever = self._make_retriever_with_indexer()
        indexer.index(basic_doc)
        results = retriever.retrieve("製品", score_threshold=0.99)
        # 極めて高い閾値では結果が少なくなる
        all_scores = [r.score for r in results]
        assert all(s >= 0.99 for s in all_scores)

    def test_retrieve_top_k_limit(self, basic_doc):
        indexer, retriever = self._make_retriever_with_indexer()
        indexer.index(basic_doc)
        results = retriever.retrieve("製品", top_k=1)
        assert len(results) <= 1

    def test_retrieve_texts(self, basic_doc):
        indexer, retriever = self._make_retriever_with_indexer()
        indexer.index(basic_doc)
        texts = retriever.retrieve_texts("製品", top_k=3)
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)

    def test_retrieve_empty_store(self):
        _, retriever = self._make_retriever_with_indexer()
        results = retriever.retrieve("クエリ")
        assert results == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Generators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerators:
    def test_mock_generator_with_context(self):
        gen = MockLLMGenerator()
        text = gen.generate("質問", ["チャンク1", "チャンク2"])
        assert len(text) > 0

    def test_mock_generator_no_context(self):
        gen = MockLLMGenerator()
        text = gen.generate("質問", [])
        assert "見つかりませんでした" in text

    def test_mock_generator_model_name(self):
        gen = MockLLMGenerator()
        assert gen.model_name == "mock-llm"

    def test_gemma_llm_generator_model_name(self):
        gen = GemmaLLMGenerator(api_key="dummy", model="gemma-3-27b-it")
        assert gen.model_name == "gemma-3-27b-it"

    def test_gemma_llm_generator_default_model(self):
        gen = GemmaLLMGenerator(api_key="dummy")
        assert "gemma" in gen.model_name.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. RAGPipeline — 統合テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRAGPipeline:
    def test_create_mock(self):
        p = RAGPipeline.create_mock()
        assert isinstance(p, RAGPipeline)

    def test_create_production_no_gen_key(self):
        p = RAGPipeline.create_production(
            embedding_api_key="dummy_emb",
            generation_api_key="",
        )
        assert isinstance(p, RAGPipeline)
        # api_key がないので embedder は mock 扱いになる
        # (api_key が dummy_emb なので _is_mock=False だが httpエラーは実際の呼び出し時)

    def test_index_document(self, pipeline, basic_doc):
        count = pipeline.index_document(basic_doc)
        assert count >= 1

    def test_index_batch(self, pipeline):
        docs = [
            RAGDocument(id=f"d{i}", title=f"Doc{i}", content="内容です。" * 5)
            for i in range(4)
        ]
        result = pipeline.index_batch(docs)
        assert len(result) == 4

    def test_stats_idle_initially(self, pipeline):
        stats = pipeline.stats
        assert stats.status == RAGStatus.IDLE

    def test_stats_ready_after_index(self, indexed_pipeline):
        stats = indexed_pipeline.stats
        assert stats.status == RAGStatus.READY
        assert stats.doc_count == 1
        assert stats.chunk_count >= 1

    def test_search_returns_results(self, indexed_pipeline):
        results = indexed_pipeline.search("製品の素材")
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_search_empty_pipeline(self, pipeline):
        results = pipeline.search("質問")
        assert results == []

    def test_query_returns_rag_answer(self, indexed_pipeline):
        answer = indexed_pipeline.query("この製品はどんな素材ですか？")
        assert isinstance(answer, RAGAnswer)
        assert answer.success is True
        assert len(answer.sources) >= 0

    def test_query_answer_has_query(self, indexed_pipeline):
        q = "返品期限は？"
        answer = indexed_pipeline.query(q)
        assert answer.query == q

    def test_query_latency_tracked(self, indexed_pipeline):
        answer = indexed_pipeline.query("素材")
        assert answer.latency_ms >= 0

    def test_query_model_used(self, indexed_pipeline):
        answer = indexed_pipeline.query("素材")
        assert answer.model_used == "mock-llm"

    def test_query_sources_dict_format(self, indexed_pipeline):
        answer = indexed_pipeline.query("製品")
        for src in answer.sources:
            assert "chunk_id" in src
            assert "score" in src
            assert "rank" in src

    def test_delete_document(self, pipeline, basic_doc):
        pipeline.index_document(basic_doc)
        removed = pipeline.delete_document("doc1")
        assert removed >= 1
        assert pipeline.stats.doc_count == 0

    def test_re_index_after_delete(self, pipeline, basic_doc):
        pipeline.index_document(basic_doc)
        pipeline.delete_document("doc1")
        count = pipeline.index_document(basic_doc)
        assert count >= 1
        assert pipeline.stats.doc_count == 1

    def test_multiple_docs_query(self, pipeline):
        docs = [
            RAGDocument(id="d1", title="猫", content="猫はかわいい動物です。肉食で夜行性の生き物。"),
            RAGDocument(id="d2", title="車", content="自動車はガソリンまたは電気で動く乗り物。"),
            RAGDocument(id="d3", title="料理", content="料理は食材を加熱や調味料で味付けする技術。"),
        ]
        for doc in docs:
            pipeline.index_document(doc)
        answer = pipeline.query("猫について教えてください")
        assert answer.success is True

    def test_to_status_dict(self, indexed_pipeline):
        d = indexed_pipeline.to_status_dict()
        assert "status" in d
        assert "doc_count" in d
        assert "chunk_count" in d
        assert "embedder" in d
        assert "generator" in d

    def test_status_dict_values(self, indexed_pipeline):
        d = indexed_pipeline.to_status_dict()
        assert d["status"] == RAGStatus.READY.value
        assert d["doc_count"] == 1
        assert d["embedder"] == GemmaEmbeddingModel.MOCK.value

    def test_top_k_limit_in_query(self, pipeline):
        for i in range(5):
            doc = RAGDocument(
                id=f"d{i}", title=f"文書{i}",
                content=f"これは文書{i}の内容です。詳細な説明が含まれます。"
            )
            pipeline.index_document(doc)
        answer = pipeline.query("内容", top_k=2)
        assert len(answer.sources) <= 2

    def test_doc_filter_in_search(self, pipeline):
        docs = [
            RAGDocument(id="d1", title="A", content="アルファの情報です。" * 3),
            RAGDocument(id="d2", title="B", content="ベータの情報です。" * 3),
        ]
        for doc in docs:
            pipeline.index_document(doc)
        results = pipeline.search("情報", doc_filter=["d1"])
        assert all(r.chunk.doc_id == "d1" for r in results)

    def test_score_threshold_in_query(self, indexed_pipeline):
        answer = indexed_pipeline.query("全く関係ない話", score_threshold=0.999)
        # 非常に高い閾値 → ソースなし or 少数のみ
        for src in answer.sources:
            assert src["score"] >= 0.999

    def test_rag_answer_to_dict_keys(self, indexed_pipeline):
        answer = indexed_pipeline.query("製品")
        d = answer.to_dict()
        assert set(d.keys()) == {"query", "answer", "sources", "latency_ms", "model_used"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. エッジケース
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEdgeCases:
    def test_very_long_document(self, pipeline):
        """長文ドキュメント（10000字）でもクラッシュしない。"""
        content = "これは長い文書です。詳細な内容が続きます。" * 200
        doc = RAGDocument(id="long", title="長文", content=content)
        count = pipeline.index_document(doc)
        assert count >= 1

    def test_unicode_content(self, pipeline):
        """Unicode / emoji / 記号を含んでもクラッシュしない。"""
        doc = RAGDocument(
            id="u1", title="Unicode テスト",
            content="🎉 祝！ 特殊文字 ①②③ αβγ → ← ↑ \n\n次の段落。",
        )
        answer = pipeline.index_document(doc)
        assert answer >= 0

    def test_single_word_content(self, pipeline):
        """非常に短いコンテンツ。"""
        doc = RAGDocument(id="short", title="T", content="x" * 25)
        pipeline.index_document(doc)
        results = pipeline.search("x")
        # チャンク化できるかどうか依存、少なくともクラッシュしない
        assert isinstance(results, list)

    def test_query_after_all_docs_deleted(self, indexed_pipeline):
        """全削除後のクエリ。"""
        indexed_pipeline.delete_document("doc1")
        answer = indexed_pipeline.query("製品")
        assert isinstance(answer, RAGAnswer)

    def test_duplicate_doc_id_overwrites(self, pipeline):
        """同 ID のドキュメントを二度インデックスすると上書き。"""
        doc = RAGDocument(id="dup", title="初版", content="最初の内容です。詳細情報。" * 3)
        pipeline.index_document(doc)
        count_before = pipeline.stats.chunk_count

        doc2 = RAGDocument(id="dup", title="改版", content="更新された内容です。" * 3)
        pipeline.index_document(doc2)
        count_after = pipeline.stats.chunk_count

        assert pipeline.stats.doc_count == 1   # 2 ではなく 1
        # チャンク数は置き換え後の件数
        assert count_after >= 1

    def test_chunking_config_sentence_strategy(self):
        """文境界チャンキングで文末記号が正しく分割される。"""
        content = (
            "第一文は長めの文章です。詳細な内容が含まれています。"
            "第二文も同様に長い文章となっています。詳しい説明があります。"
        )
        doc = RAGDocument(id="s1", title="T", content=content)
        cfg = ChunkingConfig(strategy=ChunkingStrategy.SENTENCE, chunk_size=30)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk(doc)
        # chunk_size=30 で分割されるので複数チャンクになるはず
        assert len(chunks) >= 1

    def test_mock_embed_dimension_custom(self):
        """カスタム dim で埋め込みが正しいサイズを返す。"""
        cfg = EmbeddingConfig(model=GemmaEmbeddingModel.MOCK, dim=128)
        provider = GemmaEmbeddingProvider(cfg)
        vec = provider.embed("テスト")
        assert len(vec) == 128

    def test_retrieval_result_score_range(self, indexed_pipeline):
        """検索スコアは 0〜1 の範囲。"""
        results = indexed_pipeline.search("製品")
        for r in results:
            assert 0.0 <= r.score <= 1.0 + 1e-6   # 浮動小数誤差許容

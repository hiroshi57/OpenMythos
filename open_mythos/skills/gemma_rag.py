"""
Sprint 82 — EmbeddingGemma RAG エンジン

Google EmbeddingGemma (text-embedding-004) を利用した
Retrieval-Augmented Generation パイプラインの OpenMythos 移植。

ref: awesome-gemma / Google AI Studio Embedding API
     https://ai.google.dev/gemini-api/docs/embeddings

オブジェクト:
  GemmaEmbeddingModel  : 利用モデル識別 enum
  EmbeddingConfig      : 埋め込みプロバイダー設定
  RAGDocument          : インデックス対象ドキュメント
  RAGChunk             : チャンキング後の断片
  RetrievalResult      : 検索結果 (chunk + score)
  RAGAnswer            : 最終 RAG 回答
  GemmaEmbeddingProvider: EmbeddingGemma 呼び出し (Mock / HTTP)
  DocumentChunker      : 固定長 + 文境界チャンキング
  RAGIndexer           : chunk → embed → VectorStore(FAISS in-memory)
  RAGRetriever         : query embed → similarity search → RetrievalResult
  RAGPipeline          : インデックス + 検索 + 生成の統合ファサード

使用例::
    pipeline = RAGPipeline.create_mock()
    pipeline.index_document(RAGDocument(id="d1", title="製品説明", content="..."))
    answer = pipeline.query("この製品の主な特徴は？")
    print(answer.text)
    print(answer.sources)   # [{"chunk_id": ..., "score": ...}]
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 型定義 / Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GemmaEmbeddingModel(str, Enum):
    """Google AI Studio embedding モデル"""
    TEXT_EMBEDDING_004    = "text-embedding-004"        # EmbeddingGemma 最新
    EMBEDDING_001         = "embedding-001"             # 旧世代
    MOCK                  = "mock"                      # テスト用決定論的埋め込み


class ChunkingStrategy(str, Enum):
    FIXED_SIZE   = "fixed_size"     # 固定文字数
    SENTENCE     = "sentence"       # 文境界
    PARAGRAPH    = "paragraph"      # 段落境界


class RAGStatus(str, Enum):
    IDLE         = "idle"
    INDEXING     = "indexing"
    READY        = "ready"
    ERROR        = "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EmbeddingConfig:
    """埋め込みプロバイダー設定"""
    model:       GemmaEmbeddingModel = GemmaEmbeddingModel.TEXT_EMBEDDING_004
    api_key:     str                 = ""
    timeout:     int                 = 30        # 秒
    dim:         int                 = 768       # text-embedding-004 の次元数
    task_type:   str                 = "RETRIEVAL_DOCUMENT"
    # RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY | SEMANTIC_SIMILARITY | CLASSIFICATION


@dataclass
class RAGDocument:
    """インデックス対象ドキュメント"""
    id:       str
    title:    str
    content:  str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class RAGChunk:
    """チャンキング後の断片"""
    id:          str
    doc_id:      str
    doc_title:   str
    text:        str
    chunk_index: int
    metadata:    Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """タイトルを先頭に付与した検索用テキスト"""
        return f"{self.doc_title}\n{self.text}"


@dataclass
class ChunkingConfig:
    """チャンキング設定"""
    strategy:   ChunkingStrategy = ChunkingStrategy.SENTENCE
    chunk_size: int              = 512       # 文字数 (FIXED_SIZE / SENTENCE 上限)
    overlap:    int              = 64        # オーバーラップ文字数
    min_chars:  int              = 20        # この未満は無視


@dataclass
class RetrievalResult:
    """検索結果"""
    chunk:    RAGChunk
    score:    float             # cosine similarity (0.0〜1.0)
    rank:     int               = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id":  self.chunk.id,
            "doc_id":    self.chunk.doc_id,
            "doc_title": self.chunk.doc_title,
            "text":      self.chunk.text,
            "score":     round(self.score, 4),
            "rank":      self.rank,
        }


@dataclass
class RAGAnswer:
    """RAG 最終回答"""
    query:    str
    text:     str
    sources:  List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float              = 0.0
    model_used: str                = ""

    @property
    def success(self) -> bool:
        return bool(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query":      self.query,
            "answer":     self.text,
            "sources":    self.sources,
            "latency_ms": round(self.latency_ms, 1),
            "model_used": self.model_used,
        }


@dataclass
class IndexStats:
    """インデックス統計"""
    doc_count:   int   = 0
    chunk_count: int   = 0
    status:      RAGStatus = RAGStatus.IDLE
    last_indexed: Optional[str] = None    # ISO 8601


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GemmaEmbeddingProvider
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GemmaEmbeddingProvider:
    """
    EmbeddingGemma (text-embedding-004) の埋め込み生成プロバイダー。

    mock モード時は deterministic ハッシュベース埋め込みを返す（テスト用）。
    本番モード時は Google AI Studio REST API を呼ぶ。
    """

    _ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:embedContent?key={api_key}"
    )

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._is_mock = (config.model == GemmaEmbeddingModel.MOCK or not config.api_key)

    # ── 公開 API ──────────────────────────────────────────────────

    def embed(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """テキスト 1 件を埋め込みベクターに変換する。"""
        t = task_type or self.config.task_type
        if self._is_mock:
            return self._mock_embed(text, self.config.dim)
        return self._http_embed(text, t)

    def embed_batch(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
    ) -> List[List[float]]:
        """複数テキストを一括で埋め込む。"""
        return [self.embed(t, task_type) for t in texts]

    @property
    def dim(self) -> int:
        return self.config.dim

    # ── Mock 実装 ──────────────────────────────────────────────────

    @staticmethod
    def _mock_embed(text: str, dim: int) -> List[float]:
        """
        テスト用 deterministic 埋め込み。
        同じテキストは常に同じベクターを返し、コサイン類似度が
        意味的近さをある程度反映するよう単語ハッシュで構成する。
        """
        vec = [0.0] * dim
        words = re.findall(r'\w+', text.lower())
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        # L2 正規化
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # ── HTTP 実装 ─────────────────────────────────────────────────

    def _http_embed(self, text: str, task_type: str) -> List[float]:
        model_name = self.config.model.value
        url = self._ENDPOINT.format(
            model=model_name,
            api_key=self.config.api_key,
        )
        body = json.dumps({
            "model": f"models/{model_name}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read())
            return data["embedding"]["values"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"EmbeddingGemma HTTP {e.code}: {e.read().decode()}") from e
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"EmbeddingGemma レスポンス解析失敗: {e}") from e


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DocumentChunker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DocumentChunker:
    """
    ドキュメントを検索しやすい断片に分割する。

    strategy:
      FIXED_SIZE  — chunk_size 文字ごとに分割 (overlap あり)
      SENTENCE    — 文末 (。.!?) で分割、上限 chunk_size 以内に収める
      PARAGRAPH   — 空行で段落分割
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, doc: RAGDocument) -> List[RAGChunk]:
        strategy = self.config.strategy
        if strategy == ChunkingStrategy.FIXED_SIZE:
            texts = self._fixed_size(doc.content)
        elif strategy == ChunkingStrategy.SENTENCE:
            texts = self._sentence(doc.content)
        else:
            texts = self._paragraph(doc.content)

        chunks: List[RAGChunk] = []
        for i, text in enumerate(texts):
            if len(text) < self.config.min_chars:
                continue
            chunk_id = f"{doc.id}__c{i}"
            chunks.append(RAGChunk(
                id=chunk_id,
                doc_id=doc.id,
                doc_title=doc.title,
                text=text.strip(),
                chunk_index=i,
                metadata={**doc.metadata, "doc_id": doc.id},
            ))
        return chunks

    # ── チャンキング戦略 ─────────────────────────────────────────

    def _fixed_size(self, text: str) -> List[str]:
        size = self.config.chunk_size
        overlap = self.config.overlap
        step = max(1, size - overlap)
        return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size]]

    def _sentence(self, text: str) -> List[str]:
        """文末記号で分割し、chunk_size を超えたら強制分割。"""
        # 文末記号: 。.!?！？（全角・半角両対応）
        raw_sents = re.split(r'(?<=[。.!?！？])\s*', text)
        chunks: List[str] = []
        current = ""
        for sent in raw_sents:
            if not sent:
                continue
            if len(current) + len(sent) <= self.config.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current)
                # sent 自体が chunk_size を超えるなら固定分割
                if len(sent) > self.config.chunk_size:
                    chunks.extend(self._fixed_size(sent))
                    current = ""
                else:
                    current = sent
        if current:
            chunks.append(current)
        return chunks

    def _paragraph(self, text: str) -> List[str]:
        """空行で段落分割。"""
        paras = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paras if p.strip()]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# In-Memory Vector Store (FAISS 代替)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class _ChunkVectorStore:
    """RAG チャンク専用の in-memory ベクターストア。"""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[RAGChunk, List[float]]] = {}

    def upsert(self, chunk: RAGChunk, vector: List[float]) -> None:
        self._store[chunk.id] = (chunk, vector)

    def delete_by_doc(self, doc_id: str) -> int:
        keys = [k for k, (c, _) in self._store.items() if c.doc_id == doc_id]
        for k in keys:
            del self._store[k]
        return len(keys)

    def search(
        self,
        query_vec: List[float],
        top_k: int = 5,
        doc_filter: Optional[List[str]] = None,
    ) -> List[Tuple[RAGChunk, float]]:
        results: List[Tuple[RAGChunk, float]] = []
        for chunk, vec in self._store.values():
            if doc_filter and chunk.doc_id not in doc_filter:
                continue
            score = _cosine_sim(query_vec, vec)
            results.append((chunk, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @property
    def chunk_count(self) -> int:
        return len(self._store)

    @property
    def doc_ids(self) -> List[str]:
        return list({c.doc_id for c, _ in self._store.values()})

    def clear(self) -> None:
        self._store.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAGIndexer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RAGIndexer:
    """
    ドキュメントをチャンク化 → 埋め込み → ベクターストアへ登録する。
    """

    def __init__(
        self,
        embedder: GemmaEmbeddingProvider,
        chunker: Optional[DocumentChunker] = None,
        store: Optional[_ChunkVectorStore] = None,
    ) -> None:
        self.embedder = embedder
        self.chunker  = chunker or DocumentChunker()
        self.store    = store or _ChunkVectorStore()
        self._doc_registry: Dict[str, RAGDocument] = {}

    def index(self, doc: RAGDocument) -> List[RAGChunk]:
        """1 ドキュメントをインデックスする。既存の同 ID は上書き。"""
        # 旧チャンクを削除
        if doc.id in self._doc_registry:
            self.store.delete_by_doc(doc.id)

        chunks = self.chunker.chunk(doc)
        texts = [c.full_text for c in chunks]
        vectors = self.embedder.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

        for chunk, vec in zip(chunks, vectors):
            self.store.upsert(chunk, vec)

        self._doc_registry[doc.id] = doc
        return chunks

    def index_batch(self, docs: List[RAGDocument]) -> Dict[str, int]:
        """複数ドキュメントを一括インデックス。{doc_id: chunk_count} を返す。"""
        result: Dict[str, int] = {}
        for doc in docs:
            chunks = self.index(doc)
            result[doc.id] = len(chunks)
        return result

    def delete(self, doc_id: str) -> int:
        """ドキュメントをインデックスから削除。削除チャンク数を返す。"""
        removed = self.store.delete_by_doc(doc_id)
        self._doc_registry.pop(doc_id, None)
        return removed

    @property
    def stats(self) -> IndexStats:
        return IndexStats(
            doc_count=len(self._doc_registry),
            chunk_count=self.store.chunk_count,
            status=RAGStatus.READY if self._doc_registry else RAGStatus.IDLE,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAGRetriever
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RAGRetriever:
    """
    クエリを埋め込み → ベクターストア検索 → RetrievalResult リストを返す。
    """

    def __init__(
        self,
        embedder: GemmaEmbeddingProvider,
        store: _ChunkVectorStore,
    ) -> None:
        self.embedder = embedder
        self.store    = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[List[str]] = None,
        score_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        クエリに最も関連するチャンクを返す。

        Args:
            query:           検索クエリ文字列
            top_k:           返す上位件数
            doc_filter:      特定 doc_id のみ検索（None = 全件）
            score_threshold: この未満のスコアは除外
        """
        q_vec = self.embedder.embed(query, task_type="RETRIEVAL_QUERY")
        hits = self.store.search(q_vec, top_k=top_k, doc_filter=doc_filter)
        results: List[RetrievalResult] = []
        for rank, (chunk, score) in enumerate(hits, start=1):
            if score < score_threshold:
                continue
            results.append(RetrievalResult(chunk=chunk, score=score, rank=rank))
        return results

    def retrieve_texts(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        """チャンクのテキストだけをリストで返す（LLM プロンプト組み立て用）。"""
        results = self.retrieve(query, top_k=top_k)
        return [r.chunk.text for r in results]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GenerationMixin (LLM 生成の薄いラッパー)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MockLLMGenerator:
    """テスト用 LLM — 取得チャンクを結合して返す。"""

    def generate(self, query: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return f"「{query}」に関する情報が見つかりませんでした。"
        joined = "\n---\n".join(context_chunks[:3])
        return f"[RAG 回答]\n参照情報:\n{joined}\n\n質問: {query}"

    @property
    def model_name(self) -> str:
        return "mock-llm"


class GemmaLLMGenerator:
    """
    Google AI Studio Gemma モデルを使った生成。
    (FunctionGemma / Gemma 4 等との統合ポイント)
    """

    _ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent?key={api_key}"
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gemma-3-27b-it",
        timeout: int = 60,
    ) -> None:
        self.api_key  = api_key
        self.model    = model
        self.timeout  = timeout

    def generate(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = (
            f"以下のコンテキスト情報を参照して、質問に日本語で答えてください。\n\n"
            f"コンテキスト:\n{context}\n\n"
            f"質問: {query}\n\n回答:"
        )
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
        }).encode()
        url = self._ENDPOINT.format(model=self.model, api_key=self.api_key)
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"GemmaLLM 生成失敗: {e}") from e

    @property
    def model_name(self) -> str:
        return self.model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAGPipeline (統合ファサード)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RAGPipeline:
    """
    インデックス・検索・生成を統合したメインファサード。

    使用例:
        # Mock モード（テスト）
        pipeline = RAGPipeline.create_mock()
        pipeline.index_document(RAGDocument(id="d1", title="製品Q&A", content="..."))
        answer = pipeline.query("返品ポリシーは？")
        print(answer.text)

        # 本番モード（API キーあり）
        pipeline = RAGPipeline.create_production(
            embedding_api_key="...",
            generation_api_key="...",
        )
    """

    def __init__(
        self,
        indexer:   RAGIndexer,
        retriever: RAGRetriever,
        generator: Any,             # MockLLMGenerator | GemmaLLMGenerator
        top_k:     int = 5,
    ) -> None:
        self.indexer   = indexer
        self.retriever = retriever
        self.generator = generator
        self.top_k     = top_k

    # ── ファクトリ ────────────────────────────────────────────────

    @classmethod
    def create_mock(
        cls,
        chunking_config: Optional[ChunkingConfig] = None,
        top_k: int = 5,
    ) -> "RAGPipeline":
        """テスト用 Mock パイプライン（API キー不要）。"""
        emb_cfg  = EmbeddingConfig(model=GemmaEmbeddingModel.MOCK)
        embedder = GemmaEmbeddingProvider(emb_cfg)
        store    = _ChunkVectorStore()
        chunker  = DocumentChunker(chunking_config)
        indexer  = RAGIndexer(embedder, chunker, store)
        retriever = RAGRetriever(embedder, store)
        generator = MockLLMGenerator()
        return cls(indexer, retriever, generator, top_k)

    @classmethod
    def create_production(
        cls,
        embedding_api_key: str,
        generation_api_key: str = "",
        embedding_model: GemmaEmbeddingModel = GemmaEmbeddingModel.TEXT_EMBEDDING_004,
        generation_model: str = "gemma-3-27b-it",
        chunking_config: Optional[ChunkingConfig] = None,
        top_k: int = 5,
    ) -> "RAGPipeline":
        """本番用パイプライン（Google AI Studio API キー必要）。"""
        emb_cfg  = EmbeddingConfig(model=embedding_model, api_key=embedding_api_key)
        embedder = GemmaEmbeddingProvider(emb_cfg)
        store    = _ChunkVectorStore()
        chunker  = DocumentChunker(chunking_config)
        indexer  = RAGIndexer(embedder, chunker, store)
        retriever = RAGRetriever(embedder, store)
        if generation_api_key:
            generator = GemmaLLMGenerator(generation_api_key, model=generation_model)
        else:
            generator = MockLLMGenerator()
        return cls(indexer, retriever, generator, top_k)

    # ── インデックス操作 ─────────────────────────────────────────

    def index_document(self, doc: RAGDocument) -> int:
        """1 ドキュメントをインデックス。登録チャンク数を返す。"""
        chunks = self.indexer.index(doc)
        return len(chunks)

    def index_batch(self, docs: List[RAGDocument]) -> Dict[str, int]:
        return self.indexer.index_batch(docs)

    def delete_document(self, doc_id: str) -> int:
        return self.indexer.delete(doc_id)

    # ── 検索 ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_filter: Optional[List[str]] = None,
        score_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """検索のみ（生成なし）。"""
        return self.retriever.retrieve(
            query,
            top_k=top_k or self.top_k,
            doc_filter=doc_filter,
            score_threshold=score_threshold,
        )

    # ── RAG 生成 ─────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        doc_filter: Optional[List[str]] = None,
        score_threshold: float = 0.0,
    ) -> RAGAnswer:
        """
        検索 + 生成の統合クエリ。

        1. question を embedding してチャンク検索
        2. 上位 top_k チャンクをコンテキストとして LLM へ渡す
        3. RAGAnswer を返す
        """
        t0 = time.perf_counter()
        results = self.search(
            question,
            top_k=top_k or self.top_k,
            doc_filter=doc_filter,
            score_threshold=score_threshold,
        )
        context_chunks = [r.chunk.text for r in results]
        answer_text = self.generator.generate(question, context_chunks)
        latency_ms = (time.perf_counter() - t0) * 1000

        return RAGAnswer(
            query=question,
            text=answer_text,
            sources=[r.to_dict() for r in results],
            latency_ms=latency_ms,
            model_used=self.generator.model_name,
        )

    # ── ステータス ────────────────────────────────────────────────

    @property
    def stats(self) -> IndexStats:
        return self.indexer.stats

    def to_status_dict(self) -> Dict[str, Any]:
        stats = self.stats
        return {
            "status":      stats.status.value,
            "doc_count":   stats.doc_count,
            "chunk_count": stats.chunk_count,
            "embedder":    self.indexer.embedder.config.model.value,
            "generator":   self.generator.model_name,
        }

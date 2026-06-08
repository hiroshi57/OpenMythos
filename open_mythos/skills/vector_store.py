"""
Sprint 44 — Vector DB 統合

Hermes Skills: chroma / qdrant / pinecone / faiss (enhanced)
ref: skills/vector-db/*-SKILL.md

統一インターフェース VectorStoreBackend を定義し、
4 種のバックエンド (Chroma / Qdrant / Pinecone / FAISS) を実装する。
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# データモデル
# ---------------------------------------------------------------------------

@dataclass
class VectorDocument:
    """ベクターストアに格納するドキュメント。"""
    id: str
    vector: List[float]
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    @property
    def dim(self) -> int:
        return len(self.vector)


@dataclass
class VectorStoreConfig:
    """バックエンド共通設定。"""
    collection: str = "default"
    dim: int = 768
    metric: str = "cosine"          # cosine | l2 | dot
    top_k: int = 5
    # backend-specific
    host: str = "localhost"
    port: int = 8080
    api_key: str = ""
    persist_dir: str = ".vector_store"


class VectorStoreBackend(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    FAISS = "faiss"


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _l2(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _similarity(metric: str, a: List[float], b: List[float]) -> float:
    if metric == "l2":
        return -_l2(a, b)       # 高いほど近い
    if metric == "dot":
        return sum(x * y for x, y in zip(a, b))
    return _cosine(a, b)


# ---------------------------------------------------------------------------
# 基底クラス (in-memory fallback)
# ---------------------------------------------------------------------------

class _BaseStore:
    """全バックエンド共通の in-memory フォールバック実装。"""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self.cfg = cfg
        self._docs: Dict[str, VectorDocument] = {}

    # ---- 書き込み ----

    def upsert(self, docs: List[VectorDocument]) -> int:
        """ドキュメントを挿入/更新する。返り値は処理件数。"""
        for doc in docs:
            self._docs[doc.id] = doc
        return len(docs)

    def delete(self, ids: List[str]) -> int:
        """指定 ID を削除する。返り値は実際に削除した件数。"""
        removed = 0
        for doc_id in ids:
            if doc_id in self._docs:
                del self._docs[doc_id]
                removed += 1
        return removed

    # ---- 検索 ----

    def query(
        self,
        vector: List[float],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        """近傍探索。(document, score) のリストを降順で返す。"""
        k = top_k or self.cfg.top_k
        results: List[Tuple[VectorDocument, float]] = []
        for doc in self._docs.values():
            if filter:
                if not all(doc.metadata.get(mk) == mv for mk, mv in filter.items()):
                    continue
            score = _similarity(self.cfg.metric, vector, doc.vector)
            results.append((doc, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    # ---- 管理 ----

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()

    @property
    def backend(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# ChromaStore (Skill: chroma)
# ---------------------------------------------------------------------------

class ChromaStore(_BaseStore):
    """ChromaDB バックエンド (in-memory fallback)。

    本番では `chromadb` パッケージをインポートして
    PersistentClient / HttpClient に委譲する。
    """

    def __init__(self, cfg: VectorStoreConfig) -> None:
        super().__init__(cfg)
        try:
            import chromadb  # type: ignore
            if cfg.host == "localhost":
                self._client = chromadb.Client()
            else:
                self._client = chromadb.HttpClient(host=cfg.host, port=cfg.port)
            self._col = self._client.get_or_create_collection(cfg.collection)
            self._native = True
        except ImportError:
            self._native = False

    def upsert(self, docs: List[VectorDocument]) -> int:
        if self._native:
            self._col.upsert(
                ids=[d.id for d in docs],
                embeddings=[d.vector for d in docs],
                documents=[d.text for d in docs],
                metadatas=[d.metadata for d in docs],
            )
            return len(docs)
        return super().upsert(docs)

    def query(
        self,
        vector: List[float],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        if self._native:
            k = top_k or self.cfg.top_k
            kwargs: Dict[str, Any] = {"query_embeddings": [vector], "n_results": k}
            if filter:
                kwargs["where"] = filter
            res = self._col.query(**kwargs)
            out = []
            for i, doc_id in enumerate(res["ids"][0]):
                doc = VectorDocument(
                    id=doc_id,
                    vector=vector,
                    text=(res["documents"] or [[""]])[0][i],
                    metadata=(res["metadatas"] or [[{}]])[0][i],
                )
                dist = (res["distances"] or [[0.0]])[0][i]
                score = 1.0 - dist  # cosine distance → similarity
                out.append((doc, score))
            return out
        return super().query(vector, top_k=top_k, filter=filter)


# ---------------------------------------------------------------------------
# QdrantStore (Skill: qdrant-vector-search)
# ---------------------------------------------------------------------------

class QdrantStore(_BaseStore):
    """Qdrant バックエンド (in-memory fallback)。

    本番では `qdrant-client` パッケージをインポートして
    QdrantClient に委譲する。
    """

    def __init__(self, cfg: VectorStoreConfig) -> None:
        super().__init__(cfg)
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore
            _dist = {"cosine": Distance.COSINE, "l2": Distance.EUCLID, "dot": Distance.DOT}
            self._qc = QdrantClient(host=cfg.host, port=cfg.port, api_key=cfg.api_key or None)
            existing = [c.name for c in self._qc.get_collections().collections]
            if cfg.collection not in existing:
                self._qc.create_collection(
                    cfg.collection,
                    vectors_config=VectorParams(size=cfg.dim, distance=_dist.get(cfg.metric, Distance.COSINE)),
                )
            self._native = True
        except (ImportError, Exception):
            self._native = False

    def upsert(self, docs: List[VectorDocument]) -> int:
        if self._native:
            from qdrant_client.models import PointStruct  # type: ignore
            points = [
                PointStruct(id=d.id, vector=d.vector, payload={**d.metadata, "_text": d.text})
                for d in docs
            ]
            self._qc.upsert(collection_name=self.cfg.collection, points=points)
            return len(docs)
        return super().upsert(docs)

    def query(
        self,
        vector: List[float],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        if self._native:
            from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
            k = top_k or self.cfg.top_k
            qfilter = None
            if filter:
                qfilter = Filter(
                    must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter.items()]
                )
            hits = self._qc.search(
                collection_name=self.cfg.collection,
                query_vector=vector,
                limit=k,
                query_filter=qfilter,
            )
            return [
                (VectorDocument(
                    id=str(h.id),
                    vector=vector,
                    text=h.payload.get("_text", ""),
                    metadata={k: v for k, v in h.payload.items() if k != "_text"},
                ), h.score)
                for h in hits
            ]
        return super().query(vector, top_k=top_k, filter=filter)


# ---------------------------------------------------------------------------
# PineconeStore (Skill: pinecone)
# ---------------------------------------------------------------------------

class PineconeStore(_BaseStore):
    """Pinecone バックエンド (in-memory fallback)。

    本番では `pinecone-client` パッケージをインポートして
    Pinecone Index に委譲する。
    """

    def __init__(self, cfg: VectorStoreConfig) -> None:
        super().__init__(cfg)
        self._native = False
        if cfg.api_key:
            try:
                from pinecone import Pinecone  # type: ignore
                pc = Pinecone(api_key=cfg.api_key)
                self._index = pc.Index(cfg.collection)
                self._native = True
            except (ImportError, Exception):
                pass

    def upsert(self, docs: List[VectorDocument]) -> int:
        if self._native:
            vectors = [
                {"id": d.id, "values": d.vector, "metadata": {**d.metadata, "_text": d.text}}
                for d in docs
            ]
            self._index.upsert(vectors=vectors)
            return len(docs)
        return super().upsert(docs)

    def query(
        self,
        vector: List[float],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        if self._native:
            k = top_k or self.cfg.top_k
            kwargs: Dict[str, Any] = {"vector": vector, "top_k": k, "include_metadata": True}
            if filter:
                kwargs["filter"] = filter
            res = self._index.query(**kwargs)
            return [
                (VectorDocument(
                    id=m.id,
                    vector=vector,
                    text=m.metadata.get("_text", ""),
                    metadata={k: v for k, v in m.metadata.items() if k != "_text"},
                ), m.score)
                for m in res.matches
            ]
        return super().query(vector, top_k=top_k, filter=filter)


# ---------------------------------------------------------------------------
# FaissStore (Skill: faiss — enhanced from Sprint 33)
# ---------------------------------------------------------------------------

class FaissStore(_BaseStore):
    """FAISS バックエンド (enhanced)。

    in-memory fallback は _BaseStore が担う。
    faiss パッケージがある場合は IVFFlat / HNSW インデックスを使用する。
    """

    def __init__(self, cfg: VectorStoreConfig, index_type: str = "flat") -> None:
        super().__init__(cfg)
        self._index_type = index_type  # flat | ivf | hnsw
        self._faiss_index: Any = None
        self._id_list: List[str] = []
        self._actual_dim: int = cfg.dim
        try:
            import faiss  # type: ignore
            self._faiss = faiss
            self._native = True
        except ImportError:
            self._native = False

    def _build_index(self, dim: int) -> None:
        faiss = self._faiss
        self._actual_dim = dim
        if self._index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, 32)
        elif self._index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dim)
            idx = faiss.IndexIVFFlat(quantizer, dim, 100)
        else:
            idx = faiss.IndexFlatL2(dim)
        if self.cfg.metric == "cosine":
            self._faiss_index = faiss.IndexFlatIP(dim)
        else:
            self._faiss_index = idx

    def upsert(self, docs: List[VectorDocument]) -> int:
        if self._native:
            import numpy as np  # type: ignore
            # 遅延インデックス構築: 最初の upsert 時に実際の次元で初期化
            if self._faiss_index is None and docs:
                self._build_index(len(docs[0].vector))
            if self._faiss_index is not None:
                vecs = np.array([d.vector for d in docs], dtype=np.float32)
                if self.cfg.metric == "cosine":
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    vecs = vecs / (norms + 1e-9)
                self._faiss_index.add(vecs)
                self._id_list.extend(d.id for d in docs)
                # also store in base for metadata lookup
                super().upsert(docs)
                return len(docs)
        return super().upsert(docs)

    def query(
        self,
        vector: List[float],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        if self._native and self._faiss_index is not None and self._id_list:
            import numpy as np  # type: ignore
            k = min(top_k or self.cfg.top_k, len(self._id_list))
            q = np.array([vector], dtype=np.float32)
            if self.cfg.metric == "cosine":
                q = q / (np.linalg.norm(q) + 1e-9)
            D, I = self._faiss_index.search(q, k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self._id_list):
                    continue
                doc_id = self._id_list[idx]
                doc = self._docs.get(doc_id)
                if doc is None:
                    continue
                if filter and not all(doc.metadata.get(mk) == mv for mk, mv in filter.items()):
                    continue
                score = float(dist)
                results.append((doc, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        return super().query(vector, top_k=top_k, filter=filter)


# ---------------------------------------------------------------------------
# ファクトリ
# ---------------------------------------------------------------------------

class VectorStoreFactory:
    """バックエンド種別から適切なストアを生成するファクトリ。"""

    @staticmethod
    def create(
        backend: VectorStoreBackend,
        cfg: Optional[VectorStoreConfig] = None,
        **kwargs: Any,
    ) -> _BaseStore:
        cfg = cfg or VectorStoreConfig(**kwargs)
        if backend == VectorStoreBackend.CHROMA:
            return ChromaStore(cfg)
        if backend == VectorStoreBackend.QDRANT:
            return QdrantStore(cfg)
        if backend == VectorStoreBackend.PINECONE:
            return PineconeStore(cfg)
        if backend == VectorStoreBackend.FAISS:
            return FaissStore(cfg)
        raise ValueError(f"Unknown backend: {backend}")

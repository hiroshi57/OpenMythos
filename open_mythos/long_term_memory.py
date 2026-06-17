"""
LongTermMemoryAgent — 長期記憶統合 (Sprint 26 / P7パターン).
ANN インデックス対応 (Sprint 33 / FAISS).

エピソード記憶（過去の対話履歴）とセマンティック記憶（知識・ファクト）を
統合管理し、類似クエリへの検索精度を継続的に向上させる。

設計:
    MemoryEntry         -- 単一記憶エントリ (text, embedding, score, category)
    MemoryRetrieval     -- 検索結果 (entries + relevance_scores)
    EpisodicStore       -- 対話エピソードを時系列・類似度で蓄積・検索
    SemanticStore       -- キーワードインデックス付き知識ストア
    LongTermMemoryAgent -- エピソード + セマンティック統合インターフェース

精度向上のポイント:
    - score_threshold でノイズ除去 (品質フィルタ)
    - Jaccard重複除去で記憶の多様性を確保
    - 類似度 × スコア × 鮮度の3軸で優先順を決定
    - consolidate() で古い低品質記憶を圧縮

他スプリントとの連携:
    - LLMOScorer (Sprint 10/19) でストア品質を保証
    - RAG VectorStore (Sprint 11) と同じ TF-IDF ベクトル化
    - ConversationMemory (Sprint 12) のターンを直接投入可能
    - EnsembleScorer (Sprint 27) の入力として利用

使い方::

    from open_mythos.long_term_memory import LongTermMemoryAgent

    agent = LongTermMemoryAgent(score_threshold=0.6, max_episodes=1000)
    agent.store_episode("SEO記事の最適構成は？",
                        "H1→H2×3→FAQ→CTA の順が効果的", score=0.92)
    agent.store_knowledge("llmo_definition",
                          "LLMOはAI検索向けコンテンツ最適化", tags=["llmo","seo"])

    result = agent.retrieve("記事の構成について教えて", top_k=3)
    print(result.best_entry.text)
    print(f"relevance: {result.top_relevance:.3f}")
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # 型注釈 "np.ndarray" 用 (実体は各メソッド内で遅延 import)


# ---------------------------------------------------------------------------
# TF-IDF ユーティリティ (stdlib only)
# ---------------------------------------------------------------------------
# ANN ユーティリティ / FAISS ラッパー は後述 (ANNIndex クラス)


def _tokenize(text: str) -> List[str]:
    """英語単語分割 + 日本語 bi-gram のフォールバック。"""
    import re
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    # 日本語文字があれば bi-gram を追加
    ja_chars = re.findall(r"[぀-鿿]", text)
    if ja_chars:
        for i in range(len(ja_chars) - 1):
            tokens.append(ja_chars[i] + ja_chars[i + 1])
    return tokens


def _tf(tokens: List[str]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    total = max(len(tokens), 1)
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1 / total
    return tf


def _cosine_tfidf(text_a: str, text_b: str, idf: Optional[Dict[str, float]] = None) -> float:
    """TF-IDF コサイン類似度。idf が None の場合は TF のみ使用。"""
    ta, tb = _tokenize(text_a), _tokenize(text_b)
    tf_a, tf_b = _tf(ta), _tf(tb)
    vocab = set(tf_a) | set(tf_b)
    if not vocab:
        return 0.0
    va = [tf_a.get(v, 0.0) * (idf.get(v, 1.0) if idf else 1.0) for v in vocab]
    vb = [tf_b.get(v, 0.0) * (idf.get(v, 1.0) if idf else 1.0) for v in vocab]
    dot = sum(a * b for a, b in zip(va, vb))
    na = math.sqrt(sum(a * a for a in va))
    nb = math.sqrt(sum(b * b for b in vb))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


# ---------------------------------------------------------------------------
# ANN ユーティリティ — ハッシュ TF-IDF ベクトル化 + FAISS ラッパー (Sprint 33)
# ---------------------------------------------------------------------------

ANN_DIM: int = 256   # ハッシュ TF-IDF ベクトルの固定次元数


def _text_to_vector(text: str, dim: int = ANN_DIM) -> "np.ndarray":
    """
    テキストを固定次元 L2 正規化ベクトルに変換する (ハッシュ TF-IDF)。

    トークンを hash(token) % dim でバケットに割り当て、TF を計算してから
    L2 正規化する。語彙の事前定義不要で FAISS に直接 add できる。

    Parameters
    ----------
    text : 変換対象テキスト
    dim  : ベクトル次元数 (デフォルト ANN_DIM=256)
    """
    import numpy as np
    tokens = _tokenize(text)
    vec = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        vec[hash(t) % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec /= norm
    return vec


class ANNIndex:
    """
    FAISS ANN インデックスのラッパー。

    - ``backend="auto"``  : faiss が import できる場合 → faiss、できない場合 → linear
    - ``backend="faiss"`` : faiss を強制使用 (import できなければ ImportError)
    - ``backend="linear"``: 全 entry_id を返す線形フォールバック

    検索精度:
        ``faiss.IndexFlatIP`` (内積) を使用。ベクトルが L2 正規化済みなら
        内積 == コサイン類似度であり、完全な正確度 (ANN ではなく厳密な NN)。

    Parameters
    ----------
    dim     : ベクトル次元数 (EpisodicStore の ANN_DIM と一致させること)
    backend : "auto" | "faiss" | "linear"
    """

    def __init__(self, dim: int = ANN_DIM, backend: str = "auto") -> None:
        self.dim       = dim
        self._backend  = self._resolve_backend(backend)
        self._id_map: List[str] = []
        self._faiss_idx = None

        if self._backend == "faiss":
            import faiss
            self._faiss_idx = faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------------
    # Class methods / properties
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "auto":
            try:
                import faiss  # noqa: F401
                return "faiss"
            except ImportError:
                return "linear"
        if backend == "faiss":
            try:
                import faiss  # noqa: F401
            except ImportError:
                raise ImportError(
                    "faiss が見つかりません。pip install faiss-cpu でインストールしてください。"
                )
        return backend

    @staticmethod
    def faiss_available() -> bool:
        """FAISS が利用可能かどうかを返す"""
        try:
            import faiss  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def is_faiss(self) -> bool:
        """FAISS バックエンドが有効かどうか"""
        return self._backend == "faiss"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def size(self) -> int:
        return len(self._id_map)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry_id: str, vector: "np.ndarray") -> None:
        """
        1 エントリを追加する。

        Parameters
        ----------
        entry_id : MemoryEntry.entry_id
        vector   : _text_to_vector() で生成した L2 正規化ベクトル
        """
        self._id_map.append(entry_id)
        if self._backend == "faiss":
            import numpy as np
            self._faiss_idx.add(vector.reshape(1, -1).astype(np.float32))

    def search(
        self,
        query_vec: "np.ndarray",
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        ANN 検索して ``(entry_id, score)`` のリストを返す。

        linear backend では全 entry_id を score=0 で返す
        (EpisodicStore が続けて TF-IDF で再スコアリングする)。
        """
        if not self._id_map:
            return []
        k = min(top_k, len(self._id_map))
        if self._backend == "faiss":
            import numpy as np
            qvec = query_vec.reshape(1, -1).astype(np.float32)
            scores, indices = self._faiss_idx.search(qvec, k)
            return [
                (self._id_map[int(i)], float(s))
                for i, s in zip(indices[0], scores[0])
                if i >= 0
            ]
        # linear fallback: 先頭 k 件を返す (EpisodicStore が TF-IDF で再スコアリング)
        return [(eid, 0.0) for eid in self._id_map[:k]]

    def rebuild(
        self,
        entry_ids: List[str],
        vectors: "np.ndarray",
    ) -> None:
        """
        全エントリで ANN インデックスを再構築する。

        _evict() や consolidate() 後に呼ぶ。

        Parameters
        ----------
        entry_ids : エントリ ID リスト (N,)
        vectors   : 対応するベクトル行列 (N, dim)
        """
        self._id_map = list(entry_ids)
        if self._backend == "faiss":
            import faiss, numpy as np  # noqa: E401
            self._faiss_idx = faiss.IndexFlatIP(self.dim)
            if len(entry_ids) > 0:
                self._faiss_idx.add(vectors.astype(np.float32))

    def clear(self) -> None:
        """インデックスを空にする。"""
        self._id_map.clear()
        if self._backend == "faiss":
            import faiss
            self._faiss_idx = faiss.IndexFlatIP(self.dim)


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """
    単一記憶エントリ。

    Attributes
    ----------
    text       : 記憶本文 (対話応答またはファクト)
    context    : 発火元のクエリ / コンテキスト
    score      : 品質スコア (0〜1)
    category   : "episode" / "knowledge" / カスタムラベル
    tags       : 検索キーワードタグ
    created_at : 作成 UNIX タイムスタンプ
    entry_id   : 一意ID
    access_count: 検索でヒットした回数 (鮮度×重要度のシグナル)
    freshness_half_life_h: 鮮度の半減期 (時間)。デフォルト 17h
    """

    text: str
    context: str = ""
    score: float = 1.0
    category: str = "episode"
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    access_count: int = 0
    freshness_half_life_h: float = 17.0  # 設定可能な半減期

    @property
    def freshness(self) -> float:
        """経過時間に基づく鮮度 (半減期は freshness_half_life_h で設定可能)。"""
        age_h = (time.time() - self.created_at) / 3600.0
        decay = math.log(2) / max(self.freshness_half_life_h, 1e-6)
        return math.exp(-decay * age_h)

    def priority(self, relevance: float = 0.0) -> float:
        """relevance × score × freshness の3軸優先スコア。"""
        return 0.5 * relevance + 0.3 * self.score + 0.2 * self.freshness


# ---------------------------------------------------------------------------
# MemoryRetrieval
# ---------------------------------------------------------------------------


@dataclass
class MemoryRetrieval:
    """
    retrieve() の返値。

    Attributes
    ----------
    query          : 検索クエリ
    entries        : 取得したエントリ (priority降順)
    relevance_scores: 各エントリの relevance スコア (entries と同順)
    total_searched : 検索したエントリ総数
    """

    query: str
    entries: List[MemoryEntry]
    relevance_scores: List[float]
    total_searched: int

    @property
    def best_entry(self) -> Optional[MemoryEntry]:
        return self.entries[0] if self.entries else None

    @property
    def top_relevance(self) -> float:
        return self.relevance_scores[0] if self.relevance_scores else 0.0

    def to_context_string(self) -> str:
        """検索結果を LLM 入力用テキストに変換。"""
        if not self.entries:
            return ""
        lines = ["[記憶から検索]:"]
        for i, (e, r) in enumerate(zip(self.entries, self.relevance_scores), 1):
            lines.append(f"  {i}. [{e.category}|score={e.score:.2f}|rel={r:.2f}] {e.text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EpisodicStore
# ---------------------------------------------------------------------------


class EpisodicStore:
    """
    対話エピソードを時系列・類似度で蓄積・検索するストア。

    Parameters
    ----------
    max_size      : 最大保持エントリ数 (超過時は score×freshness 下位を削除)
    score_threshold: この score 以上のエントリのみ保存
    dedup_threshold: Jaccard 類似度がこの値以上の場合は重複とみなす
    """

    def __init__(
        self,
        max_size: int = 500,
        score_threshold: float = 0.5,
        dedup_threshold: float = 0.85,
        freshness_half_life_h: float = 17.0,
        ann_backend: str = "auto",          # Sprint 33: "auto"|"faiss"|"linear"
    ) -> None:
        self.max_size = max_size
        self.score_threshold = score_threshold
        self.dedup_threshold = dedup_threshold
        self.freshness_half_life_h = freshness_half_life_h
        self._entries: List[MemoryEntry] = []
        self._idf: Dict[str, float] = {}  # コーパスレベルの IDF
        self._ann = ANNIndex(dim=ANN_DIM, backend=ann_backend)  # Sprint 33

    def _rebuild_idf(self) -> None:
        """全エントリからコーパスレベルの IDF を再計算する。"""
        n = max(len(self._entries), 1)
        df: Dict[str, int] = {}
        for e in self._entries:
            tokens = set(_tokenize(e.context + " " + e.text))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        self._idf = {
            t: math.log((n + 1) / (cnt + 1)) + 1.0  # スムージング付き IDF
            for t, cnt in df.items()
        }

    # ------------------------------------------------------------------ append

    def append(
        self,
        context: str,
        text: str,
        score: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> Optional[MemoryEntry]:
        """
        エピソードを追加する。

        score < score_threshold または重複する場合は追加しない。

        Returns
        -------
        追加した MemoryEntry、または None (フィルタ済み)
        """
        if score < self.score_threshold:
            return None
        # 重複チェック
        for e in self._entries[-50:]:  # 最近50件とのみ比較 (高速化)
            if _jaccard(e.text, text) >= self.dedup_threshold:
                # 同一に近い内容が既存 → score が高い方を残す
                # context・tags・created_at も新しい方で更新する (B5 fix)
                if score > e.score:
                    e.score = score
                    e.text = text
                    e.context = context
                    if tags:
                        e.tags = tags
                    e.created_at = time.time()
                return None
        entry = MemoryEntry(
            text=text,
            context=context,
            score=score,
            category="episode",
            tags=tags or [],
            freshness_half_life_h=self.freshness_half_life_h,
        )
        self._entries.append(entry)
        # ANN インデックスに追加 (Sprint 33)
        vec = _text_to_vector(entry.context + " " + entry.text, self._ann.dim)
        self._ann.add(entry.entry_id, vec)

        if len(self._entries) > self.max_size:
            self._evict()
        # IDF は一定件数追加ごとに再計算 (毎回は高コスト)
        if len(self._entries) % 50 == 0:
            self._rebuild_idf()
        return entry

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.0,
        category_filter: Optional[str] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        クエリに近いエントリを (entry, relevance) のリストで返す。

        Sprint 33 ハイブリッド検索:
            1. FAISS ANN で候補を top_k × 8 件に絞り込む (高速)
            2. TF-IDF コサインで精密スコアリング → priority 降順
        category_filter 指定時または線形バックエンド時は従来どおり全件 TF-IDF。
        """
        candidates = [e for e in self._entries
                      if category_filter is None or e.category == category_filter]
        if not candidates:
            return []

        # FAISS 候補絞り込み (category_filter なし かつ FAISS 使用 かつ十分な件数)
        if (
            self._ann.is_faiss
            and category_filter is None
            and len(candidates) > top_k * 2
        ):
            qvec      = _text_to_vector(query, self._ann.dim)
            ann_k     = min(top_k * 8, len(candidates))
            ann_hits  = self._ann.search(qvec, top_k=ann_k)
            cand_ids  = {eid for eid, _ in ann_hits}
            filtered  = [e for e in candidates if e.entry_id in cand_ids]
            if filtered:  # ANN 結果が空でなければ候補を絞る
                candidates = filtered

        # TF-IDF 精密スコアリング
        idf    = self._idf if self._idf else None
        scored: List[Tuple[MemoryEntry, float]] = []
        for e in candidates:
            rel = _cosine_tfidf(query, e.context + " " + e.text, idf=idf)
            if rel >= min_relevance:
                scored.append((e, rel))

        scored.sort(key=lambda x: x[0].priority(x[1]), reverse=True)
        result = scored[:top_k]
        for e, _ in result:
            e.access_count += 1
        return result

    # ------------------------------------------------------------------ evict

    def _evict(self) -> None:
        """score × freshness が最も低いエントリを削除し、ANN インデックスを再構築。"""
        self._entries.sort(key=lambda e: e.score * e.freshness, reverse=True)
        self._entries = self._entries[: self.max_size]
        self._rebuild_ann()  # Sprint 33: 削除後に ANN を再構築

    # ------------------------------------------------------------------ ANN rebuild (Sprint 33)

    def _rebuild_ann(self) -> None:
        """全エントリから ANN インデックスを再構築する。"""
        import numpy as np
        entry_ids = [e.entry_id for e in self._entries]
        if not entry_ids:
            self._ann.clear()
            return
        vectors = np.stack([
            _text_to_vector(e.context + " " + e.text, self._ann.dim)
            for e in self._entries
        ])
        self._ann.rebuild(entry_ids, vectors)

    # ------------------------------------------------------------------ stats

    def stats(self) -> Dict[str, float]:
        if not self._entries:
            return {"count": 0, "avg_score": 0.0, "avg_freshness": 0.0,
                    "ann_backend": self._ann.backend, "ann_size": 0}
        scores    = [e.score for e in self._entries]
        freshness = [e.freshness for e in self._entries]
        return {
            "count":        len(self._entries),
            "avg_score":    sum(scores)    / len(scores),
            "avg_freshness": sum(freshness) / len(freshness),
            "ann_backend":  self._ann.backend,   # Sprint 33
            "ann_size":     self._ann.size,       # Sprint 33
        }

    @property
    def entries(self) -> List[MemoryEntry]:
        return list(self._entries)

    @property
    def ann(self) -> ANNIndex:
        """ANN インデックスへの参照 (Sprint 33)"""
        return self._ann


# ---------------------------------------------------------------------------
# SemanticStore
# ---------------------------------------------------------------------------


class SemanticStore:
    """
    キーワードインデックス付き知識ストア。

    key → MemoryEntry の辞書をベースに、タグと本文テキストで
    TF-IDF 検索を行う。

    Parameters
    ----------
    max_size : 最大エントリ数
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._store: Dict[str, MemoryEntry] = {}
        self._tag_index: Dict[str, List[str]] = {}  # tag → [key]

    def store(
        self,
        key: str,
        content: str,
        tags: Optional[List[str]] = None,
        score: float = 1.0,
    ) -> MemoryEntry:
        """キー付きファクトを保存。同じ key は上書き。"""
        if len(self._store) >= self.max_size and key not in self._store:
            # 最も古いエントリを削除
            oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
            self._remove(oldest_key)

        entry = MemoryEntry(
            text=content,
            context=key,
            score=score,
            category="knowledge",
            tags=tags or [],
        )
        self._store[key] = entry
        for tag in (tags or []):
            self._tag_index.setdefault(tag, [])
            if key not in self._tag_index[tag]:
                self._tag_index[tag].append(key)
        return entry

    def search(
        self,
        query: str,
        top_k: int = 5,
        tags: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """タグフィルタ + TF-IDF 類似度で知識を検索。"""
        if tags:
            candidate_keys: set = set()
            for tag in tags:
                for k in self._tag_index.get(tag, []):
                    candidate_keys.add(k)
            candidates = [self._store[k] for k in candidate_keys if k in self._store]
        else:
            candidates = list(self._store.values())
        if not candidates:
            return []
        scored = []
        for e in candidates:
            rel = _cosine_tfidf(query, e.context + " " + e.text)
            scored.append((e, rel))
        scored.sort(key=lambda x: x[0].priority(x[1]), reverse=True)
        result = scored[:top_k]
        for e, _ in result:
            e.access_count += 1
        return result

    def _remove(self, key: str) -> None:
        if key in self._store:
            entry = self._store.pop(key)
            for tag in entry.tags:
                if tag in self._tag_index:
                    self._tag_index[tag] = [k for k in self._tag_index[tag] if k != key]

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> Dict[str, float]:
        entries = list(self._store.values())
        if not entries:
            return {"count": 0, "avg_score": 0.0}
        return {
            "count": len(entries),
            "avg_score": sum(e.score for e in entries) / len(entries),
        }


# ---------------------------------------------------------------------------
# LongTermMemoryAgent
# ---------------------------------------------------------------------------


class LongTermMemoryAgent:
    """
    エピソード記憶とセマンティック記憶を統合管理するエージェント。

    P7パターンの中核: 高品質な過去経験を蓄積し、現在のタスクへの
    文脈注入によって応答精度を継続的に向上させる。

    Parameters
    ----------
    score_threshold  : エピソードストアの品質フィルタ閾値
    max_episodes     : エピソードストア最大サイズ
    max_knowledge    : セマンティックストア最大サイズ
    dedup_threshold  : Jaccard 重複除去閾値
    """

    def __init__(
        self,
        score_threshold: float = 0.6,
        max_episodes: int = 500,
        max_knowledge: int = 1000,
        dedup_threshold: float = 0.85,
        ann_backend: str = "auto",          # Sprint 33: "auto"|"faiss"|"linear"
    ) -> None:
        self.episodes = EpisodicStore(
            max_size=max_episodes,
            score_threshold=score_threshold,
            dedup_threshold=dedup_threshold,
            ann_backend=ann_backend,
        )
        self.knowledge = SemanticStore(max_size=max_knowledge)
        self._score_threshold = score_threshold

    # ------------------------------------------------------------------ store

    def store_episode(
        self,
        context: str,
        response: str,
        score: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> Optional[MemoryEntry]:
        """
        対話エピソードを保存する。

        Args:
            context : 発火元クエリ
            response: 生成したレスポンス (高品質なもの)
            score   : LLMO スコア等の品質指標
            tags    : 検索用タグ
        """
        return self.episodes.append(context, response, score=score, tags=tags)

    def store_knowledge(
        self,
        key: str,
        content: str,
        tags: Optional[List[str]] = None,
        score: float = 1.0,
    ) -> MemoryEntry:
        """
        ファクト・ルールを知識ストアへ保存する。

        Args:
            key    : 知識の一意キー
            content: 知識本文
            tags   : 検索用タグ
            score  : 品質スコア
        """
        return self.knowledge.store(key, content, tags=tags, score=score)

    def store_conversation_memory(
        self,
        context: str,
        response: str,
        score: float = 1.0,
        evaluate_fn: Optional[Callable[[str], float]] = None,
    ) -> Optional[MemoryEntry]:
        """
        ConversationMemory のターンを受け取りエピソードとして保存。

        evaluate_fn が指定された場合、response を評価して score を上書き。
        """
        if evaluate_fn is not None:
            score = evaluate_fn(response)
        return self.store_episode(context, response, score=score)

    # ----------------------------------------------------------------- retrieve

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.0,
        include_knowledge: bool = True,
        tags: Optional[List[str]] = None,
    ) -> MemoryRetrieval:
        """
        クエリに関連する記憶を統合検索する。

        エピソードとセマンティック記憶の両方を検索し、
        priority (relevance + score + freshness) で統合ランキングする。

        Args:
            query          : 検索クエリ
            top_k          : 返却エントリ数上限
            min_relevance  : 最低 relevance 閾値
            include_knowledge: セマンティックストアを含めるか
            tags           : セマンティック検索用タグフィルタ

        Returns:
            MemoryRetrieval
        """
        total_searched = len(self.episodes.entries)
        all_results: List[Tuple[MemoryEntry, float]] = []

        # エピソード検索
        ep_results = self.episodes.search(query, top_k=top_k * 2, min_relevance=min_relevance)
        all_results.extend(ep_results)

        # セマンティック検索
        if include_knowledge:
            km_results = self.knowledge.search(query, top_k=top_k * 2, tags=tags)
            all_results.extend(km_results)
            total_searched += self.knowledge.size

        # 統合ソート
        all_results.sort(key=lambda x: x[0].priority(x[1]), reverse=True)
        top = all_results[:top_k]

        return MemoryRetrieval(
            query=query,
            entries=[e for e, _ in top],
            relevance_scores=[r for _, r in top],
            total_searched=total_searched,
        )

    # ---------------------------------------------------------------- consolidate

    def consolidate(self) -> Dict[str, int]:
        """
        記憶を整理する:
        - Jaccard ≥ 0.9 の近似重複を削除 (高スコアを保持)
        - freshness × score が下位 10% のエピソードを削除

        Returns:
            {"removed_duplicates": N, "removed_stale": M}
        """
        entries = list(self.episodes._entries)
        if len(entries) < 2:
            return {"removed_duplicates": 0, "removed_stale": 0}

        # 重複削除 (O(n²) だが max_size ≤ 500 で許容)
        keep_flags = [True] * len(entries)
        for i in range(len(entries)):
            if not keep_flags[i]:
                continue
            for j in range(i + 1, len(entries)):
                if not keep_flags[j]:
                    continue
                if _jaccard(entries[i].text, entries[j].text) >= 0.9:
                    if entries[i].score >= entries[j].score:
                        keep_flags[j] = False
                    else:
                        keep_flags[i] = False
                        break

        removed_dups = sum(1 for f in keep_flags if not f)
        self.episodes._entries = [e for e, keep in zip(entries, keep_flags) if keep]

        # 鮮度×スコアで下位 10% 除去
        if len(self.episodes._entries) > 20:
            thresh_idx = max(1, int(len(self.episodes._entries) * 0.9))
            self.episodes._entries.sort(
                key=lambda e: e.score * e.freshness, reverse=True
            )
            removed_stale = len(self.episodes._entries) - thresh_idx
            self.episodes._entries = self.episodes._entries[:thresh_idx]
        else:
            removed_stale = 0

        # ANN インデックスを再構築 (Sprint 33: consolidate 後のエントリ変化に追従)
        self.episodes._rebuild_ann()

        return {"removed_duplicates": removed_dups, "removed_stale": removed_stale}

    # ---------------------------------------------------------------- stats

    def stats(self) -> Dict[str, object]:
        """記憶ストアの統計情報。"""
        ep = self.episodes.stats()
        km = self.knowledge.stats()
        return {
            "episode_count": ep["count"],
            "episode_avg_score": ep.get("avg_score", 0.0),
            "episode_avg_freshness": ep.get("avg_freshness", 0.0),
            "knowledge_count": km["count"],
            "knowledge_avg_score": km.get("avg_score", 0.0),
            "total_entries": int(ep["count"]) + int(km["count"]),
        }

"""
LongTermMemoryAgent — 長期記憶統合 (Sprint 26 / P7パターン).

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
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# TF-IDF ユーティリティ (stdlib only)
# ---------------------------------------------------------------------------


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
    """

    text: str
    context: str = ""
    score: float = 1.0
    category: str = "episode"
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    access_count: int = 0

    @property
    def freshness(self) -> float:
        """経過時間に基づく鮮度 (1h=1.0, 24h≈0.7, 7d≈0.3)。"""
        age_h = (time.time() - self.created_at) / 3600.0
        return math.exp(-0.04 * age_h)  # half-life ≈ 17h

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
    ) -> None:
        self.max_size = max_size
        self.score_threshold = score_threshold
        self.dedup_threshold = dedup_threshold
        self._entries: List[MemoryEntry] = []

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
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            self._evict()
        return entry

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.0,
        category_filter: Optional[str] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """クエリに近いエントリを (entry, relevance) のリストで返す。"""
        candidates = [e for e in self._entries
                      if category_filter is None or e.category == category_filter]
        if not candidates:
            return []
        scored: List[Tuple[MemoryEntry, float]] = []
        for e in candidates:
            rel = _cosine_tfidf(query, e.context + " " + e.text)
            if rel >= min_relevance:
                scored.append((e, rel))
        # priority (relevance + score + freshness) で降順ソート
        scored.sort(key=lambda x: x[0].priority(x[1]), reverse=True)
        result = scored[:top_k]
        for e, _ in result:
            e.access_count += 1
        return result

    # ------------------------------------------------------------------ evict

    def _evict(self) -> None:
        """score × freshness が最も低いエントリを削除。"""
        self._entries.sort(key=lambda e: e.score * e.freshness, reverse=True)
        self._entries = self._entries[: self.max_size]

    # ------------------------------------------------------------------ stats

    def stats(self) -> Dict[str, float]:
        if not self._entries:
            return {"count": 0, "avg_score": 0.0, "avg_freshness": 0.0}
        scores = [e.score for e in self._entries]
        freshness = [e.freshness for e in self._entries]
        return {
            "count": len(self._entries),
            "avg_score": sum(scores) / len(scores),
            "avg_freshness": sum(freshness) / len(freshness),
        }

    @property
    def entries(self) -> List[MemoryEntry]:
        return list(self._entries)


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
    ) -> None:
        self.episodes = EpisodicStore(
            max_size=max_episodes,
            score_threshold=score_threshold,
            dedup_threshold=dedup_threshold,
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

"""
llmo_feedback.py — Living LLMO フィードバックループ (P1〜P4)

P1: Edit Delta Learning  — ユーザー編集差分 → 変換重み更新
P2: Acceptance Learning  — 採用された変換 → ランキング昇格
P3: Rating Calibration   — ユーザー評価とスコアのずれを補正
P4: Cross-Document Learning — 同ユーザー複数文書から共通傾向を抽出
"""

from __future__ import annotations

import difflib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# FeedbackEntry — フィードバック1件のデータ構造
# ---------------------------------------------------------------------------


@dataclass
class FeedbackEntry:
    """フィードバック1エントリ。FeedbackStore に保存される単位。"""

    entry_id: str
    user_id: str
    original_text: str
    revised_text: str  # ユーザーが編集した後のテキスト
    accepted: bool  # P2: ユーザーが最適化結果を採用したか
    rating: int  # P3: ユーザー満足度 1〜5
    domain: str  # P13連携: "marketing" / "tech" / "medical" / "general" 等
    intent_type: str  # "informational" / "transactional" / etc.
    original_score: float  # 最適化前の llmo_total スコア
    revised_score: float  # 最適化後(またはユーザー編集後)のスコア
    transformations_applied: list[str]  # 適用された変換種別
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "user_id": self.user_id,
            "original_text": self.original_text,
            "revised_text": self.revised_text,
            "accepted": self.accepted,
            "rating": self.rating,
            "domain": self.domain,
            "intent_type": self.intent_type,
            "original_score": self.original_score,
            "revised_score": self.revised_score,
            "transformations_applied": self.transformations_applied,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeedbackEntry":
        return cls(
            entry_id=d["entry_id"],
            user_id=d["user_id"],
            original_text=d["original_text"],
            revised_text=d["revised_text"],
            accepted=d["accepted"],
            rating=d["rating"],
            domain=d["domain"],
            intent_type=d["intent_type"],
            original_score=d["original_score"],
            revised_score=d["revised_score"],
            transformations_applied=d["transformations_applied"],
            timestamp=d.get("timestamp", 0.0),
        )


# ---------------------------------------------------------------------------
# FeedbackStore — フィードバック永続化ストア
# ---------------------------------------------------------------------------


class FeedbackStore:
    """
    FィードバックエントリをJSON ファイルに永続化するストア。

    Usage::

        store = FeedbackStore()
        entry = FeedbackEntry(entry_id=str(uuid.uuid4()), ...)
        store.add(entry)
        all_entries = store.get_all()
    """

    def __init__(self, path: str | Path = "data/llmo_feedback.json") -> None:
        self._path = Path(path)
        self._entries: list[FeedbackEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # 基本操作
    # ------------------------------------------------------------------

    def add(self, entry: FeedbackEntry) -> None:
        """エントリを追加してファイルに保存する。"""
        self._entries.append(entry)
        self._save()

    def add_raw(
        self,
        *,
        user_id: str = "anonymous",
        original_text: str,
        revised_text: str,
        accepted: bool = True,
        rating: int = 3,
        domain: str = "general",
        intent_type: str = "informational",
        original_score: float = 0.0,
        revised_score: float = 0.0,
        transformations_applied: list[str] | None = None,
    ) -> FeedbackEntry:
        """
        キーワード引数からエントリを生成して追加する。
        entry_id と timestamp は自動生成される。
        """
        entry = FeedbackEntry(
            entry_id=str(uuid.uuid4()),
            user_id=user_id,
            original_text=original_text,
            revised_text=revised_text,
            accepted=accepted,
            rating=rating,
            domain=domain,
            intent_type=intent_type,
            original_score=original_score,
            revised_score=revised_score,
            transformations_applied=transformations_applied or [],
        )
        self.add(entry)
        return entry

    def get_all(self) -> list[FeedbackEntry]:
        """全エントリを返す。"""
        return list(self._entries)

    def get_by_user(self, user_id: str) -> list[FeedbackEntry]:
        """特定ユーザーのエントリを返す。"""
        return [e for e in self._entries if e.user_id == user_id]

    def get_by_domain(self, domain: str) -> list[FeedbackEntry]:
        """特定ドメインのエントリを返す。"""
        return [e for e in self._entries if e.domain == domain]

    def get_accepted(self) -> list[FeedbackEntry]:
        """採用されたエントリのみを返す。"""
        return [e for e in self._entries if e.accepted]

    def get_rejected(self) -> list[FeedbackEntry]:
        """却下されたエントリのみを返す。"""
        return [e for e in self._entries if not e.accepted]

    def count(self) -> int:
        """総エントリ数を返す。"""
        return len(self._entries)

    def clear(self) -> None:
        """全エントリを削除する（テスト用）。"""
        self._entries = []
        self._save()

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """ファイルに保存する。ディレクトリがなければ作成する。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(
                [e.to_dict() for e in self._entries], f, ensure_ascii=False, indent=2
            )

    def _load(self) -> None:
        """ファイルから読み込む。ファイルがなければ空で開始する。"""
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    raw = json.load(f)
                self._entries = [FeedbackEntry.from_dict(d) for d in raw]
            except (json.JSONDecodeError, KeyError):
                self._entries = []
        else:
            self._entries = []


# ---------------------------------------------------------------------------
# FeedbackAnalyzer — P1〜P4 の分析エンジン
# ---------------------------------------------------------------------------


class FeedbackAnalyzer:
    """
    FeedbackStore に蓄積されたエントリを分析し、
    各成長パターン向けのシグナルを算出する。
    """

    # ------------------------------------------------------------------
    # P1: Edit Delta Learning — 編集差分から axis 別改善量を推定
    # ------------------------------------------------------------------

    @staticmethod
    def extract_delta(original: str, revised: str) -> dict[str, float]:
        """
        original → revised の編集差分から
        entity_density / answer_directness / citability の
        推定改善量 (delta) を返す。

        Returns::

            {
                "entity_density": 0.05,
                "answer_directness": 0.03,
                "citability": 0.01,
                "net_delta": 0.09,   # スコア差 revised-original
                "added_chars": 120,
                "removed_chars": 30,
            }
        """
        if not original and not revised:
            return {
                "entity_density": 0.0,
                "answer_directness": 0.0,
                "citability": 0.0,
                "net_delta": 0.0,
                "added_chars": 0,
                "removed_chars": 0,
            }

        # difflib で行単位の差分を取得
        orig_lines = original.splitlines(keepends=True)
        rev_lines = revised.splitlines(keepends=True)
        diff = list(difflib.unified_diff(orig_lines, rev_lines, n=0))

        added = "".join(
            line[1:]
            for line in diff
            if line.startswith("+") and not line.startswith("+++")
        )
        removed = "".join(
            line[1:]
            for line in diff
            if line.startswith("-") and not line.startswith("---")
        )

        added_chars = len(added)
        removed_chars = len(removed)

        # 追加テキストから axis 別シグナルを推定
        # entity_density: 固有名詞っぽい大文字語・数字・専門語の密度
        cap_words = sum(
            1 for w in added.split() if w and (w[0].isupper() or w[0].isdigit())
        )
        entity_signal = min(cap_words / max(len(added.split()), 1) * 0.5, 0.15)

        # answer_directness: 文頭に結論を置く編集か（先頭行に追加があるか）
        first_line_added = any(
            line.startswith("+") and not line.startswith("+++")
            for line in diff[:5]  # 最初の5行の差分
        )
        directness_signal = 0.05 if first_line_added else 0.0

        # citability: 参照・出典・数値が追加されたか
        cit_keywords = (
            "according",
            "source",
            "research",
            "study",
            "ref",
            "%",
            "http",
            "www",
        )
        cit_hits = sum(1 for kw in cit_keywords if kw in added.lower())
        citability_signal = min(cit_hits * 0.02, 0.10)

        net = entity_signal + directness_signal + citability_signal

        return {
            "entity_density": round(entity_signal, 4),
            "answer_directness": round(directness_signal, 4),
            "citability": round(citability_signal, 4),
            "net_delta": round(net, 4),
            "added_chars": added_chars,
            "removed_chars": removed_chars,
        }

    # ------------------------------------------------------------------
    # P2: Acceptance Learning — 変換別採用率を算出
    # ------------------------------------------------------------------

    @staticmethod
    def compute_acceptance_rate(store: FeedbackStore, transformation: str) -> float:
        """
        指定変換が適用されたエントリのうち採用されたものの割合を返す。
        該当エントリがなければ 0.5（中立）を返す。
        """
        relevant = [
            e for e in store.get_all() if transformation in e.transformations_applied
        ]
        if not relevant:
            return 0.5
        accepted = sum(1 for e in relevant if e.accepted)
        return accepted / len(relevant)

    # ------------------------------------------------------------------
    # P3: Rating Calibration — 予測スコアとユーザー評価のずれを算出
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rating_bias(store: FeedbackStore) -> float:
        """
        全エントリについて
        (ユーザー評価を 0〜1 に正規化した値) - (revised_score) の
        平均乖離を返す。

        正の値 → システムが過小評価している
        負の値 → システムが過大評価している
        """
        entries = store.get_all()
        if not entries:
            return 0.0
        biases = [(e.rating / 5.0) - e.revised_score for e in entries]
        return round(sum(biases) / len(biases), 4)

    # ------------------------------------------------------------------
    # P4: Cross-Document Learning — ユーザー別の傾向プロファイルを生成
    # ------------------------------------------------------------------

    @staticmethod
    def extract_user_profile(store: FeedbackStore, user_id: str) -> dict[str, Any]:
        """
        同一ユーザーの複数エントリから共通の編集傾向を抽出する。

        Returns::

            {
                "user_id": "user123",
                "entry_count": 5,
                "preferred_domain": "marketing",
                "preferred_intent": "informational",
                "avg_rating": 4.2,
                "preferred_transformations": ["add_structure", "citation_cues"],
                "avg_score_improvement": 0.08,
            }
        """
        entries = store.get_by_user(user_id)
        if not entries:
            return {"user_id": user_id, "entry_count": 0}

        # 最頻ドメイン・意図
        domains = [e.domain for e in entries]
        intents = [e.intent_type for e in entries]
        preferred_domain = max(set(domains), key=domains.count)
        preferred_intent = max(set(intents), key=intents.count)

        # 採用済みエントリで使われた変換の頻度
        accepted_entries = [e for e in entries if e.accepted]
        transformation_counts: dict[str, int] = {}
        for e in accepted_entries:
            for t in e.transformations_applied:
                transformation_counts[t] = transformation_counts.get(t, 0) + 1
        preferred_transformations = sorted(
            transformation_counts, key=lambda k: transformation_counts[k], reverse=True
        )[:3]

        avg_rating = sum(e.rating for e in entries) / len(entries)
        avg_improvement = sum(
            e.revised_score - e.original_score for e in entries
        ) / len(entries)

        return {
            "user_id": user_id,
            "entry_count": len(entries),
            "preferred_domain": preferred_domain,
            "preferred_intent": preferred_intent,
            "avg_rating": round(avg_rating, 2),
            "preferred_transformations": preferred_transformations,
            "avg_score_improvement": round(avg_improvement, 4),
        }

    # ------------------------------------------------------------------
    # 統合: 全フィードバックから重み更新シグナルを集約
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_weight_signals(store: FeedbackStore) -> dict[str, float]:
        """
        全変換についての採用率を辞書で返す。
        LLMOScorer.adapt_weights() に渡すための集約データ。
        """
        all_transformations = {
            "add_structure",
            "boost_entity_density",
            "add_citation_cues",
            "expand_content",
            "inject_query_keyword",
            "rewrite_for_answer_first",
        }
        return {
            t: FeedbackAnalyzer.compute_acceptance_rate(store, t)
            for t in all_transformations
        }

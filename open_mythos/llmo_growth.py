"""
llmo_growth.py — Living LLMO 成功/失敗学習 (P5〜P12)

P5:  Rejection Memory       — 却下された変換を NG パターンとして記録
P6:  Failure Memory         — スコアが下がった変換をペナルティ付きで記録
P7:  Loop Escape Memory     — optimize() が収束しなかったパターンを記録
P8:  Anti-Pattern Registry  — 繰り返し失敗する変換シーケンスを禁止リスト化
P9:  Pattern Mining         — 成功変換シーケンスの頻度×効果量ランキング
P10: Entity Vocabulary Growth — 高スコアコンテンツから entity 辞書を自動拡張
P11: Template Crystallization — 高スコアを生む文章構造をテンプレート化
P12: Champion Promotion     — 最高スコア改善の変換組み合わせをチャンピオン固定
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------


def _text_hash(text: str) -> str:
    """テキストの先頭200文字のハッシュを返す（入力パターンの識別に使用）。"""
    return hashlib.md5(text[:200].encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# P5: Rejection Memory — 却下された変換を記録
# ---------------------------------------------------------------------------


class RejectionMemory:
    """
    ユーザーが「却下」した変換結果を記録する。
    同じ入力パターンで同じ変換を再適用しないために使用する。

    Usage::

        mem = RejectionMemory()
        mem.record_rejection("abc12345", "add_structure")
        assert mem.is_rejected("abc12345", "add_structure")
    """

    def __init__(self) -> None:
        # key: "{text_hash}:{transformation}" → 却下回数
        self._rejected: dict[str, int] = defaultdict(int)

    def record_rejection(self, text_hash: str, transformation: str) -> None:
        """却下を記録する。"""
        key = f"{text_hash}:{transformation}"
        self._rejected[key] += 1

    def is_rejected(self, text_hash: str, transformation: str) -> bool:
        """この入力×変換の組み合わせが過去に却下されていれば True。"""
        key = f"{text_hash}:{transformation}"
        return self._rejected[key] > 0

    def get_rejection_count(self, transformation: str) -> int:
        """指定変換の総却下回数を返す（全入力パターンの合計）。"""
        return sum(
            v for k, v in self._rejected.items() if k.endswith(f":{transformation}")
        )

    def total_rejections(self) -> int:
        """記録された総却下件数を返す。"""
        return sum(self._rejected.values())

    def to_dict(self) -> dict[str, Any]:
        return {"rejected": dict(self._rejected)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RejectionMemory":
        obj = cls()
        obj._rejected = defaultdict(int, d.get("rejected", {}))
        return obj


# ---------------------------------------------------------------------------
# P6: Failure Memory — スコアが下がった変換をペナルティ付きで記録
# ---------------------------------------------------------------------------


class FailureMemory:
    """
    最適化後にスコアが下がった（逆効果だった）変換を記録し、
    次回同じ入力に対して適用ペナルティを付与する。

    Usage::

        mem = FailureMemory()
        mem.record_failure("abc12345", "expand_content", score_delta=-0.05)
        pen = mem.get_penalty("expand_content")
    """

    def __init__(self) -> None:
        # key: transformation → 累積 penalty (正の値 → 回避すべき度合い)
        self._penalties: dict[str, float] = defaultdict(float)
        # key: "{text_hash}:{transformation}" → 失敗フラグ
        self._failures: dict[str, bool] = {}

    def record_failure(
        self, text_hash: str, transformation: str, score_delta: float
    ) -> None:
        """
        score_delta が負のとき（スコアが下がった）に記録する。
        ペナルティは失敗の絶対量だけ加算する。
        """
        if score_delta < 0:
            self._failures[f"{text_hash}:{transformation}"] = True
            self._penalties[transformation] += abs(score_delta)

    def should_skip(self, text_hash: str, transformation: str) -> bool:
        """この入力×変換が過去に失敗していれば True。"""
        return self._failures.get(f"{text_hash}:{transformation}", False)

    def get_penalty(self, transformation: str) -> float:
        """変換の累積ペナルティを返す（0.0 = ペナルティなし）。"""
        return round(self._penalties.get(transformation, 0.0), 4)

    def total_failures(self) -> int:
        """失敗として記録された (入力, 変換) ペアの総数を返す。"""
        return len(self._failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "penalties": dict(self._penalties),
            "failures": dict(self._failures),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FailureMemory":
        obj = cls()
        obj._penalties = defaultdict(float, d.get("penalties", {}))
        obj._failures = d.get("failures", {})
        return obj


# ---------------------------------------------------------------------------
# P7: Loop Escape Memory — optimize() が収束しなかったパターンを記録
# ---------------------------------------------------------------------------

# 収束失敗時に代わりに使う戦略マップ
_ESCAPE_STRATEGY: dict[str, str] = {
    "default": "rewrite_for_answer_first",
    "rewrite_for_answer_first": "add_structure",
    "add_structure": "boost_entity_density",
    "boost_entity_density": "add_citation_cues",
    "add_citation_cues": "expand_content",
}


class LoopEscapeMemory:
    """
    optimize() が max_iterations を超えても収束しなかった入力を記録し、
    次回同じ入力に対して別の戦略を先行して使用するよう誘導する。

    Usage::

        mem = LoopEscapeMemory()
        mem.record_no_convergence("abc12345", strategy_used="add_structure")
        alt = mem.get_alternative_strategy("abc12345")
    """

    def __init__(self) -> None:
        # text_hash → 最後に失敗した戦略
        self._escape_log: dict[str, str] = {}

    def record_no_convergence(self, text_hash: str, strategy_used: str) -> None:
        """収束失敗を記録する。"""
        self._escape_log[text_hash] = strategy_used

    def get_alternative_strategy(self, text_hash: str) -> str | None:
        """
        前回失敗した戦略の次の候補を返す。
        記録がなければ None を返す。
        """
        last = self._escape_log.get(text_hash)
        if last is None:
            return None
        return _ESCAPE_STRATEGY.get(last, "rewrite_for_answer_first")

    def has_escape_record(self, text_hash: str) -> bool:
        """この入力が収束失敗の記録を持つかどうかを返す。"""
        return text_hash in self._escape_log

    def total_escapes(self) -> int:
        """記録された収束失敗の総数を返す。"""
        return len(self._escape_log)

    def to_dict(self) -> dict[str, Any]:
        return {"escape_log": self._escape_log}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoopEscapeMemory":
        obj = cls()
        obj._escape_log = d.get("escape_log", {})
        return obj


# ---------------------------------------------------------------------------
# P8: Anti-Pattern Registry — 繰り返し失敗するシーケンスを禁止
# ---------------------------------------------------------------------------


class AntiPatternRegistry:
    """
    2連続以上で失敗しやすい変換シーケンスを禁止リストとして管理する。
    例: ("expand_content", "add_structure") は相性が悪い、など。

    Usage::

        reg = AntiPatternRegistry()
        reg.register_bad_sequence(["expand_content", "add_structure"])
        assert reg.is_forbidden_sequence(["expand_content", "add_structure"])
    """

    def __init__(self) -> None:
        # 禁止されている (t1, t2) ペアのセット
        self._forbidden_pairs: set[tuple[str, str]] = set()

    def register_bad_sequence(self, seq: list[str]) -> None:
        """
        シーケンス内の連続する2要素をすべて禁止ペアとして登録する。
        例: ["a", "b", "c"] → ("a","b"), ("b","c") が禁止される。
        """
        for i in range(len(seq) - 1):
            self._forbidden_pairs.add((seq[i], seq[i + 1]))

    def is_forbidden_sequence(self, seq: list[str]) -> bool:
        """シーケンス内に禁止ペアが含まれれば True を返す。"""
        for i in range(len(seq) - 1):
            if (seq[i], seq[i + 1]) in self._forbidden_pairs:
                return True
        return False

    def forbidden_next(self, transformation: str) -> set[str]:
        """指定変換の直後に使ってはいけない変換セットを返す。"""
        return {b for (a, b) in self._forbidden_pairs if a == transformation}

    def total_forbidden_pairs(self) -> int:
        """登録された禁止ペア数を返す。"""
        return len(self._forbidden_pairs)

    def to_dict(self) -> dict[str, Any]:
        return {"forbidden_pairs": [list(p) for p in self._forbidden_pairs]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AntiPatternRegistry":
        obj = cls()
        obj._forbidden_pairs = {tuple(p) for p in d.get("forbidden_pairs", [])}  # type: ignore[assignment]
        return obj


# ---------------------------------------------------------------------------
# P9: Pattern Mining — 成功変換シーケンスのランキング
# ---------------------------------------------------------------------------


@dataclass
class TransformationRecord:
    """変換1種別の成功・失敗統計。"""

    name: str
    success_count: int = 0
    failure_count: int = 0
    total_delta: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def avg_delta(self) -> float:
        total = self.success_count + self.failure_count
        return self.total_delta / total if total > 0 else 0.0

    @property
    def score(self) -> float:
        """頻度 × 効果量の複合スコア（ランキング用）。"""
        freq = math.log1p(self.success_count)
        return round(freq * self.avg_delta * self.success_rate, 6)


class PatternMiner:
    """
    成功した変換シーケンスを記録・集約し、
    頻度 × 効果量 × 成功率でランキングを自動更新する。(P9)

    Usage::

        miner = PatternMiner()
        miner.record_success(["add_structure", "citation_cues"], delta=0.12)
        top = miner.top_k(k=3)
    """

    def __init__(self) -> None:
        self._records: dict[str, TransformationRecord] = {}

    def record_success(self, transformations: list[str], delta: float) -> None:
        """成功した変換シーケンスを記録する。delta は total スコア増分。"""
        for t in transformations:
            if t not in self._records:
                self._records[t] = TransformationRecord(name=t)
            rec = self._records[t]
            rec.success_count += 1
            rec.total_delta += max(delta, 0.0)

    def record_failure(self, transformations: list[str]) -> None:
        """失敗した変換（スコアが下がった）を記録する。"""
        for t in transformations:
            if t not in self._records:
                self._records[t] = TransformationRecord(name=t)
            self._records[t].failure_count += 1

    def get_ranking(self) -> list[tuple[str, float]]:
        """(変換名, スコア) のリストをスコア降順で返す。"""
        return sorted(
            [(name, rec.score) for name, rec in self._records.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    def top_k(self, k: int = 5) -> list[str]:
        """スコア上位 k 件の変換名リストを返す。"""
        return [name for name, _ in self.get_ranking()[:k]]

    def get_record(self, transformation: str) -> TransformationRecord | None:
        """変換名のレコードを返す。存在しなければ None。"""
        return self._records.get(transformation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": {
                name: {
                    "name": rec.name,
                    "success_count": rec.success_count,
                    "failure_count": rec.failure_count,
                    "total_delta": rec.total_delta,
                }
                for name, rec in self._records.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PatternMiner":
        obj = cls()
        for name, rd in d.get("records", {}).items():
            rec = TransformationRecord(
                name=name,
                success_count=rd["success_count"],
                failure_count=rd["failure_count"],
                total_delta=rd["total_delta"],
            )
            obj._records[name] = rec
        return obj


# ---------------------------------------------------------------------------
# P10: Entity Vocabulary Growth — entity 辞書の自動拡張
# ---------------------------------------------------------------------------

# entity 候補として除外する一般的なストップワード（英語）
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "not",
        "very",
        "also",
        "just",
        "more",
        "most",
        "some",
        "any",
        "all",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "it's",
        "i",
        "we",
        "you",
        "he",
        "she",
        "they",
        "them",
        "their",
        "our",
        "your",
        "my",
    }
)


def _extract_candidate_entities(text: str) -> list[str]:
    """テキストから entity 候補語（大文字開始・数値含む語）を抽出する。"""
    # 単語境界で分割
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]\b|\b[A-Z][a-z]+\b", text)
    candidates = []
    for w in words:
        lower = w.lower()
        if lower in _STOPWORDS:
            continue
        if len(w) < 3:
            continue
        candidates.append(lower)
    return candidates


class EntityKnowledgeBase:
    """
    高スコアコンテンツの TF-IDF 上位語を entity 辞書に自動追加する。(P10)

    Usage::

        kb = EntityKnowledgeBase()
        added = kb.ingest_high_score_content("Google launched AI...", score=0.82)
        entities = kb.get_entities(min_confidence=0.6)
    """

    def __init__(self) -> None:
        # word → 信頼度スコア (0〜1)
        self._entities: dict[str, float] = {}
        # word → 出現回数
        self._counts: dict[str, int] = defaultdict(int)
        self._total_ingested: int = 0

    def ingest_high_score_content(
        self, text: str, score: float, threshold: float = 0.70
    ) -> int:
        """
        スコアが threshold 以上のコンテンツから entity 候補を抽出して追加する。
        追加された語数を返す。
        """
        if score < threshold:
            return 0

        candidates = _extract_candidate_entities(text)
        if not candidates:
            return 0

        # TF (このテキスト内での頻度) を計算
        total_words = max(len(text.split()), 1)
        tf: dict[str, float] = defaultdict(float)
        for w in candidates:
            tf[w] += 1.0 / total_words

        added = 0
        for word, tf_score in tf.items():
            self._counts[word] += 1
            # 信頼度 = (現在の信頼度 × 既存回数 + score × tf) / (既存回数 + 1)
            # → 露出が増えるほど高くなる加重平均
            old_conf = self._entities.get(word, 0.0)
            old_count = self._counts[word] - 1
            new_conf = (old_conf * old_count + score * tf_score * 10) / self._counts[
                word
            ]
            new_conf = min(new_conf, 1.0)
            if word not in self._entities:
                added += 1
            self._entities[word] = round(new_conf, 4)

        self._total_ingested += 1
        return added

    def get_entities(self, min_confidence: float = 0.5) -> list[str]:
        """min_confidence 以上の entity 語を信頼度降順で返す。"""
        filtered = [(w, c) for w, c in self._entities.items() if c >= min_confidence]
        return [w for w, _ in sorted(filtered, key=lambda x: x[1], reverse=True)]

    def size(self) -> int:
        """登録済み entity 総数を返す。"""
        return len(self._entities)

    def get_confidence(self, word: str) -> float:
        """指定語の信頼度を返す。未登録なら 0.0。"""
        return self._entities.get(word.lower(), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": self._entities,
            "counts": dict(self._counts),
            "total_ingested": self._total_ingested,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EntityKnowledgeBase":
        obj = cls()
        obj._entities = d.get("entities", {})
        obj._counts = defaultdict(int, d.get("counts", {}))
        obj._total_ingested = d.get("total_ingested", 0)
        return obj


# ---------------------------------------------------------------------------
# P11: Template Crystallization — 高スコアを生む文章構造のテンプレート化
# ---------------------------------------------------------------------------


@dataclass
class ContentTemplate:
    """結晶化されたコンテンツテンプレート。"""

    pattern: str  # 文章構造の抽象表現（例: "intro + bullet_list + conclusion"）
    avg_score: float
    count: int
    sample_text: str  # 代表テキストの冒頭 120 文字


class TemplateLibrary:
    """
    高スコアを繰り返し生む文章構造を抽出してテンプレートとして登録する。(P11)

    Usage::

        lib = TemplateLibrary()
        lib.crystallize("## Overview\\n- point1\\n- point2\\n## Summary", score=0.81)
        templates = lib.get_best_templates(k=3)
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self._threshold = threshold
        self._templates: list[ContentTemplate] = []

    def _extract_pattern(self, text: str) -> str:
        """テキストの構造パターンを抽象化して文字列で返す。"""
        parts = []
        if re.search(r"^#{1,3}\s", text, re.MULTILINE):
            parts.append("headings")
        if re.search(r"^\s*[-*]\s", text, re.MULTILINE):
            parts.append("bullets")
        if re.search(r"^\s*\d+\.\s", text, re.MULTILINE):
            parts.append("numbered_list")
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) >= 5:
            parts.append("multi_sentence")
        if len(text) > 500:
            parts.append("long_form")
        return "+".join(parts) if parts else "plain"

    def crystallize(self, text: str, score: float) -> bool:
        """
        スコアが threshold 以上ならテンプレートとして登録する。
        既存テンプレートと同じパターンなら avg_score を更新する。
        追加または更新されれば True を返す。
        """
        if score < self._threshold:
            return False

        pattern = self._extract_pattern(text)
        sample = text[:120].replace("\n", " ")

        # 既存パターンと一致するか確認
        for tmpl in self._templates:
            if tmpl.pattern == pattern:
                # 加重平均で avg_score を更新
                tmpl.avg_score = (tmpl.avg_score * tmpl.count + score) / (
                    tmpl.count + 1
                )
                tmpl.count += 1
                if score > tmpl.avg_score:
                    tmpl.sample_text = sample
                return True

        # 新規テンプレートとして追加
        self._templates.append(
            ContentTemplate(
                pattern=pattern, avg_score=round(score, 4), count=1, sample_text=sample
            )
        )
        return True

    def get_best_templates(self, k: int = 3) -> list[ContentTemplate]:
        """avg_score 降順で上位 k 件のテンプレートを返す。"""
        return sorted(self._templates, key=lambda t: t.avg_score, reverse=True)[:k]

    def size(self) -> int:
        """登録テンプレート数を返す。"""
        return len(self._templates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self._threshold,
            "templates": [
                {
                    "pattern": t.pattern,
                    "avg_score": t.avg_score,
                    "count": t.count,
                    "sample_text": t.sample_text,
                }
                for t in self._templates
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TemplateLibrary":
        obj = cls(threshold=d.get("threshold", 0.75))
        for td in d.get("templates", []):
            obj._templates.append(
                ContentTemplate(
                    pattern=td["pattern"],
                    avg_score=td["avg_score"],
                    count=td["count"],
                    sample_text=td["sample_text"],
                )
            )
        return obj


# ---------------------------------------------------------------------------
# P12: Champion Promotion — 最高スコア変換セットをチャンピオンとして固定
# ---------------------------------------------------------------------------


class ChampionPromoter:
    """
    最高スコア改善をもたらした変換シーケンスを「champion」として記録し、
    新しいケースで最初に試す候補として提供する。(P12)

    Usage::

        promoter = ChampionPromoter()
        promoter.promote(["add_structure", "citation_cues"], score_delta=0.20)
        champ = promoter.get_champion()
    """

    # champion の最大保持数
    MAX_CHAMPIONS = 5

    def __init__(self) -> None:
        # (score_delta, transformations) のリスト。スコア降順で管理。
        self._champions: list[tuple[float, list[str]]] = []

    def promote(self, transformations: list[str], score_delta: float) -> bool:
        """
        score_delta が現在の最低チャンピオンを上回れば champion として追加する。
        champion リストは MAX_CHAMPIONS 件を超えないよう末尾を切り捨てる。
        追加されれば True を返す。
        """
        if not transformations or score_delta <= 0:
            return False

        self._champions.append((score_delta, list(transformations)))
        self._champions.sort(key=lambda x: x[0], reverse=True)
        self._champions = self._champions[: self.MAX_CHAMPIONS]
        return True

    def get_champion(self) -> list[str] | None:
        """最高スコアの変換シーケンスを返す。なければ None。"""
        if not self._champions:
            return None
        return list(self._champions[0][1])

    def all_champions(self) -> list[list[str]]:
        """全 champion シーケンスをスコア降順で返す。"""
        return [list(seq) for _, seq in self._champions]

    def best_score_delta(self) -> float:
        """champion の最高スコア改善量を返す。なければ 0.0。"""
        return self._champions[0][0] if self._champions else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "champions": [
                {"score_delta": delta, "transformations": seq}
                for delta, seq in self._champions
            ]
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChampionPromoter":
        obj = cls()
        obj._champions = [
            (item["score_delta"], item["transformations"])
            for item in d.get("champions", [])
        ]
        return obj


# ---------------------------------------------------------------------------
# GrowthStore — P5〜P12 を一括管理するコンテナ
# ---------------------------------------------------------------------------


class GrowthStore:
    """
    失敗/成功学習の 8 パターン (P5〜P12) を一括保持するコンテナ。
    LLMOOptimizer や GrowthCycle から利用される。
    """

    def __init__(self) -> None:
        self.rejection_memory = RejectionMemory()
        self.failure_memory = FailureMemory()
        self.loop_escape_memory = LoopEscapeMemory()
        self.anti_pattern_registry = AntiPatternRegistry()
        self.pattern_miner = PatternMiner()
        self.entity_kb = EntityKnowledgeBase()
        self.template_library = TemplateLibrary()
        self.champion_promoter = ChampionPromoter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rejection_memory": self.rejection_memory.to_dict(),
            "failure_memory": self.failure_memory.to_dict(),
            "loop_escape_memory": self.loop_escape_memory.to_dict(),
            "anti_pattern_registry": self.anti_pattern_registry.to_dict(),
            "pattern_miner": self.pattern_miner.to_dict(),
            "entity_kb": self.entity_kb.to_dict(),
            "template_library": self.template_library.to_dict(),
            "champion_promoter": self.champion_promoter.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GrowthStore":
        obj = cls()
        obj.rejection_memory = RejectionMemory.from_dict(d.get("rejection_memory", {}))
        obj.failure_memory = FailureMemory.from_dict(d.get("failure_memory", {}))
        obj.loop_escape_memory = LoopEscapeMemory.from_dict(
            d.get("loop_escape_memory", {})
        )
        obj.anti_pattern_registry = AntiPatternRegistry.from_dict(
            d.get("anti_pattern_registry", {})
        )
        obj.pattern_miner = PatternMiner.from_dict(d.get("pattern_miner", {}))
        obj.entity_kb = EntityKnowledgeBase.from_dict(d.get("entity_kb", {}))
        obj.template_library = TemplateLibrary.from_dict(d.get("template_library", {}))
        obj.champion_promoter = ChampionPromoter.from_dict(
            d.get("champion_promoter", {})
        )
        return obj

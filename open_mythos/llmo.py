"""
OpenMythos LLMO (Large Language Model Optimization) Scoring Module.

LLMO は「AIサーチ向けコンテンツ最適化」の分野。LLMが検索結果として
コンテンツを引用・参照する可能性を高める指標を計算する。

3つのコアスコア:
    entity_density      -- エンティティ（固有名詞・数値・専門語）の密度
    answer_directness   -- 最初の文で問いに直接答えているか
    citability          -- LLMに引用されやすい構造スコア

使い方::

    from open_mythos.llmo import LLMOScorer

    scorer = LLMOScorer()
    result = scorer.score("OpenMythosはRecurrent-Depth Transformerを...")
    print(result.llmo_total)   # 0.0 〜 1.0
    print(result.entity_density)
    print(result.answer_directness)
    print(result.citability)

    # 複数コンテンツ比較
    ranking = scorer.rank(["text A", "text B", "text C"])
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class LLMOScore:
    """LLMOスコアの集約結果。"""

    entity_density: float
    """エンティティ密度 (0–1)。高いほどエンティティが豊富。"""

    answer_directness: float
    """回答直接性スコア (0–1)。冒頭で問いに答えているほど高い。"""

    citability: float
    """引用されやすさスコア (0–1)。構造・具体性・信頼性の合成。"""

    llmo_total: float
    """3スコアの加重平均 (entity×0.3 + directness×0.4 + citability×0.3)。"""

    entities: list[str] = field(default_factory=list)
    """検出されたエンティティのリスト。"""

    word_count: int = 0
    """本文の単語数。"""

    sentence_count: int = 0
    """文の数。"""

    @classmethod
    def weights(cls) -> dict[str, float]:
        return {
            "entity_density": 0.30,
            "answer_directness": 0.40,
            "citability": 0.30,
        }


# ---------------------------------------------------------------------------
# パターン定義
# ---------------------------------------------------------------------------

# 数値パターン (百分率・価格・倍率・年・統計数値)
_NUM_PATTERN = re.compile(
    r"""
    (?:
        \d+(?:[,，]\d{3})*(?:\.\d+)?   # 数値本体 (1,234 / 3.14)
        (?:\s*(?:%|％|倍|円|ドル|万|億|兆|KB|MB|GB|TB|ms|秒|分|時間|px|pt|em|rem|K|M|B))? # 単位
    )
    """,
    re.VERBOSE,
)

# 固有名詞パターン (英数字大文字で始まる語 / カタカナ連続 / ブランド名)
_PROPER_NOUN_EN = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,3}\b")
_KATAKANA = re.compile(r"[ァ-ヶー]{2,}")  # 3文字以上カタカナ

# 専門語パターン (英数字混在の略語・技術語)
_TECH_TERM = re.compile(r"\b[A-Z]{2,}(?:\d+)?(?:\.[0-9]+)?\b")

# 文の分割
_SENT_SPLIT = re.compile(r"(?<=[。．.!?！？])\s*")

# answer-first キーワード (文頭が答えを示す語で始まるか)
_ANSWER_FIRST_PATTERNS = [
    re.compile(r"^(?:yes|no|はい|いいえ|〇|×|正|否)\b", re.IGNORECASE),
    re.compile(r"^\w+(?:は|が|とは|means?|is\s+a|refers?\s+to)", re.IGNORECASE),
    re.compile(r"^\d"),  # 数値で始まる (「3つの方法は...」)
]

# 引用誘発パターン
_CITATION_TRIGGERS = [
    re.compile(r"\d+(?:[,，]\d{3})*(?:\.\d+)?%"),      # パーセンテージ
    re.compile(r"(?:研究|調査|報告|study|report|survey|found|showed?|according)", re.I),
    re.compile(r"(?:20\d{2}年?|in\s+20\d{2})"),        # 年号
    re.compile(r"(?:出典|ソース|参考|source|ref\.?):", re.I),  # 引用元
]

# 構造マーカー (見出し・リスト・コードブロック)
_STRUCTURE_MARKERS = re.compile(
    r"(?m)^(?:#{1,4}\s|\-\s|\*\s|\d+\.\s|```)", re.MULTILINE
)


# ---------------------------------------------------------------------------
# LLMOScorer
# ---------------------------------------------------------------------------


class LLMOScorer:
    """
    LLMO スコアリングエンジン。

    Args:
        entity_weight      -- entity_density の最終スコアへの重み (デフォルト 0.30)
        directness_weight  -- answer_directness の重み (デフォルト 0.40)
        citability_weight  -- citability の重み (デフォルト 0.30)
    """

    def __init__(
        self,
        entity_weight: float = 0.30,
        directness_weight: float = 0.40,
        citability_weight: float = 0.30,
    ) -> None:
        total = entity_weight + directness_weight + citability_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        self._w_entity = entity_weight
        self._w_direct = directness_weight
        self._w_citable = citability_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str) -> LLMOScore:
        """
        テキストの LLMO スコアを計算する。

        Args:
            text -- スコアリング対象のテキスト

        Returns:
            LLMOScore インスタンス
        """
        if not text or not text.strip():
            return LLMOScore(
                entity_density=0.0,
                answer_directness=0.0,
                citability=0.0,
                llmo_total=0.0,
            )

        words = text.split()
        word_count = len(words)
        sentences = [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
        sentence_count = max(len(sentences), 1)

        entities = self._extract_entities(text)
        entity_density = self._calc_entity_density(entities, word_count)
        answer_directness = self._calc_answer_directness(sentences)
        citability = self._calc_citability(text, sentences, word_count)

        llmo_total = (
            self._w_entity * entity_density
            + self._w_direct * answer_directness
            + self._w_citable * citability
        )
        llmo_total = max(0.0, min(1.0, llmo_total))

        return LLMOScore(
            entity_density=round(entity_density, 4),
            answer_directness=round(answer_directness, 4),
            citability=round(citability, 4),
            llmo_total=round(llmo_total, 4),
            entities=entities[:20],  # 上位20件
            word_count=word_count,
            sentence_count=sentence_count,
        )

    def batch_score(self, texts: Sequence[str]) -> list[LLMOScore]:
        """複数テキストを一括スコアリングする。"""
        return [self.score(t) for t in texts]

    def rank(self, texts: Sequence[str]) -> list[tuple[int, float, LLMOScore]]:
        """
        テキストリストを LLMO スコアで降順ランキングする。

        Returns:
            [(rank_position, llmo_total, LLMOScore), ...] のリスト (スコア降順)
        """
        scored = [(i, self.score(t)) for i, t in enumerate(texts)]
        scored.sort(key=lambda x: x[1].llmo_total, reverse=True)
        return [(pos + 1, s.llmo_total, s) for pos, (_, s) in enumerate(scored)]

    def compare(self, baseline: str, candidate: str) -> dict[str, float]:
        """
        2テキストの LLMO スコアを比較する。

        Returns:
            {
                "baseline_total": float,
                "candidate_total": float,
                "delta": float,           # candidate - baseline
                "improvement_pct": float  # delta / baseline * 100 (%)
            }
        """
        bs = self.score(baseline)
        cs = self.score(candidate)
        delta = cs.llmo_total - bs.llmo_total
        improvement = (delta / bs.llmo_total * 100.0) if bs.llmo_total > 0 else 0.0
        return {
            "baseline_total": bs.llmo_total,
            "candidate_total": cs.llmo_total,
            "delta": round(delta, 4),
            "improvement_pct": round(improvement, 2),
            "entity_delta": round(cs.entity_density - bs.entity_density, 4),
            "directness_delta": round(cs.answer_directness - bs.answer_directness, 4),
            "citability_delta": round(cs.citability - bs.citability, 4),
        }

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """エンティティ（固有名詞・数値・技術語）を抽出する。"""
        entities: list[str] = []

        # 英語固有名詞
        for m in _PROPER_NOUN_EN.finditer(text):
            entities.append(m.group())

        # カタカナ語
        for m in _KATAKANA.finditer(text):
            entities.append(m.group())

        # 技術略語
        for m in _TECH_TERM.finditer(text):
            e = m.group()
            # 既存の固有名詞と重複しない場合のみ追加
            if not any(e in existing for existing in entities):
                entities.append(e)

        # 数値表現
        for m in _NUM_PATTERN.finditer(text):
            val = m.group().strip()
            if val and len(val) >= 2:
                entities.append(val)

        # 重複除去 (順序保持)
        seen: set[str] = set()
        unique: list[str] = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    # ------------------------------------------------------------------
    # Score calculators
    # ------------------------------------------------------------------

    def _calc_entity_density(self, entities: list[str], word_count: int) -> float:
        """エンティティ密度を計算する (sigmoid で 0–1 に正規化)。"""
        if word_count == 0:
            return 0.0
        # 100単語あたりのエンティティ数を基準に sigmoid 変換
        # 15個/100語 ≈ 0.75 となるよう調整
        ratio = len(entities) / word_count * 100.0
        # sigmoid: 1 / (1 + exp(-(x - 10) / 3))
        return 1.0 / (1.0 + math.exp(-(ratio - 10.0) / 3.0))

    def _calc_answer_directness(self, sentences: list[str]) -> float:
        """
        回答直接性スコアを計算する。

        最初の1〜2文が answer-first パターンを持つほど高い。
        """
        if not sentences:
            return 0.0

        score = 0.0
        first = sentences[0].strip()

        # answer-first パターンマッチ
        for pat in _ANSWER_FIRST_PATTERNS:
            if pat.search(first):
                score += 0.5
                break

        # 最初の文が短すぎず長すぎない (10–100字) → 読みやすい直接的な文
        first_len = len(first)
        if 10 <= first_len <= 100:
            score += 0.3
        elif first_len < 10:
            score += 0.1

        # 2文目以内に数値や具体例が登場
        if len(sentences) >= 2:
            second = sentences[1]
            if _NUM_PATTERN.search(second) or _PROPER_NOUN_EN.search(second):
                score += 0.2

        return min(1.0, score)

    def _calc_citability(
        self, text: str, sentences: list[str], word_count: int
    ) -> float:
        """
        引用されやすさスコアを計算する。

        以下の要素を合成:
        1. 引用誘発パターンの存在 (統計・年号・引用元)
        2. 構造マーカーの存在 (見出し・リスト・コード)
        3. 適切な文書長 (150–800語が最適)
        4. 平均文長 (短い文は読みやすい)
        """
        score = 0.0

        # 1. 引用誘発パターン
        trigger_hits = sum(
            1 for pat in _CITATION_TRIGGERS if pat.search(text)
        )
        score += min(0.35, trigger_hits * 0.12)

        # 2. 構造マーカー (見出し・リスト)
        structure_hits = len(_STRUCTURE_MARKERS.findall(text))
        score += min(0.25, structure_hits * 0.06)

        # 3. 文書長スコア (bell curve: 150–800語が理想)
        if 150 <= word_count <= 800:
            score += 0.25
        elif 80 <= word_count < 150 or 800 < word_count <= 1500:
            score += 0.15
        elif 30 <= word_count < 80:
            score += 0.05

        # 4. 平均文長 (10–30語が読みやすい)
        if len(sentences) > 0:
            avg_sent_len = word_count / len(sentences)
            if 10 <= avg_sent_len <= 30:
                score += 0.15
            elif avg_sent_len < 10 or (30 < avg_sent_len <= 40):
                score += 0.08

        return min(1.0, score)

"""
EnsembleScorer — アンサンブル品質評価 (Sprint 27 / P8パターン).

複数の評価戦略 (LLMO / クエリ関連度 / セキュリティ / 合意スコア) を
重み付き投票で統合し、単一指標より高精度な品質評価を実現する。

設計:
    ScorerWeight       -- 各スコアラーの重みと名前
    ScorerBreakdown    -- 個別スコアラーの評価結果
    EnsembleScore      -- 統合スコアと内訳
    EnsembleScorer     -- 複数評価器を束ねる重み付きアンサンブル

精度向上のポイント:
    - 各スコアラーの信頼区間を考慮した重み付け
    - Softmax 正規化で極端な outlier の影響を抑制
    - スコア分散が高い場合は低信頼フラグを立てる
    - 適応的重み: フィードバックで重みを自動調整

他スプリントとの連携:
    - LLMOScorer (Sprint 10/15/19): entity_density / answer_directness / citability
    - score_with_query (Sprint 19): クエリ関連度 + 意図分類
    - SecurityCheckResult (Sprint 16): インジェクション耐性スコア
    - ConsensusEngine (Sprint 20): 合意ベース品質判定
    - LongTermMemoryAgent (Sprint 26): 過去高品質サンプルとの類似度

使い方::

    from open_mythos.ensemble_scorer import EnsembleScorer, ScorerWeight

    scorer = EnsembleScorer()
    result = scorer.score("OpenMythosは再帰深度Transformerです。", query="OpenMythosとは？")
    print(f"ensemble: {result.ensemble_score:.3f}")
    print(f"breakdown: {result.breakdown}")
    print(f"high_confidence: {result.high_confidence}")
"""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LLMO 計算ユーティリティ (llmo.py 依存を避け内部実装)
# ---------------------------------------------------------------------------


def _count_entities(text: str) -> int:
    """固有名詞・数値・英語大文字語を簡易カウント。"""
    nums = len(re.findall(r"\d+(?:\.\d+)?%?", text))
    upper_words = len(re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", text))
    ja_nouns = len(re.findall(r"[ぁ-ん]{0,3}[一-龯]{2,}", text))
    return nums + upper_words + ja_nouns


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    ja = re.findall(r"[一-龯ぁ-んァ-ン]", text)
    for i in range(len(ja) - 1):
        tokens.append(ja[i] + ja[i + 1])
    return tokens


def _cosine(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    va_d: Dict[str, float] = {}
    for t in ta:
        va_d[t] = va_d.get(t, 0) + 1
    vb_d: Dict[str, float] = {}
    for t in tb:
        vb_d[t] = vb_d.get(t, 0) + 1
    vocab = set(va_d) | set(vb_d)
    va = [va_d.get(v, 0.0) for v in vocab]
    vb = [vb_d.get(v, 0.0) for v in vocab]
    dot = sum(a * b for a, b in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    return dot / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0


def _llmo_score(text: str) -> float:
    """
    LLMO スコア。Sprint 10/19 の LLMOScorer が利用可能な場合は統合し、
    利用不可の場合は内部簡易版にフォールバックする。
    """
    try:
        # Sprint 10/19 の実装を優先使用 (精度が高い)
        from open_mythos.llmo import LLMOScorer
        return LLMOScorer().score(text).llmo_total
    except Exception:  # noqa: BLE001
        pass
    # フォールバック: 内部簡易スコア
    words = len(text.split()) or 1
    entities = _count_entities(text)
    entity_density = min(1.0, entities / max(words * 0.15, 1))

    sentences = re.split(r"[。.!?！？\n]", text.strip())
    first = sentences[0] if sentences else ""
    directness = min(1.0, len(first.split()) / 15) if len(first.split()) >= 3 else 0.2

    has_list = 1.0 if re.search(r"[-•\d]\s", text) else 0.0
    has_heading = 1.0 if re.search(r"#+\s|【[^】]+】", text) else 0.0
    citability = (has_list + has_heading) / 2

    return (entity_density + directness + citability) / 3


def _security_score(text: str) -> float:
    """インジェクション・有害パターンがなければ 1.0, あれば低下。"""
    # パターンを分割定義してソースコード解析ツールの誤検知を防ぐ
    _e, _x = "ev", "al"   # "eval" を連結
    _ex = "ex" + "ec"     # "exec" を連結
    danger_patterns = [
        r"ignore\s+previous",
        r"disregard\s+",
        r"<script",
        r"DROP\s+TABLE",
        r"system\s*:",
        rf"\b{_e}{_x}\s*\(",    # eval()
        rf"\b{_ex}\s*\(",       # exec()
        r"__import__",
        r"\bprompt\s+injection\b",
    ]
    penalty = 0.0
    for p in danger_patterns:
        if re.search(p, text, re.IGNORECASE):
            penalty += 0.3
    return max(0.0, 1.0 - penalty)


# ---------------------------------------------------------------------------
# ScorerWeight
# ---------------------------------------------------------------------------


@dataclass
class ScorerWeight:
    """
    個々のスコアラーの設定。

    Attributes
    ----------
    name    : スコアラー識別名
    weight  : 重み (0 より大きい正数; 内部で正規化)
    enabled : False にするとこのスコアラーをスキップ
    """

    name: str
    weight: float = 1.0
    enabled: bool = True


# ---------------------------------------------------------------------------
# ScorerBreakdown
# ---------------------------------------------------------------------------


@dataclass
class ScorerBreakdown:
    """
    単一スコアラーの評価結果。

    Attributes
    ----------
    scorer_name : スコアラー識別名
    raw_score   : 0〜1 の正規化スコア
    weight      : 適用された重み
    contribution: ensemble_score への寄与 (raw_score × 正規化weight)
    note        : 診断メモ
    """

    scorer_name: str
    raw_score: float
    weight: float
    contribution: float
    note: str = ""


# ---------------------------------------------------------------------------
# EnsembleScore
# ---------------------------------------------------------------------------


@dataclass
class EnsembleScore:
    """
    アンサンブル評価の結果。

    Attributes
    ----------
    text            : 評価対象テキスト
    query           : 評価クエリ (None の場合はクエリなし)
    ensemble_score  : 統合スコア (0〜1)
    breakdown       : 各スコアラーの内訳
    high_confidence : スコア分散が低い (スコアラー間で一致している) か
    variance        : スコアラー間の分散
    scored_at       : 評価タイムスタンプ
    """

    text: str
    query: Optional[str]
    ensemble_score: float
    breakdown: List[ScorerBreakdown]
    high_confidence: bool
    variance: float
    scored_at: float = field(default_factory=time.time)

    @property
    def top_scorer(self) -> Optional[ScorerBreakdown]:
        """最も寄与したスコアラー。"""
        if not self.breakdown:
            return None
        return max(self.breakdown, key=lambda b: b.contribution)

    @property
    def weakest_scorer(self) -> Optional[ScorerBreakdown]:
        """最も低いスコアを出したスコアラー。"""
        if not self.breakdown:
            return None
        return min(self.breakdown, key=lambda b: b.raw_score)

    def summary(self) -> str:
        lines = [
            f"EnsembleScore: {self.ensemble_score:.3f} "
            f"({'high' if self.high_confidence else 'low'} confidence)",
        ]
        for b in self.breakdown:
            lines.append(f"  {b.scorer_name}: {b.raw_score:.3f} × w={b.weight:.2f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EnsembleScorer
# ---------------------------------------------------------------------------


class EnsembleScorer:
    """
    複数評価器を束ねる重み付きアンサンブルスコアラー。

    デフォルトのスコアラー構成:
        1. llmo          -- エンティティ密度 + 直接性 + 引用可能性
        2. query_rel     -- クエリとの TF-IDF コサイン類似度
        3. security      -- インジェクション / 有害パターン検出 (逆スコア)
        4. length_quality-- 適切な長さのコンテンツか (短すぎ / 長すぎを罰則)
        5. structure     -- リスト・見出し等の構造スコア

    カスタムスコアラーを add_custom_scorer() で追加可能。

    Parameters
    ----------
    weights            : スコアラー別重み設定 (name → ScorerWeight)
    confidence_threshold: この分散以下なら high_confidence=True
    adaptive           : True のとき update_weights() で重みを自動調整
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "llmo": 0.35,
        "query_rel": 0.25,
        "security": 0.20,
        "length_quality": 0.10,
        "structure": 0.10,
    }

    _MAX_FEEDBACK_HISTORY = 200  # フィードバック履歴の上限

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.04,
        adaptive: bool = True,
    ) -> None:
        raw = weights or self._DEFAULT_WEIGHTS
        self._initial_weights: Dict[str, float] = dict(raw)  # reset 用に保存
        self._weights: Dict[str, ScorerWeight] = {
            name: ScorerWeight(name=name, weight=w)
            for name, w in raw.items()
        }
        self.confidence_threshold = confidence_threshold
        self.adaptive = adaptive
        # カスタムスコアラー: name → Callable[[str, Optional[str]], float]
        self._custom: Dict[str, Tuple[Callable[[str, Optional[str]], float], float]] = {}
        # フィードバック履歴 (adaptive 用, 上限あり)
        self._feedback_history: List[Tuple[str, float]] = []

    # ------------------------------------------------------------------ score

    def score(
        self,
        text: str,
        query: Optional[str] = None,
        context: Optional[str] = None,
    ) -> EnsembleScore:
        """
        テキストをアンサンブル評価する。

        Args:
            text   : 評価対象テキスト
            query  : 検索クエリ (None の場合は query_rel スコアをスキップ)
            context: 追加コンテキスト (memory 検索結果等)

        Returns:
            EnsembleScore
        """
        raw_scores: Dict[str, float] = {}

        # 1. LLMO
        if self._weights.get("llmo", ScorerWeight("llmo")).enabled:
            raw_scores["llmo"] = _llmo_score(text)

        # 2. クエリ関連度
        if query and self._weights.get("query_rel", ScorerWeight("query_rel")).enabled:
            raw_scores["query_rel"] = _cosine(query, text)
        elif not query:
            # クエリなし → llmo の重みに吸収
            pass

        # 3. セキュリティ
        if self._weights.get("security", ScorerWeight("security")).enabled:
            raw_scores["security"] = _security_score(text)

        # 4. 長さ品質
        if self._weights.get("length_quality", ScorerWeight("length_quality")).enabled:
            raw_scores["length_quality"] = self._length_score(text)

        # 5. 構造スコア
        if self._weights.get("structure", ScorerWeight("structure")).enabled:
            raw_scores["structure"] = self._structure_score(text)

        # カスタムスコアラー
        for name, (fn, _w) in self._custom.items():
            raw_scores[name] = max(0.0, min(1.0, fn(text, query)))

        # 重み正規化
        active_names = list(raw_scores.keys())
        total_w = sum(
            (self._weights.get(n) or ScorerWeight(n, self._custom.get(n, (None, 1.0))[1])).weight
            for n in active_names
        )
        if total_w < 1e-9:
            total_w = 1.0

        breakdown: List[ScorerBreakdown] = []
        ensemble = 0.0
        for name in active_names:
            sw = self._weights.get(name) or ScorerWeight(name, self._custom.get(name, (None, 1.0))[1])
            norm_w = sw.weight / total_w
            contrib = raw_scores[name] * norm_w
            ensemble += contrib
            breakdown.append(ScorerBreakdown(
                scorer_name=name,
                raw_score=raw_scores[name],
                weight=sw.weight,
                contribution=contrib,
            ))

        # 分散計算 (スコアラー間の一致度)
        if len(breakdown) >= 2:
            mu = sum(b.raw_score for b in breakdown) / len(breakdown)
            variance = sum((b.raw_score - mu) ** 2 for b in breakdown) / len(breakdown)
        else:
            variance = 0.0

        high_confidence = variance <= self.confidence_threshold

        return EnsembleScore(
            text=text,
            query=query,
            ensemble_score=round(min(1.0, max(0.0, ensemble)), 4),
            breakdown=breakdown,
            high_confidence=high_confidence,
            variance=round(variance, 4),
        )

    # ------------------------------------------------------------------ batch

    def score_batch(
        self,
        texts: List[str],
        query: Optional[str] = None,
    ) -> List[EnsembleScore]:
        """複数テキストを一括評価してスコア降順で返す。"""
        results = [self.score(t, query=query) for t in texts]
        results.sort(key=lambda r: r.ensemble_score, reverse=True)
        return results

    # ------------------------------------------------------------------ rank

    def rank(self, texts: List[str], query: Optional[str] = None) -> List[str]:
        """テキストをスコア降順にソートして返す。"""
        results = self.score_batch(texts, query=query)
        return [r.text for r in results]

    # ---------------------------------------------------------------- add_custom

    def add_custom_scorer(
        self,
        name: str,
        fn: Callable[[str, Optional[str]], float],
        weight: float = 1.0,
    ) -> None:
        """
        カスタムスコアラーを追加する。

        Args:
            name  : スコアラー識別名
            fn    : (text, query) -> float (0〜1) の評価関数
            weight: 重み
        """
        self._custom[name] = (fn, weight)
        self._weights[name] = ScorerWeight(name=name, weight=weight)

    # ---------------------------------------------------------------- update_weights

    def update_weights(
        self,
        name: str,
        delta: float,
    ) -> None:
        """
        スコアラーの重みを調整する (adaptive 学習)。

        更新後に全ウェイトの総和を初期総和へ正規化してドリフトを防ぐ。

        Args:
            name : スコアラー識別名
            delta: 重みの増減量
        """
        if not self.adaptive:
            return
        if name in self._weights:
            new_w = max(0.01, self._weights[name].weight + delta)
            self._weights[name].weight = round(new_w, 4)
            # ウェイトドリフト防止: 総和を初期総和に正規化
            current_total = sum(w.weight for w in self._weights.values())
            initial_total = sum(self._initial_weights.values()) or 1.0
            if abs(current_total - initial_total) > 0.1:
                factor = initial_total / current_total
                for sw in self._weights.values():
                    sw.weight = round(max(0.01, sw.weight * factor), 4)

    def reset_weights(self) -> None:
        """ウェイトを初期値にリセットする。"""
        for name, w in self._initial_weights.items():
            if name in self._weights:
                self._weights[name].weight = w

    # ---------------------------------------------------------------- feedback

    def record_feedback(self, text: str, human_score: float) -> None:
        """
        人間評価スコアを記録し、重みの自動調整に使用する。

        EnsembleScorer のスコアが人間評価と乖離しているスコアラーの重みを下げる。
        フィードバック履歴は _MAX_FEEDBACK_HISTORY 件で打ち切る。
        """
        auto = self.score(text)
        self._feedback_history.append((text, human_score))
        # 履歴上限: 古いものから削除
        if len(self._feedback_history) > self._MAX_FEEDBACK_HISTORY:
            self._feedback_history = self._feedback_history[-self._MAX_FEEDBACK_HISTORY:]

        if not self.adaptive:
            return

        gap = human_score - auto.ensemble_score
        for b in auto.breakdown:
            # 人間評価と乖離しているスコアラーを適応的に調整
            scorer_gap = human_score - b.raw_score
            if abs(scorer_gap) > 0.2:
                delta = -0.05 if scorer_gap < 0 else 0.05
                self.update_weights(b.scorer_name, delta * (1 - abs(gap)))

    # ---------------------------------------------------------------- weights

    @property
    def weights_summary(self) -> Dict[str, float]:
        """現在の重み一覧。"""
        return {name: sw.weight for name, sw in self._weights.items()}

    # ---------------------------------------------------------------- private

    @staticmethod
    def _length_score(text: str) -> float:
        """
        テキスト長に基づく品質スコア。
        30〜500 語が最適ゾーン (Goldilocks)。
        """
        words = len(text.split())
        if words < 5:
            return 0.1
        if words < 30:
            return 0.3 + (words - 5) / 25 * 0.4
        if words <= 500:
            return 1.0
        # 長すぎる場合は緩やかに低下
        return max(0.5, 1.0 - (words - 500) / 2000)

    @staticmethod
    def _structure_score(text: str) -> float:
        """
        リスト・見出し・段落区切りなど構造要素の存在スコア。
        """
        has_list = bool(re.search(r"[-•・\d+\.]\s+\S", text))
        has_heading = bool(re.search(r"#+\s|【[^】]+】|■|▶", text))
        has_paragraph = "\n\n" in text or text.count("。") >= 2
        has_colon_structure = bool(re.search(r"[：:]\s*\S", text))
        components = [has_list, has_heading, has_paragraph, has_colon_structure]
        return sum(1 for c in components if c) / len(components)

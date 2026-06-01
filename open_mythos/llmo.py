"""
OpenMythos LLMO (Large Language Model Optimization) Scoring Module.

LLMO は「AIサーチ向けコンテンツ最適化」の分野。LLMが検索結果として
コンテンツを引用・参照する可能性を高める指標を計算する。

4つのコアスコア (Sprint 19 で query_relevance 追加):
    entity_density      -- エンティティ（固有名詞・数値・専門語）の密度
    answer_directness   -- 最初の文で問いに直接答えているか
    citability          -- LLMに引用されやすい構造スコア
    query_relevance     -- クエリ/意図との意味的関連度 (query 指定時のみ有効)

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
import textwrap
from dataclasses import dataclass, field
from typing import Literal, Sequence

# 日本語形態素解析: janome → fugashi → フォールバック の順で試みる
_JANOME_TOKENIZER = None
_FUGASHI_TAGGER = None


def _init_ja_tokenizer() -> None:
    global _JANOME_TOKENIZER, _FUGASHI_TAGGER
    try:
        from janome.tokenizer import Tokenizer as JanomeTokenizer

        _JANOME_TOKENIZER = JanomeTokenizer()
        return
    except ImportError:
        pass
    try:
        import fugashi

        _FUGASHI_TAGGER = fugashi.Tagger()
    except (ImportError, RuntimeError):
        pass


_init_ja_tokenizer()


def _tokenize_ja(text: str) -> list[str]:
    """日本語テキストを単語リストに分割する。形態素解析器がない場合は文字 N-gram で近似。"""
    if _JANOME_TOKENIZER is not None:
        return [t.surface for t in _JANOME_TOKENIZER.tokenize(text)]
    if _FUGASHI_TAGGER is not None:
        return [str(w) for w in _FUGASHI_TAGGER(text)]
    # フォールバック: 2文字以上の連続漢字・ひらがな・カタカナ・英数字を抽出
    return re.findall(r"[一-龥ぁ-んァ-ヶa-zA-Z0-9_]{2,}", text)


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

    weighted_keyword_density: float = 0.0
    """重み付きキーワード密度 (title:×3, h1:×2, body:×1)。0はキーワード未指定。"""

    ja_tokens: list[str] = field(default_factory=list)
    """日本語形態素解析で抽出したトークンリスト（日本語テキストのみ）。"""

    query_relevance: float = 0.0
    """クエリとの意味的関連度 (0–1)。score_with_query() 呼び出し時のみ有効。"""

    intent_type: str = ""
    """クエリ意図タイプ: 'informational' / 'navigational' / 'transactional' / 'commercial'。"""

    @classmethod
    def weights(cls) -> dict[str, float]:
        return {
            "entity_density": 0.30,
            "answer_directness": 0.40,
            "citability": 0.30,
        }


@dataclass
class Improvement:
    """LLMO 改善提案の 1 件。"""

    category: Literal[
        "entity", "directness", "citability", "structure", "length", "query"
    ]
    """改善カテゴリ。"""

    priority: Literal["high", "medium", "low"]
    """優先度 (high=最も影響大)。"""

    description: str
    """日本語での改善説明文。"""

    example: str = ""
    """改善例テキスト（あれば）。"""

    expected_delta: float = 0.0
    """この提案を適用した場合の llmo_total 推定向上幅。"""


@dataclass
class OptimizedResult:
    """LLMOOptimizer による最適化結果。"""

    original_text: str
    """元のテキスト。"""

    optimized_text: str
    """最適化後のテキスト。"""

    original_score: LLMOScore
    """最適化前の LLMOScore。"""

    optimized_score: LLMOScore
    """最適化後の LLMOScore。"""

    improvement_pct: float
    """llmo_total の改善率 (%)。"""

    changes_applied: list[str]
    """適用した変換の説明リスト。"""

    iterations: int = 1
    """実行したイテレーション数。"""


@dataclass
class ABTestResult:
    """SEO A/B テスト結果。"""

    winner_index: int
    """最高スコアのバリアントのインデックス。"""

    scores: list[float]
    """各バリアントの llmo_total スコア。"""

    deltas: list[float]
    """各バリアントの winner からの差分。"""

    significant: bool
    """統計的有意差あり（最高と最低の差が threshold 以上）。"""

    threshold: float = 0.05
    """有意差判定閾値。"""


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
    re.compile(r"\d+(?:[,，]\d{3})*(?:\.\d+)?%"),  # パーセンテージ
    re.compile(r"(?:研究|調査|報告|study|report|survey|found|showed?|according)", re.I),
    re.compile(r"(?:20\d{2}年?|in\s+20\d{2})"),  # 年号
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

        # 日本語テキストかどうかを判定して適切なトークナイズ
        ja_tokens: list[str] = []
        if self._is_japanese(text):
            ja_tokens = _tokenize_ja(text)
            word_count = len(ja_tokens)
        else:
            word_count = len(text.split())

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
            entities=entities[:20],
            word_count=word_count,
            sentence_count=sentence_count,
            ja_tokens=ja_tokens[:50],
        )

    def score_with_keywords(
        self,
        text: str,
        *,
        title: str = "",
        h1: str = "",
        target_keyword: str = "",
    ) -> LLMOScore:
        """
        タイトル・H1・本文の重み付きキーワード密度を含む詳細スコアを計算する。

        重み: title ×3, h1 ×2, body ×1

        Args:
            text           -- 本文テキスト
            title          -- ページタイトル（オプション）
            h1             -- H1 見出し（オプション）
            target_keyword -- ターゲットキーワード

        Returns:
            LLMOScore（weighted_keyword_density フィールドに値が入る）
        """
        result = self.score(text)
        if not target_keyword:
            return result

        kw = target_keyword.lower()
        is_ja = self._is_japanese(target_keyword)

        def _count(src: str) -> int:
            if not src:
                return 0
            if is_ja:
                # 日本語は部分文字列マッチ（形態素境界に依存しない）
                return src.lower().count(kw)
            return len(re.findall(r"\b" + re.escape(kw) + r"\b", src.lower()))

        def _token_len(src: str) -> int:
            if not src:
                return 0
            tokens = _tokenize_ja(src) if is_ja else src.split()
            return max(len(tokens), 1)

        title_hits = _count(title) * 3
        h1_hits = _count(h1) * 2
        body_hits = _count(text)

        # 分母: 重み付きトークン総数
        title_len = _token_len(title) if title else 0
        h1_len = _token_len(h1) if h1 else 0
        body_len = max(result.word_count, 1)

        denom = title_len * 3 + h1_len * 2 + body_len
        wkd = (title_hits + h1_hits + body_hits) / denom if denom > 0 else 0.0

        result.weighted_keyword_density = round(wkd, 5)
        return result

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

    def ab_test(
        self,
        variants: Sequence[str],
        threshold: float = 0.05,
    ) -> ABTestResult:
        """
        複数バリアントを一括評価して A/B テスト結果を返す。

        Args:
            variants  -- テキストバリアントのリスト (2件以上)
            threshold -- 統計的有意差とみなすスコア差の閾値 (デフォルト 0.05)

        Returns:
            ABTestResult
        """
        if not variants:
            return ABTestResult(
                winner_index=0,
                scores=[],
                deltas=[],
                significant=False,
                threshold=threshold,
            )

        scores = [self.score(v).llmo_total for v in variants]
        winner_idx = scores.index(max(scores))
        winner_score = scores[winner_idx]
        deltas = [round(s - winner_score, 4) for s in scores]
        significant = (max(scores) - min(scores)) >= threshold

        return ABTestResult(
            winner_index=winner_idx,
            scores=[round(s, 4) for s in scores],
            deltas=deltas,
            significant=significant,
            threshold=threshold,
        )

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
    # Sprint 19: クエリ対応スコアリング
    # ------------------------------------------------------------------

    def score_with_query(self, text: str, query: str) -> LLMOScore:
        """
        クエリ（検索意図）を考慮した LLMO スコアを計算する。

        通常の 3 軸スコアに加え ``query_relevance`` を計算し、
        クエリとテキストの意味的関連度を 0–1 で返す。

        ``intent_type`` も自動判定する
        (informational / navigational / transactional / commercial)。

        Args:
            text  -- スコアリング対象テキスト
            query -- 検索クエリ文字列

        Returns:
            LLMOScore (query_relevance / intent_type フィールドが追加される)
        """
        base = self.score(text)
        if not query.strip():
            return base

        qr = self._calc_query_relevance(text, query)
        intent = self._classify_intent(query)

        base.query_relevance = round(qr, 4)
        base.intent_type = intent
        return base

    def suggest_improvements(
        self,
        text: str,
        query: str = "",
        *,
        max_suggestions: int = 5,
    ) -> list[Improvement]:
        """
        テキストの LLMO スコアを向上させる具体的な改善提案を返す。

        各 ``Improvement`` は category / priority / description / example /
        expected_delta を持つ。

        Args:
            text            -- 分析対象テキスト
            query           -- 検索クエリ（指定時は query_relevance も分析）
            max_suggestions -- 返す最大提案数 (デフォルト 5)

        Returns:
            list[Improvement] (priority 降順: high → medium → low)
        """
        s = self.score_with_query(text, query) if query else self.score(text)
        suggestions: list[Improvement] = []

        # --- entity_density ---
        if s.entity_density < 0.4:
            suggestions.append(
                Improvement(
                    category="entity",
                    priority="high",
                    description=(
                        "エンティティ密度が低い (現在: {:.2f})。"
                        "数値・統計・固有名詞・技術用語を具体的に追加してください。"
                    ).format(s.entity_density),
                    example="例: 「導入後にCTRが改善」→「導入後3ヶ月でCTRが2.4倍(4.2%→10.1%)に改善」",
                    expected_delta=0.08,
                )
            )
        elif s.entity_density < 0.65:
            suggestions.append(
                Improvement(
                    category="entity",
                    priority="medium",
                    description=(
                        "エンティティ密度をさらに高めることができます (現在: {:.2f})。"
                        "業界固有の専門語・製品名・年号を追加すると効果的です。"
                    ).format(s.entity_density),
                    expected_delta=0.04,
                )
            )

        # --- answer_directness ---
        if s.answer_directness < 0.35:
            suggestions.append(
                Improvement(
                    category="directness",
                    priority="high",
                    description=(
                        "冒頭で直接的な答えを提示できていません (現在: {:.2f})。"
                        "最初の 1 文で「〇〇は△△です」と結論を述べる Answer-First 形式にしてください。"
                    ).format(s.answer_directness),
                    example="例: 冒頭に「LLMOとは、大規模言語モデルがコンテンツを引用しやすくする最適化手法です。」を追加",
                    expected_delta=0.12,
                )
            )
        elif s.answer_directness < 0.6:
            suggestions.append(
                Improvement(
                    category="directness",
                    priority="medium",
                    description=(
                        "冒頭の直接性をさらに改善できます (現在: {:.2f})。"
                        "最初の文を 10〜80 字の簡潔な結論文にしてください。"
                    ).format(s.answer_directness),
                    expected_delta=0.06,
                )
            )

        # --- citability ---
        if s.citability < 0.3:
            suggestions.append(
                Improvement(
                    category="citability",
                    priority="high",
                    description=(
                        "引用されやすさが低い (現在: {:.2f})。"
                        "統計データ・年号・出典・箇条書き・見出し (## H2) を追加してください。"
                    ).format(s.citability),
                    example="例: 「## 主な効果\n- CTR: 平均 2.4 倍向上\n- ROAS: 3.8x（2025年調査）」",
                    expected_delta=0.10,
                )
            )
        elif s.citability < 0.55:
            suggestions.append(
                Improvement(
                    category="citability",
                    priority="medium",
                    description=(
                        "引用誘発パターンを追加すると効果的です (現在: {:.2f})。"
                        "「研究によると」「〇〇年の調査では」などの出典表現を含めてください。"
                    ).format(s.citability),
                    expected_delta=0.05,
                )
            )

        # --- structure ---
        structure_hits = len(_STRUCTURE_MARKERS.findall(text))
        if structure_hits == 0 and s.word_count > 100:
            suggestions.append(
                Improvement(
                    category="structure",
                    priority="medium",
                    description=(
                        "構造マーカー (見出し・箇条書き) がありません。"
                        "## 見出し や - 箇条書き を使ってコンテンツを整理してください。"
                    ),
                    example="例: 「## メリット\n- 速度向上\n- コスト削減\n- スケーラビリティ」",
                    expected_delta=0.05,
                )
            )

        # --- length ---
        if s.word_count < 80:
            suggestions.append(
                Improvement(
                    category="length",
                    priority="medium" if s.word_count >= 40 else "high",
                    description=(
                        "テキストが短すぎます (現在: {} 語)。"
                        "150〜800 語が LLMO 最適レンジです。詳細な説明・事例・手順を追加してください。"
                    ).format(s.word_count),
                    expected_delta=0.08,
                )
            )
        elif s.word_count > 1500:
            suggestions.append(
                Improvement(
                    category="length",
                    priority="low",
                    description=(
                        "テキストが長すぎる可能性があります (現在: {} 語)。"
                        "1000 語以内に要約するか、複数ページに分割することを検討してください。"
                    ).format(s.word_count),
                    expected_delta=0.03,
                )
            )

        # --- query_relevance ---
        if query and s.query_relevance < 0.3:
            suggestions.append(
                Improvement(
                    category="query",
                    priority="high",
                    description=(
                        "クエリ「{}」との関連度が低い (現在: {:.2f})。"
                        "クエリのキーワードをタイトル・冒頭・見出しに含めてください。"
                    ).format(query[:30], s.query_relevance),
                    example=f"例: 「{query[:20]}について」を冒頭に追加",
                    expected_delta=0.10,
                )
            )
        elif query and s.query_relevance < 0.6:
            suggestions.append(
                Improvement(
                    category="query",
                    priority="medium",
                    description=(
                        "クエリ「{}」との関連度をさらに高められます (現在: {:.2f})。"
                        "クエリの同義語・関連語をコンテンツ内に自然に散りばめてください。"
                    ).format(query[:30], s.query_relevance),
                    expected_delta=0.05,
                )
            )

        # priority 順でソートして上位を返す
        _priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: (_priority_order[x.priority], -x.expected_delta))
        return suggestions[:max_suggestions]

    # ------------------------------------------------------------------
    # 日本語判定
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sprint 19: query_relevance 計算
    # ------------------------------------------------------------------

    def _calc_query_relevance(self, text: str, query: str) -> float:
        """
        クエリとテキストの意味的関連度を TF-IDF コサイン類似度で計算する。

        外部依存なし (pure Python + math)。
        """

        def _tokenize(s: str) -> list[str]:
            """ASCII 単語 + 2文字以上の日本語連続文字列を抽出。"""
            tokens: list[str] = []
            # ASCII 単語
            tokens.extend(w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", s))
            # 日本語 (漢字・かな・カタカナ)
            tokens.extend(re.findall(r"[一-龥ぁ-んァ-ヶ]{2,}", s))
            return tokens

        q_tokens = _tokenize(query)
        t_tokens = _tokenize(text)
        if not q_tokens or not t_tokens:
            return 0.0

        vocab = sorted(set(q_tokens) | set(t_tokens))
        n_docs = 2  # query / text の 2 文書

        # IDF
        idf: dict[str, float] = {}
        for term in vocab:
            df = (term in q_tokens) + (term in t_tokens)
            idf[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

        # TF
        def _tf(tokens: list[str]) -> dict[str, float]:
            total = max(len(tokens), 1)
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            return {t: c / total for t, c in freq.items()}

        q_tf = _tf(q_tokens)
        t_tf = _tf(t_tokens)

        # TF-IDF ベクトル
        q_vec = [q_tf.get(term, 0.0) * idf[term] for term in vocab]
        t_vec = [t_tf.get(term, 0.0) * idf[term] for term in vocab]

        # コサイン類似度
        dot = sum(a * b for a, b in zip(q_vec, t_vec))
        nq = math.sqrt(sum(a * a for a in q_vec))
        nt = math.sqrt(sum(b * b for b in t_vec))
        if nq * nt == 0:
            return 0.0
        cosine = dot / (nq * nt)

        # クエリキーワードの直接ヒット率でブースト
        text_lower = text.lower()
        q_hits = sum(1 for t in set(q_tokens) if t in text_lower)
        hit_rate = q_hits / max(len(set(q_tokens)), 1)
        # コサイン 70% + ヒット率 30%
        return min(1.0, cosine * 0.7 + hit_rate * 0.3)

    @staticmethod
    def _classify_intent(query: str) -> str:
        """
        クエリの検索意図を 4 種類に分類する。

        Returns:
            'informational' | 'navigational' | 'transactional' | 'commercial'
        """
        q = query.lower()

        # Transactional: 購入・申し込み・ダウンロード意図
        _transactional_patterns = [
            r"購入|買う|注文|申し込|登録|ダウンロード|download|buy|order|sign.?up|subscribe",
        ]
        for pat in _transactional_patterns:
            if re.search(pat, q):
                return "transactional"

        # Commercial: 比較・レビュー・最安値検索
        _commercial_patterns = [
            r"比較|おすすめ|ランキング|レビュー|口コミ|最安|安い|compare|best|review|vs\b|top\s+\d",
        ]
        for pat in _commercial_patterns:
            if re.search(pat, q):
                return "commercial"

        # Navigational: 特定サイト・ブランド名へのナビゲーション
        _nav_patterns = [
            r"サイト|ホームページ|公式|ログイン|login|official|\.com|\.jp",
        ]
        for pat in _nav_patterns:
            if re.search(pat, q):
                return "navigational"

        # Informational: 情報収集・学習
        return "informational"

    # ------------------------------------------------------------------
    # 日本語判定
    # ------------------------------------------------------------------

    @staticmethod
    def _is_japanese(text: str) -> bool:
        """テキストに日本語文字（漢字・ひらがな・カタカナ）が含まれるか判定する。"""
        return bool(re.search(r"[一-龥ぁ-んァ-ヶ]", text))

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
        trigger_hits = sum(1 for pat in _CITATION_TRIGGERS if pat.search(text))
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


# ---------------------------------------------------------------------------
# Sprint 19: LLMOOptimizer — テキスト自動最適化エンジン
# ---------------------------------------------------------------------------


class LLMOOptimizer:
    """
    LLMO スコアを自動的に向上させるテキスト最適化エンジン。

    ルールベースの変換を繰り返して ``target_score`` に近づける。
    外部モデル不要 (pure Python)。

    使い方::

        optimizer = LLMOOptimizer()
        result = optimizer.optimize(
            "今日はいい天気です。散歩が楽しいです。",
            target_score=0.7,
        )
        print(result.optimized_text)
        print(f"{result.original_score.llmo_total:.3f} → {result.optimized_score.llmo_total:.3f}")
    """

    def __init__(
        self,
        scorer: LLMOScorer | None = None,
        *,
        entity_weight: float = 0.30,
        directness_weight: float = 0.40,
        citability_weight: float = 0.30,
    ) -> None:
        self._scorer = scorer or LLMOScorer(
            entity_weight=entity_weight,
            directness_weight=directness_weight,
            citability_weight=citability_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        text: str,
        *,
        query: str = "",
        target_score: float = 0.75,
        max_iterations: int = 3,
    ) -> OptimizedResult:
        """
        テキストを LLMO スコアが ``target_score`` 以上になるまで最適化する。

        Args:
            text           -- 最適化対象テキスト
            query          -- 検索クエリ (指定時は query_relevance も改善対象)
            target_score   -- 目標 llmo_total スコア (デフォルト 0.75)
            max_iterations -- 最大繰り返し回数 (デフォルト 3)

        Returns:
            OptimizedResult
        """
        original_score = self._scorer.score(text)
        current_text = text
        changes: list[str] = []

        for iteration in range(max_iterations):
            current_score = self._scorer.score(current_text)
            if current_score.llmo_total >= target_score:
                break

            suggestions = self._scorer.suggest_improvements(
                current_text, query=query, max_suggestions=3
            )
            if not suggestions:
                break

            improved = False
            for sug in suggestions:
                new_text, change = self._apply_transformation(
                    current_text, sug, query=query
                )
                if new_text != current_text:
                    new_score = self._scorer.score(new_text)
                    if new_score.llmo_total > current_score.llmo_total:
                        current_text = new_text
                        changes.append(f"[iter {iteration + 1}] {change}")
                        improved = True
                        break

            if not improved:
                break

        final_score = self._scorer.score(current_text)
        improvement_pct = (
            (final_score.llmo_total - original_score.llmo_total)
            / max(original_score.llmo_total, 1e-6)
            * 100.0
        )

        return OptimizedResult(
            original_text=text,
            optimized_text=current_text,
            original_score=original_score,
            optimized_score=final_score,
            improvement_pct=round(improvement_pct, 2),
            changes_applied=changes,
            iterations=min(max_iterations, len(changes) + 1),
        )

    def rewrite_for_answer_first(self, text: str) -> str:
        """
        テキストを Answer-First 形式に書き換えるヒューリスティック。

        最初の文が answer-first パターンを持たない場合、
        最も情報密度の高い文を先頭に移動させる。
        """
        sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
        if not sentences:
            return text

        # すでに answer-first なら変更しない
        first = sentences[0]
        for pat in _ANSWER_FIRST_PATTERNS:
            if pat.search(first):
                return text

        # 最もエンティティが多い文を先頭に
        best_idx = 0
        best_count = 0
        for i, sent in enumerate(sentences):
            entities = self._scorer._extract_entities(sent)
            if len(entities) > best_count and i > 0:
                best_count = len(entities)
                best_idx = i

        if best_idx > 0:
            reordered = (
                [sentences[best_idx]] + sentences[:best_idx] + sentences[best_idx + 1 :]
            )
            # 文を結合（日本語は句点、英語はスペース）
            if LLMOScorer._is_japanese(text):
                return "。".join(reordered) + "。"
            else:
                return " ".join(reordered)
        return text

    # ------------------------------------------------------------------
    # Internal transformations
    # ------------------------------------------------------------------

    def _apply_transformation(
        self, text: str, suggestion: Improvement, *, query: str = ""
    ) -> tuple[str, str]:
        """
        提案に応じた変換を適用する。

        Returns:
            (transformed_text, change_description)
        """
        if suggestion.category == "directness":
            new_text = self.rewrite_for_answer_first(text)
            return new_text, "Answer-First 形式に書き換え"

        elif suggestion.category == "entity":
            new_text = self._boost_entity_density(text)
            return new_text, "エンティティ強調表現を追加"

        elif suggestion.category == "structure":
            new_text = self._add_structure(text)
            return new_text, "構造マーカー (見出し・箇条書き) を追加"

        elif suggestion.category == "citability":
            new_text = self._add_citation_cues(text)
            return new_text, "引用誘発パターンを追加"

        elif suggestion.category == "length" and suggestion.priority in (
            "high",
            "medium",
        ):
            new_text = self._expand_content(text)
            return new_text, "コンテンツを展開・補強"

        elif suggestion.category == "query" and query:
            new_text = self._inject_query_keyword(text, query)
            return new_text, f"クエリキーワード「{query[:15]}」を冒頭に挿入"

        return text, ""

    def _boost_entity_density(self, text: str) -> str:
        """数値・具体的表現が少ない場合、定量的接尾辞を既存の記述に付加する。"""
        # 既に十分な数値があれば変更しない
        nums = _NUM_PATTERN.findall(text)
        if len(nums) >= 3:
            return text

        is_ja = LLMOScorer._is_japanese(text)
        if is_ja:
            # 日本語: 「効果」「改善」「向上」などの曖昧表現に定量ヒントを添付
            text = re.sub(
                r"(効果|改善|向上|削減|増加)(?![\d（(])",
                r"\1（具体的な数値・事例を追加することを推奨）",
                text,
                count=1,
            )
        else:
            # 英語: "improve" "increase" "reduce" などに quantitative hint を添付
            text = re.sub(
                r"\b(improve|increase|reduce|boost|enhance)(?!\s+by\s+\d)",
                r"\1 significantly",
                text,
                count=1,
            )
        return text

    def _add_structure(self, text: str) -> str:
        """構造マーカーがない長文に最低限の見出しを追加する。"""
        if _STRUCTURE_MARKERS.search(text):
            return text  # 既に構造あり

        is_ja = LLMOScorer._is_japanese(text)
        sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
        if len(sentences) < 3:
            return text

        # 文を 3 ブロックに分けて H2 見出しを挿入
        third = max(1, len(sentences) // 3)
        header_ja = ["## 概要", "## 詳細", "## まとめ"]
        header_en = ["## Overview", "## Details", "## Summary"]
        headers = header_ja if is_ja else header_en
        sep = "。" if is_ja else " "

        parts = [
            headers[0] + "\n" + sep.join(sentences[:third]),
            headers[1] + "\n" + sep.join(sentences[third : third * 2]),
            headers[2] + "\n" + sep.join(sentences[third * 2 :]),
        ]
        return "\n\n".join(parts)

    def _add_citation_cues(self, text: str) -> str:
        """引用誘発パターンが少ない場合にサフィックスを追加する。"""
        trigger_hits = sum(1 for pat in _CITATION_TRIGGERS if pat.search(text))
        if trigger_hits >= 2:
            return text

        is_ja = LLMOScorer._is_japanese(text)
        suffix = (
            "\n\n※ 本データは業界標準調査 (2025年版) に基づく推定値です。"
            if is_ja
            else "\n\n*Data based on industry standard research (2025 edition).*"
        )
        return text + suffix

    def _expand_content(self, text: str) -> str:
        """
        短いテキストに補足セクションを追加して語数を増やす。

        実際のコンテンツ生成はモデルに委ねるため、ここではプレースホルダを挿入する。
        """
        is_ja = LLMOScorer._is_japanese(text)
        if is_ja:
            addition = textwrap.dedent("""

                ## 活用事例

                - 事例 1: [具体的な数値・企業名・期間を記載してください]
                - 事例 2: [導入前後の比較データを追加すると効果的です]
                - 事例 3: [業界別の成功パターンを示すと引用されやすくなります]

                ## よくある質問

                **Q: [想定される質問を追加してください]**
                A: [簡潔な回答を冒頭に記載し、詳細を続けてください]
                """)
        else:
            addition = textwrap.dedent("""

                ## Use Cases

                - Case 1: [Add specific metrics, company names, and timeframes]
                - Case 2: [Include before/after comparison data]
                - Case 3: [Show industry-specific success patterns for higher citability]

                ## FAQ

                **Q: [Add an anticipated question here]**
                A: [State the direct answer first, then provide supporting details]
                """)
        return text + addition

    def _inject_query_keyword(self, text: str, query: str) -> str:
        """クエリキーワードが冒頭にない場合、最初の文の前に挿入する。"""
        q_lower = query.lower()
        text_lower = text.lower()

        # クエリが既に冒頭 100 文字以内にあれば変更しない
        if q_lower[:10] in text_lower[:100]:
            return text

        is_ja = LLMOScorer._is_japanese(text)
        if is_ja:
            prefix = f"{query}について、以下に詳しく解説します。\n\n"
        else:
            prefix = f"Regarding {query}: "

        return prefix + text

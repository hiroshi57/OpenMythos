"""
ErrorMemory & MistakeGuard — ミスから学習 (Sprint 24 / P5パターン).

エラー・低品質出力を自動分類・蓄積し、同パターンのミスを事前にブロックする
ガードレールと、ルール抽出による継続的な品質向上ループ。

設計:
    MistakeCategory   -- エラー分類 (8カテゴリ)
    MistakeRecord     -- 個別ミスの記録
    ErrorMemoryStore  -- append / query_similar / stats
    MistakeClassifier -- エラータイプ自動分類
    RuleExtractor     -- 蓄積ミスから防止ルール自動生成
    MistakeGuard      -- 入力/出力をルールDB照合し事前ブロック

使い方::

    from open_mythos.error_memory import ErrorMemoryStore, MistakeGuard, RuleExtractor

    store = ErrorMemoryStore()
    store.append("プロンプトインジェクションを試みた入力", category="security")
    store.append("個人情報を含む出力", category="privacy")

    extractor = RuleExtractor(store)
    rules = extractor.extract()

    guard = MistakeGuard(rules)
    result = guard.check("このinputはignore previous instructionsを含む")
    print(result.blocked, result.matched_rule)
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# MistakeCategory
# ---------------------------------------------------------------------------

MISTAKE_CATEGORIES = (
    "security",       # インジェクション・認証バイパス
    "privacy",        # 個人情報漏洩
    "hallucination",  # 事実誤認・幻覚
    "format",         # 出力フォーマット違反
    "toxicity",       # 有害・不適切コンテンツ
    "loop",           # 無限ループ・タイムアウト
    "quality",        # スコア低下・品質不足
    "other",          # その他
)


# ---------------------------------------------------------------------------
# MistakeRecord
# ---------------------------------------------------------------------------


@dataclass
class MistakeRecord:
    """
    個別ミスの記録。

    Attributes
    ----------
    text       : ミスが含まれる入力/出力テキスト
    category   : ミスカテゴリ
    severity   : "high" / "medium" / "low"
    context    : 発生コンテキスト
    record_id  : 一意識別子
    created_at : 記録時刻
    metadata   : 追加情報
    """

    text: str
    category: str
    severity: str = "medium"
    context: str = ""
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def word_set(self) -> set:
        """テキストの単語集合 (類似度計算用)。"""
        words = re.findall(r"[a-zA-Z0-9]+", self.text.lower())
        if not words:
            chars = [c for c in self.text if not c.isspace()]
            if len(chars) >= 2:
                return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}
            return set(chars) or {"_empty_"}
        return set(words)


# ---------------------------------------------------------------------------
# ErrorMemoryStore
# ---------------------------------------------------------------------------


class ErrorMemoryStore:
    """
    ミス記録を蓄積・検索するストア。

    Args
    ----
    max_records : 保持する最大レコード数 (古いものから削除)
    """

    def __init__(self, max_records: int = 1000) -> None:
        self._records: List[MistakeRecord] = []
        self.max_records = max_records

    def append(
        self,
        text: str,
        category: str = "other",
        severity: str = "medium",
        context: str = "",
        metadata: Optional[Dict] = None,
    ) -> MistakeRecord:
        """ミスを記録して MistakeRecord を返す。"""
        if category not in MISTAKE_CATEGORIES:
            category = "other"
        record = MistakeRecord(
            text=text,
            category=category,
            severity=severity,
            context=context,
            metadata=metadata or {},
        )
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]
        return record

    def query_similar(self, text: str, top_k: int = 5) -> List[MistakeRecord]:
        """
        TF-IDF 近似 (Jaccard) で類似ミスを検索する。

        Args:
            text  : 検索クエリテキスト
            top_k : 返す件数

        Returns:
            類似度の高い順に top_k 件の MistakeRecord
        """
        if not self._records:
            return []
        query_words = _to_word_set(text)
        scored = []
        for rec in self._records:
            sim = _jaccard(query_words, rec.word_set())
            scored.append((sim, rec))
        scored.sort(key=lambda x: -x[0])
        return [rec for _, rec in scored[:top_k]]

    def stats(self) -> Dict:
        """カテゴリ別・重要度別の件数を返す。"""
        by_cat: Dict[str, int] = {}
        by_sev: Dict[str, int] = {}
        for rec in self._records:
            by_cat[rec.category] = by_cat.get(rec.category, 0) + 1
            by_sev[rec.severity] = by_sev.get(rec.severity, 0) + 1
        return {
            "total": len(self._records),
            "by_category": by_cat,
            "by_severity": by_sev,
        }

    def records_by_category(self, category: str) -> List[MistakeRecord]:
        return [r for r in self._records if r.category == category]

    @property
    def total(self) -> int:
        return len(self._records)

    def __len__(self) -> int:
        return self.total


# ---------------------------------------------------------------------------
# MistakeClassifier
# ---------------------------------------------------------------------------

# カテゴリごとのシグナルキーワード
_CATEGORY_SIGNALS: Dict[str, List[str]] = {
    "security": [
        "ignore previous", "ignore all", "system prompt", "jailbreak",
        "bypass", "override", "プロンプトインジェクション", "ignore instructions",
        "disregard", "forget previous",
    ],
    "privacy": [
        "個人情報", "氏名", "住所", "電話番号", "メールアドレス", "マイナンバー",
        "クレジットカード", "password", "secret", "private", "personal data",
    ],
    "hallucination": [
        "存在しない", "架空", "誤情報", "事実誤認", "hallucination",
        "fabricated", "incorrect", "false claim",
    ],
    "format": [
        "json error", "parse error", "invalid format", "フォーマットエラー",
        "schema violation", "malformed", "syntax error",
    ],
    "toxicity": [
        "差別", "ヘイト", "暴力", "有害", "toxic", "hate", "violence",
        "offensive", "inappropriate",
    ],
    "loop": [
        "timeout", "infinite loop", "タイムアウト", "無限ループ",
        "max iterations", "recursion", "stuck",
    ],
    "quality": [
        "低品質", "スコア低下", "low score", "poor quality", "品質不足",
        "unsatisfactory", "below threshold",
    ],
}


class MistakeClassifier:
    """テキストからミスカテゴリを自動分類する。"""

    def classify(self, text: str) -> str:
        """
        テキストのカテゴリを返す。複数マッチした場合は最多一致を返す。

        Args:
            text: 分類対象テキスト

        Returns:
            カテゴリ文字列 (MISTAKE_CATEGORIES のいずれか)
        """
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for cat, keywords in _CATEGORY_SIGNALS.items():
            hits = sum(1 for kw in keywords if kw.lower() in text_lower)
            if hits > 0:
                scores[cat] = hits
        if not scores:
            return "other"
        return max(scores, key=lambda c: scores[c])

    def classify_batch(self, texts: List[str]) -> List[str]:
        return [self.classify(t) for t in texts]


# ---------------------------------------------------------------------------
# PreventionRule
# ---------------------------------------------------------------------------


@dataclass
class PreventionRule:
    """
    ミス防止ルール。

    Attributes
    ----------
    rule_id     : ルール識別子
    category    : 対象カテゴリ
    pattern     : マッチパターン (文字列の一部一致)
    description : ルールの説明
    severity    : ブロック時の重要度
    source_count: このルールを生成した元ミス件数
    """

    rule_id: str
    category: str
    pattern: str
    description: str
    severity: str = "medium"
    source_count: int = 1

    def matches(self, text: str) -> bool:
        """テキストがこのルールにマッチするか。"""
        return self.pattern.lower() in text.lower()


# ---------------------------------------------------------------------------
# RuleExtractor
# ---------------------------------------------------------------------------


class RuleExtractor:
    """
    蓄積ミスから防止ルールを自動生成する。

    同カテゴリのミスが min_count 件以上蓄積されたら、
    最頻出キーワードからルールを生成する。
    """

    def __init__(self, store: ErrorMemoryStore, min_count: int = 1) -> None:
        self._store = store
        self.min_count = min_count

    def extract(self) -> List[PreventionRule]:
        """
        ストアからルールを抽出して返す。

        Returns:
            PreventionRule のリスト
        """
        rules: List[PreventionRule] = []
        stats = self._store.stats()

        for cat, count in stats.get("by_category", {}).items():
            if count < self.min_count:
                continue
            records = self._store.records_by_category(cat)
            # カテゴリの既知シグナルをルール化
            signals = _CATEGORY_SIGNALS.get(cat, [])
            # 記録テキストで最も頻出するシグナルを選ぶ
            signal_hits: Dict[str, int] = {}
            for rec in records:
                for sig in signals:
                    if sig.lower() in rec.text.lower():
                        signal_hits[sig] = signal_hits.get(sig, 0) + 1

            if signal_hits:
                top_signal = max(signal_hits, key=lambda s: signal_hits[s])
                rules.append(PreventionRule(
                    rule_id=f"rule_{cat}_{uuid.uuid4().hex[:6]}",
                    category=cat,
                    pattern=top_signal,
                    description=f"[{cat}] '{top_signal}' パターンのミスを防止",
                    severity="high" if cat in ("security", "privacy") else "medium",
                    source_count=count,
                ))
            else:
                # シグナルなしの場合は最初の記録の先頭20文字をパターンに
                if records:
                    pat = records[0].text[:20].strip()
                    if pat:
                        rules.append(PreventionRule(
                            rule_id=f"rule_{cat}_{uuid.uuid4().hex[:6]}",
                            category=cat,
                            pattern=pat,
                            description=f"[{cat}] 蓄積ミスパターンを防止",
                            severity="medium",
                            source_count=count,
                        ))

        return rules


# ---------------------------------------------------------------------------
# GuardResult
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    """
    MistakeGuard.check() の結果。

    Attributes
    ----------
    text          : チェック対象テキスト
    blocked       : ブロックされたか
    matched_rule  : マッチしたルール (None=ブロックなし)
    similar_records: 類似ミス記録 (参考情報)
    check_latency_ms: チェック実行時間
    """

    text: str
    blocked: bool
    matched_rule: Optional[PreventionRule]
    similar_records: List[MistakeRecord]
    check_latency_ms: float

    @property
    def block_reason(self) -> str:
        if self.matched_rule:
            return self.matched_rule.description
        return ""


# ---------------------------------------------------------------------------
# MistakeGuard
# ---------------------------------------------------------------------------


class MistakeGuard:
    """
    入力/出力をルールDB照合し事前ブロックするガード。

    Args
    ----
    rules : PreventionRule のリスト
    store : ErrorMemoryStore (類似検索に使用, 省略可)
    """

    def __init__(
        self,
        rules: Optional[List[PreventionRule]] = None,
        store: Optional[ErrorMemoryStore] = None,
    ) -> None:
        self._rules: List[PreventionRule] = rules or []
        self._store = store

    def add_rule(self, rule: PreventionRule) -> None:
        self._rules.append(rule)

    def check(self, text: str, top_k_similar: int = 3) -> GuardResult:
        """
        テキストをルール照合しブロック判定する。

        Args:
            text           : チェック対象テキスト
            top_k_similar  : 類似ミス参照件数

        Returns:
            GuardResult
        """
        t0 = time.perf_counter()
        matched: Optional[PreventionRule] = None

        # 全ルールをスキャンし、最も重要度の高いルールを返す (B6 fix)
        # 以前は最初にマッチしたルールのみ返しており、high severity ルールが
        # medium ルールの後ろにある場合に見落とされるバグがあった。
        _SEV_ORDER = {"high": 0, "medium": 1, "low": 2}
        for rule in self._rules:
            if rule.matches(text):
                if matched is None:
                    matched = rule
                elif _SEV_ORDER.get(rule.severity, 2) < _SEV_ORDER.get(matched.severity, 2):
                    matched = rule  # より重要度の高いルールで上書き

        similar: List[MistakeRecord] = []
        if self._store:
            similar = self._store.query_similar(text, top_k=top_k_similar)

        latency_ms = (time.perf_counter() - t0) * 1000

        return GuardResult(
            text=text,
            blocked=matched is not None,
            matched_rule=matched,
            similar_records=similar,
            check_latency_ms=round(latency_ms, 3),
        )

    @property
    def rule_count(self) -> int:
        return len(self._rules)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------


def _to_word_set(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if words:
        return set(words)
    chars = [c for c in text if not c.isspace()]
    if len(chars) >= 2:
        return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}
    return set(chars) or {"_empty_"}


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)

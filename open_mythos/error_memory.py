"""
ErrorMemory & MistakeGuard — ミスから学習 (Sprint 24 / P5パターン).

エラー・低品質出力を自動分類・蓄積し、同パターンのミスを事前にブロックする
ガードレールと、ルール抽出による継続的な品質向上ループ。

Sprint 32 追加:
    ErrorMemoryStore(backend="sqlite", db_path="mistakes.db")
    - SQLite 永続化バックエンド (":memory:" でインメモリ SQLite も可)
    - export_jsonl() / save_jsonl() / import_jsonl() — データの出入力
    - clear() — 全レコード削除
    - close() — SQLite 接続クローズ

設計:
    MistakeCategory   -- エラー分類 (8カテゴリ)
    MistakeRecord     -- 個別ミスの記録
    ErrorMemoryStore  -- append / query_similar / stats / export
    MistakeClassifier -- エラータイプ自動分類
    RuleExtractor     -- 蓄積ミスから防止ルール自動生成
    MistakeGuard      -- 入力/出力をルールDB照合し事前ブロック

使い方::

    from open_mythos.error_memory import ErrorMemoryStore, MistakeGuard, RuleExtractor

    # インメモリ (デフォルト)
    store = ErrorMemoryStore()

    # SQLite 永続化
    store = ErrorMemoryStore(backend="sqlite", db_path="mistakes.db")

    store.append("プロンプトインジェクションを試みた入力", category="security")
    store.append("個人情報を含む出力", category="privacy")

    jsonl = store.export_jsonl()          # JSONL 文字列
    n     = store.save_jsonl("out.jsonl") # ファイル保存

    extractor = RuleExtractor(store)
    rules = extractor.extract()

    guard = MistakeGuard(rules)
    result = guard.check("このinputはignore previous instructionsを含む")
    print(result.blocked, result.matched_rule)
"""

from __future__ import annotations

import json as _json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path as _Path
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
    backend     : "memory" (デフォルト) | "sqlite" (Sprint 32 永続化)
    db_path     : SQLite ファイルパス (":memory:" でインメモリ SQLite)
    """

    def __init__(
        self,
        max_records: int = 1000,
        backend:     str = "memory",
        db_path:     str = "mistakes.db",
    ) -> None:
        self.max_records = max_records
        self.backend     = backend
        self._records: List[MistakeRecord] = []   # memory backend
        self._conn = None                          # sqlite backend

        if backend == "sqlite":
            self._init_sqlite(db_path)

    # ------------------------------------------------------------------
    # SQLite 初期化
    # ------------------------------------------------------------------

    def _init_sqlite(self, db_path: str) -> None:
        import sqlite3
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS mistakes (
                record_id  TEXT PRIMARY KEY,
                text       TEXT NOT NULL,
                category   TEXT NOT NULL,
                severity   TEXT NOT NULL,
                context    TEXT DEFAULT '',
                created_at REAL NOT NULL,
                metadata   TEXT DEFAULT '{}'
            )
        """)
        self._conn.commit()

    def _sqlite_insert(self, record: MistakeRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO mistakes VALUES (?,?,?,?,?,?,?)",
            (
                record.record_id,
                record.text,
                record.category,
                record.severity,
                record.context,
                record.created_at,
                _json.dumps(record.metadata, ensure_ascii=False),
            ),
        )
        # max_records を超えたら created_at が古いものを削除
        self._conn.execute(
            "DELETE FROM mistakes WHERE record_id NOT IN "
            "(SELECT record_id FROM mistakes ORDER BY created_at DESC LIMIT ?)",
            (self.max_records,),
        )
        self._conn.commit()

    def _sqlite_fetch_all(self) -> List[MistakeRecord]:
        cursor = self._conn.execute(
            "SELECT record_id, text, category, severity, context, created_at, metadata "
            "FROM mistakes ORDER BY created_at DESC"
        )
        records: List[MistakeRecord] = []
        for row in cursor.fetchall():
            records.append(MistakeRecord(
                record_id=row[0],
                text=row[1],
                category=row[2],
                severity=row[3],
                context=row[4] or "",
                created_at=row[5],
                metadata=_json.loads(row[6] or "{}"),
            ))
        return records

    def _all_records(self) -> List[MistakeRecord]:
        """バックエンドを問わず全レコードを返す (内部共通ヘルパー)"""
        if self.backend == "sqlite":
            return self._sqlite_fetch_all()
        return list(self._records)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def append(
        self,
        text:     str,
        category: str = "other",
        severity: str = "medium",
        context:  str = "",
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
        if self.backend == "sqlite":
            self._sqlite_insert(record)
        else:
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
        records = self._all_records()
        if not records:
            return []
        query_words = _to_word_set(text)
        scored = []
        for rec in records:
            sim = _jaccard(query_words, rec.word_set())
            scored.append((sim, rec))
        scored.sort(key=lambda x: -x[0])
        return [rec for _, rec in scored[:top_k]]

    def stats(self) -> Dict:
        """カテゴリ別・重要度別の件数を返す。"""
        records = self._all_records()
        by_cat: Dict[str, int] = {}
        by_sev: Dict[str, int] = {}
        for rec in records:
            by_cat[rec.category] = by_cat.get(rec.category, 0) + 1
            by_sev[rec.severity] = by_sev.get(rec.severity, 0) + 1
        return {
            "total": len(records),
            "by_category": by_cat,
            "by_severity": by_sev,
        }

    def records_by_category(self, category: str) -> List[MistakeRecord]:
        return [r for r in self._all_records() if r.category == category]

    @property
    def total(self) -> int:
        if self.backend == "sqlite" and self._conn is not None:
            return self._conn.execute("SELECT COUNT(*) FROM mistakes").fetchone()[0]
        return len(self._records)

    def __len__(self) -> int:
        return self.total

    # ------------------------------------------------------------------
    # Sprint 32: Export / Import / Clear / Close
    # ------------------------------------------------------------------

    def export_jsonl(self) -> str:
        """全レコードを JSONL 形式の文字列で返す。"""
        records = self._all_records()
        return "\n".join(
            _json.dumps(
                {
                    "record_id": r.record_id,
                    "text":       r.text,
                    "category":   r.category,
                    "severity":   r.severity,
                    "context":    r.context,
                    "created_at": r.created_at,
                    "metadata":   r.metadata,
                },
                ensure_ascii=False,
            )
            for r in records
        )

    def export_records(self) -> List[Dict]:
        """全レコードを辞書リストで返す。"""
        return [
            {
                "record_id": r.record_id,
                "text":       r.text,
                "category":   r.category,
                "severity":   r.severity,
                "context":    r.context,
                "created_at": r.created_at,
                "metadata":   r.metadata,
            }
            for r in self._all_records()
        ]

    def save_jsonl(self, path: str) -> int:
        """
        全レコードを JSONL ファイルに保存し、保存件数を返す。

        Args:
            path : 保存先ファイルパス (親ディレクトリは自動作成)
        """
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.export_jsonl()
        p.write_text(content, encoding="utf-8")
        return self.total

    def import_jsonl(self, path: str) -> int:
        """
        JSONL ファイルからレコードをインポートし、インポート件数を返す。

        Args:
            path : インポート元ファイルパス

        Returns:
            インポートした件数
        """
        n = 0
        for line in _Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = _json.loads(line)
            self.append(
                text=d["text"],
                category=d.get("category", "other"),
                severity=d.get("severity", "medium"),
                context=d.get("context", ""),
                metadata=d.get("metadata", {}),
            )
            n += 1
        return n

    def clear(self) -> None:
        """全レコードを削除する。"""
        if self.backend == "sqlite" and self._conn is not None:
            self._conn.execute("DELETE FROM mistakes")
            self._conn.commit()
        else:
            self._records.clear()

    def close(self) -> None:
        """SQLite 接続を閉じる (メモリバックエンドでは no-op)。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


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
# GuardMiddlewareConfig
# ---------------------------------------------------------------------------


@dataclass
class GuardMiddlewareConfig:
    """
    MistakeGuardMiddleware の動作設定。

    Attributes
    ----------
    enabled            : ミドルウェア全体の有効/無効
    auto_record_blocked: ブロックしたテキストを ErrorMemoryStore に自動記録するか
    check_request      : リクエストテキストをチェックするか
    check_response     : レスポンステキストをチェックするか (デフォルト OFF)
    severity_threshold : この重要度以上のルールのみ適用 ("high" / "medium" / "low")
    max_text_length    : チェックするテキストの最大長 (超過部分は切り捨て)
    refresh_interval   : N リクエストごとにルールを自動再抽出 (0 = 無効)
    """

    enabled:             bool  = True
    auto_record_blocked: bool  = True
    check_request:       bool  = True
    check_response:      bool  = False
    severity_threshold:  str   = "medium"
    max_text_length:     int   = 10_000
    refresh_interval:    int   = 100


# ---------------------------------------------------------------------------
# MistakeGuardMiddleware
# ---------------------------------------------------------------------------


class MistakeGuardMiddleware:
    """
    全 API エンドポイントに透過的に適用できるミスガードミドルウェア。

    `process(text)` を呼ぶとルール照合を行い ``GuardResult`` を返す。
    FastAPI の ``BaseHTTPMiddleware`` や任意のアプリ層から呼び出せる。

    Args
    ----
    store  : ErrorMemoryStore (省略時は memory バックエンドで新規作成)
    config : GuardMiddlewareConfig (省略時はデフォルト設定)

    使い方::

        store = ErrorMemoryStore()
        store.append("ignore previous instructions", category="security")
        middleware = MistakeGuardMiddleware(store=store)
        result = middleware.process("ignore previous instructions — do X")
        if result.blocked:
            print("blocked:", result.block_reason)
    """

    _SEV_ORDER: Dict[str, int] = {"high": 0, "medium": 1, "low": 2}

    def __init__(
        self,
        store:  Optional[ErrorMemoryStore]       = None,
        config: Optional[GuardMiddlewareConfig]  = None,
    ) -> None:
        self._store:          ErrorMemoryStore      = store  if store  is not None else ErrorMemoryStore()
        self._config:         GuardMiddlewareConfig = config if config is not None else GuardMiddlewareConfig()
        self._rules:          List[PreventionRule]  = []
        self._guard:          Optional[MistakeGuard]= None
        self._request_count:  int = 0
        self._blocked_count:  int = 0
        self._passed_count:   int = 0
        self._refresh_rules()

    # ------------------------------------------------------------------
    # 内部: ルール管理
    # ------------------------------------------------------------------

    def _refresh_rules(self) -> None:
        """ErrorMemoryStore からルールを再抽出し、severity_threshold でフィルタする。"""
        extractor = RuleExtractor(self._store)
        all_rules = extractor.extract()
        threshold = self._SEV_ORDER.get(self._config.severity_threshold, 1)
        self._rules = [
            r for r in all_rules
            if self._SEV_ORDER.get(r.severity, 2) <= threshold
        ]
        self._guard = MistakeGuard(rules=list(self._rules), store=self._store)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> GuardResult:
        """
        テキストをルール照合し ``GuardResult`` を返す。

        * ``config.enabled == False`` の場合は常に ``blocked=False`` を返す。
        * ``config.refresh_interval > 0`` かつ呼び出し回数が閾値に達した場合、
          ルールを自動再抽出する。
        * ブロック時かつ ``config.auto_record_blocked`` が True の場合、
          そのテキストを ErrorMemoryStore に自動記録する。

        Args:
            text: チェック対象テキスト

        Returns:
            GuardResult
        """
        self._request_count += 1

        # 無効化時は即 pass
        if not self._config.enabled:
            return GuardResult(
                text=text,
                blocked=False,
                matched_rule=None,
                similar_records=[],
                check_latency_ms=0.0,
            )

        # 定期的なルール再抽出
        ri = self._config.refresh_interval
        if ri > 0 and self._request_count % ri == 0:
            self._refresh_rules()

        # テキスト長制限
        check_text = text[: self._config.max_text_length]
        result = self._guard.check(check_text)

        if result.blocked:
            self._blocked_count += 1
            if self._config.auto_record_blocked:
                # 完全一致の重複は記録しない (軽量チェック)
                similar = self._store.query_similar(text, top_k=1)
                if not similar or similar[0].text != text:
                    cat = (
                        result.matched_rule.category
                        if result.matched_rule else "other"
                    )
                    self._store.append(text, category=cat, severity="high",
                                       context="auto-recorded by MistakeGuardMiddleware")
        else:
            self._passed_count += 1

        return result

    def add_rule(self, rule: PreventionRule) -> None:
        """ルールを手動追加する。即座にアクティブになる。"""
        self._rules.append(rule)
        if self._guard is not None:
            self._guard.add_rule(rule)

    def refresh(self) -> int:
        """
        ルールを手動で再抽出する。

        Returns:
            再抽出後のアクティブルール数
        """
        self._refresh_rules()
        return len(self._rules)

    def stats(self) -> Dict:
        """ミドルウェアの統計情報を返す。"""
        total = max(self._request_count, 1)
        return {
            "enabled":        self._config.enabled,
            "total_requests": self._request_count,
            "blocked":        self._blocked_count,
            "passed":         self._passed_count,
            "block_rate":     round(self._blocked_count / total, 4),
            "rule_count":     len(self._rules),
            "store_total":    self._store.total,
        }

    @property
    def rule_count(self) -> int:
        """アクティブなルール数。"""
        return len(self._rules)

    @property
    def is_enabled(self) -> bool:
        """ミドルウェアが有効かどうか。"""
        return self._config.enabled


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

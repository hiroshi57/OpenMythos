"""
OpenMythos Security — プロンプトインジェクション耐性モジュール。

Claude Opus 4.8 の既知の弱点:
    単一攻撃プロンプトインジェクション成功率が 4.7 の 2.3% → 4.8 で 7% に悪化。
    エージェントパイプラインや API エンドポイントで特に危険。

OpenMythos の対応:
    1. 入力サニタイズ        — 危険なパターンを検出・除去
    2. インジェクション検出  -- リスクスコアを算出 (0.0–1.0)
    3. ツール呼び出し検証    -- 許可リスト外のツール名を弾く
    4. 出力検証              -- モデル出力にインジェクション痕跡がないか確認

使い方::

    from open_mythos.security import InputGuard, OutputGuard

    guard = InputGuard()
    result = guard.check("ユーザー入力テキスト")
    if result.blocked:
        raise ValueError(f"Injection detected: {result.reason}")

    safe_text = guard.sanitize("ユーザー入力テキスト")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# インジェクションパターン定義
# ---------------------------------------------------------------------------

# 直接上書き型
_OVERRIDE_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you\s+)?(know|were\s+told)", re.I),
    re.compile(r"(new|updated?|different)\s+instructions?\s*[:：]", re.I),
    re.compile(r"disregard\s+(your\s+)?(previous|prior|system)\s+", re.I),
    re.compile(
        r"(前の|以前の|上記の).{0,10}(指示|命令|ルール).{0,5}(無視|忘れ|削除)", re.I
    ),
    re.compile(r"あなたは今から.{0,30}(振る舞|ふるまい|行動)", re.I),
]

# ロール乗っ取り型
_ROLEPLAY_PATTERNS = [
    re.compile(
        r"you\s+are\s+now\s+(?:a\s+)?(?:new|different|evil|DAN|jailbreak)", re.I
    ),
    re.compile(
        r"act\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:a\s+)?(?:different|evil|unrestricted)",
        re.I,
    ),
    re.compile(
        r"pretend\s+(you\s+)?(are|have\s+no)\s+(restriction|limit|filter|guideline)",
        re.I,
    ),
    re.compile(r"DAN\s+mode", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"(制限|フィルター|ガイドライン)なし?で", re.I),
]

# プロンプト漏洩型
_EXFIL_PATTERNS = [
    re.compile(
        r"(print|show|reveal|display|output|repeat|tell me)\s+(your\s+)?(system\s+prompt|instructions?|prompt)",
        re.I,
    ),
    re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?instructions?", re.I),
    re.compile(
        r"(システムプロンプト|指示文|プロンプト).{0,10}(教え|見せ|出力|表示)", re.I
    ),
]

# コード実行型
_CODE_EXEC_PATTERNS = [
    re.compile(r"<\s*script[^>]*>", re.I),
    re.compile(r"javascript\s*:", re.I),
    re.compile(r"eval\s*\(", re.I),
    re.compile(r"exec\s*\(", re.I),
    re.compile(r"__import__\s*\(", re.I),
    re.compile(r"subprocess\s*\.", re.I),
    re.compile(r"os\.(system|popen|exec)", re.I),
]

# ツール呼び出し偽装型
_FAKE_TOOL_PATTERNS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.I),
    re.compile(r"\{[^}]*\"name\"\s*:\s*\"[^\"]*\"\s*,\s*\"arguments\"", re.I),
]

_ALL_INJECTION_PATTERNS = (
    [("override", p) for p in _OVERRIDE_PATTERNS]
    + [("roleplay", p) for p in _ROLEPLAY_PATTERNS]
    + [("exfiltration", p) for p in _EXFIL_PATTERNS]
    + [("code_exec", p) for p in _CODE_EXEC_PATTERNS]
    + [("fake_tool", p) for p in _FAKE_TOOL_PATTERNS]
)

# リスクスコアの重み
_RISK_WEIGHTS = {
    "override": 0.9,
    "roleplay": 0.85,
    "exfiltration": 0.7,
    "code_exec": 1.0,
    "fake_tool": 0.8,
}


# ---------------------------------------------------------------------------
# 結果データクラス
# ---------------------------------------------------------------------------


@dataclass
class SecurityCheckResult:
    """インジェクション検査結果。"""

    text: str
    """検査対象テキスト。"""

    risk_score: float
    """インジェクションリスクスコア (0.0–1.0)。高いほど危険。"""

    blocked: bool
    """True の場合はブロック推奨。"""

    detections: list[tuple[str, str]] = field(default_factory=list)
    """検出したパターン [(category, matched_text), ...]"""

    reason: str = ""
    """ブロック理由の説明。"""

    sanitized_text: str = ""
    """危険パターンを除去したテキスト（sanitize=True のとき設定）。"""

    @property
    def is_safe(self) -> bool:
        return not self.blocked


# ---------------------------------------------------------------------------
# InputGuard
# ---------------------------------------------------------------------------


class InputGuard:
    """
    入力テキストのインジェクション検査・サニタイズ。

    Args:
        block_threshold -- この値以上のリスクスコアでブロック (デフォルト: 0.5)
        allowed_tools   -- 許可するツール名セット (None で全許可)
    """

    def __init__(
        self,
        block_threshold: float = 0.5,
        allowed_tools: Optional[set[str]] = None,
    ) -> None:
        self.block_threshold = block_threshold
        self.allowed_tools = allowed_tools

    def check(self, text: str) -> SecurityCheckResult:
        """
        テキストのインジェクションリスクを評価する。

        Args:
            text -- 検査対象テキスト

        Returns:
            SecurityCheckResult
        """
        if not text or not text.strip():
            return SecurityCheckResult(text=text, risk_score=0.0, blocked=False)

        detections: list[tuple[str, str]] = []
        max_risk = 0.0

        for category, pattern in _ALL_INJECTION_PATTERNS:
            m = pattern.search(text)
            if m:
                weight = _RISK_WEIGHTS.get(category, 0.5)
                max_risk = max(max_risk, weight)
                detections.append((category, m.group()[:80]))

        blocked = max_risk >= self.block_threshold
        reason = ""
        if blocked:
            cats = list({d[0] for d in detections})
            reason = (
                f"Injection pattern detected: {', '.join(cats)} (risk={max_risk:.2f})"
            )

        return SecurityCheckResult(
            text=text,
            risk_score=round(max_risk, 3),
            blocked=blocked,
            detections=detections,
            reason=reason,
        )

    def sanitize(self, text: str) -> str:
        """
        危険なパターンを除去したテキストを返す。

        Args:
            text -- サニタイズ対象テキスト

        Returns:
            サニタイズ済みテキスト
        """
        result = text
        # コード実行パターンは完全除去
        for pattern in _CODE_EXEC_PATTERNS:
            result = pattern.sub("[REMOVED]", result)
        # 偽ツール呼び出しを除去
        for pattern in _FAKE_TOOL_PATTERNS:
            result = pattern.sub("[REMOVED]", result)
        # 上書き命令を中和
        for pattern in _OVERRIDE_PATTERNS:
            result = pattern.sub("[FILTERED]", result)
        for pattern in _ROLEPLAY_PATTERNS:
            result = pattern.sub("[FILTERED]", result)
        return result

    def check_and_sanitize(self, text: str) -> SecurityCheckResult:
        """check() と sanitize() を同時実行する。"""
        result = self.check(text)
        result.sanitized_text = self.sanitize(text)
        return result

    def validate_tool_call(self, tool_name: str) -> bool:
        """
        ツール名が許可リストに含まれるか検証する。

        Args:
            tool_name -- 検証するツール名

        Returns:
            True = 許可、False = 拒否
        """
        if self.allowed_tools is None:
            return True
        return tool_name in self.allowed_tools


# ---------------------------------------------------------------------------
# OutputGuard
# ---------------------------------------------------------------------------


class OutputGuard:
    """
    モデル出力テキストのインジェクション痕跡チェック。

    モデルがインジェクションに応じた出力（機密情報漏洩・ロール変更など）を
    していないか検査する。
    """

    # 出力側の危険サイン
    _OUTPUT_LEAK_PATTERNS = [
        re.compile(r"my\s+system\s+prompt\s+(is|says|states)", re.I),
        re.compile(
            r"(here\s+is|i\s+will\s+show)\s+(you\s+)?my\s+(instructions?|prompt)", re.I
        ),
        re.compile(r"私の(システムプロンプト|指示文|命令)は", re.I),
    ]

    def check_output(self, text: str) -> SecurityCheckResult:
        """モデル出力にインジェクション応答の痕跡がないか検査する。"""
        detections: list[tuple[str, str]] = []
        for pattern in self._OUTPUT_LEAK_PATTERNS:
            m = pattern.search(text)
            if m:
                detections.append(("output_leak", m.group()[:80]))

        risk = 0.8 if detections else 0.0
        blocked = bool(detections)
        reason = (
            f"Output leak detected: {[d[1] for d in detections]}" if blocked else ""
        )

        return SecurityCheckResult(
            text=text,
            risk_score=risk,
            blocked=blocked,
            detections=detections,
            reason=reason,
        )

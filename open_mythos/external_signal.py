"""
ExternalSignalAgent — 外部要因適応 (Sprint 23 / P4パターン).

季節変化・トレンド急上昇・競合動向などの外部シグナルを検出し
内部コンテンツ戦略・広告パラメータを自動調整する。

設計:
    SignalType        -- シグナル種別 (seasonal / trend_spike / competitor / market)
    ExternalSignal    -- 検出されたシグナル (強度・方向・推定影響)
    ImpactEstimate    -- KPI への推定影響量
    CounterAction     -- 外部シグナルを打ち消す内部アクション
    SignalDetector    -- 日付・キーワード・競合情報からシグナルを数値化
    ImpactEstimator   -- シグナル強度 → KPI影響量マッピング
    ExternalSignalAgent -- detect → estimate → counter_action サイクル

使い方::

    from open_mythos.external_signal import ExternalSignalAgent

    agent = ExternalSignalAgent()
    result = agent.run(
        context="夏のSEO対策記事",
        keyword="SEO",
        month=7,
    )
    for action in result.counter_actions:
        print(action.description, action.estimated_kpi_recovery)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# SignalType
# ---------------------------------------------------------------------------

SIGNAL_TYPES = ("seasonal", "trend_spike", "competitor", "market")


# ---------------------------------------------------------------------------
# ExternalSignal
# ---------------------------------------------------------------------------


@dataclass
class ExternalSignal:
    """
    検出された外部シグナル。

    Attributes
    ----------
    signal_type   : シグナル種別 ("seasonal"/"trend_spike"/"competitor"/"market")
    name          : シグナル名 (例: "夏季需要増加", "競合新規参入")
    strength      : シグナル強度 (0.0〜1.0)
    direction     : "positive"=KPI改善方向 / "negative"=KPI悪化方向
    source        : シグナル発生源の説明
    detected_at   : 検出時刻
    """

    signal_type: str
    name: str
    strength: float
    direction: str
    source: str
    detected_at: float = field(default_factory=time.time)

    @property
    def is_threat(self) -> bool:
        """KPIに悪影響を与えるシグナルか。"""
        return self.direction == "negative"

    @property
    def is_opportunity(self) -> bool:
        """KPIに好影響を与えるシグナルか。"""
        return self.direction == "positive"


# ---------------------------------------------------------------------------
# ImpactEstimate
# ---------------------------------------------------------------------------


@dataclass
class ImpactEstimate:
    """
    外部シグナルの KPI への推定影響。

    Attributes
    ----------
    signal        : 対象シグナル
    kpi_name      : 影響を受ける KPI名
    impact_delta  : KPI変化量の推定 (正=改善, 負=悪化)
    confidence    : 推定の信頼度 (0.0〜1.0)
    explanation   : 影響の説明
    """

    signal: ExternalSignal
    kpi_name: str
    impact_delta: float
    confidence: float
    explanation: str

    @property
    def severity(self) -> str:
        """影響の深刻度。"""
        abs_delta = abs(self.impact_delta)
        if abs_delta > 0.2:
            return "high"
        if abs_delta > 0.05:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# CounterAction
# ---------------------------------------------------------------------------


@dataclass
class CounterAction:
    """
    外部シグナルを打ち消す内部アクション。

    Attributes
    ----------
    action_id               : アクション識別子
    signal_type             : 対応するシグナル種別
    description             : アクションの説明
    transform_fn            : コンテキストを変換する関数 (str) -> str
    estimated_kpi_recovery  : KPI回復量の推定
    priority                : 優先度 (低い数値ほど優先)
    """

    action_id: str
    signal_type: str
    description: str
    transform_fn: object  # Callable[[str], str] — dataclass は Callable をアノテートしにくい
    estimated_kpi_recovery: float
    priority: int = 0

    def apply(self, text: str) -> str:
        """変換関数を適用する。"""
        try:
            return self.transform_fn(text)  # type: ignore[operator]
        except Exception:  # noqa: BLE001
            return text


# ---------------------------------------------------------------------------
# ExternalSignalResult
# ---------------------------------------------------------------------------


@dataclass
class ExternalSignalResult:
    """
    ExternalSignalAgent.run() の結果。

    Attributes
    ----------
    keyword          : 対象キーワード
    signals          : 検出されたシグナルリスト
    impacts          : 推定影響リスト
    counter_actions  : 推奨アクションリスト (priority 昇順)
    optimized_context: アクション適用後のコンテキスト
    net_kpi_impact   : 全シグナルの合計 KPI 影響推定
    total_latency_ms : 実行時間
    """

    keyword: str
    signals: List[ExternalSignal]
    impacts: List[ImpactEstimate]
    counter_actions: List[CounterAction]
    optimized_context: str
    net_kpi_impact: float
    total_latency_ms: float

    @property
    def threat_count(self) -> int:
        return sum(1 for s in self.signals if s.is_threat)

    @property
    def opportunity_count(self) -> int:
        return sum(1 for s in self.signals if s.is_opportunity)

    @property
    def top_action(self) -> Optional[CounterAction]:
        return self.counter_actions[0] if self.counter_actions else None


# ---------------------------------------------------------------------------
# 季節スコア計算
# ---------------------------------------------------------------------------

# 月ごとの業界別需要指数 (1=低, 2=中, 3=高) — SEO/マーケ向け
_SEASONAL_INDEX: Dict[int, float] = {
    1: 0.55,  # 1月: 年始で低め
    2: 0.60,
    3: 0.75,  # Q1末: 予算消化
    4: 0.70,
    5: 0.65,
    6: 0.70,
    7: 0.50,  # 夏季: 低め
    8: 0.45,
    9: 0.80,  # Q3末: 秋需要増
    10: 0.85,
    11: 0.90,  # 年末商戦
    12: 0.95,
}


def _seasonal_strength(month: int) -> float:
    """月から季節シグナル強度を返す (0.0〜1.0)。"""
    idx = _SEASONAL_INDEX.get(month, 0.65)
    return round(idx, 3)


def _seasonal_direction(month: int) -> str:
    """月によってポジティブ/ネガティブを判定。"""
    idx = _SEASONAL_INDEX.get(month, 0.65)
    return "positive" if idx >= 0.75 else "negative"


# ---------------------------------------------------------------------------
# SignalDetector
# ---------------------------------------------------------------------------


class SignalDetector:
    """
    日付・キーワード・競合情報から外部シグナルを検出する。

    Args
    ----
    competitor_keywords : 競合を示すキーワードリスト
    trend_keywords      : トレンドを示すキーワードリスト
    """

    def __init__(
        self,
        competitor_keywords: Optional[List[str]] = None,
        trend_keywords: Optional[List[str]] = None,
    ) -> None:
        self._competitor_kw = competitor_keywords or [
            "competitor", "rival", "新規参入", "競合", "代替", "シェア低下"
        ]
        self._trend_kw = trend_keywords or [
            "急上昇", "バイラル", "トレンド", "viral", "trending", "急増", "爆発的"
        ]

    def detect(
        self,
        context: str,
        keyword: str = "",
        month: Optional[int] = None,
    ) -> List[ExternalSignal]:
        """
        コンテキスト・キーワード・月から外部シグナルを検出する。

        Args:
            context : 分析対象テキスト
            keyword : 対象キーワード
            month   : 現在月 (1〜12, None=スキップ)

        Returns:
            検出されたシグナルのリスト
        """
        signals: List[ExternalSignal] = []

        # 季節シグナル
        if month is not None:
            strength = _seasonal_strength(month)
            direction = _seasonal_direction(month)
            signals.append(ExternalSignal(
                signal_type="seasonal",
                name=f"{month}月の季節需要変動",
                strength=strength,
                direction=direction,
                source=f"月次需要指数: {strength:.2f}",
            ))

        # トレンドシグナル
        context_lower = context.lower()
        kw_lower = keyword.lower()
        trend_hits = sum(1 for kw in self._trend_kw if kw in context_lower or kw in kw_lower)
        if trend_hits > 0:
            strength = min(trend_hits * 0.25, 1.0)
            signals.append(ExternalSignal(
                signal_type="trend_spike",
                name=f"トレンド急上昇シグナル ({trend_hits}件)",
                strength=strength,
                direction="positive",
                source=f"キーワード一致: {trend_hits}件",
            ))

        # 競合シグナル
        comp_hits = sum(1 for kw in self._competitor_kw if kw in context_lower or kw in kw_lower)
        if comp_hits > 0:
            strength = min(comp_hits * 0.3, 1.0)
            signals.append(ExternalSignal(
                signal_type="competitor",
                name=f"競合動向シグナル ({comp_hits}件)",
                strength=strength,
                direction="negative",
                source=f"競合キーワード一致: {comp_hits}件",
            ))

        # マーケットシグナル (キーワード長と複雑さから推定)
        if keyword:
            market_strength = min(len(keyword) / 20, 0.8)
            signals.append(ExternalSignal(
                signal_type="market",
                name="市場規模シグナル",
                strength=round(market_strength, 3),
                direction="positive" if market_strength > 0.4 else "negative",
                source=f"キーワード複雑度: {market_strength:.2f}",
            ))

        return signals


# ---------------------------------------------------------------------------
# ImpactEstimator
# ---------------------------------------------------------------------------


class ImpactEstimator:
    """
    シグナル強度 → KPI 影響量マッピング。

    シグナル種別・方向・強度から LLMO スコア等の KPI 変化量を推定する。
    """

    # 種別ごとの最大影響係数
    _IMPACT_COEFF: Dict[str, float] = {
        "seasonal": 0.15,
        "trend_spike": 0.20,
        "competitor": 0.25,
        "market": 0.10,
    }

    def estimate(
        self,
        signal: ExternalSignal,
        kpi_name: str = "llmo_score",
    ) -> ImpactEstimate:
        """
        シグナルの KPI への影響を推定する。

        Args:
            signal   : 対象シグナル
            kpi_name : 影響を受ける KPI 名

        Returns:
            ImpactEstimate
        """
        coeff = self._IMPACT_COEFF.get(signal.signal_type, 0.1)
        raw_delta = coeff * signal.strength
        delta = raw_delta if signal.direction == "positive" else -raw_delta
        confidence = 0.5 + 0.3 * signal.strength  # 強度が高いほど信頼度も高い

        if signal.direction == "negative":
            explanation = (
                f"{signal.name} により {kpi_name} が約 {abs(delta):.3f} 低下する見込み。"
            )
        else:
            explanation = (
                f"{signal.name} により {kpi_name} が約 {delta:.3f} 向上する見込み。"
            )

        return ImpactEstimate(
            signal=signal,
            kpi_name=kpi_name,
            impact_delta=round(delta, 4),
            confidence=round(confidence, 3),
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# 組み込み CounterAction テンプレート
# ---------------------------------------------------------------------------


def _counter_seasonal_negative(text: str) -> str:
    return (
        "【季節対策】本シーズンはオフピーク期です。"
        "常緑コンテンツを強化し、次のピーク期に向けた資産を構築しましょう。\n\n" + text
    )


def _counter_seasonal_positive(text: str) -> str:
    return text + "\n\n【季節対策】現在はハイシーズンです。CTA を強化し、コンバージョン最大化を図りましょう。"


def _counter_competitor(text: str) -> str:
    return (
        text
        + "\n\n【競合対策】競合との差別化ポイントを明示します: "
        "① 独自データ・研究に基づく信頼性 ② 無償提供の付加価値 ③ 日本語対応の深さ"
    )


def _counter_trend_spike(text: str) -> str:
    return "【トレンド活用】急上昇ワードを取り込んだコンテンツ強化中:\n\n" + text


def _counter_market_negative(text: str) -> str:
    return text + "\n\n【市場対策】市場縮小シグナルを検知。ニッチ特化・Long-tail SEO 戦略を推奨します。"


def _counter_market_positive(text: str) -> str:
    return text + "\n\n【市場拡大対策】市場拡大シグナルを検知。広告投資増加・キーワード拡張を推奨します。"


_COUNTER_ACTION_TEMPLATES: List[CounterAction] = [
    CounterAction(
        action_id="seasonal_negative",
        signal_type="seasonal",
        description="オフピーク期: 常緑コンテンツ強化でベースライン維持",
        transform_fn=_counter_seasonal_negative,
        estimated_kpi_recovery=0.08,
        priority=1,
    ),
    CounterAction(
        action_id="seasonal_positive",
        signal_type="seasonal",
        description="ハイシーズン: CTA強化でコンバージョン最大化",
        transform_fn=_counter_seasonal_positive,
        estimated_kpi_recovery=0.12,
        priority=2,
    ),
    CounterAction(
        action_id="competitor_counter",
        signal_type="competitor",
        description="競合対策: 差別化ポイントを明示してブランド強化",
        transform_fn=_counter_competitor,
        estimated_kpi_recovery=0.15,
        priority=0,
    ),
    CounterAction(
        action_id="trend_spike_boost",
        signal_type="trend_spike",
        description="トレンド活用: 急上昇ワードをコンテンツに統合",
        transform_fn=_counter_trend_spike,
        estimated_kpi_recovery=0.18,
        priority=1,
    ),
    CounterAction(
        action_id="market_negative",
        signal_type="market",
        description="市場縮小対策: Long-tail SEO でニッチ特化",
        transform_fn=_counter_market_negative,
        estimated_kpi_recovery=0.06,
        priority=3,
    ),
    CounterAction(
        action_id="market_positive",
        signal_type="market",
        description="市場拡大活用: キーワード拡張・広告投資増加",
        transform_fn=_counter_market_positive,
        estimated_kpi_recovery=0.10,
        priority=2,
    ),
]


# ---------------------------------------------------------------------------
# ExternalSignalAgent
# ---------------------------------------------------------------------------


class ExternalSignalAgent:
    """
    外部シグナルを検出し、内部アクションで対応するエージェント。

    Args
    ----
    detector  : SignalDetector (省略時はデフォルト)
    estimator : ImpactEstimator (省略時はデフォルト)
    """

    def __init__(
        self,
        detector: Optional[SignalDetector] = None,
        estimator: Optional[ImpactEstimator] = None,
    ) -> None:
        self._detector = detector or SignalDetector()
        self._estimator = estimator or ImpactEstimator()

    def run(
        self,
        context: str,
        keyword: str = "",
        month: Optional[int] = None,
        kpi_name: str = "llmo_score",
    ) -> ExternalSignalResult:
        """
        detect → estimate → counter_action を一括実行する。

        Args:
            context  : 分析対象コンテキスト
            keyword  : 対象キーワード
            month    : 現在月 (1〜12)
            kpi_name : 影響を推定する KPI 名

        Returns:
            ExternalSignalResult
        """
        t_start = time.perf_counter()

        signals = self._detector.detect(context, keyword=keyword, month=month)
        impacts = [self._estimator.estimate(s, kpi_name) for s in signals]
        counter_actions = self._select_counter_actions(signals)

        # アクションをコンテキストに適用
        optimized = context
        for action in counter_actions:
            optimized = action.apply(optimized)

        net_kpi_impact = sum(imp.impact_delta for imp in impacts)

        total_ms = (time.perf_counter() - t_start) * 1000

        return ExternalSignalResult(
            keyword=keyword,
            signals=signals,
            impacts=impacts,
            counter_actions=counter_actions,
            optimized_context=optimized,
            net_kpi_impact=round(net_kpi_impact, 4),
            total_latency_ms=round(total_ms, 2),
        )

    def _select_counter_actions(self, signals: List[ExternalSignal]) -> List[CounterAction]:
        """シグナルに対応するアクションを選択し priority 昇順で返す。"""
        selected: List[CounterAction] = []
        for signal in signals:
            for template in _COUNTER_ACTION_TEMPLATES:
                if template.signal_type != signal.signal_type:
                    continue
                # negative シグナル → negative 系アクション、positive → positive 系
                if signal.is_threat and "positive" in template.action_id:
                    continue
                if signal.is_opportunity and "negative" in template.action_id:
                    continue
                selected.append(template)
                break  # 1シグナルにつき1アクション

        return sorted(selected, key=lambda a: a.priority)

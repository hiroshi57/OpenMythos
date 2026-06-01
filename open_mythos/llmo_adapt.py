"""
llmo_adapt.py — Living LLMO 環境・自律学習 (P13〜P20) + 自動トリガー

P13: Domain Specialization      — ドメイン別に最適化パラメータを蓄積
P14: Temporal Decay             — 古いパターンの重みを指数減衰させて陳腐化を防ぐ
P15: Trend Adaptation           — 新コンテンツから entity 信頼度を動的更新
P16: Regret Minimization        — 過去の「別の選択肢が良かった」を後から学習
P17: Adaptive Target Calibration— intent×domain 別の target_score を自動調整
P18: Self-Benchmark             — 内蔵テストケースで定期的に自己評価
P19: Growth Cycle Scheduler     — フィードバック数/時間/ドリフトで成長タイミングを自動判定
P20: Meta-Learning              — どのパターンが最も効果的かを学習し、学習順序を自律最適化

自動トリガー:
  FeedbackCountTrigger — N 件溜まったら発火
  TimerTrigger         — 指定時間間隔で発火
  ScoreDriftTrigger    — Self-Benchmark でドリフト検出時に発火
  CompositeTrigger     — AND / OR の組み合わせ
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# ===========================================================================
# P13: Domain Specialization — ドメイン別プロファイル
# ===========================================================================


class DomainSpecializer:
    """
    ドメイン (例: marketing / tech / medical / general) ごとに
    最適化パラメータ (entity_weight / directness_weight / citability_weight) を
    個別に蓄積・更新する。(P13)

    Usage::

        ds = DomainSpecializer()
        ds.update("marketing", {"entity_weight": 0.3, "directness_weight": 0.4}, score=0.78)
        params = ds.get_params("marketing")
    """

    _DEFAULT_PARAMS: dict[str, float] = {
        "entity_weight": 0.30,
        "directness_weight": 0.40,
        "citability_weight": 0.30,
        "target_score": 0.75,
    }

    def __init__(self) -> None:
        # domain → {param_name → (weighted_sum, total_count)}
        self._profiles: dict[str, dict[str, list[float]]] = {}

    def update(self, domain: str, params: dict[str, float], score: float) -> None:
        """
        ドメインのパラメータをスコアで重み付けして更新する。
        score が高いほどこのパラメータセットの影響を大きくする。
        """
        if domain not in self._profiles:
            self._profiles[domain] = {k: [0.0, 0.0] for k in self._DEFAULT_PARAMS}

        for key, val in params.items():
            if key in self._profiles[domain]:
                self._profiles[domain][key][0] += val * score  # weighted sum
                self._profiles[domain][key][1] += score  # weight sum

    def get_params(self, domain: str) -> dict[str, float]:
        """
        ドメインの学習済みパラメータを返す。
        データがなければデフォルト値を返す。
        """
        if domain not in self._profiles:
            return dict(self._DEFAULT_PARAMS)

        result = {}
        for key, (wsum, wcount) in self._profiles[domain].items():
            result[key] = (
                round(wsum / wcount, 4) if wcount > 0 else self._DEFAULT_PARAMS[key]
            )
        return result

    def known_domains(self) -> list[str]:
        """学習済みドメインの一覧を返す。"""
        return list(self._profiles.keys())

    def to_dict(self) -> dict[str, Any]:
        return {"profiles": self._profiles}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DomainSpecializer":
        obj = cls()
        obj._profiles = d.get("profiles", {})
        return obj


# ===========================================================================
# P14: Temporal Decay — 古いパターンの重みを指数減衰
# ===========================================================================


class TemporalDecay:
    """
    PatternMiner 等の各レコードに対して時間経過に応じた指数減衰を適用する。
    古くなった（信頼性が下がった）パターンを自動的に排除する。(P14)

    Usage::

        td = TemporalDecay(decay_rate=0.95)
        records = {"add_structure": 1.0, "expand_content": 0.3}
        decayed = td.apply_decay(records)
        pruned = td.prune_stale(decayed, threshold=0.1)
    """

    def __init__(self, decay_rate: float = 0.95) -> None:
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}")
        self._decay_rate = decay_rate

    def apply_decay(self, records: dict[str, float]) -> dict[str, float]:
        """
        各レコードの値に decay_rate を乗算して返す。
        元の辞書は変更しない（コピーを返す）。
        """
        return {k: round(v * self._decay_rate, 6) for k, v in records.items()}

    def prune_stale(
        self, records: dict[str, float], threshold: float = 0.10
    ) -> dict[str, float]:
        """
        threshold 未満の値を持つエントリを除去して返す。
        元の辞書は変更しない。
        """
        return {k: v for k, v in records.items() if v >= threshold}

    @property
    def decay_rate(self) -> float:
        return self._decay_rate


# ===========================================================================
# P15: Trend Adaptation — entity 信頼度を動的更新
# ===========================================================================


class TrendAdapter:
    """
    新しく届くテキスト群から新キーワードを検出し、
    EntityKnowledgeBase の信頼度スコアを動的に更新する。(P15)

    Usage::

        ta = TrendAdapter()
        updated = ta.update_from_texts(entity_kb, ["AI is transforming marketing..."])
    """

    def update_from_texts(
        self, entity_kb: Any, texts: list[str], boost: float = 0.05
    ) -> int:
        """
        テキストリスト内に entity_kb の登録語が出現したら
        その信頼度を boost 分だけ引き上げる。
        更新されたエントリ数を返す。
        """
        if not texts or not hasattr(entity_kb, "_entities"):
            return 0

        updated = 0
        combined = " ".join(texts).lower()
        for word in list(entity_kb._entities.keys()):
            if word in combined:
                old = entity_kb._entities[word]
                entity_kb._entities[word] = round(min(old + boost, 1.0), 4)
                updated += 1
        return updated


# ===========================================================================
# P16: Regret Minimization — 後知恵学習
# ===========================================================================


@dataclass
class RegretRecord:
    """後知恵学習の1レコード。"""

    text_hash: str
    applied_transformation: str
    applied_delta: float
    alternative_transformation: str
    estimated_alternative_delta: float

    @property
    def regret(self) -> float:
        """逃した改善量（後悔量）を返す。"""
        return max(self.estimated_alternative_delta - self.applied_delta, 0.0)


class RegretMinimizer:
    """
    過去の最適化結果を遡り、「別の変換を選んでいればもっと良かった」
    ケースを推定して記録し、次の選択に反映させる。(P16)

    Usage::

        rm = RegretMinimizer()
        rm.observe("abc12345", "add_structure", 0.08, {"citation_cues": 0.15})
        lost = rm.total_regret()
    """

    def __init__(self) -> None:
        self._records: list[RegretRecord] = []

    def observe(
        self,
        text_hash: str,
        applied: str,
        applied_delta: float,
        alternatives: dict[str, float],
    ) -> None:
        """
        実際に適用した変換と、その他の候補の推定スコアを記録する。
        alternatives: {変換名 → 推定スコア改善量}
        """
        for alt, est_delta in alternatives.items():
            if alt == applied:
                continue
            if est_delta > applied_delta:
                self._records.append(
                    RegretRecord(
                        text_hash=text_hash,
                        applied_transformation=applied,
                        applied_delta=applied_delta,
                        alternative_transformation=alt,
                        estimated_alternative_delta=est_delta,
                    )
                )

    def get_regrettable_transformations(self) -> list[str]:
        """後悔量が大きい「採用したが良くなかった」変換を返す。"""
        regrets: dict[str, float] = {}
        for r in self._records:
            regrets[r.applied_transformation] = (
                regrets.get(r.applied_transformation, 0.0) + r.regret
            )
        return sorted(regrets, key=lambda k: regrets[k], reverse=True)

    def total_regret(self) -> float:
        """全レコードの後悔量の合計を返す。"""
        return round(sum(r.regret for r in self._records), 4)

    def record_count(self) -> int:
        return len(self._records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [
                {
                    "text_hash": r.text_hash,
                    "applied_transformation": r.applied_transformation,
                    "applied_delta": r.applied_delta,
                    "alternative_transformation": r.alternative_transformation,
                    "estimated_alternative_delta": r.estimated_alternative_delta,
                }
                for r in self._records
            ]
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RegretMinimizer":
        obj = cls()
        for rd in d.get("records", []):
            obj._records.append(RegretRecord(**rd))
        return obj


# ===========================================================================
# P17: Adaptive Target Calibration — intent×domain 別 target_score 自動調整
# ===========================================================================


class AdaptiveTargetCalibrator:
    """
    intent_type × domain の組み合わせごとに、過去に実際に達成できた
    スコアの平均を追跡し、target_score を自動調整する。(P17)

    Usage::

        cal = AdaptiveTargetCalibrator()
        cal.update("informational", "tech", achieved_score=0.78)
        target = cal.get_target("informational", "tech")
    """

    def __init__(self, default_target: float = 0.75) -> None:
        self._default = default_target
        # key: "intent:domain" → [sum, count]
        self._calibration: dict[str, list[float]] = {}

    def update(self, intent: str, domain: str, achieved_score: float) -> None:
        """達成スコアを記録して平均を更新する。"""
        key = f"{intent}:{domain}"
        if key not in self._calibration:
            self._calibration[key] = [0.0, 0.0]
        self._calibration[key][0] += achieved_score
        self._calibration[key][1] += 1.0

    def get_target(
        self, intent: str, domain: str, *, default: float | None = None
    ) -> float:
        """
        過去の達成スコア平均に基づく target_score を返す。
        データがなければ default または初期化時の default_target を返す。
        """
        key = f"{intent}:{domain}"
        if key not in self._calibration or self._calibration[key][1] == 0:
            return default if default is not None else self._default
        avg = self._calibration[key][0] / self._calibration[key][1]
        # 達成平均の 105% を目標として設定（無理なく少し上を狙う）
        return round(min(avg * 1.05, 1.0), 4)

    def known_keys(self) -> list[str]:
        """学習済みの intent:domain キー一覧を返す。"""
        return list(self._calibration.keys())

    def to_dict(self) -> dict[str, Any]:
        return {"default": self._default, "calibration": self._calibration}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AdaptiveTargetCalibrator":
        obj = cls(default_target=d.get("default", 0.75))
        obj._calibration = d.get("calibration", {})
        return obj


# ===========================================================================
# P18: Self-Benchmark — 内蔵テストケースで定期的に自己評価
# ===========================================================================

# 内蔵テストケース (テキスト, 期待スコア帯)
_SELF_BENCHMARK_CASES = [
    ("OpenAI released GPT-4 in 2023. It supports multimodal inputs.", 0.45),
    ("The quick brown fox jumps over the lazy dog.", 0.20),
    (
        "## Overview\nAccording to research by Stanford AI Lab, transformer models achieve "
        "state-of-the-art results on NLP benchmarks.\n\n## Details\nThe study used 50,000 "
        "samples. Results show 92% accuracy.\n\n## Summary\nTransformers outperform RNNs.",
        0.70,
    ),
    ("Buy now! Limited offer. Click here.", 0.30),
    (
        "Google DeepMind published AlphaFold 2 results in Nature. "
        "The model predicts protein structures with atomic accuracy.",
        0.55,
    ),
]


class SelfBenchmark:
    """
    内蔵テストケース (5件) に対してスコアラーを実行し、
    スコア基準のドリフト（時系列的なずれ）を検出・補正する。(P18)

    Usage::

        sb = SelfBenchmark()
        result = sb.run(scorer)
        if sb.detect_drift(baseline_result, result):
            print("drift detected")
    """

    def run(self, scorer: Any) -> dict[str, Any]:
        """
        全内蔵ケースに対してスコアリングを実行し結果を返す。

        Returns::

            {
                "avg_score": 0.44,
                "scores": [0.45, 0.21, 0.72, ...],
                "timestamp": 1748780000.0,
            }
        """
        scores = []
        for text, _ in _SELF_BENCHMARK_CASES:
            try:
                result = scorer.score(text)
                scores.append(result.llmo_total)
            except Exception:
                scores.append(0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "avg_score": round(avg, 4),
            "scores": [round(s, 4) for s in scores],
            "case_count": len(scores),
            "timestamp": time.time(),
        }

    def detect_drift(
        self, baseline: dict[str, Any], current: dict[str, Any], threshold: float = 0.05
    ) -> bool:
        """
        baseline と current の avg_score 差が threshold を超えれば True を返す。
        どちらかが None / 空なら False。
        """
        if not baseline or not current:
            return False
        b = baseline.get("avg_score", 0.0)
        c = current.get("avg_score", 0.0)
        return abs(c - b) >= threshold


# ===========================================================================
# 自動トリガー (P19 に連携)
# ===========================================================================


class GrowthTrigger(ABC):
    """自動トリガーの基底クラス。"""

    @abstractmethod
    def should_fire(self, context: dict[str, Any]) -> bool:
        """トリガー条件を満たせば True を返す。"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """トリガーの識別名。"""
        ...


class FeedbackCountTrigger(GrowthTrigger):
    """
    フィードバック件数が threshold を超えたら発火する。(P19)

    context["feedback_count"] を参照する。
    発火するたびに _last_fired_count を更新し、
    次の threshold 件が溜まるまで再発火しない。
    """

    def __init__(self, threshold: int = 20) -> None:
        self._threshold = threshold
        self._last_fired_count: int = 0

    def should_fire(self, context: dict[str, Any]) -> bool:
        count = context.get("feedback_count", 0)
        if count - self._last_fired_count >= self._threshold:
            self._last_fired_count = count
            return True
        return False

    @property
    def name(self) -> str:
        return f"FeedbackCountTrigger(threshold={self._threshold})"


class TimerTrigger(GrowthTrigger):
    """
    前回の発火から interval_hours 時間経過したら発火する。(P19)

    context["current_time"] を参照する（デフォルトは time.time()）。
    """

    def __init__(self, interval_hours: float = 24.0) -> None:
        self._interval = interval_hours * 3600.0
        self._last_fired: float = 0.0

    def should_fire(self, context: dict[str, Any]) -> bool:
        now = context.get("current_time", time.time())
        if now - self._last_fired >= self._interval:
            self._last_fired = now
            return True
        return False

    @property
    def name(self) -> str:
        return f"TimerTrigger(interval={self._interval/3600:.1f}h)"


class ScoreDriftTrigger(GrowthTrigger):
    """
    Self-Benchmark の avg_score が baseline から threshold 以上ずれたら発火。(P19)

    context["score_drift"] を参照する（SelfBenchmark が計算して渡す）。
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self._threshold = threshold

    def should_fire(self, context: dict[str, Any]) -> bool:
        drift = context.get("score_drift", 0.0)
        return abs(drift) >= self._threshold

    @property
    def name(self) -> str:
        return f"ScoreDriftTrigger(threshold={self._threshold})"


class CompositeTrigger(GrowthTrigger):
    """
    複数トリガーを AND / OR で組み合わせる。(P19)

    mode="any"  → いずれか1つが発火すれば発火（OR）
    mode="all"  → 全てが発火して初めて発火（AND）
    """

    def __init__(self, triggers: list[GrowthTrigger], mode: str = "any") -> None:
        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got {mode!r}")
        self._triggers = triggers
        self._mode = mode

    def should_fire(self, context: dict[str, Any]) -> bool:
        results = [t.should_fire(context) for t in self._triggers]
        if self._mode == "any":
            return any(results)
        return all(results)

    @property
    def name(self) -> str:
        names = ", ".join(t.name for t in self._triggers)
        return f"CompositeTrigger({self._mode}: [{names}])"


# ===========================================================================
# P19: Growth Cycle Scheduler — 成長タイミングを自動判定
# ===========================================================================


class GrowthCycleScheduler:
    """
    登録されたトリガー群を監視し、いずれかが発火したら
    GrowthCycle の実行を推奨する。(P19)

    Usage::

        scheduler = GrowthCycleScheduler([
            FeedbackCountTrigger(threshold=20),
            TimerTrigger(interval_hours=24),
        ])
        ctx = {"feedback_count": 25, "current_time": time.time()}
        if scheduler.check(ctx):
            growth_cycle.run()
    """

    def __init__(self, triggers: list[GrowthTrigger] | None = None) -> None:
        self._triggers: list[GrowthTrigger] = triggers or [
            FeedbackCountTrigger(threshold=20),
            TimerTrigger(interval_hours=24.0),
            ScoreDriftTrigger(threshold=0.05),
        ]
        self._cycle_count: int = 0
        self._last_cycle_time: float = 0.0
        self._fired_trigger_names: list[str] = []

    def check(self, context: dict[str, Any]) -> bool:
        """
        いずれかのトリガーが発火すれば True を返す。
        発火したトリガー名を fired_trigger_names に記録する。
        """
        self._fired_trigger_names = []
        fired = False
        for trigger in self._triggers:
            if trigger.should_fire(context):
                self._fired_trigger_names.append(trigger.name)
                fired = True
        return fired

    def record_cycle(self) -> None:
        """Growth Cycle が実行されたことを記録する。"""
        self._cycle_count += 1
        self._last_cycle_time = time.time()

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_cycle_time(self) -> float:
        return self._last_cycle_time

    @property
    def fired_trigger_names(self) -> list[str]:
        return list(self._fired_trigger_names)


# ===========================================================================
# P20: Meta-Learning — 学習戦略そのものを学習
# ===========================================================================


class MetaLearner:
    """
    どの成長パターン (P1〜P19) が最も効果的（= スコア改善量が大きい）かを追跡し、
    Growth Cycle での実行順序を自律最適化する。(P20)

    Usage::

        ml = MetaLearner()
        ml.observe("PatternMiner", before_score=0.52, after_score=0.61)
        order = ml.get_priority_order()
    """

    # 全パターン名（デフォルト実行順序）
    ALL_PATTERNS = [
        "FeedbackAnalyzer",  # P1-P4
        "RejectionMemory",  # P5
        "FailureMemory",  # P6
        "LoopEscapeMemory",  # P7
        "AntiPatternRegistry",  # P8
        "PatternMiner",  # P9
        "EntityKnowledgeBase",  # P10
        "TemplateLibrary",  # P11
        "ChampionPromoter",  # P12
        "DomainSpecializer",  # P13
        "TemporalDecay",  # P14
        "TrendAdapter",  # P15
        "RegretMinimizer",  # P16
        "AdaptiveTargetCalibrator",  # P17
        "SelfBenchmark",  # P18
        "GrowthCycleScheduler",  # P19
    ]

    def __init__(self) -> None:
        # pattern_name → [total_improvement, observation_count]
        self._effectiveness: dict[str, list[float]] = {
            p: [0.0, 0.0] for p in self.ALL_PATTERNS
        }

    def observe(
        self, pattern_name: str, before_score: float, after_score: float
    ) -> None:
        """パターン適用前後のスコアを記録する。"""
        delta = max(after_score - before_score, 0.0)
        if pattern_name in self._effectiveness:
            self._effectiveness[pattern_name][0] += delta
            self._effectiveness[pattern_name][1] += 1.0

    def get_priority_order(self) -> list[str]:
        """
        学習速度（平均スコア改善量）が高い順にパターン名を並べて返す。
        観測データがないパターンはデフォルト順序の末尾に置く。
        """

        def sort_key(p: str) -> float:
            total, count = self._effectiveness.get(p, [0.0, 0.0])
            return total / count if count > 0 else -1.0

        observed = [
            p
            for p in self.ALL_PATTERNS
            if self._effectiveness.get(p, [0.0, 1.0])[1] > 0
        ]
        unobserved = [p for p in self.ALL_PATTERNS if p not in observed]
        return sorted(observed, key=sort_key, reverse=True) + unobserved

    def get_effectiveness(self, pattern_name: str) -> float:
        """指定パターンの平均スコア改善量を返す。"""
        total, count = self._effectiveness.get(pattern_name, [0.0, 0.0])
        return round(total / count, 4) if count > 0 else 0.0

    def summary(self) -> list[dict[str, Any]]:
        """全パターンの効果量サマリーを効果量降順で返す。"""
        rows = []
        for p in self.get_priority_order():
            rows.append(
                {
                    "pattern": p,
                    "avg_improvement": self.get_effectiveness(p),
                    "observations": int(self._effectiveness.get(p, [0.0, 0.0])[1]),
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {"effectiveness": self._effectiveness}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetaLearner":
        obj = cls()
        for k, v in d.get("effectiveness", {}).items():
            obj._effectiveness[k] = v
        return obj


# ===========================================================================
# AdaptStore — P13〜P20 を一括管理するコンテナ
# ===========================================================================


class AdaptStore:
    """
    環境・自律学習の 8 パターン (P13〜P20) を一括保持するコンテナ。
    """

    def __init__(self) -> None:
        self.domain_specializer = DomainSpecializer()
        self.temporal_decay = TemporalDecay()
        self.trend_adapter = TrendAdapter()
        self.regret_minimizer = RegretMinimizer()
        self.adaptive_target = AdaptiveTargetCalibrator()
        self.self_benchmark = SelfBenchmark()
        self.scheduler = GrowthCycleScheduler()
        self.meta_learner = MetaLearner()
        # Self-Benchmark の基準スナップショット（初回実行時に設定）
        self.benchmark_baseline: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_specializer": self.domain_specializer.to_dict(),
            "regret_minimizer": self.regret_minimizer.to_dict(),
            "adaptive_target": self.adaptive_target.to_dict(),
            "meta_learner": self.meta_learner.to_dict(),
            "benchmark_baseline": self.benchmark_baseline,
            "scheduler_cycle_count": self.scheduler.cycle_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AdaptStore":
        obj = cls()
        obj.domain_specializer = DomainSpecializer.from_dict(
            d.get("domain_specializer", {})
        )
        obj.regret_minimizer = RegretMinimizer.from_dict(d.get("regret_minimizer", {}))
        obj.adaptive_target = AdaptiveTargetCalibrator.from_dict(
            d.get("adaptive_target", {})
        )
        obj.meta_learner = MetaLearner.from_dict(d.get("meta_learner", {}))
        obj.benchmark_baseline = d.get("benchmark_baseline", {})
        return obj

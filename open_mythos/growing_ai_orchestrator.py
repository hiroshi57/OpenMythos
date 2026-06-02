"""
Sprint 30 — P1〜P10 統合オーケストレーター
GrowingAIOrchestrator: ゴールと文脈から最適なパターンを自動選択・実行する統合エンジン

Patterns:
    P1  debate          討議型集合知 (DebateOrchestrator)
    P2  kpi             KPI 駆動自己改善 (KPIAgent)
    P3  profiler        ボトルネック発見・解消 (ProfilerAgent)
    P4  signal          外部要因適応 (ExternalSignalAgent)
    P5  mistake         ミスから学習 (MistakeGuard)
    P6  distill         継続的自己蒸留 (SelfDistillLoop)
    P7  memory          長期記憶統合 (LongTermMemoryAgent)
    P8  ensemble        アンサンブル品質評価 (EnsembleScorer)
    P9  evolve          適応型プロンプト進化 (PromptEvolution)
    P10 plan            自律タスク計画 (TaskPlanner)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pattern enum
# ---------------------------------------------------------------------------

class PatternType(str, Enum):
    """P1〜P10 の 10 パターン識別子"""
    DEBATE   = "debate"    # P1
    KPI      = "kpi"       # P2
    PROFILER = "profiler"  # P3
    SIGNAL   = "signal"    # P4
    MISTAKE  = "mistake"   # P5
    DISTILL  = "distill"   # P6
    MEMORY   = "memory"    # P7
    ENSEMBLE = "ensemble"  # P8
    EVOLVE   = "evolve"    # P9
    PLAN     = "plan"      # P10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GrowthContext:
    """オーケストレーターに渡すコンテキスト"""
    goal: str
    hints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PatternResult:
    """1 パターンの実行結果"""
    pattern: PatternType
    output: Any
    score: float                      # 0.0 〜 1.0
    latency_ms: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class OrchestratorResult:
    """GrowingAIOrchestrator.run() の最終結果"""
    goal: str
    patterns_used: List[PatternType]
    results: List[PatternResult]
    final_output: str
    overall_score: float              # 0.0 〜 1.0
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Pattern Selector
# ---------------------------------------------------------------------------

_PATTERN_KEYWORDS: Dict[PatternType, List[str]] = {
    PatternType.DEBATE:   [
        "議論", "意見", "compare", "debate", "pros", "cons", "対比", "検討",
        "どちら", "比較", "which", "versus", "vs",
    ],
    PatternType.KPI:      [
        "改善", "最適化", "optimize", "kpi", "metric", "performance", "向上",
        "目標", "target", "指標", "達成",
    ],
    PatternType.PROFILER: [
        "遅い", "ボトルネック", "bottleneck", "slow", "latency", "速度",
        "パフォーマンス", "profile", "プロファイル", "チューニング",
    ],
    PatternType.SIGNAL:   [
        "外部", "市場", "trend", "signal", "environment", "市況",
        "トレンド", "競合", "external", "market", "情勢",
    ],
    PatternType.MISTAKE:  [
        "エラー", "ミス", "error", "mistake", "prevent", "guard",
        "失敗", "防止", "リスク", "risk", "回避",
    ],
    PatternType.DISTILL:  [
        "蒸留", "要約", "distill", "compress", "summarize", "要点",
        "精錬", "refine", "洗練", "品質向上",
    ],
    PatternType.MEMORY:   [
        "記憶", "履歴", "memory", "history", "past", "context",
        "過去", "recall", "思い出", "参照",
    ],
    PatternType.ENSEMBLE: [
        "評価", "スコア", "score", "evaluate", "quality", "品質評価",
        "採点", "rating", "判定", "assess",
    ],
    PatternType.EVOLVE:   [
        "プロンプト", "prompt", "evolve", "進化", "改良",
        "最適化プロンプト", "世代", "generation", "genetic",
    ],
    PatternType.PLAN:     [
        "計画", "タスク", "plan", "task", "decompose", "分解", "実行",
        "ステップ", "step", "workflow", "手順", "フロー",
    ],
}


class PatternSelector:
    """ゴール文字列と hints からどのパターンを適用するかを選択する"""

    def __init__(self, keywords: Optional[Dict[PatternType, List[str]]] = None) -> None:
        self._keywords = keywords or _PATTERN_KEYWORDS

    def score_all(self, context: GrowthContext) -> Dict[PatternType, int]:
        """全パターンのスコアを返す"""
        goal_lower = context.goal.lower()
        hint_text  = " ".join(context.hints).lower()

        scores: Dict[PatternType, int] = {}
        for pattern, keywords in self._keywords.items():
            s = sum(1 for kw in keywords if kw in goal_lower)
            s += sum(1 for kw in keywords if kw in hint_text)
            if s > 0:
                scores[pattern] = s
        return scores

    def select(
        self,
        context: GrowthContext,
        max_patterns: int = 3,
    ) -> List[PatternType]:
        """スコア上位 max_patterns のパターンを返す（最低 1 件保証）"""
        scores = self.score_all(context)
        if not scores:
            # デフォルト: 品質評価 → タスク計画
            return [PatternType.ENSEMBLE, PatternType.PLAN][:max_patterns]

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_items[:max_patterns]]


# ---------------------------------------------------------------------------
# GrowingAIOrchestrator
# ---------------------------------------------------------------------------

class GrowingAIOrchestrator:
    """
    P1〜P10 を統合するオーケストレーター。

    run(goal) でゴールを受け取り、PatternSelector が最適パターンを選び、
    各パターンを順次実行して結果を統合する。

    Parameters
    ----------
    max_patterns : int
        1 回の run() で起動するパターン上限 (default: 3)
    timeout_s : float
        1 パターンあたりのタイムアウト秒 (default: 30.0)
    selector : PatternSelector | None
        カスタムセレクタ（None で標準セレクタを使用）
    """

    def __init__(
        self,
        max_patterns: int = 3,
        timeout_s: float = 30.0,
        selector: Optional[PatternSelector] = None,
    ) -> None:
        self.max_patterns = max_patterns
        self.timeout_s    = timeout_s
        self._selector    = selector or PatternSelector()
        self._log: List[OrchestratorResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        goal: str,
        hints: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """
        ゴールに対してパターンを自動選択・実行し OrchestratorResult を返す。

        Parameters
        ----------
        goal : str
            達成したい目標・質問・タスク記述
        hints : list[str] | None
            パターン選択を補助するヒントキーワード群
        metadata : dict | None
            任意の付加情報（各パターンには渡さない）
        """
        t_total = time.perf_counter()
        ctx = GrowthContext(
            goal=goal,
            hints=hints or [],
            metadata=metadata or {},
        )

        patterns = self._selector.select(ctx, self.max_patterns)
        results  = self._execute_patterns(goal, patterns, ctx)

        final_output  = self._synthesize(goal, results)
        overall_score = self._aggregate_score(results)
        total_ms      = round((time.perf_counter() - t_total) * 1000, 2)

        orch_result = OrchestratorResult(
            goal=goal,
            patterns_used=patterns,
            results=results,
            final_output=final_output,
            overall_score=overall_score,
            total_latency_ms=total_ms,
        )
        self._log.append(orch_result)
        return orch_result

    def history(self) -> List[OrchestratorResult]:
        """これまでの run() 結果リストを返す"""
        return list(self._log)

    def clear_history(self) -> None:
        """実行ログをリセットする"""
        self._log.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_patterns(
        self,
        goal: str,
        patterns: List[PatternType],
        ctx: GrowthContext,
    ) -> List[PatternResult]:
        results: List[PatternResult] = []
        for pattern in patterns:
            t = time.perf_counter()
            try:
                output, score = self._run_single_pattern(pattern, goal, ctx)
                error = None
            except Exception as exc:  # noqa: BLE001
                output = ""
                score  = 0.0
                error  = str(exc)
            ms = round((time.perf_counter() - t) * 1000, 2)
            results.append(
                PatternResult(
                    pattern=pattern,
                    output=output,
                    score=float(max(0.0, min(1.0, score))),
                    latency_ms=ms,
                    error=error,
                )
            )
        return results

    def _run_single_pattern(
        self,
        pattern: PatternType,
        goal: str,
        ctx: GrowthContext,
    ) -> tuple[Any, float]:
        """パターン別の実行ロジック。(output, score) を返す"""

        if pattern == PatternType.DEBATE:
            from open_mythos.debate import DebateOrchestrator
            orch   = DebateOrchestrator(n_agents=2, n_rounds=1)
            result = orch.run(goal)
            return result.consensus, result.agreement_score

        if pattern == PatternType.KPI:
            from open_mythos.kpi_agent import KPIAgent, KPIDefinition
            kpi    = KPIDefinition(name="quality", current=0.5, target=0.8, weight=1.0)
            agent  = KPIAgent([kpi])
            snap   = agent.snapshot()
            return f"KPI snapshot: overall={snap.overall_score:.3f}", snap.overall_score

        if pattern == PatternType.PROFILER:
            from open_mythos.profiler import ProfilerAgent
            agent  = ProfilerAgent()
            prof   = agent.profile(goal)
            report = agent.detect(prof)
            bn     = report.bottleneck_stage or "none"
            score  = 0.5 if report.has_bottleneck else 1.0
            return f"Bottleneck detected: {bn}", score

        if pattern == PatternType.SIGNAL:
            from open_mythos.external_signal import ExternalSignalAgent
            agent   = ExternalSignalAgent()
            summary = agent.summarize()
            score   = max(0.0, min(1.0, 0.5 + summary.net_sentiment * 0.5))
            return summary.recommendation, score

        if pattern == PatternType.MISTAKE:
            from open_mythos.error_memory import MistakeGuard
            guard   = MistakeGuard()
            matched = guard.check(goal)
            if matched:
                return f"Mistake guard: {matched.rule_name}", 0.3
            return "No known mistake patterns detected", 0.9

        if pattern == PatternType.DISTILL:
            from open_mythos.self_distill import SelfDistillLoop
            loop   = SelfDistillLoop()
            result = loop.run([goal], n_iterations=1)   # Sprint 31: List[str] + n_iterations
            best   = result.best_output
            if best:
                return best.output, float(best.score)   # .text → .output (DistillSample)
            return goal, 0.5

        if pattern == PatternType.MEMORY:
            from open_mythos.long_term_memory import LongTermMemoryAgent
            agent   = LongTermMemoryAgent()
            entries = agent.retrieve(goal, top_k=3)
            if entries:
                texts = "; ".join(e.text[:40] for e in entries[:3])
                return f"Memory ({len(entries)} entries): {texts}", 0.8
            return "No relevant memories found", 0.5

        if pattern == PatternType.ENSEMBLE:
            from open_mythos.ensemble_scorer import EnsembleScorer
            scorer = EnsembleScorer()
            result = scorer.score(goal)
            return f"Ensemble quality: {result.final_score:.3f}", result.final_score

        if pattern == PatternType.EVOLVE:
            from open_mythos.prompt_evolution import PromptEvolution, EvolutionConfig
            cfg    = EvolutionConfig(population_size=4, n_generations=2)
            evo    = PromptEvolution(config=cfg)
            result = evo.evolve(goal)
            best   = result.best_gene
            if best:
                return best.text, float(best.fitness)
            return goal, 0.5

        if pattern == PatternType.PLAN:
            from open_mythos.task_planner import TaskPlanner
            planner = TaskPlanner()
            plan    = planner.decompose(goal)
            n       = len(plan.tasks)
            score   = min(1.0, n * 0.2) if n else 0.3
            return f"Plan: {n} tasks decomposed", score

        # フォールバック (到達不能だが安全のため)
        return goal, 0.5

    @staticmethod
    def _synthesize(goal: str, results: List[PatternResult]) -> str:
        """成功した結果の中から最高スコアの output を返す。全失敗時は goal を返す"""
        successful = [r for r in results if r.success and r.output]
        if not successful:
            return goal
        best = max(successful, key=lambda r: r.score)
        return str(best.output)

    @staticmethod
    def _aggregate_score(results: List[PatternResult]) -> float:
        """成功結果の平均スコアを返す"""
        scores = [r.score for r in results if r.success]
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)

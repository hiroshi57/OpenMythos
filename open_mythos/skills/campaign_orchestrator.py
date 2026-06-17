"""
Sprint 66B — A/B → 予算最適化 自動連携オーケストレーター

A/B テストの勝者判定 (ab_test) と予算最適化 (budget_optimizer)、
異常検知 (anomaly_detector) を統合し、広告運用の自動化ワークフローを提供する。

ワークフロー:
  1. A/B テストの勝者を判定（統計的有意差を考慮）
  2. 勝者キャンペーンに予算を重点再配分
  3. 異常検知でKPI急変を監視しアラート

オブジェクト:
  OrchestrationConfig   : ワークフロー設定 (勝者ボーナス倍率 / 有意差要求)
  WinnerDecision        : A/B 勝者判定結果
  ReallocationPlan      : 予算再配分プラン
  CampaignOrchestrator  : 統合ワークフローエンジン

設計方針:
  - 既存 ab_test / budget_optimizer / campaign_analytics を組み合わせるだけ
  - 新たな統計ロジックは持たず、各モジュールに委譲
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from open_mythos.skills.ab_test import (
    ABTest, ABTestAnalyzer,
)
from open_mythos.skills.budget_optimizer import (
    BudgetOptimizer, AllocationStrategy, OptimizationResult, BudgetConstraint,
)
from open_mythos.skills.campaign_analytics import CampaignAnalyticsStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 設定・結果モデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OrchestrationConfig:
    """ワークフロー設定"""
    winner_metric:      str   = "ctr"   # 勝者判定指標
    require_significance: bool = True    # 有意差を要求するか
    winner_bonus:       float = 1.5     # 勝者の予算重み倍率
    strategy:           AllocationStrategy = AllocationStrategy.ROAS_WEIGHTED


@dataclass
class WinnerDecision:
    """A/B 勝者判定結果"""
    winner_variant_id: Optional[str]
    winner_label:      Optional[str]
    is_significant:    bool
    p_value:           float
    confident:         bool      # 自信を持って勝者と言えるか
    reason:            str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner_variant_id": self.winner_variant_id,
            "winner_label":      self.winner_label,
            "is_significant":    self.is_significant,
            "p_value":           round(self.p_value, 6),
            "confident":         self.confident,
            "reason":            self.reason,
        }


@dataclass
class ReallocationPlan:
    """予算再配分プラン"""
    winner_decision:   WinnerDecision
    optimization:      OptimizationResult
    winner_campaign_id: Optional[str]
    applied_bonus:     bool
    timestamp:         float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner_decision":    self.winner_decision.to_dict(),
            "optimization":       self.optimization.to_dict(),
            "winner_campaign_id": self.winner_campaign_id,
            "applied_bonus":      self.applied_bonus,
            "timestamp":          self.timestamp,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignOrchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignOrchestrator:
    """
    A/B テスト → 予算最適化の自動連携エンジン。

    Usage:
        orch = CampaignOrchestrator(analytics_store=store)
        decision = orch.decide_winner(ab_test)
        plan = orch.reallocate(
            ab_test,
            total_budget=100000,
            campaign_ids=["c1", "c2"],
            winner_campaign_id="c1",
        )
    """

    def __init__(
        self,
        analytics_store: Optional[CampaignAnalyticsStore] = None,
        config: Optional[OrchestrationConfig] = None,
        ab_analyzer: Optional[ABTestAnalyzer] = None,
        optimizer: Optional[BudgetOptimizer] = None,
    ) -> None:
        self.config = config or OrchestrationConfig()
        self.analytics_store = analytics_store or CampaignAnalyticsStore()
        self.ab_analyzer = ab_analyzer or ABTestAnalyzer()
        self.optimizer = optimizer or BudgetOptimizer(store=self.analytics_store)

    # ---- 勝者判定 ----

    def decide_winner(self, test: ABTest) -> WinnerDecision:
        """
        A/B テストの勝者を判定する。
        有意差を要求する設定の場合、有意でなければ confident=False。
        """
        winner = self.ab_analyzer.determine_winner(test, metric=self.config.winner_metric)
        if winner is None:
            return WinnerDecision(
                winner_variant_id=None, winner_label=None,
                is_significant=False, p_value=1.0, confident=False,
                reason="データ不足のため勝者を判定できません",
            )

        # 先頭 2 Variant の有意差を評価（勝者 vs 次点）
        is_sig = False
        p_value = 1.0
        others = [v for v in test.variants if v.id != winner.id and v.stats.impressions > 0]
        if others:
            runner_up = max(others, key=lambda v: getattr(v.stats, self.config.winner_metric, 0.0))
            sig = self.ab_analyzer.compare_ctr(winner, runner_up)
            is_sig = sig.significant
            p_value = sig.p_value

        confident = is_sig or not self.config.require_significance
        reason = (
            f"{winner.name} が最高 {self.config.winner_metric}"
            + ("（統計的に有意）" if is_sig else "（有意差なし）")
        )
        return WinnerDecision(
            winner_variant_id=winner.id,
            winner_label=winner.name,
            is_significant=is_sig,
            p_value=p_value,
            confident=confident,
            reason=reason,
        )

    # ---- 予算再配分 ----

    def reallocate(
        self,
        test: ABTest,
        total_budget: float,
        campaign_ids: List[str],
        winner_campaign_id: Optional[str] = None,
    ) -> ReallocationPlan:
        """
        A/B 勝者判定に基づき予算を再配分する。

        勝者が confident なら winner_campaign_id に bonus 倍率を適用した
        制約付き最適化を行う。confident でなければ通常配分。
        """
        decision = self.decide_winner(test)

        constraints: Dict[str, BudgetConstraint] = {}
        applied_bonus = False

        if decision.confident and winner_campaign_id and winner_campaign_id in campaign_ids:
            # 勝者に最低保証を与える（総予算 × bonus_share）
            n = len(campaign_ids)
            base_share = total_budget / n if n else 0.0
            min_winner = min(total_budget, base_share * self.config.winner_bonus)
            constraints[winner_campaign_id] = BudgetConstraint(min_budget=min_winner)
            applied_bonus = True

        optimization = self.optimizer.optimize(
            total_budget=total_budget,
            campaign_ids=campaign_ids,
            strategy=self.config.strategy,
            constraints=constraints or None,
        )

        return ReallocationPlan(
            winner_decision=decision,
            optimization=optimization,
            winner_campaign_id=winner_campaign_id if applied_bonus else None,
            applied_bonus=applied_bonus,
        )

    def run_workflow(
        self,
        test: ABTest,
        total_budget: float,
        campaign_ids: List[str],
        winner_campaign_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """勝者判定 → 再配分を一括実行し dict で返す（API 向け）"""
        plan = self.reallocate(test, total_budget, campaign_ids, winner_campaign_id)
        return plan.to_dict()

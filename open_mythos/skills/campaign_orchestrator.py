"""
Sprint 66B / 67A — A/B → 予算最適化 自動連携 + 異常検知 → 自動予算停止

A/B テストの勝者判定 (ab_test) と予算最適化 (budget_optimizer)、
異常検知 (anomaly_detector) を統合し、広告運用の自動化ワークフローを提供する。

ワークフロー (Sprint 66B):
  1. A/B テストの勝者を判定（統計的有意差を考慮）
  2. 勝者キャンペーンに予算を重点再配分
  3. 異常検知でKPI急変を監視しアラート

自動凍結ワークフロー (Sprint 67A):
  1. AnomalyDetector で Critical アラートを検知
  2. Critical 対象キャンペーンの予算配分を 0 に凍結
  3. FreezeDecision / FrozenBudgetPlan で結果を返す

オブジェクト:
  OrchestrationConfig   : ワークフロー設定 (勝者ボーナス倍率 / 有意差要求)
  WinnerDecision        : A/B 勝者判定結果
  ReallocationPlan      : 予算再配分プラン
  FreezeDecision        : 凍結判定結果 (Sprint 67A)
  FrozenBudgetPlan      : 凍結後の予算プラン (Sprint 67A)
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
from open_mythos.skills.anomaly_detector import (
    AnomalyDetector, AlertSeverity, AlertStore,
)


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


@dataclass
class FreezeDecision:
    """
    異常検知 → 自動予算凍結の判定結果 (Sprint 67A)

    freeze_campaign_ids : Critical アラートが検知されたキャンペーン ID リスト
    frozen              : 少なくとも 1 件を凍結した場合 True
    alert_count         : 検知された Critical アラート総数
    reason              : 判定理由
    """
    freeze_campaign_ids: List[str]
    frozen:              bool
    alert_count:         int
    reason:              str
    timestamp:           float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freeze_campaign_ids": self.freeze_campaign_ids,
            "frozen":              self.frozen,
            "alert_count":         self.alert_count,
            "reason":              self.reason,
            "timestamp":           self.timestamp,
        }


@dataclass
class FrozenBudgetPlan:
    """
    凍結後の予算配分プラン (Sprint 67A)

    凍結対象キャンペーンは amount=0 / share=0 に強制。
    残りの総予算は非凍結キャンペーンへ再分配する。
    """
    freeze_decision:  FreezeDecision
    optimization:     OptimizationResult
    remaining_budget: float
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freeze_decision":  self.freeze_decision.to_dict(),
            "optimization":     self.optimization.to_dict(),
            "remaining_budget": self.remaining_budget,
            "timestamp":        self.timestamp,
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
        anomaly_detector: Optional[AnomalyDetector] = None,
    ) -> None:
        self.config = config or OrchestrationConfig()
        self.analytics_store = analytics_store or CampaignAnalyticsStore()
        self.ab_analyzer = ab_analyzer or ABTestAnalyzer()
        self.optimizer = optimizer or BudgetOptimizer(store=self.analytics_store)
        self.anomaly_detector = anomaly_detector or AnomalyDetector()

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

    # ---- Sprint 67A: 異常検知 → 自動予算停止 ----

    def freeze_if_critical(
        self,
        campaign_ids: List[str],
        total_budget: float,
        alert_store: Optional[AlertStore] = None,
        metric_list: Optional[List[str]] = None,
    ) -> FrozenBudgetPlan:
        """
        Critical アラートが検知されたキャンペーンの予算を 0 に凍結し、
        残予算を非凍結キャンペーンへ再配分する。

        Args:
            campaign_ids  : 対象キャンペーン ID リスト
            total_budget  : 総予算
            alert_store   : 既存 AlertStore（指定時はそのアラートを参照）。
                            None のときは analytics_store から再検知する。
            metric_list   : 検知対象指標（None = デフォルト全指標）

        Returns:
            FrozenBudgetPlan
        """
        # 1) Critical アラート対象キャンペーンを特定
        if alert_store is not None:
            critical = alert_store.list_by_severity(AlertSeverity.CRITICAL)
            freeze_ids = list({a.campaign_id for a in critical if a.campaign_id in campaign_ids})
            alert_count = len(critical)
        else:
            freeze_ids = []
            alert_count = 0
            for cid in campaign_ids:
                metrics = self.analytics_store.get(cid)
                if metrics is None:
                    continue
                alerts = self.anomaly_detector.detect_multi(metrics, metric_list)
                critical_here = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
                alert_count += len(critical_here)
                if critical_here:
                    freeze_ids.append(cid)

        frozen = bool(freeze_ids)
        if frozen:
            reason = f"Critical アラート検知: {', '.join(freeze_ids)} を凍結"
        else:
            reason = "Critical アラートなし — 凍結対象なし"

        freeze_decision = FreezeDecision(
            freeze_campaign_ids=freeze_ids,
            frozen=frozen,
            alert_count=alert_count,
            reason=reason,
        )

        # 2) 非凍結キャンペーンのみで予算最適化
        active_ids = [c for c in campaign_ids if c not in freeze_ids]
        remaining = total_budget  # 凍結分も含む総額をそのまま残余とする

        if active_ids:
            optimization = self.optimizer.optimize(
                total_budget=remaining,
                campaign_ids=active_ids,
                strategy=self.config.strategy,
            )
        else:
            # 全キャンペーンが凍結 → 空の最適化結果
            from open_mythos.skills.budget_optimizer import OptimizationResult
            optimization = OptimizationResult(
                strategy=self.config.strategy,
                total_budget=remaining,
                allocations=[],
                allocated_total=0.0,
                unallocated=remaining,
            )

        return FrozenBudgetPlan(
            freeze_decision=freeze_decision,
            optimization=optimization,
            remaining_budget=remaining,
        )

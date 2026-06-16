"""
Sprint 64C — 広告予算最適化

キャンペーン分析データ (campaign_analytics) を入力として、総予算を
各キャンペーンに最適配分する。ROAS（広告費用対効果）ベースの自動配分を
中心に、複数の配分戦略をサポートする。

オブジェクト:
  AllocationStrategy : 配分戦略 (Equal/RoasWeighted/Performance/Proportional)
  BudgetAllocation   : キャンペーン 1 件への配分結果
  OptimizationResult : 最適化結果全体 (配分リスト + メタデータ)
  BudgetOptimizer    : 予算最適化エンジン
  BudgetConstraint   : 配分制約 (min/max per campaign)

設計方針:
  - 外部依存なし
  - campaign_analytics.CampaignAnalyticsStore / KpiCalculator を利用
  - ROAS が高いキャンペーンに多く配分（実績ベース最適化）
  - 制約 (min/max) を満たしつつ正規化
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from open_mythos.skills.campaign_analytics import (
    CampaignAnalyticsStore,
    KpiCalculator,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AllocationStrategy(str, Enum):
    EQUAL        = "equal"          # 均等配分
    ROAS_WEIGHTED = "roas_weighted" # ROAS 比例配分
    PERFORMANCE  = "performance"    # CVR ベース配分
    PROPORTIONAL = "proportional"   # 過去消化額比例配分


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 制約・結果モデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BudgetConstraint:
    """キャンペーン配分制約"""
    min_budget: float = 0.0
    max_budget: Optional[float] = None  # None = 上限なし

    def clamp(self, value: float) -> float:
        v = max(self.min_budget, value)
        if self.max_budget is not None:
            v = min(self.max_budget, v)
        return v


@dataclass
class BudgetAllocation:
    """キャンペーン 1 件への配分結果"""
    campaign_id: str
    amount:      float
    share:       float          # 総予算に占める割合 (0.0〜1.0)
    roas:        float = 0.0
    rationale:   str   = ""     # 配分根拠

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "amount":      round(self.amount, 2),
            "share":       round(self.share, 6),
            "roas":        round(self.roas, 4),
            "rationale":   self.rationale,
        }


@dataclass
class OptimizationResult:
    """最適化結果全体"""
    total_budget: float
    strategy:     AllocationStrategy
    allocations:  List[BudgetAllocation]
    allocated_total: float
    unallocated:  float
    timestamp:    float = field(default_factory=time.time)

    def get(self, campaign_id: str) -> Optional[BudgetAllocation]:
        for a in self.allocations:
            if a.campaign_id == campaign_id:
                return a
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_budget":    round(self.total_budget, 2),
            "strategy":        self.strategy.value,
            "allocations":     [a.to_dict() for a in self.allocations],
            "allocated_total": round(self.allocated_total, 2),
            "unallocated":     round(self.unallocated, 2),
            "timestamp":       self.timestamp,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BudgetOptimizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BudgetOptimizer:
    """
    予算最適化エンジン。

    Usage:
        store = CampaignAnalyticsStore()
        store.record("c1", impressions=10000, clicks=500, conversions=50, spend=1000, revenue=5000)
        store.record("c2", impressions=10000, clicks=300, conversions=20, spend=1000, revenue=2000)

        optimizer = BudgetOptimizer(store)
        result = optimizer.optimize(
            total_budget=10000,
            campaign_ids=["c1", "c2"],
            strategy=AllocationStrategy.ROAS_WEIGHTED,
        )
        # c1 (ROAS 5) に c2 (ROAS 2) より多く配分される
    """

    def __init__(
        self,
        store: Optional[CampaignAnalyticsStore] = None,
        calculator: Optional[KpiCalculator] = None,
    ) -> None:
        self.store = store or CampaignAnalyticsStore()
        self.calculator = calculator or KpiCalculator()

    def optimize(
        self,
        total_budget: float,
        campaign_ids: List[str],
        strategy: AllocationStrategy = AllocationStrategy.ROAS_WEIGHTED,
        constraints: Optional[Dict[str, BudgetConstraint]] = None,
    ) -> OptimizationResult:
        """
        総予算を campaign_ids に配分する。

        Args:
            total_budget : 配分する総予算
            campaign_ids : 配分対象のキャンペーン ID リスト
            strategy     : 配分戦略
            constraints  : キャンペーン別の min/max 制約
        """
        if total_budget < 0:
            raise ValueError("total_budget must be non-negative")
        if not campaign_ids:
            return OptimizationResult(
                total_budget=total_budget,
                strategy=strategy,
                allocations=[],
                allocated_total=0.0,
                unallocated=total_budget,
            )

        constraints = constraints or {}

        # 戦略ごとに重みを算出
        weights = self._compute_weights(campaign_ids, strategy)

        total_weight = sum(weights.values())
        allocations: List[BudgetAllocation] = []

        for cid in campaign_ids:
            if total_weight > 0:
                raw = total_budget * (weights[cid] / total_weight)
            else:
                raw = total_budget / len(campaign_ids)

            constraint = constraints.get(cid)
            amount = constraint.clamp(raw) if constraint else raw

            roas = self._roas(cid)
            allocations.append(BudgetAllocation(
                campaign_id=cid,
                amount=amount,
                share=0.0,  # 後で正規化
                roas=roas,
                rationale=self._rationale(strategy, weights[cid], roas),
            ))

        # 制約適用後に share を再計算
        allocated_total = sum(a.amount for a in allocations)
        for a in allocations:
            a.share = (a.amount / allocated_total) if allocated_total > 0 else 0.0

        return OptimizationResult(
            total_budget=total_budget,
            strategy=strategy,
            allocations=allocations,
            allocated_total=allocated_total,
            unallocated=max(0.0, total_budget - allocated_total),
        )

    # ---- 内部実装 ----

    def _compute_weights(
        self, campaign_ids: List[str], strategy: AllocationStrategy
    ) -> Dict[str, float]:
        if strategy == AllocationStrategy.EQUAL:
            return {cid: 1.0 for cid in campaign_ids}

        if strategy == AllocationStrategy.ROAS_WEIGHTED:
            weights = {cid: self._roas(cid) for cid in campaign_ids}
            # 全 ROAS が 0 の場合は均等配分にフォールバック
            if sum(weights.values()) <= 0:
                return {cid: 1.0 for cid in campaign_ids}
            return weights

        if strategy == AllocationStrategy.PERFORMANCE:
            weights = {cid: self._cvr(cid) for cid in campaign_ids}
            if sum(weights.values()) <= 0:
                return {cid: 1.0 for cid in campaign_ids}
            return weights

        if strategy == AllocationStrategy.PROPORTIONAL:
            weights = {cid: self._spend(cid) for cid in campaign_ids}
            if sum(weights.values()) <= 0:
                return {cid: 1.0 for cid in campaign_ids}
            return weights

        return {cid: 1.0 for cid in campaign_ids}

    def _roas(self, campaign_id: str) -> float:
        m = self.store.get(campaign_id)
        if m is None:
            return 0.0
        return self.calculator.compute(m).roas

    def _cvr(self, campaign_id: str) -> float:
        m = self.store.get(campaign_id)
        if m is None:
            return 0.0
        return self.calculator.compute(m).cvr

    def _spend(self, campaign_id: str) -> float:
        m = self.store.get(campaign_id)
        if m is None:
            return 0.0
        return m.total_spend

    def _rationale(
        self, strategy: AllocationStrategy, weight: float, roas: float
    ) -> str:
        if strategy == AllocationStrategy.EQUAL:
            return "均等配分"
        if strategy == AllocationStrategy.ROAS_WEIGHTED:
            return f"ROAS {roas:.2f} に比例配分"
        if strategy == AllocationStrategy.PERFORMANCE:
            return f"CVR ベース配分 (weight={weight:.4f})"
        if strategy == AllocationStrategy.PROPORTIONAL:
            return f"過去消化額比例配分 (weight={weight:.2f})"
        return "デフォルト配分"

    def recommend_strategy(self, campaign_ids: List[str]) -> AllocationStrategy:
        """
        データの揃い具合に応じて推奨戦略を返す。

        - 売上データあり → ROAS_WEIGHTED
        - クリック/CVデータあり → PERFORMANCE
        - 消化額のみ → PROPORTIONAL
        - データなし → EQUAL
        """
        has_revenue = False
        has_conversions = False
        has_spend = False
        for cid in campaign_ids:
            m = self.store.get(cid)
            if m is None:
                continue
            if m.total_revenue > 0:
                has_revenue = True
            if m.total_conversions > 0:
                has_conversions = True
            if m.total_spend > 0:
                has_spend = True

        if has_revenue:
            return AllocationStrategy.ROAS_WEIGHTED
        if has_conversions:
            return AllocationStrategy.PERFORMANCE
        if has_spend:
            return AllocationStrategy.PROPORTIONAL
        return AllocationStrategy.EQUAL

"""
Sprint 63A — A/B テスト基盤

複数の広告コピー案（Variant）の効果を測定し、統計的有意差を判定する
フレームワーク。serve/ab_router.py（スタンドアロン A/B サーバ）とは独立した
ドメインロジック層。

オブジェクト:
  VariantStatus   : Variant 状態 (Draft/Running/Winner/Loser/Stopped)
  Variant         : A/B テストの 1 案 (id/name/content/weight)
  VariantStats    : Variant の累積統計 (impressions/clicks/conversions + 派生指標)
  ABTest          : A/B テスト本体 (variants + ステータス)
  ABTestStore     : A/B テスト CRUD ストア
  TrafficAllocator: 重み付きトラフィック振り分け (決定論的 hash ベース)
  SignificanceResult: 統計的有意差判定結果 (z値/p値/有意か)
  ABTestAnalyzer  : CTR/CVR 計算・2 標本比率 z 検定・勝者判定
  ABTestReportEngine: レポート生成 (Markdown / JSON)

設計方針:
  - 外部依存なし (math のみで正規分布 CDF を実装)
  - トラフィック振り分けは hash ベースで決定論的（再現可能）
  - 有意差は 2 標本比率 z 検定（既定 α=0.05, 両側）
"""
from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VariantStatus(str, Enum):
    DRAFT   = "draft"
    RUNNING = "running"
    WINNER  = "winner"
    LOSER   = "loser"
    STOPPED = "stopped"


class ABTestStatus(str, Enum):
    DRAFT     = "draft"
    RUNNING   = "running"
    COMPLETED = "completed"
    STOPPED   = "stopped"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Variant / 統計
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class VariantStats:
    """Variant の累積統計"""
    impressions: int = 0
    clicks:      int = 0
    conversions: int = 0

    def record_impression(self, n: int = 1) -> None:
        self.impressions += n

    def record_click(self, n: int = 1) -> None:
        self.clicks += n

    def record_conversion(self, n: int = 1) -> None:
        self.conversions += n

    @property
    def ctr(self) -> float:
        """Click-Through Rate"""
        return self.clicks / self.impressions if self.impressions else 0.0

    @property
    def cvr(self) -> float:
        """Conversion Rate (conversions / clicks)"""
        return self.conversions / self.clicks if self.clicks else 0.0

    @property
    def cvr_per_impression(self) -> float:
        """Conversion Rate (conversions / impressions)"""
        return self.conversions / self.impressions if self.impressions else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "impressions":        self.impressions,
            "clicks":             self.clicks,
            "conversions":        self.conversions,
            "ctr":                round(self.ctr, 6),
            "cvr":                round(self.cvr, 6),
            "cvr_per_impression": round(self.cvr_per_impression, 6),
        }


@dataclass
class Variant:
    """A/B テストの 1 案"""
    id:      str
    name:    str
    content: str
    weight:  float                 = 1.0   # トラフィック配分の重み
    status:  VariantStatus         = VariantStatus.DRAFT
    stats:   VariantStats          = field(default_factory=VariantStats)
    metadata: Dict[str, Any]       = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":       self.id,
            "name":     self.name,
            "content":  self.content,
            "weight":   self.weight,
            "status":   self.status.value,
            "stats":    self.stats.to_dict(),
            "metadata": self.metadata,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABTest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ABTest:
    """A/B テスト本体"""
    id:          str
    name:        str
    variants:    List[Variant]      = field(default_factory=list)
    status:      ABTestStatus       = ABTestStatus.DRAFT
    campaign_id: Optional[str]      = None
    description: str                = ""
    created_at:  float              = field(default_factory=time.time)
    updated_at:  float              = field(default_factory=time.time)

    def add_variant(self, variant: Variant) -> Variant:
        self.variants.append(variant)
        self.updated_at = time.time()
        return variant

    def get_variant(self, variant_id: str) -> Optional[Variant]:
        for v in self.variants:
            if v.id == variant_id:
                return v
        return None

    def start(self) -> None:
        if self.status != ABTestStatus.DRAFT:
            raise ValueError(f"Cannot start from status={self.status.value}")
        if len(self.variants) < 2:
            raise ValueError("A/B テストには最低 2 つの Variant が必要です")
        self.status = ABTestStatus.RUNNING
        for v in self.variants:
            v.status = VariantStatus.RUNNING
        self.updated_at = time.time()

    def stop(self) -> None:
        if self.status != ABTestStatus.RUNNING:
            raise ValueError(f"Cannot stop from status={self.status.value}")
        self.status = ABTestStatus.STOPPED
        for v in self.variants:
            if v.status == VariantStatus.RUNNING:
                v.status = VariantStatus.STOPPED
        self.updated_at = time.time()

    def complete(self, winner_id: Optional[str] = None) -> None:
        if self.status not in (ABTestStatus.RUNNING, ABTestStatus.STOPPED):
            raise ValueError(f"Cannot complete from status={self.status.value}")
        self.status = ABTestStatus.COMPLETED
        if winner_id is not None:
            for v in self.variants:
                v.status = (
                    VariantStatus.WINNER if v.id == winner_id else VariantStatus.LOSER
                )
        self.updated_at = time.time()

    @property
    def total_impressions(self) -> int:
        return sum(v.stats.impressions for v in self.variants)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":           self.id,
            "name":         self.name,
            "variants":     [v.to_dict() for v in self.variants],
            "status":       self.status.value,
            "campaign_id":  self.campaign_id,
            "description":  self.description,
            "total_impressions": self.total_impressions,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABTestStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ABTestStore:
    """A/B テスト CRUD ストア（インメモリ）"""

    def __init__(self) -> None:
        self._tests: Dict[str, ABTest] = {}

    def add(self, test: ABTest) -> ABTest:
        self._tests[test.id] = test
        return test

    def get(self, test_id: str) -> Optional[ABTest]:
        return self._tests.get(test_id)

    def list_all(self) -> List[ABTest]:
        return list(self._tests.values())

    def list_by_status(self, status: ABTestStatus) -> List[ABTest]:
        return [t for t in self._tests.values() if t.status == status]

    def find_by_campaign(self, campaign_id: str) -> List[ABTest]:
        return [t for t in self._tests.values() if t.campaign_id == campaign_id]

    def delete(self, test_id: str) -> bool:
        if test_id in self._tests:
            del self._tests[test_id]
            return True
        return False

    def count(self) -> int:
        return len(self._tests)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TrafficAllocator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrafficAllocator:
    """
    重み付きトラフィック振り分け。

    ユーザー ID の hash 値に基づき決定論的に Variant を割り当てる。
    同一 ユーザー ID は常に同じ Variant に振り分けられる（再現性）。
    """

    def __init__(self, test: ABTest) -> None:
        self._test = test

    def allocate(self, user_id: str) -> Optional[Variant]:
        """user_id を Variant に振り分ける"""
        variants = self._test.variants
        if not variants:
            return None

        total_weight = sum(v.weight for v in variants)
        if total_weight <= 0:
            return variants[0]

        # hash を [0, 1) に正規化
        bucket = self._hash_to_unit(user_id, self._test.id)
        cumulative = 0.0
        for v in variants:
            cumulative += v.weight / total_weight
            if bucket < cumulative:
                return v
        return variants[-1]

    @staticmethod
    def _hash_to_unit(user_id: str, salt: str) -> float:
        """user_id を [0, 1) の float に決定論的にマップする"""
        h = hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()
        # 先頭 8 桁 (32bit) を使う
        n = int(h[:8], 16)
        return n / 0x100000000

    def allocation_distribution(
        self, user_ids: List[str]
    ) -> Dict[str, int]:
        """複数 user_id を振り分けた結果の分布を返す（検証用）"""
        dist: Dict[str, int] = {v.id: 0 for v in self._test.variants}
        for uid in user_ids:
            v = self.allocate(uid)
            if v is not None:
                dist[v.id] += 1
        return dist


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 統計解析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SignificanceResult:
    """2 標本比率 z 検定の結果"""
    variant_a_id:  str
    variant_b_id:  str
    rate_a:        float
    rate_b:        float
    z_score:       float
    p_value:       float
    significant:   bool
    alpha:         float
    better_variant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_a_id":      self.variant_a_id,
            "variant_b_id":      self.variant_b_id,
            "rate_a":            round(self.rate_a, 6),
            "rate_b":            round(self.rate_b, 6),
            "z_score":           round(self.z_score, 4),
            "p_value":           round(self.p_value, 6),
            "significant":       self.significant,
            "alpha":             self.alpha,
            "better_variant_id": self.better_variant_id,
        }


def _normal_cdf(x: float) -> float:
    """標準正規分布の累積分布関数 (math.erf ベース)"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class ABTestAnalyzer:
    """
    A/B テストの統計解析。

    - CTR / CVR の計算
    - 2 標本比率 z 検定で有意差を判定
    - 勝者を決定
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def compare_ctr(
        self, variant_a: Variant, variant_b: Variant
    ) -> SignificanceResult:
        """2 つの Variant の CTR を比較して有意差を判定する"""
        return self._z_test(
            variant_a.id, variant_b.id,
            variant_a.stats.clicks, variant_a.stats.impressions,
            variant_b.stats.clicks, variant_b.stats.impressions,
        )

    def compare_cvr(
        self, variant_a: Variant, variant_b: Variant
    ) -> SignificanceResult:
        """2 つの Variant の CVR (conversions/clicks) を比較する"""
        return self._z_test(
            variant_a.id, variant_b.id,
            variant_a.stats.conversions, variant_a.stats.clicks,
            variant_b.stats.conversions, variant_b.stats.clicks,
        )

    def _z_test(
        self,
        a_id: str, b_id: str,
        success_a: int, total_a: int,
        success_b: int, total_b: int,
    ) -> SignificanceResult:
        rate_a = success_a / total_a if total_a else 0.0
        rate_b = success_b / total_b if total_b else 0.0

        z_score = 0.0
        p_value = 1.0
        significant = False

        if total_a > 0 and total_b > 0:
            # プールされた比率
            pooled = (success_a + success_b) / (total_a + total_b)
            se = math.sqrt(pooled * (1 - pooled) * (1 / total_a + 1 / total_b))
            if se > 0:
                z_score = (rate_a - rate_b) / se
                # 両側 p 値
                p_value = 2 * (1 - _normal_cdf(abs(z_score)))
                significant = p_value < self.alpha

        better = None
        if significant:
            better = a_id if rate_a > rate_b else b_id

        return SignificanceResult(
            variant_a_id=a_id,
            variant_b_id=b_id,
            rate_a=rate_a,
            rate_b=rate_b,
            z_score=z_score,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            better_variant_id=better,
        )

    def determine_winner(
        self, test: ABTest, metric: str = "ctr"
    ) -> Optional[Variant]:
        """
        テスト内の Variant から勝者を決定する。

        metric: "ctr" | "cvr" | "cvr_per_impression"
        最高指標の Variant を返す（インプレッションが 0 の Variant は除外）。
        """
        eligible = [v for v in test.variants if v.stats.impressions > 0]
        if not eligible:
            return None

        def score(v: Variant) -> float:
            return getattr(v.stats, metric, 0.0)

        return max(eligible, key=score)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABTestReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ABTestReportEngine:
    """A/B テストレポート生成"""

    def __init__(self, analyzer: Optional[ABTestAnalyzer] = None) -> None:
        self._analyzer = analyzer or ABTestAnalyzer()

    def summary_json(self, test: ABTest) -> Dict[str, Any]:
        """テストサマリーを JSON で返す"""
        winner = self._analyzer.determine_winner(test, metric="ctr")
        return {
            "test":            test.to_dict(),
            "winner_id":       winner.id if winner else None,
            "variant_count":   len(test.variants),
            "total_impressions": test.total_impressions,
        }

    def markdown(self, test: ABTest) -> str:
        """テストレポートを Markdown で返す"""
        lines = [
            f"# A/B テスト: {test.name}",
            "",
            f"**ID**: `{test.id}`  ",
            f"**状態**: {test.status.value}  ",
            f"**総インプレッション**: {test.total_impressions:,}  ",
            "",
            "## Variant 成績",
            "| Variant | IMP | Click | CV | CTR | CVR |",
            "|---------|-----|-------|----|----|----|",
        ]
        for v in test.variants:
            s = v.stats
            lines.append(
                f"| {v.name} | {s.impressions:,} | {s.clicks:,} | "
                f"{s.conversions:,} | {s.ctr:.2%} | {s.cvr:.2%} |"
            )

        winner = self._analyzer.determine_winner(test, metric="ctr")
        lines += ["", "## 勝者 (CTR 基準)"]
        if winner:
            lines.append(f"**{winner.name}** (CTR {winner.stats.ctr:.2%})")
        else:
            lines.append("*データ不足のため判定できません*")

        # 先頭 2 Variant の有意差
        if len(test.variants) >= 2:
            sig = self._analyzer.compare_ctr(test.variants[0], test.variants[1])
            lines += [
                "",
                "## 統計的有意差 (先頭 2 Variant の CTR)",
                f"- z 値: {sig.z_score:.4f}",
                f"- p 値: {sig.p_value:.6f}",
                f"- 有意 (α={sig.alpha}): {'はい' if sig.significant else 'いいえ'}",
            ]
        return "\n".join(lines)

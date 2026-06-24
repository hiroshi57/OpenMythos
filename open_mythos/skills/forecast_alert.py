"""
Sprint 70A — 予測アラート統合 (ForecastAlert)

ForecastStore (Sprint 69) に格納された予測値を閾値チェックし、
将来の KPI 違反を事前検知する。

オブジェクト:
  AlertThreshold         : 閾値設定 (upper_limit / lower_limit)
  ForecastAlertRule      : ルール (campaign + metric + threshold + severity)
  ForecastAlertCheck     : チェック結果 1 件
  ForecastAlertRuleStore : ルール CRUD ストア
  ForecastAlertEngine    : チェックエンジン

設計方針:
  - 外部依存なし
  - ForecastStore の latest() を使って最新予測を参照
  - enabled=False のルールは check_all() でスキップ
  - 予測データが無い場合は triggered=False を返す（graceful）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from open_mythos.skills.time_series import ForecastStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AlertThreshold:
    """予測値の閾値設定"""
    metric: str
    upper_limit: Optional[float] = None   # 予測値 > upper_limit でアラート
    lower_limit: Optional[float] = None   # 予測値 < lower_limit でアラート

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric":      self.metric,
            "upper_limit": self.upper_limit,
            "lower_limit": self.lower_limit,
        }


@dataclass
class ForecastAlertRule:
    """予測アラートルール 1 件"""
    id:          str
    campaign_id: str
    threshold:   AlertThreshold
    severity:    str          # "info" / "warning" / "critical"
    enabled:     bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "campaign_id": self.campaign_id,
            "threshold":   self.threshold.to_dict(),
            "severity":    self.severity,
            "enabled":     self.enabled,
        }


@dataclass
class ForecastAlertCheck:
    """チェック結果 1 件"""
    rule_id:          str
    campaign_id:      str
    metric:           str
    triggered:        bool
    severity:         Optional[str]
    predicted_values: List[float]
    violations:       List[Dict[str, Any]]
    checked_at:       float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id":          self.rule_id,
            "campaign_id":      self.campaign_id,
            "metric":           self.metric,
            "triggered":        self.triggered,
            "severity":         self.severity,
            "predicted_values": self.predicted_values,
            "violations":       self.violations,
            "checked_at":       self.checked_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ストア
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForecastAlertRuleStore:
    """予測アラートルール インメモリストア"""

    def __init__(self) -> None:
        self._rules: Dict[str, ForecastAlertRule] = {}

    def add(self, rule: ForecastAlertRule) -> None:
        self._rules[rule.id] = rule

    def get(self, rule_id: str) -> Optional[ForecastAlertRule]:
        return self._rules.get(rule_id)

    def list(self) -> List[ForecastAlertRule]:
        return list(self._rules.values())

    def list_by_campaign(self, campaign_id: str) -> List[ForecastAlertRule]:
        return [r for r in self._rules.values() if r.campaign_id == campaign_id]

    def delete(self, rule_id: str) -> None:
        self._rules.pop(rule_id, None)

    def set_enabled(self, rule_id: str, enabled: bool) -> None:
        rule = self._rules.get(rule_id)
        if rule is not None:
            rule.enabled = enabled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# エンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForecastAlertEngine:
    """予測値を閾値チェックしてアラートを生成するエンジン"""

    def __init__(
        self,
        forecast_store: ForecastStore,
        rule_store: ForecastAlertRuleStore,
    ) -> None:
        self._fs = forecast_store
        self._rs = rule_store

    # ── 単一ルールチェック ──────────────────────────────────────

    def check(self, rule_id: str) -> ForecastAlertCheck:
        """指定ルールを評価して ForecastAlertCheck を返す"""
        rule = self._rs.get(rule_id)
        if rule is None:
            return ForecastAlertCheck(
                rule_id=rule_id, campaign_id="", metric="",
                triggered=False, severity=None,
                predicted_values=[], violations=[],
            )

        threshold = rule.threshold
        result = self._fs.latest(rule.campaign_id, threshold.metric)
        if result is None:
            return ForecastAlertCheck(
                rule_id=rule_id,
                campaign_id=rule.campaign_id,
                metric=threshold.metric,
                triggered=False,
                severity=None,
                predicted_values=[],
                violations=[],
            )

        predicted = result.values  # List[float]
        violations: List[Dict[str, Any]] = []

        for i, val in enumerate(predicted, start=1):
            if threshold.upper_limit is not None and val > threshold.upper_limit:
                violations.append({
                    "step":  i,
                    "value": round(val, 4),
                    "limit": threshold.upper_limit,
                    "type":  "upper",
                })
            if threshold.lower_limit is not None and val < threshold.lower_limit:
                violations.append({
                    "step":  i,
                    "value": round(val, 4),
                    "limit": threshold.lower_limit,
                    "type":  "lower",
                })

        triggered = len(violations) > 0
        return ForecastAlertCheck(
            rule_id=rule_id,
            campaign_id=rule.campaign_id,
            metric=threshold.metric,
            triggered=triggered,
            severity=rule.severity if triggered else None,
            predicted_values=[round(v, 4) for v in predicted],
            violations=violations,
        )

    # ── キャンペーン全ルール一括チェック ───────────────────────

    def check_all(self, campaign_id: str) -> List[ForecastAlertCheck]:
        """campaign_id に紐付く有効なルールを全て評価する"""
        rules = [r for r in self._rs.list_by_campaign(campaign_id) if r.enabled]
        return [self.check(r.id) for r in rules]

"""
Sprint 70B — レポート自動配信 Webhook (ReportDispatcher)

定期レポートを Slack / 汎用 Webhook エンドポイントに POST する。
テスト・オフライン環境では dispatch_mock() を使用して HTTP コールを回避。

オブジェクト:
  WebhookTarget    : 配信先設定 (id / name / url / type / enabled)
  DispatchPayload  : 送信内容 (webhook_id / report_type / campaign_id / content)
  DispatchResult   : 送信結果 (success / status_code / error)
  WebhookStore     : Webhook CRUD ストア
  ReportDispatcher : レポート生成 + 配信エンジン

設計方針:
  - 外部 HTTP 依存なし (dispatch_mock が実使用。実 HTTP は dispatch() で urllib 使用)
  - campaign_analytics.CampaignAnalyticsStore からレポート内容を生成
  - enabled=False の Webhook は dispatch_all_mock() でスキップ
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from open_mythos.skills.campaign_analytics import CampaignAnalyticsStore, KpiCalculator as _KpiCalc

_kpi_calc = _KpiCalc()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class WebhookTarget:
    """Webhook 配信先"""
    id:      str
    name:    str
    url:     str
    type:    str          # "slack" / "generic"
    enabled: bool = True
    headers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":      self.id,
            "name":    self.name,
            "url":     self.url,
            "type":    self.type,
            "enabled": self.enabled,
        }


@dataclass
class DispatchPayload:
    """送信ペイロード"""
    webhook_id:  str
    report_type: str
    campaign_id: Optional[str]
    content:     str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "webhook_id":  self.webhook_id,
            "report_type": self.report_type,
            "campaign_id": self.campaign_id,
            "content":     self.content,
        }


@dataclass
class DispatchResult:
    """配信結果"""
    webhook_id:    str
    success:       bool
    status_code:   Optional[int] = None
    error:         Optional[str] = None
    dispatched_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "webhook_id":    self.webhook_id,
            "success":       self.success,
            "status_code":   self.status_code,
            "error":         self.error,
            "dispatched_at": self.dispatched_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ストア
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WebhookStore:
    """Webhook 設定 インメモリストア"""

    def __init__(self) -> None:
        self._webhooks: Dict[str, WebhookTarget] = {}

    def add(self, wh: WebhookTarget) -> None:
        self._webhooks[wh.id] = wh

    def get(self, webhook_id: str) -> Optional[WebhookTarget]:
        return self._webhooks.get(webhook_id)

    def list(self) -> List[WebhookTarget]:
        return list(self._webhooks.values())

    def list_enabled(self) -> List[WebhookTarget]:
        return [wh for wh in self._webhooks.values() if wh.enabled]

    def delete(self, webhook_id: str) -> None:
        self._webhooks.pop(webhook_id, None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ディスパッチャー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReportDispatcher:
    """レポート生成 + Webhook 配信エンジン"""

    def __init__(
        self,
        webhook_store: WebhookStore,
        analytics_store: CampaignAnalyticsStore,
    ) -> None:
        self._ws = webhook_store
        self._analytics = analytics_store
        self._history: List[DispatchResult] = []

    # ── ペイロード生成 ──────────────────────────────────────────

    def build_payload(
        self,
        webhook_id: str,
        report_type: str,
        campaign_id: Optional[str],
    ) -> DispatchPayload:
        """レポートタイプに応じたペイロードを生成する"""
        content = self._build_content(report_type, campaign_id)
        return DispatchPayload(
            webhook_id=webhook_id,
            report_type=report_type,
            campaign_id=campaign_id,
            content=content,
        )

    def _build_content(self, report_type: str, campaign_id: Optional[str]) -> str:
        if report_type == "campaign_summary" and campaign_id is not None:
            metrics = self._analytics.get(campaign_id)
            if metrics is not None:
                kpi = _kpi_calc.compute(metrics).to_dict()
                lines = [
                    f"# Campaign Summary: {campaign_id}",
                    f"- CTR:  {kpi.get('ctr', 0):.4f}",
                    f"- CVR:  {kpi.get('cvr', 0):.4f}",
                    f"- ROAS: {kpi.get('roas', 0):.4f}",
                    f"- Points: {len(metrics.points)}",
                ]
                return "\n".join(lines)
        # generic fallback
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        header = f"# OpenMythos Report — {report_type}"
        body = f"Generated at {ts} UTC"
        if campaign_id:
            body += f"\nCampaign: {campaign_id}"
        return f"{header}\n{body}"

    # ── モック配信（テスト・オフライン用）───────────────────────

    def dispatch_mock(
        self,
        webhook_id: str,
        report_type: str,
        campaign_id: Optional[str],
    ) -> DispatchResult:
        """HTTP コールなし。常に success=True を返す（webhook が存在する場合）"""
        wh = self._ws.get(webhook_id)
        if wh is None:
            result = DispatchResult(
                webhook_id=webhook_id,
                success=False,
                error=f"Webhook not found: {webhook_id}",
            )
        else:
            result = DispatchResult(
                webhook_id=webhook_id,
                success=True,
                status_code=200,
            )
        self._history.append(result)
        return result

    def dispatch_all_mock(
        self,
        report_type: str,
        campaign_id: Optional[str],
    ) -> List[DispatchResult]:
        """有効な全 Webhook にモック配信する"""
        return [
            self.dispatch_mock(wh.id, report_type, campaign_id)
            for wh in self._ws.list_enabled()
        ]

    # ── 実 HTTP 配信（urllib 使用）──────────────────────────────

    def dispatch(
        self,
        webhook_id: str,
        report_type: str,
        campaign_id: Optional[str],
    ) -> DispatchResult:
        """実際の HTTP POST を試みる。失敗時は success=False を返す"""
        import json
        import urllib.request
        import urllib.error

        wh = self._ws.get(webhook_id)
        if wh is None:
            result = DispatchResult(
                webhook_id=webhook_id,
                success=False,
                error=f"Webhook not found: {webhook_id}",
            )
            self._history.append(result)
            return result

        payload = self.build_payload(webhook_id, report_type, campaign_id)
        body: bytes
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        headers.update(wh.headers)

        if wh.type == "slack":
            data = {"text": payload.content}
        else:
            data = payload.to_dict()

        body = json.dumps(data).encode("utf-8")

        try:
            req = urllib.request.Request(wh.url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
            result = DispatchResult(webhook_id=webhook_id, success=True, status_code=status)
        except urllib.error.HTTPError as e:
            result = DispatchResult(webhook_id=webhook_id, success=False,
                                    status_code=e.code, error=str(e))
        except Exception as e:  # noqa: BLE001
            result = DispatchResult(webhook_id=webhook_id, success=False, error=str(e))

        self._history.append(result)
        return result

    # ── 履歴 ───────────────────────────────────────────────────

    def history(self) -> List[DispatchResult]:
        return list(self._history)

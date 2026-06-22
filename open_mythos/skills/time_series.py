"""
Sprint 69 — 時系列予測統合 (TimesFM + マルチモデル)

Google Research 製の事前学習済み時系列基盤モデル TimesFM を OpenMythos に統合する。
参照: https://github.com/google-research/timesfm
      https://docs.cloud.google.com/bigquery/docs/timesfm-model

ユースケース:
  - キャンペーン KPI (clicks/impressions/spend/revenue) の将来予測
  - 異常検知の前段として将来トレンドを把握
  - 予算最適化への予測値フィード

オブジェクト:
  ForecastPoint         : 予測 1 点 (step / value / lower / upper)
  ForecastResult        : 予測結果全体 (ポイントリスト + メタデータ)
  BaseForecaster        : 抽象基底クラス
  LinearTrendForecaster : 線形トレンド外挿 (外部依存なし / フォールバック)
  TimesFMForecaster     : TimesFM_2p5_200M_torch ラッパー (lazy load)
  TimesFMForecasterFactory : from_pretrained / from_mock / rule_based ファクトリ
  CampaignForecaster    : campaign_analytics.CampaignMetrics → 予測の統合層
  ForecastStore         : 予測結果インメモリストア
  ForecastReportEngine  : Markdown / JSON レポート生成

設計方針:
  - TimesFM が使えない環境では LinearTrendForecaster に自動フォールバック
  - テスト・オフライン環境では MockForecaster で予測シミュレーション
  - 既存 TrendAnalyzer (campaign_analytics) とは補完関係: 過去分析 vs 将来予測
  - TimesFM のモデルロードは lazy (初回 forecast 時): 起動時間への影響ゼロ
"""
from __future__ import annotations

import math
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from open_mythos.skills.campaign_analytics import CampaignMetrics, CampaignAnalyticsStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ForecastPoint:
    """予測 1 ステップ"""
    step:  int              # 1-based (1 = 次の期間)
    value: float            # 予測中央値
    lower: Optional[float] = None   # 予測下限 (10th percentile)
    upper: Optional[float] = None   # 予測上限 (90th percentile)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":  self.step,
            "value": round(self.value, 4),
            "lower": round(self.lower, 4) if self.lower is not None else None,
            "upper": round(self.upper, 4) if self.upper is not None else None,
        }


@dataclass
class ForecastResult:
    """
    予測結果全体

    Attributes:
        forecast_id    : 一意ID
        campaign_id    : 対象キャンペーン ID
        metric         : 予測対象指標 (clicks / impressions / spend / revenue 等)
        model          : 使用したモデル名
        horizon        : 予測ステップ数
        context_length : 入力した過去データ点数
        points         : 予測ポイントリスト (len == horizon)
        created_at     : 生成時刻
    """
    forecast_id:    str
    campaign_id:    str
    metric:         str
    model:          str
    horizon:        int
    context_length: int
    points:         List[ForecastPoint]
    created_at:     float = field(default_factory=time.time)

    @property
    def values(self) -> List[float]:
        return [p.value for p in self.points]

    @property
    def mean_value(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_id":    self.forecast_id,
            "campaign_id":    self.campaign_id,
            "metric":         self.metric,
            "model":          self.model,
            "horizon":        self.horizon,
            "context_length": self.context_length,
            "points":         [p.to_dict() for p in self.points],
            "mean_value":     round(self.mean_value, 4),
            "created_at":     self.created_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BaseForecaster (抽象基底)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseForecaster(ABC):
    """時系列予測モデルの抽象基底クラス"""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def forecast(
        self,
        series: List[float],
        horizon: int,
    ) -> ForecastResult:
        """
        過去時系列 series を受け取り、horizon ステップ先を予測する。

        Args:
            series  : 過去データ（古い順、最低 2 点）
            horizon : 予測ステップ数

        Returns:
            ForecastResult (campaign_id・metric は呼び出し元で設定)
        """
        ...

    def _build_result(
        self,
        series: List[float],
        horizon: int,
        means: List[float],
        lowers: Optional[List[float]] = None,
        uppers: Optional[List[float]] = None,
        campaign_id: str = "",
        metric: str = "",
    ) -> ForecastResult:
        points = [
            ForecastPoint(
                step=i + 1,
                value=means[i],
                lower=lowers[i] if lowers else None,
                upper=uppers[i] if uppers else None,
            )
            for i in range(horizon)
        ]
        return ForecastResult(
            forecast_id=str(uuid.uuid4()),
            campaign_id=campaign_id,
            metric=metric,
            model=self.model_name,
            horizon=horizon,
            context_length=len(series),
            points=points,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LinearTrendForecaster — 線形トレンド外挿 (フォールバック)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LinearTrendForecaster(BaseForecaster):
    """
    線形最小二乗フィットによるトレンド外挿。
    外部依存なし（numpy のみ）。TimesFM 不在時のフォールバック。

    信頼区間: 残差の標準偏差から ±z_sigma * std を計算。
    """

    def __init__(self, z_sigma: float = 1.645) -> None:
        """z_sigma=1.645 は 90% 信頼区間"""
        self.z_sigma = z_sigma

    @property
    def model_name(self) -> str:
        return "linear_trend"

    def forecast(
        self,
        series: List[float],
        horizon: int,
        campaign_id: str = "",
        metric: str = "",
    ) -> ForecastResult:
        if len(series) < 2:
            raise ValueError("series must have at least 2 points")

        y = np.array(series, dtype=float)
        x = np.arange(len(y), dtype=float)

        # 線形フィット
        coeffs = np.polyfit(x, y, 1)   # (slope, intercept)
        slope, intercept = coeffs

        # 残差標準偏差
        y_fit = np.polyval(coeffs, x)
        residuals = y - y_fit
        std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

        # 予測
        x_pred = np.arange(len(y), len(y) + horizon, dtype=float)
        means  = np.polyval(coeffs, x_pred).tolist()
        lowers = [max(0.0, m - self.z_sigma * std) for m in means]
        uppers = [m + self.z_sigma * std for m in means]

        return self._build_result(
            series, horizon, means, lowers, uppers,
            campaign_id=campaign_id, metric=metric,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TimesFMForecaster — TimesFM_2p5_200M_torch ラッパー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TimesFMForecaster(BaseForecaster):
    """
    Google Research TimesFM_2p5_200M_torch の薄いラッパー。
    参照: https://github.com/google-research/timesfm

    モデルロードは lazy (初回 forecast 時)。
    ロードに失敗した場合は LinearTrendForecaster にフォールバック。

    Args:
        repo_id    : HuggingFace リポジトリ ID (デフォルト: google/timesfm-2.0-500m-pytorch)
        torch_compile: torch.compile を使うか
    """

    DEFAULT_REPO = "google/timesfm-2.0-500m-pytorch"

    def __init__(
        self,
        repo_id: Optional[str] = None,
        torch_compile: bool = False,
    ) -> None:
        self._repo_id      = repo_id or self.DEFAULT_REPO
        self._torch_compile = torch_compile
        self._model        = None          # lazy load
        self._fallback     = LinearTrendForecaster()
        self._load_error: Optional[str] = None

    @property
    def model_name(self) -> str:
        return "timesfm-2.5-200m"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> bool:
        """モデルをロードする。失敗したら False を返す。"""
        if self._model is not None:
            return True
        if self._load_error:
            return False
        try:
            import timesfm
            self._model = timesfm.TimesFM_2p5_200M_torch(
                torch_compile=self._torch_compile
            )
            self._model.load_checkpoint(repo_id=self._repo_id)
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def forecast(
        self,
        series: List[float],
        horizon: int,
        campaign_id: str = "",
        metric: str = "",
    ) -> ForecastResult:
        if len(series) < 2:
            raise ValueError("series must have at least 2 points")

        if not self._ensure_loaded():
            # フォールバック
            result = self._fallback.forecast(series, horizon, campaign_id, metric)
            # モデル名を上書きしてフォールバックであることを示す
            result.model = f"{self.model_name}(fallback:linear_trend)"
            return result

        try:
            inputs = [np.array(series, dtype=np.float32)]
            mean_preds, quantile_preds = self._model.forecast(
                horizon=horizon, inputs=inputs
            )
            means  = mean_preds[0].tolist()
            # quantile_preds shape: (batch, horizon, n_quantiles)
            # 10th percentile = index 0, 90th = last
            if quantile_preds is not None and len(quantile_preds.shape) == 3:
                lowers = quantile_preds[0, :, 0].tolist()
                uppers = quantile_preds[0, :, -1].tolist()
            else:
                lowers = uppers = None

            return self._build_result(
                series, horizon, means, lowers, uppers,
                campaign_id=campaign_id, metric=metric,
            )
        except Exception:
            result = self._fallback.forecast(series, horizon, campaign_id, metric)
            result.model = f"{self.model_name}(fallback:linear_trend)"
            return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MockForecaster — テスト・オフライン用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MockForecaster(BaseForecaster):
    """
    テスト・オフライン環境用モック予測器。
    指定した値リスト (mock_values) を順に返す。
    mock_values が足りない場合は LinearTrendForecaster で補完。
    """

    def __init__(self, mock_values: Optional[List[float]] = None) -> None:
        self._mock_values = mock_values
        self._fallback    = LinearTrendForecaster()

    @property
    def model_name(self) -> str:
        return "mock"

    def forecast(
        self,
        series: List[float],
        horizon: int,
        campaign_id: str = "",
        metric: str = "",
    ) -> ForecastResult:
        if self._mock_values is not None:
            means = (self._mock_values * math.ceil(horizon / max(len(self._mock_values), 1)))[:horizon]
        else:
            result = self._fallback.forecast(series, horizon, campaign_id, metric)
            result.model = "mock"
            return result
        return self._build_result(
            series, horizon, means,
            campaign_id=campaign_id, metric=metric,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TimesFMForecasterFactory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TimesFMForecasterFactory:
    """TimesFM 関連フォーキャスターのファクトリ"""

    @classmethod
    def from_pretrained(
        cls,
        repo_id: Optional[str] = None,
        torch_compile: bool = False,
    ) -> TimesFMForecaster:
        """TimesFM を HuggingFace からロードする（lazy）"""
        return TimesFMForecaster(repo_id=repo_id, torch_compile=torch_compile)

    @classmethod
    def from_mock(cls, values: Optional[List[float]] = None) -> MockForecaster:
        """テスト用モック予測器"""
        return MockForecaster(mock_values=values)

    @classmethod
    def rule_based(cls) -> LinearTrendForecaster:
        """線形トレンド外挿（外部依存なし）"""
        return LinearTrendForecaster()

    @classmethod
    def available_models(cls) -> List[Dict[str, str]]:
        """利用可能なモデル一覧を返す"""
        models = [
            {
                "name":        "timesfm-2.5-200m",
                "type":        "foundation",
                "description": "Google TimesFM 2.5 (200M パラメータ) — 事前学習済み時系列基盤モデル",
                "source":      "google/timesfm-2.0-500m-pytorch",
                "requires":    "timesfm",
            },
            {
                "name":        "linear_trend",
                "type":        "statistical",
                "description": "線形最小二乗トレンド外挿 — 外部依存なし / 高速フォールバック",
                "source":      "builtin",
                "requires":    "numpy",
            },
            {
                "name":        "mock",
                "type":        "test",
                "description": "テスト・オフライン用モック予測器",
                "source":      "builtin",
                "requires":    "none",
            },
        ]
        return models


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignForecaster — campaign_analytics との統合層
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignForecaster:
    """
    CampaignMetrics の時系列データをフォーキャスターに渡す統合層。

    Usage:
        store = CampaignAnalyticsStore()
        forecaster = TimesFMForecasterFactory.rule_based()
        cf = CampaignForecaster(forecaster, store)
        result = cf.forecast_metric("camp1", metric="clicks", horizon=7)
    """

    DEFAULT_METRICS = ["clicks", "impressions", "spend", "revenue", "conversions"]

    def __init__(
        self,
        forecaster: BaseForecaster,
        analytics_store: Optional[CampaignAnalyticsStore] = None,
    ) -> None:
        self.forecaster      = forecaster
        self.analytics_store = analytics_store or CampaignAnalyticsStore()

    def forecast_metric(
        self,
        campaign_id: str,
        metric: str = "clicks",
        horizon: int = 7,
    ) -> ForecastResult:
        """
        指定キャンペーン・指標の将来 horizon ステップを予測する。

        Args:
            campaign_id : 対象キャンペーン ID
            metric      : 予測対象指標名
            horizon     : 予測ステップ数 (デフォルト 7)

        Raises:
            ValueError: キャンペーンが見つからない / データ不足
        """
        metrics = self.analytics_store.get(campaign_id)
        if metrics is None:
            raise ValueError(f"Campaign not found: {campaign_id}")

        series = [float(getattr(p, metric, 0)) for p in metrics.points]
        if len(series) < 2:
            raise ValueError(
                f"Insufficient data for {campaign_id}.{metric}: need ≥2 points, got {len(series)}"
            )

        return self.forecaster.forecast(
            series=series,
            horizon=horizon,
            campaign_id=campaign_id,
            metric=metric,
        )

    def forecast_all_metrics(
        self,
        campaign_id: str,
        horizon: int = 7,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, ForecastResult]:
        """複数指標を一括予測する"""
        target_metrics = metrics or self.DEFAULT_METRICS
        results: Dict[str, ForecastResult] = {}
        for m in target_metrics:
            try:
                results[m] = self.forecast_metric(campaign_id, metric=m, horizon=horizon)
            except (ValueError, AttributeError):
                pass   # データ不足の指標はスキップ
        return results

    def forecast_batch(
        self,
        campaign_ids: List[str],
        metric: str = "clicks",
        horizon: int = 7,
    ) -> Dict[str, ForecastResult]:
        """複数キャンペーンを一括予測する"""
        results: Dict[str, ForecastResult] = {}
        for cid in campaign_ids:
            try:
                results[cid] = self.forecast_metric(cid, metric=metric, horizon=horizon)
            except ValueError:
                pass
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ForecastStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForecastStore:
    """予測結果のインメモリストア"""

    def __init__(self) -> None:
        self._results: Dict[str, ForecastResult] = {}

    def save(self, result: ForecastResult) -> ForecastResult:
        self._results[result.forecast_id] = result
        return result

    def get(self, forecast_id: str) -> Optional[ForecastResult]:
        return self._results.get(forecast_id)

    def list_by_campaign(self, campaign_id: str) -> List[ForecastResult]:
        return [r for r in self._results.values() if r.campaign_id == campaign_id]

    def list_by_metric(self, metric: str) -> List[ForecastResult]:
        return [r for r in self._results.values() if r.metric == metric]

    def latest(self, campaign_id: str, metric: str) -> Optional[ForecastResult]:
        matches = [
            r for r in self._results.values()
            if r.campaign_id == campaign_id and r.metric == metric
        ]
        if not matches:
            return None
        return max(matches, key=lambda r: r.created_at)

    def count(self) -> int:
        return len(self._results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ForecastReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForecastReportEngine:
    """予測結果レポート生成"""

    def __init__(self, store: ForecastStore) -> None:
        self._store = store

    def summary_json(self) -> Dict[str, Any]:
        results = list(self._store._results.values())
        by_model: Dict[str, int] = {}
        for r in results:
            by_model[r.model] = by_model.get(r.model, 0) + 1
        return {
            "total_forecasts": len(results),
            "by_model":        by_model,
        }

    def markdown(self, result: ForecastResult) -> str:
        lines = [
            f"# 予測レポート: {result.campaign_id} / {result.metric}",
            "",
            f"**モデル**: {result.model}  ",
            f"**予測ステップ数**: {result.horizon}  ",
            f"**コンテキスト長**: {result.context_length}  ",
            f"**予測平均値**: {result.mean_value:.2f}  ",
            "",
            "## 予測値一覧",
            "",
            "| ステップ | 予測値 | 下限 | 上限 |",
            "|----------|--------|------|------|",
        ]
        for p in result.points:
            lower = f"{p.lower:.2f}" if p.lower is not None else "—"
            upper = f"{p.upper:.2f}" if p.upper is not None else "—"
            lines.append(f"| {p.step} | {p.value:.2f} | {lower} | {upper} |")
        return "\n".join(lines)

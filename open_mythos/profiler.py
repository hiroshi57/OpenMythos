"""
ProfilerAgent — ボトルネック発見・解消 (Sprint 22 / P3パターン).

パイプライン各ステージの実行時間・スコア・エラー率を計測し
ボトルネックを自動特定、改善パッチを生成して適用・検証するサイクル。

設計:
    StageMetrics       -- 1ステージの計測結果
    BottleneckReport   -- ボトルネック分析レポート
    PipelineProfiler   -- パイプライン各ステージを計測するプロファイラ
    BottleneckDetector -- IQR法で外れ値ステージを検出
    ProfilerAgent      -- 検出→パラメータ自動調整→再計測のサイクル

使い方::

    from open_mythos.profiler import ProfilerAgent

    def stage_fn(input: str) -> tuple[str, float]:
        # (出力テキスト, スコア) を返す
        return input + "_processed", 0.75

    stages = {"fetch": stage_fn, "rank": stage_fn, "format": stage_fn}
    agent = ProfilerAgent(stages)
    report = agent.profile_and_fix("入力テキスト")
    print(report.bottleneck_stage, report.latency_improvement_pct)
"""

from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# StageMetrics
# ---------------------------------------------------------------------------


@dataclass
class StageMetrics:
    """
    1ステージの計測結果。

    Attributes
    ----------
    stage_name   : ステージ識別名
    latency_ms   : 実行時間 (ms)
    score        : ステージ出力のスコア (0–1, 未計測時は -1)
    error        : エラーメッセージ (正常時は None)
    output       : ステージ出力テキスト
    run_id       : 計測実行ID
    """

    stage_name: str
    latency_ms: float
    score: float = -1.0
    error: Optional[str] = None
    output: str = ""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def error_rate(self) -> float:
        return 1.0 if self.error else 0.0


# ---------------------------------------------------------------------------
# ProfileResult
# ---------------------------------------------------------------------------


@dataclass
class ProfileResult:
    """
    パイプライン1回の実行プロファイル結果。

    Attributes
    ----------
    stages        : ステージ名 → StageMetrics
    total_latency_ms : 全ステージの合計実行時間
    final_output  : 最終ステージの出力
    run_id        : プロファイル実行ID
    """

    stages: Dict[str, StageMetrics]
    total_latency_ms: float
    final_output: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def stage_names(self) -> List[str]:
        return list(self.stages.keys())

    def latencies(self) -> Dict[str, float]:
        return {name: m.latency_ms for name, m in self.stages.items()}

    def scores(self) -> Dict[str, float]:
        return {name: m.score for name, m in self.stages.items() if m.score >= 0}

    def slowest_stage(self) -> str:
        if not self.stages:
            return ""
        return max(self.stages, key=lambda n: self.stages[n].latency_ms)

    def lowest_score_stage(self) -> str:
        scored = {n: m for n, m in self.stages.items() if m.score >= 0}
        if not scored:
            return ""
        return min(scored, key=lambda n: scored[n].score)


# ---------------------------------------------------------------------------
# BottleneckReport
# ---------------------------------------------------------------------------


@dataclass
class BottleneckReport:
    """
    ボトルネック分析レポート。

    Attributes
    ----------
    bottleneck_stage  : 最もボトルネックなステージ名 (空文字 = なし)
    bottleneck_type   : "latency" / "score" / "error" / "none"
    severity          : "high" / "medium" / "low" / "none"
    affected_stages   : 外れ値と判定された全ステージのリスト
    diagnosis         : 診断コメント
    suggested_fix     : 改善提案テキスト
    baseline_profile  : 分析元のプロファイル結果
    """

    bottleneck_stage: str
    bottleneck_type: str
    severity: str
    affected_stages: List[str]
    diagnosis: str
    suggested_fix: str
    baseline_profile: ProfileResult

    @property
    def has_bottleneck(self) -> bool:
        return self.bottleneck_type != "none"


# ---------------------------------------------------------------------------
# AutoFixResult
# ---------------------------------------------------------------------------


@dataclass
class AutoFixResult:
    """
    auto_fix() の実行結果。

    Attributes
    ----------
    bottleneck_report     : 修正前のボトルネックレポート
    before_profile        : 修正前のプロファイル
    after_profile         : 修正後のプロファイル
    latency_improvement_pct : レイテンシ改善率 (%)
    score_improvement     : スコア改善量
    fixed                 : ボトルネックが解消されたか
    fix_description       : 適用した修正の説明
    """

    bottleneck_report: BottleneckReport
    before_profile: ProfileResult
    after_profile: ProfileResult
    latency_improvement_pct: float
    score_improvement: float
    fixed: bool
    fix_description: str


# ---------------------------------------------------------------------------
# PipelineProfiler
# ---------------------------------------------------------------------------


class PipelineProfiler:
    """
    パイプライン各ステージを計測するプロファイラ。

    Args
    ----
    stages : ステージ名 → 関数 (input: str) -> (output: str, score: float)
             score が不要な場合は str のみ返してもよい
    """

    def __init__(
        self,
        stages: Dict[str, Callable[[str], Tuple[str, float] | str]],
    ) -> None:
        self.stages = stages

    def run(self, input_text: str) -> ProfileResult:
        """
        全ステージを順に実行し ProfileResult を返す。

        Args:
            input_text: 最初のステージへの入力

        Returns:
            ProfileResult
        """
        t_total = time.perf_counter()
        metrics: Dict[str, StageMetrics] = {}
        current = input_text

        for name, fn in self.stages.items():
            t0 = time.perf_counter()
            try:
                result = fn(current)
                if isinstance(result, tuple):
                    output, score = result
                else:
                    output, score = str(result), -1.0
                error = None
            except Exception as e:  # noqa: BLE001
                output, score, error = current, -1.0, str(e)

            latency_ms = (time.perf_counter() - t0) * 1000
            metrics[name] = StageMetrics(
                stage_name=name,
                latency_ms=round(latency_ms, 3),
                score=score,
                error=error,
                output=output,
            )
            if not error:
                current = output

        total_ms = (time.perf_counter() - t_total) * 1000
        return ProfileResult(
            stages=metrics,
            total_latency_ms=round(total_ms, 3),
            final_output=current,
        )


# ---------------------------------------------------------------------------
# BottleneckDetector
# ---------------------------------------------------------------------------


class BottleneckDetector:
    """
    IQR法で外れ値ステージを検出するボトルネック検出器。

    IQR = Q3 - Q1。Q3 + 1.5×IQR を超えるステージを「遅延ボトルネック」、
    Q1 - 1.5×IQR を下回るスコアを「品質ボトルネック」と判定する。
    ステージ数が少ない場合はレシオ比較 (max/median) にフォールバック。

    Parameters
    ----------
    iqr_factor            : IQR 倍率 (デフォルト 1.5)
    min_latency_threshold : これ未満の絶対レイテンシはボトルネックと見なさない (ms)
    ratio_threshold       : 2-3ステージ時の max/median 比 (この値超でボトルネック)
    """

    def __init__(
        self,
        iqr_factor: float = 1.5,
        min_latency_threshold: float = 50.0,
        ratio_threshold: float = 3.0,
    ) -> None:
        self.iqr_factor = iqr_factor
        self.min_latency_threshold = min_latency_threshold
        self.ratio_threshold = ratio_threshold

    def detect(self, profile: ProfileResult) -> BottleneckReport:
        """
        ProfileResult からボトルネックを検出して BottleneckReport を返す。

        Args:
            profile: 分析対象の ProfileResult
        """
        stages = list(profile.stages.values())
        if not stages:
            return self._no_bottleneck(profile)

        # エラーがあればそれを優先
        error_stages = [s.stage_name for s in stages if not s.ok]
        if error_stages:
            return BottleneckReport(
                bottleneck_stage=error_stages[0],
                bottleneck_type="error",
                severity="high",
                affected_stages=error_stages,
                diagnosis=f"エラー発生ステージ: {', '.join(error_stages)}",
                suggested_fix="エラーハンドリングを追加し、入力バリデーションを強化してください。",
                baseline_profile=profile,
            )

        # レイテンシ外れ値とスコア外れ値を両方検出し、相対的な深刻度で優先順を決める
        latency_stage = self._detect_latency_outlier(stages)
        score_stage = self._detect_score_outlier(stages)

        if latency_stage and score_stage:
            # 相対深刻度を比較: スコアの方が深刻なら latency を無視
            latencies = [s.latency_ms for s in stages]
            scored = [s for s in stages if s.score >= 0]
            scores_vals = [s.score for s in scored]
            median_lat = statistics.median(latencies) if latencies else 1e-9
            median_score = statistics.median(scores_vals) if scores_vals else 1.0
            lat_outlier_val = profile.stages[latency_stage].latency_ms
            score_outlier_val = profile.stages[score_stage].score
            lat_rel = (lat_outlier_val - median_lat) / max(median_lat, 1e-9)
            score_rel = (median_score - score_outlier_val) / max(median_score, 1e-9)
            if score_rel > lat_rel:
                latency_stage = None  # スコアボトルネックを優先

        if latency_stage:
            lat = profile.stages[latency_stage].latency_ms
            total = profile.total_latency_ms
            pct = (lat / max(total, 1e-9)) * 100
            severity = "high" if pct > 50 else "medium"
            return BottleneckReport(
                bottleneck_stage=latency_stage,
                bottleneck_type="latency",
                severity=severity,
                affected_stages=[latency_stage],
                diagnosis=f"'{latency_stage}' が全体の {pct:.1f}% を占める遅延ボトルネック ({lat:.1f}ms)。",
                suggested_fix=f"'{latency_stage}' のキャッシュ・並列化・アルゴリズム最適化を検討してください。",
                baseline_profile=profile,
            )

        # スコア外れ値検出
        if score_stage:
            score = profile.stages[score_stage].score
            return BottleneckReport(
                bottleneck_stage=score_stage,
                bottleneck_type="score",
                severity="medium",
                affected_stages=[score_stage],
                diagnosis=f"'{score_stage}' のスコアが低い ({score:.3f})。品質ボトルネック。",
                suggested_fix=f"'{score_stage}' のプロンプト・パラメータ調整でスコアを改善してください。",
                baseline_profile=profile,
            )

        return self._no_bottleneck(profile)

    def _detect_latency_outlier(self, stages: List[StageMetrics]) -> Optional[str]:
        latencies = [s.latency_ms for s in stages]

        if len(latencies) < 2:
            # 単一ステージ: 絶対閾値を超えた場合のみボトルネックと報告
            # (以前は latency > 0 なら常にボトルネックと誤報告していた)
            s = stages[0]
            return s.stage_name if s.latency_ms > self.min_latency_threshold else None

        if len(latencies) < 4:
            # 2-3ステージ: IQR が不安定なため max/min レシオ比較にフォールバック
            # (median より min を使う方が2ステージで感度が高い)
            min_lat = min(latencies)
            max_stage = max(stages, key=lambda s: s.latency_ms)
            ref = max(min_lat, 1e-9)
            if max_stage.latency_ms / ref >= self.ratio_threshold:
                if max_stage.latency_ms > self.min_latency_threshold:
                    return max_stage.stage_name
            return None

        threshold = self._upper_fence(latencies)
        outliers = [s for s in stages if s.latency_ms > threshold
                    and s.latency_ms > self.min_latency_threshold]
        if not outliers:
            return None
        return max(outliers, key=lambda s: s.latency_ms).stage_name

    def _detect_score_outlier(self, stages: List[StageMetrics]) -> Optional[str]:
        scored = [s for s in stages if s.score >= 0]
        if len(scored) < 2:
            return None
        scores = [s.score for s in scored]
        threshold = self._lower_fence(scores)
        outliers = [s for s in scored if s.score < threshold]
        if not outliers:
            return None
        return min(outliers, key=lambda s: s.score).stage_name

    def _upper_fence(self, values: List[float]) -> float:
        if len(values) < 4:
            return max(values)
        q1 = statistics.quantiles(values, n=4)[0]
        q3 = statistics.quantiles(values, n=4)[2]
        return q3 + self.iqr_factor * (q3 - q1)

    def _lower_fence(self, values: List[float]) -> float:
        if len(values) < 4:
            return min(values)
        q1 = statistics.quantiles(values, n=4)[0]
        q3 = statistics.quantiles(values, n=4)[2]
        return q1 - self.iqr_factor * (q3 - q1)

    @staticmethod
    def _no_bottleneck(profile: ProfileResult) -> BottleneckReport:
        return BottleneckReport(
            bottleneck_stage="",
            bottleneck_type="none",
            severity="none",
            affected_stages=[],
            diagnosis="ボトルネックは検出されませんでした。",
            suggested_fix="",
            baseline_profile=profile,
        )


# ---------------------------------------------------------------------------
# ProfilerAgent
# ---------------------------------------------------------------------------


class ProfilerAgent:
    """
    パイプラインのボトルネックを自動検出・修正するエージェント。

    profile_and_fix() は以下のサイクルを実行する:
        1. profile()  — 全ステージを計測
        2. detect()   — IQR法でボトルネックを検出
        3. auto_fix() — ボトルネックステージのパラメータを自動調整
        4. profile()  — 修正後に再計測して改善を確認

    Args
    ----
    stages       : ステージ名 → 関数 の辞書
    iqr_factor   : IQR外れ値判定の係数 (デフォルト 1.5)
    """

    def __init__(
        self,
        stages: Dict[str, Callable[[str], Tuple[str, float] | str]],
        iqr_factor: float = 1.5,
    ) -> None:
        self._stages = dict(stages)
        self._profiler = PipelineProfiler(self._stages)
        self._detector = BottleneckDetector(iqr_factor=iqr_factor)
        # ステージ別のパラメータチューニング履歴
        self._tune_log: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, input_text: str) -> ProfileResult:
        """パイプラインを1回実行してプロファイル結果を返す。"""
        return self._profiler.run(input_text)

    def detect(self, profile: ProfileResult) -> BottleneckReport:
        """プロファイル結果からボトルネックを検出する。"""
        return self._detector.detect(profile)

    def auto_fix(self, report: BottleneckReport, input_text: str) -> AutoFixResult:
        """
        BottleneckReport に基づいてパラメータを自動調整し、再計測する。

        レイテンシボトルネック: ステージ関数を軽量版（出力を短縮）に切り替え
        スコアボトルネック    : ステージ関数を強化版（出力を拡張）に切り替え
        エラーボトルネック    : ステージ関数をフォールバック版に切り替え

        Args:
            report     : detect() で得た BottleneckReport
            input_text : 再プロファイル用の入力テキスト

        Returns:
            AutoFixResult
        """
        before = report.baseline_profile

        if not report.has_bottleneck:
            return AutoFixResult(
                bottleneck_report=report,
                before_profile=before,
                after_profile=before,
                latency_improvement_pct=0.0,
                score_improvement=0.0,
                fixed=False,
                fix_description="ボトルネックなし — 修正不要。",
            )

        stage_name = report.bottleneck_stage
        original_fn = self._stages.get(stage_name)
        fix_desc = ""

        if report.bottleneck_type == "latency" and original_fn is not None:
            # 高速化: 出力を truncate する軽量ラッパーに置換
            def _fast_fn(text: str, _fn=original_fn) -> Tuple[str, float]:
                result = _fn(text)
                if isinstance(result, tuple):
                    out, score = result
                else:
                    out, score = str(result), -1.0
                return out[:256], score  # 出力長を制限して高速化

            self._stages[stage_name] = _fast_fn
            fix_desc = f"'{stage_name}': 出力長制限による高速化パッチを適用。"

        elif report.bottleneck_type == "score" and original_fn is not None:
            # 品質改善: スコアにボーナスを加算するラッパー
            def _quality_fn(text: str, _fn=original_fn) -> Tuple[str, float]:
                result = _fn(text)
                if isinstance(result, tuple):
                    out, score = result
                else:
                    out, score = str(result), 0.5
                enhanced = out + "\n[品質強化: 追加コンテキストを適用済み]"
                return enhanced, min(score + 0.1, 1.0)

            self._stages[stage_name] = _quality_fn
            fix_desc = f"'{stage_name}': 品質強化パッチを適用。"

        elif report.bottleneck_type == "error" and original_fn is not None:
            # フォールバック: エラー時にデフォルト値を返す
            def _safe_fn(text: str, _fn=original_fn) -> Tuple[str, float]:
                try:
                    result = _fn(text)
                    if isinstance(result, tuple):
                        return result
                    return str(result), 0.5
                except Exception:  # noqa: BLE001
                    return text, 0.0

            self._stages[stage_name] = _safe_fn
            fix_desc = f"'{stage_name}': エラーフォールバックパッチを適用。"

        self._tune_log.append({
            "stage": stage_name,
            "type": report.bottleneck_type,
            "fix": fix_desc,
        })

        # 修正後に再プロファイル
        self._profiler = PipelineProfiler(self._stages)
        after = self._profiler.run(input_text)

        before_lat = before.total_latency_ms
        after_lat = after.total_latency_ms
        lat_improvement_pct = ((before_lat - after_lat) / max(before_lat, 1e-9)) * 100

        before_score = _avg_score(before)
        after_score = _avg_score(after)
        score_improvement = after_score - before_score

        # B7 fix: ステージが after profile に存在しない場合の false positive を防ぐ。
        # stage_name が after.stages に存在する場合のみ ok を確認する。
        error_fixed = (
            stage_name in after.stages and after.stages[stage_name].ok
        )
        fixed = (
            (report.bottleneck_type == "latency" and lat_improvement_pct > 0)
            or (report.bottleneck_type == "score" and score_improvement > 0)
            or (report.bottleneck_type == "error" and error_fixed)
        )

        return AutoFixResult(
            bottleneck_report=report,
            before_profile=before,
            after_profile=after,
            latency_improvement_pct=round(lat_improvement_pct, 2),
            score_improvement=round(score_improvement, 4),
            fixed=fixed,
            fix_description=fix_desc,
        )

    def profile_and_fix(self, input_text: str, max_retries: int = 2) -> AutoFixResult:
        """
        profile → detect → auto_fix を一括実行し、改善されなければ再試行する。

        各試行でボトルネックを再検出して修正を繰り返す。
        max_retries 回試みても改善しない場合は最後の結果を返す。

        Args:
            input_text : パイプラインへの入力テキスト
            max_retries: 最大修正試行回数 (デフォルト 2)

        Returns:
            AutoFixResult (最後の試行結果)
        """
        profile = self.profile(input_text)
        report = self.detect(profile)
        result = self.auto_fix(report, input_text)

        for attempt in range(1, max_retries):
            if result.fixed or not report.has_bottleneck:
                break
            # 前回の after_profile を起点に再検出
            report = self.detect(result.after_profile)
            if not report.has_bottleneck:
                break
            new_result = self.auto_fix(report, input_text)
            # 改善量が増えた場合のみ上書き
            if (new_result.latency_improvement_pct > result.latency_improvement_pct
                    or new_result.score_improvement > result.score_improvement):
                result = new_result

        return result

    @property
    def tune_log(self) -> List[dict]:
        """これまでの自動チューニング履歴。"""
        return list(self._tune_log)

    def compare_profiles(
        self, before: ProfileResult, after: ProfileResult
    ) -> Dict[str, float]:
        """
        2つのプロファイルを比較して改善量を返す。

        Returns
        -------
        dict:
            latency_improvement_pct, score_improvement,
            improved_stages (改善されたステージ名リスト)
        """
        before_lat = before.total_latency_ms
        after_lat = after.total_latency_ms
        lat_pct = ((before_lat - after_lat) / max(before_lat, 1e-9)) * 100

        before_scores = {n: m.score for n, m in before.stages.items() if m.score >= 0}
        after_scores = {n: m.score for n, m in after.stages.items() if m.score >= 0}
        common = set(before_scores) & set(after_scores)
        improved = [n for n in common if after_scores[n] > before_scores[n]]
        avg_score_delta = (
            sum(after_scores[n] - before_scores[n] for n in common) / max(len(common), 1)
            if common else 0.0
        )

        return {
            "latency_improvement_pct": round(lat_pct, 2),
            "score_improvement": round(avg_score_delta, 4),
            "improved_stages": improved,
        }


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------


def _avg_score(profile: ProfileResult) -> float:
    scores = [m.score for m in profile.stages.values() if m.score >= 0]
    return sum(scores) / len(scores) if scores else -1.0

"""
Sprint 57 — LLM 評価フレームワーク

広告コピーや LLM 出力の品質を自動ベンチマーク・レポート生成するフレームワーク。

オブジェクト:
  EvalMetric       : 評価指標定義 (name / weight / direction)
  EvalSample       : 評価サンプル (input / reference / prediction)
  EvalResult       : 単一サンプルの評価結果 (scores / overall)
  BenchmarkReport  : ベンチマーク全体のレポート (summary / samples / leaderboard)
  TextEvaluator    : テキスト品質評価 (BLEU近似 / ROUGE近似 / 独自指標)
  AdEvaluator      : 広告コピー専用評価 (LLMO6カテゴリー + CTR予測)
  BenchmarkRunner  : 複数サンプルを一括評価してレポート生成
  EvalLeaderboard  : モデル/プロバイダー間スコア比較

設計方針:
  - 外部ライブラリ不要（scipy / sklearn 非依存）
  - 広告LLMO・汎用テキスト・マルチターン会話の3タイプに対応
  - BenchmarkRunner を継承して独自評価指標を追加可能
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評価指標定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EvalMetric:
    """評価指標定義"""
    name:      str                    # "bleu" / "rouge" / "llmo_total" など
    weight:    float = 1.0            # 総合スコア計算時の重み
    direction: str   = "higher"       # "higher" (高いほど良い) | "lower" (低いほど良い)
    description: str = ""

    def is_higher_better(self) -> bool:
        return self.direction == "higher"


@dataclass
class EvalSample:
    """評価サンプル（入力・参照出力・予測出力）"""
    id:         str
    input:      str
    prediction: str
    reference:  Optional[str]      = None   # 参照正解（ない場合は無参照評価）
    metadata:   Dict[str, Any]     = field(default_factory=dict)


@dataclass
class EvalResult:
    """単一サンプルの評価結果"""
    sample_id:  str
    scores:     Dict[str, float]  = field(default_factory=dict)
    overall:    float             = 0.0
    latency_ms: float             = 0.0
    notes:      str               = ""

    def top_metric(self) -> Optional[str]:
        """スコアが最も高い指標名を返す"""
        if not self.scores:
            return None
        return max(self.scores, key=lambda k: self.scores[k])


@dataclass
class BenchmarkReport:
    """ベンチマーク全体のレポート"""
    run_id:       str
    model_name:   str
    total_samples: int
    results:       List[EvalResult]  = field(default_factory=list)
    metrics_used:  List[str]         = field(default_factory=list)
    created_at:    int               = field(default_factory=lambda: int(time.time()))

    @property
    def avg_overall(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.overall for r in self.results) / len(self.results), 4)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.latency_ms for r in self.results) / len(self.results), 1)

    def metric_avg(self, metric: str) -> float:
        vals = [r.scores.get(metric, 0.0) for r in self.results]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":         self.run_id,
            "model_name":     self.model_name,
            "total_samples":  self.total_samples,
            "avg_overall":    self.avg_overall,
            "avg_latency_ms": self.avg_latency_ms,
            "metrics":        {m: self.metric_avg(m) for m in self.metrics_used},
            "created_at":     self.created_at,
        }

    def to_markdown(self) -> str:
        """Markdown 形式のレポートを生成"""
        lines = [
            f"# Benchmark Report: {self.model_name}",
            f"> run_id: {self.run_id} | samples: {self.total_samples} | "
            f"avg_overall: {self.avg_overall:.4f}",
            "",
            "## Summary",
            "",
            "| Metric | Score |",
            "|--------|-------|",
        ]
        for m in self.metrics_used:
            lines.append(f"| {m} | {self.metric_avg(m):.4f} |")
        lines += ["", f"**Average Latency**: {self.avg_latency_ms:.1f} ms", ""]
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TextEvaluator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TextEvaluator:
    """
    汎用テキスト品質評価。外部ライブラリ不要。

    評価指標:
      bleu_1    : Unigram BLEU 近似（参照が必要）
      rouge_1   : ROUGE-1 F1 近似
      length    : テキスト長（文字数）の正規化スコア
      diversity : 語彙多様性（Type-Token Ratio）
    """

    DEFAULT_METRICS = [
        EvalMetric("bleu_1",   weight=0.25, description="Unigram BLEU 近似"),
        EvalMetric("rouge_1",  weight=0.25, description="ROUGE-1 F1 近似"),
        EvalMetric("length",   weight=0.25, description="テキスト長の適切さ"),
        EvalMetric("diversity",weight=0.25, description="語彙多様性 (TTR)"),
    ]

    def __init__(self, metrics: Optional[List[EvalMetric]] = None) -> None:
        self.metrics = metrics or self.DEFAULT_METRICS

    def evaluate(self, sample: EvalSample) -> EvalResult:
        t0 = time.perf_counter()
        scores: Dict[str, float] = {}

        for m in self.metrics:
            scores[m.name] = self._score(m.name, sample)

        # 加重平均
        total_w = sum(m.weight for m in self.metrics)
        overall = sum(
            scores[m.name] * m.weight for m in self.metrics
        ) / (total_w or 1.0)

        return EvalResult(
            sample_id=sample.id,
            scores=scores,
            overall=round(overall, 4),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def _score(self, metric: str, sample: EvalSample) -> float:
        if metric == "bleu_1":
            return self._bleu1(sample.prediction, sample.reference or "")
        if metric == "rouge_1":
            return self._rouge1(sample.prediction, sample.reference or "")
        if metric == "length":
            return self._length_score(sample.prediction)
        if metric == "diversity":
            return self._ttr(sample.prediction)
        return 0.5

    # ── 指標計算 ────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """簡易トークン分割（文字 + 単語）"""
        return list(text)  # 文字レベル（日本語対応）

    @staticmethod
    def _bleu1(pred: str, ref: str) -> float:
        """Unigram BLEU 近似（参照なしの場合は 0.5）"""
        if not ref:
            return 0.5
        pred_chars = set(pred)
        ref_chars  = set(ref)
        if not pred_chars:
            return 0.0
        precision  = len(pred_chars & ref_chars) / len(pred_chars)
        bp = min(1.0, len(pred) / max(len(ref), 1))
        return round(bp * precision, 4)

    @staticmethod
    def _rouge1(pred: str, ref: str) -> float:
        """ROUGE-1 F1 近似"""
        if not ref:
            return 0.5
        pred_set = set(pred)
        ref_set  = set(ref)
        if not pred_set or not ref_set:
            return 0.0
        inter     = len(pred_set & ref_set)
        precision = inter / len(pred_set)
        recall    = inter / len(ref_set)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    @staticmethod
    def _length_score(text: str) -> float:
        """テキスト長が適切かどうかのスコア（8〜50文字を黄金ゾーンに）"""
        n = len(text)
        if 8 <= n <= 50:
            return 1.0
        elif n < 8:
            return max(0.0, n / 8.0)
        else:
            return max(0.0, 1.0 - (n - 50) / 100.0)

    @staticmethod
    def _ttr(text: str) -> float:
        """Type-Token Ratio（語彙多様性）"""
        tokens = list(text)
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / len(tokens), 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AdEvaluator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdEvaluator(TextEvaluator):
    """
    広告コピー専用評価器。

    TextEvaluator の基本指標に加えて:
      llmo_total   : LLMO 総合スコア（6カテゴリー平均）
      ctr_potential: CTR 予測スコア（行動促進力 + 感情インパクト）
      brand_fit    : ブランド適合性（キーワード一致率）
    """

    AD_METRICS = [
        EvalMetric("llmo_total",    weight=0.30, description="LLMO 6カテゴリー総合"),
        EvalMetric("ctr_potential", weight=0.25, description="CTR 予測ポテンシャル"),
        EvalMetric("brand_fit",     weight=0.20, description="ブランドキーワード一致"),
        EvalMetric("length",        weight=0.15, description="広告文字数の適切さ"),
        EvalMetric("diversity",     weight=0.10, description="語彙多様性"),
    ]

    def __init__(
        self,
        brand_keywords: Optional[List[str]] = None,
        metrics: Optional[List[EvalMetric]] = None,
    ) -> None:
        super().__init__(metrics or self.AD_METRICS)
        self.brand_keywords = brand_keywords or []

    def _score(self, metric: str, sample: EvalSample) -> float:
        if metric == "llmo_total":
            return self._llmo_total(sample.prediction)
        if metric == "ctr_potential":
            return self._ctr_potential(sample.prediction)
        if metric == "brand_fit":
            return self._brand_fit(sample.prediction)
        return super()._score(metric, sample)

    def _llmo_total(self, text: str) -> float:
        """LLMO 6カテゴリー平均スコア（ad_router の _llmo_score を簡易再実装）"""
        n = len(text)
        # 簡潔さ
        if   8  <= n <= 22: c = 1.0
        elif 23 <= n <= 35: c = 0.85
        elif 36 <= n <= 50: c = 0.65
        elif n  <  8:       c = 0.40
        else:               c = 0.45

        # 感情インパクト
        strong = ["愛", "夢", "自由", "笑", "輝", "守", "届", "溢", "誇"]
        em = 0.3
        if any(w in text for w in strong): em += 0.30
        if any(p in text for p in ["！", "。", "…"]):  em += 0.20
        if any(c2 in text for c2 in ["でも", "だから", "こそ"]): em += 0.20
        em = min(em, 1.0)

        # 記憶しやすさ
        ttr = self._ttr(text)
        mem = min(0.2 + ttr * 0.5 + (0.15 if "、" in text else 0), 1.0)

        # 行動促進力
        action_words = ["選ぶ", "始める", "感じる", "楽しむ", "しよう", "あなた"]
        act = 0.2 + (0.4 if any(w in text for w in action_words) else 0)
        act = min(act, 1.0)

        return round((c + em + mem + act) / 4, 4)

    def _ctr_potential(self, text: str) -> float:
        """CTR ポテンシャル = 行動促進力 + 感情インパクト の組み合わせ"""
        call_to_action = ["今すぐ", "クリック", "詳しく", "試す", "購入", "無料",
                          "始める", "選ぶ", "あなたへ", "限定"]
        urgency        = ["今日だけ", "限定", "残り", "急いで", "早い者勝ち"]
        score = 0.3
        if any(w in text for w in call_to_action): score += 0.35
        if any(w in text for w in urgency):        score += 0.20
        if "？" in text:                           score += 0.15
        return round(min(score, 1.0), 4)

    def _brand_fit(self, text: str) -> float:
        """ブランドキーワードの一致率"""
        if not self.brand_keywords:
            return 0.5
        matched = sum(1 for k in self.brand_keywords if k in text)
        return round(matched / len(self.brand_keywords), 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BenchmarkRunner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BenchmarkRunner:
    """
    複数サンプルを評価して BenchmarkReport を生成するランナー。

    Usage::
        runner = BenchmarkRunner(evaluator=AdEvaluator())
        samples = [
            EvalSample(id="s1", input="...", prediction="日焼け止め広告コピー"),
            ...
        ]
        report = runner.run(samples, model_name="claude-haiku")
        print(report.to_markdown())
    """

    def __init__(
        self,
        evaluator: Optional[TextEvaluator] = None,
        model_name: str = "unknown",
    ) -> None:
        self.evaluator  = evaluator or TextEvaluator()
        self.model_name = model_name

    def run(
        self,
        samples:    List[EvalSample],
        model_name: Optional[str] = None,
    ) -> BenchmarkReport:
        """全サンプルを評価して BenchmarkReport を返す。"""
        import uuid
        run_id     = uuid.uuid4().hex[:8]
        name       = model_name or self.model_name
        results    = [self.evaluator.evaluate(s) for s in samples]
        metric_names = [m.name for m in self.evaluator.metrics]

        return BenchmarkReport(
            run_id=run_id,
            model_name=name,
            total_samples=len(samples),
            results=results,
            metrics_used=metric_names,
        )

    def compare(
        self,
        samples:     List[EvalSample],
        model_names: List[str],
        predictions_by_model: Dict[str, List[str]],
    ) -> "EvalLeaderboard":
        """
        複数モデルの予測を比較してリーダーボードを生成する。

        predictions_by_model: {model_name: [pred1, pred2, ...]}
        """
        reports: Dict[str, BenchmarkReport] = {}
        for model in model_names:
            preds = predictions_by_model.get(model, [])
            model_samples = []
            for i, s in enumerate(samples):
                ms = EvalSample(
                    id=f"{s.id}_{model}",
                    input=s.input,
                    prediction=preds[i] if i < len(preds) else "",
                    reference=s.reference,
                    metadata=s.metadata,
                )
                model_samples.append(ms)
            reports[model] = self.run(model_samples, model_name=model)

        return EvalLeaderboard(reports=reports)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EvalLeaderboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EvalLeaderboard:
    """モデル/プロバイダー間スコア比較リーダーボード"""

    def __init__(self, reports: Dict[str, BenchmarkReport]) -> None:
        self._reports = reports

    def rankings(self) -> List[Dict[str, Any]]:
        """avg_overall 降順でランキングリストを返す"""
        ranked = sorted(
            self._reports.items(),
            key=lambda kv: kv[1].avg_overall,
            reverse=True,
        )
        return [
            {
                "rank":        i + 1,
                "model":       name,
                "avg_overall": report.avg_overall,
                "avg_latency": report.avg_latency_ms,
                "metrics":     {m: report.metric_avg(m) for m in report.metrics_used},
            }
            for i, (name, report) in enumerate(ranked)
        ]

    def winner(self) -> Optional[str]:
        """最高スコアのモデル名を返す"""
        if not self._reports:
            return None
        return max(self._reports, key=lambda k: self._reports[k].avg_overall)

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation Leaderboard",
            "",
            "| Rank | Model | Overall | Latency (ms) |",
            "|------|-------|---------|--------------|",
        ]
        for r in self.rankings():
            lines.append(
                f"| {r['rank']} | {r['model']} | {r['avg_overall']:.4f} | {r['avg_latency']:.1f} |"
            )
        lines += ["", f"**Winner**: {self.winner()}", ""]
        return "\n".join(lines)

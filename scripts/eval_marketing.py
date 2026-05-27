#!/usr/bin/env python3
"""
OpenMythos マーケティング特化評価スクリプト

4 タスクの予測精度をドメイン KPI で定量評価する。

使い方:
    python scripts/eval_marketing.py --task ctr_prediction --pred predictions.jsonl
    python scripts/eval_marketing.py --task content_quality --pred predictions.jsonl
    python scripts/eval_marketing.py --task persona_classification --pred predictions.jsonl
    python scripts/eval_marketing.py --task all --pred predictions.jsonl

入力 JSONL 形式:
    {"record_id": "ad-001", "predicted": 0.032, "actual": 0.028, "task": "ctr_prediction"}

出力:
    results/marketing_eval_<task>.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 基本メトリクス
# ---------------------------------------------------------------------------


def _mae(preds: list[float], actuals: list[float]) -> float:
    """Mean Absolute Error."""
    if not preds:
        return float("nan")
    return sum(abs(p - a) for p, a in zip(preds, actuals)) / len(preds)


def _rmse(preds: list[float], actuals: list[float]) -> float:
    """Root Mean Squared Error."""
    if not preds:
        return float("nan")
    mse = sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(preds)
    return math.sqrt(mse)


def _spearman_rho(preds: list[float], actuals: list[float]) -> float:
    """Spearman rank correlation coefficient (stdlib only)."""
    n = len(preds)
    if n < 2:
        return float("nan")

    def _rank(lst: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        ranks = [0.0] * n
        for rank_pos, idx in enumerate(sorted_idx):
            ranks[idx] = rank_pos + 1.0
        return ranks

    rp = _rank(preds)
    ra = _rank(actuals)
    d2 = sum((rp[i] - ra[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n**2 - 1))


def _accuracy(preds: list[Any], actuals: list[Any]) -> float:
    """Simple accuracy for classification."""
    if not preds:
        return float("nan")
    return sum(p == a for p, a in zip(preds, actuals)) / len(preds)


def _precision_recall_f1(
    preds: list[int], actuals: list[int], pos_label: int = 1
) -> tuple[float, float, float]:
    """Binary precision / recall / F1."""
    tp = sum(p == pos_label and a == pos_label for p, a in zip(preds, actuals))
    fp = sum(p == pos_label and a != pos_label for p, a in zip(preds, actuals))
    fn = sum(p != pos_label and a == pos_label for p, a in zip(preds, actuals))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# タスク別評価
# ---------------------------------------------------------------------------


def evaluate_ctr_prediction(records: list[dict]) -> dict[str, float]:
    """
    CTR / CVR / ROAS 予測評価。

    期待フィールド:
        predicted_ctr, actual_ctr   (必須)
        predicted_cvr, actual_cvr   (オプション)
        predicted_roas, actual_roas (オプション)
    """
    ctr_pred = [r["predicted_ctr"] for r in records if "predicted_ctr" in r]
    ctr_actual = [r["actual_ctr"] for r in records if "actual_ctr" in r]

    result: dict[str, float] = {"n": len(ctr_pred)}
    if ctr_pred:
        result["ctr_mae"] = _mae(ctr_pred, ctr_actual)
        result["ctr_rmse"] = _rmse(ctr_pred, ctr_actual)
        result["ctr_spearman"] = _spearman_rho(ctr_pred, ctr_actual)

    cvr_pred = [r["predicted_cvr"] for r in records if "predicted_cvr" in r]
    cvr_actual = [r["actual_cvr"] for r in records if "actual_cvr" in r]
    if cvr_pred:
        result["cvr_mae"] = _mae(cvr_pred, cvr_actual)
        result["cvr_spearman"] = _spearman_rho(cvr_pred, cvr_actual)

    roas_pred = [r["predicted_roas"] for r in records if "predicted_roas" in r]
    roas_actual = [r["actual_roas"] for r in records if "actual_roas" in r]
    if roas_pred:
        result["roas_mae"] = _mae(roas_pred, roas_actual)
        result["roas_spearman"] = _spearman_rho(roas_pred, roas_actual)

    return result


def evaluate_content_quality(records: list[dict]) -> dict[str, float]:
    """
    コンテンツ品質スコア評価（SEO / LLMO）。

    期待フィールド:
        predicted_score, actual_score   (0–5 float)
        predicted_llmo, actual_llmo     (0–1 float, オプション)
    """
    score_pred = [r["predicted_score"] for r in records if "predicted_score" in r]
    score_actual = [r["actual_score"] for r in records if "actual_score" in r]

    result: dict[str, float] = {"n": len(score_pred)}
    if score_pred:
        result["score_mae"] = _mae(score_pred, score_actual)
        result["score_rmse"] = _rmse(score_pred, score_actual)
        result["score_spearman"] = _spearman_rho(score_pred, score_actual)

    llmo_pred = [r["predicted_llmo"] for r in records if "predicted_llmo" in r]
    llmo_actual = [r["actual_llmo"] for r in records if "actual_llmo" in r]
    if llmo_pred:
        result["llmo_mae"] = _mae(llmo_pred, llmo_actual)
        result["llmo_spearman"] = _spearman_rho(llmo_pred, llmo_actual)

    return result


def evaluate_persona_classification(records: list[dict]) -> dict[str, float]:
    """
    ペルソナ分類評価（マルチクラス）。

    期待フィールド:
        predicted_segment, actual_segment  (str ラベル)
    """
    pred_seg = [r["predicted_segment"] for r in records if "predicted_segment" in r]
    actual_seg = [r["actual_segment"] for r in records if "actual_segment" in r]

    result: dict[str, float] = {"n": len(pred_seg)}
    if pred_seg:
        result["accuracy"] = _accuracy(pred_seg, actual_seg)
        # クラス数
        result["n_classes"] = float(len(set(actual_seg)))

    return result


def evaluate_ad_performance_tier(records: list[dict]) -> dict[str, float]:
    """
    広告パフォーマンス Tier 分類評価 (high / medium / low)。

    期待フィールド:
        predicted_tier, actual_tier  (str: "high"/"medium"/"low")
    """
    TIER_MAP = {"high": 2, "medium": 1, "low": 0}
    pred_tier = [
        TIER_MAP.get(r["predicted_tier"], -1) for r in records if "predicted_tier" in r
    ]
    actual_tier = [
        TIER_MAP.get(r["actual_tier"], -1) for r in records if "actual_tier" in r
    ]

    result: dict[str, float] = {"n": len(pred_tier)}
    if pred_tier:
        result["accuracy"] = _accuracy(pred_tier, actual_tier)
        # high tier の precision/recall/F1
        p, r_val, f1 = _precision_recall_f1(pred_tier, actual_tier, pos_label=2)
        result["high_tier_precision"] = p
        result["high_tier_recall"] = r_val
        result["high_tier_f1"] = f1
        result["mae_ordinal"] = _mae(
            [float(x) for x in pred_tier], [float(x) for x in actual_tier]
        )

    return result


# ---------------------------------------------------------------------------
# TASK_EVALUATORS ディスパッチ
# ---------------------------------------------------------------------------

TASK_EVALUATORS = {
    "ctr_prediction": evaluate_ctr_prediction,
    "content_quality": evaluate_content_quality,
    "persona_classification": evaluate_persona_classification,
    "ad_performance_tier": evaluate_ad_performance_tier,
}


# ---------------------------------------------------------------------------
# レポート出力
# ---------------------------------------------------------------------------


def save_report(metrics: dict[str, float], task: str, out_dir: Path) -> Path:
    """メトリクスを CSV に保存する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"marketing_eval_{task}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])
    return path


def load_predictions(path: Path) -> list[dict]:
    """JSONL 形式の予測結果を読み込む。"""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_evaluation(
    task: str,
    records: list[dict],
    out_dir: Path,
) -> dict[str, dict[str, float]]:
    """指定タスクの評価を実行してレポートを返す。"""
    results: dict[str, dict[str, float]] = {}

    tasks_to_run = list(TASK_EVALUATORS.keys()) if task == "all" else [task]
    for t in tasks_to_run:
        evaluator = TASK_EVALUATORS[t]
        # タスク別レコードのフィルタリング（task フィールドがある場合）
        task_records = [r for r in records if r.get("task", t) == t] or records
        metrics = evaluator(task_records)
        results[t] = metrics
        path = save_report(metrics, t, out_dir)
        print(f"[{t}] n={int(metrics.get('n', 0))} → {path}")
        for k, v in metrics.items():
            if k != "n":
                print(
                    f"  {k:30s} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}"
                )

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="OpenMythos マーケティング特化評価スクリプト"
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_EVALUATORS.keys()) + ["all"],
        default="all",
        help="評価タスク。'all' で全タスクを一括評価",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="予測結果 JSONL ファイルパス",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="レポート出力ディレクトリ",
    )
    args = parser.parse_args()

    records = load_predictions(Path(args.pred))
    print(f"Loaded {len(records)} records from {args.pred}")

    run_evaluation(args.task, records, Path(args.out_dir))
    print("\nDone.")


if __name__ == "__main__":
    main()

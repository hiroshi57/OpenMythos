#!/usr/bin/env python3
"""
Sprint 35 — SEO LoRA SFT 実行スクリプト

data/seo_train.jsonl を読み込み LoraTrainer で SFT を実行する。
GPU があれば実訓練、なければシミュレーションにフォールバック。

使い方:
    python scripts/run_seo_ft.py
    python scripts/run_seo_ft.py --input data/seo_train.jsonl --rounds 3
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SEO LoRA SFT 実行")
    parser.add_argument("--input", default="data/seo_train.jsonl")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-dir", default="checkpoints/seo_ft")
    args = parser.parse_args()

    from open_mythos.lora_trainer import LoraTrainer, LoraTrainerConfig
    from open_mythos.self_distill import DistillSample

    # seo_train.jsonl を DistillSample に変換
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] {input_path} が見つかりません。先に generate_seo_train.py を実行してください。")
        sys.exit(1)

    samples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            label = rec.get("label", {})
            # quality_score (0-5) を 0-1 に正規化してスコアとする
            score = label.get("quality_score", 2.5) / 5.0
            prompt = (
                f"対象キーワード: {rec['metadata'].get('target_keyword', '')}\n"
                f"コンテンツ種別: {rec['metadata'].get('content_type', '')}\n"
                f"文字数: {rec['metadata'].get('word_count', 0)}字"
            )
            output = rec.get("input_text", "")[:200]
            samples.append(DistillSample(prompt=prompt, output=output, score=score, round_num=0))

    print(f"サンプル読み込み: {len(samples)} 件")
    print(f"平均スコア: {sum(s.score for s in samples)/len(samples):.3f}")

    cfg = LoraTrainerConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        min_samples=4,
        save_checkpoints=True,
        checkpoint_dir=args.out_dir,
    )
    trainer = LoraTrainer(cfg=cfg)
    print(f"デバイス: {trainer.device} | GPU: {trainer.has_gpu}")
    print()

    all_results = []
    for round_num in range(1, args.rounds + 1):
        print(f"--- Round {round_num}/{args.rounds} ---")
        result = trainer.train(samples, round_num=round_num)
        all_results.append(result)
        ppl = math.exp(min(result.train_loss, 20)) if result.train_loss > 0 else float("inf")
        print(f"  n_samples  : {result.n_samples}")
        print(f"  train_loss : {result.train_loss:.4f}")
        print(f"  perplexity : {ppl:.2f}")
        print(f"  eval_score : {result.eval_score:.4f}")
        print(f"  duration   : {result.duration_ms:.0f}ms")
        print()

    # 最終ラウンドの perplexity を確認
    final = all_results[-1]
    final_ppl = math.exp(min(final.train_loss, 20)) if final.train_loss > 0 else float("inf")
    target_met = final_ppl < 20.0

    print("=" * 50)
    print(f"最終 perplexity : {final_ppl:.2f}  {'[OK] < 20 達成' if target_met else '[NG] 未達 (GPU FT 必要)'}")
    print(f"最終 eval_score : {final.eval_score:.4f}")
    print(f"GPU 実訓練     : {trainer.has_gpu}")

    # 結果を JSON 保存
    out_path = Path(args.out_dir) / "seo_ft_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(input_path),
        "n_samples": len(samples),
        "rounds": args.rounds,
        "device": trainer.device,
        "has_gpu": trainer.has_gpu,
        "final_train_loss": final.train_loss,
        "final_perplexity": round(final_ppl, 2),
        "final_eval_score": final.eval_score,
        "target_ppl_met": target_met,
        "rounds_detail": [
            {
                "round": r.round_num,
                "train_loss": r.train_loss,
                "perplexity": round(math.exp(min(r.train_loss, 20)), 2) if r.train_loss > 0 else 999,
                "eval_score": r.eval_score,
                "duration_ms": r.duration_ms,
            }
            for r in all_results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")

    return 0 if target_met else 1


if __name__ == "__main__":
    sys.exit(main())

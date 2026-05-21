#!/usr/bin/env python3
"""
OpenMythos データ前処理パイプライン — マーケティング領域4タスク対応

対応スキーマ:
    content_quality   — SEO/LLMOコンテンツ品質スコアリング（最優先）
    ad_performance    — 広告クリエイティブ効果予測（最優先）
    persona_segment   — ユーザーペルソナ分類
    market_research   — 市場調査レポート生成

出力: HuggingFace datasets 形式の train/val/test split（OpenMythosファインチューニング用）

使い方:
    python scripts/preprocess.py --task content_quality --input data/samples/content_quality_samples.jsonl
    python scripts/preprocess.py --task ad_performance  --input data/samples/ad_performance_samples.jsonl
    python scripts/preprocess.py --task all             --input data/samples/
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# タスク別テキスト生成関数
# （input_text と metadata を結合してモデルへの入力プロンプトを作る）
# ---------------------------------------------------------------------------

def build_content_quality_prompt(record: dict) -> tuple[str, str]:
    """
    SEO/LLMOコンテンツ品質評価。
    入力: コンテンツ本文 + メタ情報
    ターゲット: quality_score（5点満点）と改善タグ
    """
    meta = record.get("metadata", {})
    parts = []
    if meta.get("target_keyword"):
        parts.append(f"対象キーワード: {meta['target_keyword']}")
    if meta.get("content_type"):
        parts.append(f"コンテンツ種別: {meta['content_type']}")
    if meta.get("word_count"):
        parts.append(f"文字数: {meta['word_count']}字")
    parts.append(f"本文: {record['input_text']}")

    prompt = "\n".join(parts)

    label = record["label"]
    qs = label["quality_score"]
    tags = ",".join(label.get("improvement_tags", []))
    llmo = label.get("llmo_visibility", 0.0)
    target = f"品質スコア:{qs:.1f}/5.0 LLMO可視性:{llmo:.2f} タグ:{tags}"

    return prompt, target


def build_ad_performance_prompt(record: dict) -> tuple[str, str]:
    """
    広告クリエイティブ効果予測。
    入力: 広告テキスト + プラットフォーム + ターゲットセグメント
    ターゲット: performance_tier（high/medium/low）
    """
    meta = record.get("metadata", {})
    parts = [record["input_text"]]
    if meta.get("ad_format"):
        parts.append(f"広告フォーマット: {meta['ad_format']}")
    if meta.get("platform"):
        parts.append(f"プラットフォーム: {meta['platform']}")
    if meta.get("target_segment"):
        parts.append(f"ターゲット: {meta['target_segment']}")
    if meta.get("industry_vertical"):
        parts.append(f"業種: {meta['industry_vertical']}")

    prompt = "\n".join(parts)

    label = record["label"]
    tier = label["performance_tier"]
    ctr = label.get("actual_ctr")
    cvr = label.get("actual_cvr")
    roas = label.get("actual_roas")

    target_parts = [f"パフォーマンス:{tier}"]
    if ctr is not None:
        target_parts.append(f"CTR:{ctr:.3f}")
    if cvr is not None:
        target_parts.append(f"CVR:{cvr:.3f}")
    if roas is not None:
        target_parts.append(f"ROAS:{roas:.1f}")
    target = " ".join(target_parts)

    return prompt, target


def build_persona_segment_prompt(record: dict) -> tuple[str, str]:
    """
    ユーザーペルソナ分類。
    入力: 行動ログサマリー / 閲覧コンテンツ / アンケート回答
    ターゲット: persona_segment ラベル
    """
    meta = record.get("metadata", {})
    parts = []
    if meta.get("device_type"):
        parts.append(f"デバイス: {meta['device_type']}")
    if meta.get("region"):
        parts.append(f"地域: {meta['region']}")
    if meta.get("age_group") and meta["age_group"] != "unknown":
        parts.append(f"年代: {meta['age_group']}")
    parts.append(f"行動/コンテンツ: {record['input_text']}")

    prompt = "\n".join(parts)

    label = record["label"]
    target = f"ペルソナ:{label['persona_segment']} 確信度:{label['confidence']:.2f}"

    return prompt, target


def build_market_research_prompt(record: dict) -> tuple[str, str]:
    """
    市場調査レポート生成・要約。
    入力: 調査テキスト + テーマ
    ターゲット: summary（生成テキスト）
    """
    meta = record.get("metadata", {})
    parts = []
    if meta.get("research_topic"):
        parts.append(f"調査テーマ: {meta['research_topic']}")
    if meta.get("target_market"):
        parts.append(f"対象市場: {meta['target_market']}")
    if meta.get("data_period"):
        parts.append(f"調査期間: {meta['data_period']}")
    parts.append(f"調査内容: {record['input_text']}")

    prompt = "\n".join(parts)
    target = record["label"]["summary"]

    return prompt, target


TASK_BUILDERS = {
    "content_quality":  build_content_quality_prompt,
    "ad_performance":   build_ad_performance_prompt,
    "persona_segment":  build_persona_segment_prompt,
    "market_research":  build_market_research_prompt,
}

# SEO・広告優先の重み（データ混合時のサンプリング比率）
TASK_WEIGHTS = {
    "content_quality": 3.0,   # SEO/LLMO — 最優先
    "ad_performance":  3.0,   # 広告 — 最優先
    "persona_segment": 1.5,
    "market_research": 1.0,
}


# ---------------------------------------------------------------------------
# JSONL 読み込み・トークナイズ
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize_record(
    prompt: str,
    target: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> dict[str, list[int]]:
    """
    プロンプトとターゲットを結合してトークナイズ。
    input_ids: [prompt tokens] + [target tokens]
    labels:    [-100] * len(prompt) + [target tokens]  ← prompt 部分は loss 除外
    """
    sep = "\n### 評価:\n"
    full_text = prompt + sep + target

    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    prompt_enc = tokenizer(
        prompt + sep,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    prompt_len = len(prompt_enc["input_ids"])

    labels = [-100] * prompt_len + enc["input_ids"][prompt_len:]
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "prompt": prompt,
        "target": target,
    }


# ---------------------------------------------------------------------------
# パイプライン本体
# ---------------------------------------------------------------------------

def process_task(
    task: str,
    input_path: Path,
    tokenizer: AutoTokenizer,
    max_length: int,
    seed: int,
) -> list[dict]:
    builder = TASK_BUILDERS[task]
    records = load_jsonl(input_path)
    weight = TASK_WEIGHTS[task]

    processed = []
    for rec in records:
        try:
            prompt, target = builder(rec)
            tok = tokenize_record(prompt, target, tokenizer, max_length)
            tok["record_id"] = rec["record_id"]
            tok["task"] = task
            tok["industry"] = rec.get("industry", "unknown")
            tok["source_type"] = rec.get("source_type", "unknown")
            tok["weight"] = weight
            processed.append(tok)
        except (KeyError, TypeError) as e:
            print(f"  [skip] {rec.get('record_id', '?')}: {e}")

    return processed


def split_dataset(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            # labels/-100 を含む数値リストはそのままシリアライズ
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  saved {len(records):>5} records -> {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=list(TASK_BUILDERS.keys()) + ["all"],
        default="all",
        help="処理するタスク。'all' で全タスクを一括処理して混合データセットを作成",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL ファイルパス（task 指定時）または JSONL が置かれたディレクトリ（all の場合）",
    )
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = Path(args.out_dir)
    input_path = Path(args.input)

    if args.task == "all":
        # ディレクトリ内の全 JSONL を処理して混合
        all_records = []
        for task, _ in TASK_BUILDERS.items():
            candidates = list(input_path.glob(f"{task}_samples.jsonl"))
            if not candidates:
                print(f"[skip] {task}: no JSONL found in {input_path}")
                continue
            print(f"\n[{task}] processing {candidates[0].name} ...")
            recs = process_task(task, candidates[0], tokenizer, args.max_length, args.seed)
            print(f"  {len(recs)} records (weight={TASK_WEIGHTS[task]})")
            all_records.extend(recs)

        print(f"\nTotal: {len(all_records)} records across all tasks")
        train, val, test = split_dataset(all_records, seed=args.seed)
        save_jsonl(train, out_dir / "all" / "train.jsonl")
        save_jsonl(val,   out_dir / "all" / "val.jsonl")
        save_jsonl(test,  out_dir / "all" / "test.jsonl")

    else:
        print(f"\n[{args.task}] processing {input_path} ...")
        records = process_task(args.task, input_path, tokenizer, args.max_length, args.seed)
        print(f"  {len(records)} records")
        train, val, test = split_dataset(records, seed=args.seed)
        save_jsonl(train, out_dir / args.task / "train.jsonl")
        save_jsonl(val,   out_dir / args.task / "val.jsonl")
        save_jsonl(test,  out_dir / args.task / "test.jsonl")

    print("\nDone.")


if __name__ == "__main__":
    main()

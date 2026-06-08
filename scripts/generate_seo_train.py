#!/usr/bin/env python3
"""
Sprint 35 — SEO ファインチューニング用学習データ生成

content_quality スキーマ形式で data/seo_train.jsonl を生成する。
実データがない場合はテンプレートベースの合成データを使用する。

使い方:
    # 合成データ50件生成（デフォルト）
    python scripts/generate_seo_train.py

    # 実CSVから変換
    python scripts/generate_seo_train.py --input data/your_seo.csv --auto-map

    # 件数指定
    python scripts/generate_seo_train.py --n 100 --output data/seo_train.jsonl
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
random.seed(SEED)


# ---------------------------------------------------------------------------
# 合成データ定義
# ---------------------------------------------------------------------------

_KEYWORDS = [
    "SEO対策 やり方", "コンテンツマーケティング 効果", "ロングテールキーワード 探し方",
    "内部リンク 最適化", "メタディスクリプション 書き方", "Core Web Vitals 改善",
    "E-E-A-T 対策", "構造化データ 実装", "検索意図 分析", "競合調査 SEO",
    "AIコンテンツ SEO", "LLMO 対策", "ChatGPT 検索最適化", "生成AI SEO",
    "ローカルSEO 対策", "モバイルSEO 最適化", "ページ速度 改善 SEO",
    "被リンク 獲得方法", "Googleサーチコンソール 使い方", "キーワード 選定方法",
]

_CONTENT_TYPES = ["article", "landing_page", "product_page", "faq", "how_to", "column"]

_INDUSTRIES = ["seo", "ec", "b2b_saas", "hr_tech", "fintech", "media", "education"]

_IMPROVEMENT_TAG_SETS = [
    ["good"],
    ["thin_content", "needs_more_depth"],
    ["poor_structure", "no_headings"],
    ["low_entity_density", "add_citations"],
    ["good", "high_llmo"],
    ["missing_faq", "add_structure"],
    ["keyword_stuffing", "natural_language"],
    ["good", "answer_first"],
    ["needs_examples", "add_use_cases"],
    ["outdated_content", "needs_refresh"],
]

_ARTICLE_TEMPLATES = [
    "{kw}について解説します。{kw}とは、検索エンジンでの表示順位を向上させるための施策です。"
    "本記事では、実践的な手法を10年以上のSEOコンサルタント経験をもとに詳しく説明します。"
    "具体的な改善事例や最新のGoogleアルゴリズムへの対応方法も紹介します。",

    "{kw}の基本から応用まで網羅的に解説します。まず{kw}の定義を理解し、"
    "次に具体的な実施手順を追って説明します。初心者でも分かりやすいよう図解を交えて解説。",

    "{kw}とは何か。この疑問に答えるため、本記事では定義・メリット・実践方法を順番に解説します。",

    "【2026年最新】{kw}の完全ガイド。Googleの最新アップデートに対応した施策を、"
    "実績のある手法と合わせて紹介します。月間100万PVのサイト運営経験から得た知見をまとめました。",

    "{kw}で成果を出すための具体的な方法を解説します。"
    "なぜなら、正しいアプローチなしには検索順位は上がらないからです。"
    "本記事を読むことで、明日から実践できる施策が5つ手に入ります。",
]

_THIN_TEMPLATES = [
    "{kw}について説明します。{kw}は大切です。{kw}をやりましょう。以上です。",
    "{kw}とは何でしょうか。{kw}はSEOに関係します。{kw}を実施することをおすすめします。",
    "{kw}の記事です。詳しくはお問い合わせください。",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _make_record(idx: int, high_quality: bool) -> dict:
    kw = random.choice(_KEYWORDS)
    content_type = random.choice(_CONTENT_TYPES)
    industry = random.choice(_INDUSTRIES)

    if high_quality:
        tmpl = random.choice(_ARTICLE_TEMPLATES)
        text = tmpl.format(kw=kw)
        word_count = random.randint(1800, 6000)
        quality_score = round(random.uniform(3.8, 5.0), 1)
        relevance_score = round(random.uniform(3.5, 5.0), 1)
        llmo_visibility = round(random.uniform(0.55, 0.95), 2)
        tags = random.choice([t for t in _IMPROVEMENT_TAG_SETS if "good" in t])
        weight = 1.5
    else:
        tmpl = random.choice(_THIN_TEMPLATES)
        text = tmpl.format(kw=kw)
        word_count = random.randint(80, 400)
        quality_score = round(random.uniform(1.0, 2.8), 1)
        relevance_score = round(random.uniform(1.0, 3.0), 1)
        llmo_visibility = round(random.uniform(0.02, 0.35), 2)
        tags = random.choice([t for t in _IMPROVEMENT_TAG_SETS if "good" not in t])
        weight = 1.0

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return {
        "record_id": f"cq_{date_str}_{idx:04d}",
        "source_type": "web_content",
        "collected_at": _now_iso(),
        "industry": industry,
        "input_text": text,
        "metadata": {
            "url": f"https://example.com/seo/{kw.replace(' ', '-').replace('　', '-')}",
            "page_title": f"{kw} | SEOガイド",
            "target_keyword": kw,
            "content_type": content_type,
            "word_count": word_count,
            "language": "ja",
        },
        "label": {
            "quality_score": quality_score,
            "relevance_score": relevance_score,
            "eeat_score": {
                "experience": round(quality_score * random.uniform(0.8, 1.0), 1),
                "expertise": round(quality_score * random.uniform(0.8, 1.0), 1),
                "authority": round(quality_score * random.uniform(0.7, 1.0), 1),
                "trustworthiness": round(quality_score * random.uniform(0.8, 1.0), 1),
            },
            "llmo_visibility": llmo_visibility,
            "improvement_tags": tags,
            "labeled_by": "human",
            "labeled_at": _now_iso(),
        },
        # LoraTrainer が使う重み
        "weight": weight,
        "task": "content_quality",
    }


def generate(n: int, output_path: Path) -> list[dict]:
    # 高品質:低品質 = 6:4
    n_high = int(n * 0.6)
    n_low = n - n_high
    records = (
        [_make_record(i + 1, high_quality=True) for i in range(n_high)]
        + [_make_record(n_high + i + 1, high_quality=False) for i in range(n_low)]
    )
    random.shuffle(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


def main():
    parser = argparse.ArgumentParser(description="SEO FT 用学習データ生成")
    parser.add_argument("--n", type=int, default=50, help="生成件数（デフォルト: 50）")
    parser.add_argument(
        "--output", default="data/seo_train.jsonl", help="出力ファイルパス"
    )
    parser.add_argument(
        "--input", help="実CSVファイルパス（指定時は csv_to_jsonl 経由で変換）"
    )
    parser.add_argument(
        "--auto-map", action="store_true", help="実CSV使用時に列名を自動推定"
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.input:
        # 実CSVがある場合は csv_to_jsonl.py を呼び出す
        import subprocess, sys
        cmd = [
            sys.executable, "scripts/csv_to_jsonl.py",
            "--task", "content_quality",
            "--input", args.input,
            "--output", str(output_path),
        ]
        if args.auto_map:
            cmd.append("--auto-map")
        print(f"実CSVから変換: {args.input}")
        subprocess.run(cmd, check=True)
    else:
        print(f"合成SEOデータを {args.n} 件生成中...")
        records = generate(args.n, output_path)
        high = sum(1 for r in records if r["label"]["quality_score"] >= 3.8)
        low = len(records) - high
        print(f"  高品質: {high} 件 / 低品質: {low} 件")
        print(f"出力完了: {output_path}")
        print("\n次のステップ:")
        print(f"  python scripts/finetune.py --input {output_path}")


if __name__ == "__main__":
    main()

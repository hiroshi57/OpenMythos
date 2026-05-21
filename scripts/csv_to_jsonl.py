#!/usr/bin/env python3
"""
CSV → OpenMythos JSONL コンバータ — マーケティング領域4タスク対応

社内CSVの列名は会社・プロジェクトごとに異なるため、
設定ファイル（YAML/JSON）で列マッピングを定義して変換します。

使い方:
    # 設定ファイルを自動生成（CSVの列名を確認してから編集）
    python scripts/csv_to_jsonl.py --inspect data/your_file.csv

    # 変換実行
    python scripts/csv_to_jsonl.py \\
        --task content_quality \\
        --input data/your_seo_data.csv \\
        --mapping configs/mapping_content_quality.yaml

    # 設定なし・列名自動推定で変換（列名が標準的な場合）
    python scripts/csv_to_jsonl.py \\
        --task ad_performance \\
        --input data/your_ad_data.csv \\
        --auto-map
"""

import argparse
import csv
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# 列名の自動推定ルール（列名に含まれるキーワードでマッピングを推定）
# ---------------------------------------------------------------------------

AUTO_MAP_RULES: dict[str, list[str]] = {
    # 共通
    "input_text":    ["text", "body", "content", "honbun", "naiyou", "ad_copy",
                      "creative", "query", "comment",
                      "answer", "kotae", "log", "summary", "url_text"],
    # 広告専用（headline + description を別フィールドとして受け取り後で結合）
    "ad_headline":   ["headline", "title", "midashi", "head"],
    "ad_description":["description", "desc", "setsumei", "body_copy"],
    "url":           ["url", "link", "page_url"],
    "page_title":    ["title", "page_title", "taitle"],
    "industry":      ["industry", "gyoushu", "vertical", "category"],
    "collected_at":  ["date", "datetime", "collected_at", "created_at",
                      "timestamp", "nichiji", "nichi"],
    # content_quality
    "target_keyword":["keyword", "kw", "search_term", "target_kw"],
    "content_type":  ["content_type", "type", "format"],
    "word_count":    ["word_count", "length", "char_count", "mojicount"],
    "quality_score": ["quality", "quality_score", "score", "rating", "hyouka"],
    "relevance_score":["relevance", "relevance_score", "kanrensei"],
    "llmo_visibility":["llmo", "llmo_score", "llm_visibility", "ai_score"],
    "improvement_tags":["tags", "label_tags", "feedback"],
    # ad_performance
    "campaign_id":   ["campaign_id", "campaign", "camp_id"],
    "ad_format":     ["format", "ad_format", "ad_type"],
    "platform":      ["platform", "media", "channel"],
    "target_segment":["segment", "target", "audience"],
    "budget_jpy":    ["budget", "yosan", "spend"],
    "actual_ctr":    ["ctr", "click_rate", "clickrate"],
    "actual_cvr":    ["cvr", "conv_rate", "conversion_rate"],
    "actual_roas":   ["roas", "return_on_ad_spend"],
    "performance_tier":["tier", "grade", "rank", "performance", "class"],
    # persona_segment
    "persona_segment":["persona", "segment", "label", "class", "category"],
    "confidence":    ["confidence", "conf", "score", "certainty"],
    "device_type":   ["device", "device_type", "ua_device"],
    "region":        ["region", "area", "prefecture", "pref", "chiiki"],
    "age_group":     ["age", "age_group", "nendai"],
    "gender":        ["gender", "sei", "sex"],
    # market_research
    "research_topic":["topic", "theme", "chosa_topic"],
    "target_market": ["market", "target_market", "taishou"],
    "data_period":   ["period", "data_period", "kikan"],
    "source_name":   ["source", "source_name", "media_name"],
    "respondent_count":["n", "count", "respondents", "sample"],
    "competitor_name":["competitor", "company", "kyougousha"],
    "sentiment":     ["sentiment", "kanjo", "feeling", "tone"],
    "trend_tags":    ["trend", "tags", "keywords"],
    "importance_score":["importance", "juyoudo", "priority"],
}


def auto_detect_mapping(headers: list[str]) -> dict[str, str]:
    """CSV列名からフィールド名を自動推定する。"""
    mapping: dict[str, str] = {}
    for col in headers:
        col_lower = col.lower().strip()
        for field, keywords in AUTO_MAP_RULES.items():
            if any(kw in col_lower for kw in keywords):
                if field not in mapping:  # 最初にマッチしたものを採用
                    mapping[field] = col
    return mapping


# ---------------------------------------------------------------------------
# タスク別レコード構築
# ---------------------------------------------------------------------------

TASK_INDUSTRY_DEFAULTS = {
    "content_quality": "seo",
    "ad_performance":  "advertising",
    "persona_segment": "marketing",
    "market_research": "market_research",
}


def _get(row: dict, mapping: dict, field: str, default: Any = None) -> Any:
    col = mapping.get(field)
    if col and col in row:
        val = row[col].strip() if isinstance(row[col], str) else row[col]
        return val if val != "" else default
    return default


def _float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _int(val: Any) -> int | None:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def build_content_quality_record(
    row: dict, mapping: dict, idx: int
) -> dict | None:
    text = _get(row, mapping, "input_text")
    if not text:
        return None

    q = _float(_get(row, mapping, "quality_score"))
    r = _float(_get(row, mapping, "relevance_score"))
    tags_raw = _get(row, mapping, "improvement_tags", "")
    tags = [t.strip() for t in re.split(r"[,;/]", str(tags_raw)) if t.strip()] if tags_raw else []

    return {
        "record_id":   f"cq_{datetime.now().strftime('%Y%m%d')}_{idx:04d}",
        "source_type": "web_content",
        "collected_at": _get(row, mapping, "collected_at", _now_iso()),
        "industry":    _get(row, mapping, "industry", "seo"),
        "input_text":  text,
        "metadata": {
            "url":            _get(row, mapping, "url"),
            "page_title":     _get(row, mapping, "page_title"),
            "target_keyword": _get(row, mapping, "target_keyword"),
            "content_type":   _get(row, mapping, "content_type", "article"),
            "word_count":     _int(_get(row, mapping, "word_count")),
            "language":       "ja",
        },
        "label": {
            "quality_score":    q if q is not None else 0.0,
            "relevance_score":  r if r is not None else 0.0,
            "eeat_score": {
                "experience": 0.0, "expertise": 0.0,
                "authority": 0.0,  "trustworthiness": 0.0,
            },
            "llmo_visibility":  _float(_get(row, mapping, "llmo_visibility")) or 0.0,
            "improvement_tags": tags,
            "labeled_by":  "human",
            "labeled_at":  _now_iso(),
        },
    }


def build_ad_performance_record(
    row: dict, mapping: dict, idx: int
) -> dict | None:
    # headline + description を結合（広告CSVの典型パターン）
    headline = _get(row, mapping, "ad_headline", "")
    desc     = _get(row, mapping, "ad_description", "")
    text     = _get(row, mapping, "input_text")
    if not text:
        if headline or desc:
            text = f"見出し: {headline} | 説明: {desc}".strip(" |")
        else:
            return None

    ctr  = _float(_get(row, mapping, "actual_ctr"))
    cvr  = _float(_get(row, mapping, "actual_cvr"))
    roas = _float(_get(row, mapping, "actual_roas"))

    tier = _get(row, mapping, "performance_tier")
    if tier is None and ctr is not None:
        tier = "high" if ctr >= 0.05 else ("medium" if ctr >= 0.02 else "low")
    tier = tier or "medium"

    return {
        "record_id":   f"ap_{datetime.now().strftime('%Y%m%d')}_{idx:04d}",
        "source_type": "ad_creative",
        "collected_at": _get(row, mapping, "collected_at", _now_iso()),
        "industry":    _get(row, mapping, "industry", "advertising"),
        "input_text":  text,
        "metadata": {
            "campaign_id":    _get(row, mapping, "campaign_id"),
            "ad_format":      _get(row, mapping, "ad_format", "search"),
            "platform":       _get(row, mapping, "platform", "google"),
            "target_segment": _get(row, mapping, "target_segment"),
            "budget_jpy":     _int(_get(row, mapping, "budget_jpy")),
            "industry_vertical": _get(row, mapping, "industry"),
            "creative_id":    _get(row, mapping, "campaign_id"),
            "landing_url":    _get(row, mapping, "url"),
        },
        "label": {
            "performance_tier": tier,
            "actual_ctr":  ctr,
            "actual_cvr":  cvr,
            "actual_roas": roas,
            "predicted_ctr": None,
            "labeled_by":  "human",
            "labeled_at":  _now_iso(),
        },
    }


def build_persona_segment_record(
    row: dict, mapping: dict, idx: int
) -> dict | None:
    text = _get(row, mapping, "input_text")
    if not text:
        return None

    return {
        "record_id":   f"ps_{datetime.now().strftime('%Y%m%d')}_{idx:04d}",
        "source_type": "behavior_log",
        "collected_at": _get(row, mapping, "collected_at", _now_iso()),
        "industry":    _get(row, mapping, "industry", "marketing"),
        "input_text":  text,
        "metadata": {
            "session_id":  _get(row, mapping, "campaign_id"),
            "device_type": _get(row, mapping, "device_type", "unknown"),
            "region":      _get(row, mapping, "region"),
            "age_group":   _get(row, mapping, "age_group", "unknown"),
            "gender":      _get(row, mapping, "gender", "unknown"),
        },
        "label": {
            "persona_segment": _get(row, mapping, "persona_segment", "other"),
            "confidence":  _float(_get(row, mapping, "confidence")) or 0.8,
            "labeled_by":  "human",
            "labeled_at":  _now_iso(),
        },
    }


def build_market_research_record(
    row: dict, mapping: dict, idx: int
) -> dict | None:
    text = _get(row, mapping, "input_text")
    if not text:
        return None

    tags_raw = _get(row, mapping, "trend_tags", "")
    tags = [t.strip() for t in re.split(r"[,;/]", str(tags_raw)) if t.strip()] if tags_raw else []

    return {
        "record_id":   f"mr_{datetime.now().strftime('%Y%m%d')}_{idx:04d}",
        "source_type": "web_content",
        "collected_at": _get(row, mapping, "collected_at", _now_iso()),
        "industry":    _get(row, mapping, "industry", "market_research"),
        "input_text":  text,
        "metadata": {
            "research_topic":   _get(row, mapping, "research_topic"),
            "target_market":    _get(row, mapping, "target_market"),
            "data_period":      _get(row, mapping, "data_period"),
            "source_name":      _get(row, mapping, "source_name"),
            "respondent_count": _int(_get(row, mapping, "respondent_count")),
            "competitor_name":  _get(row, mapping, "competitor_name"),
        },
        "label": {
            "summary":        _get(row, mapping, "input_text", text)[:300],
            "sentiment":      _get(row, mapping, "sentiment", "neutral"),
            "trend_tags":     tags,
            "importance_score": _float(_get(row, mapping, "importance_score")) or 3.0,
            "report_section": "market_size",
            "labeled_by":     "human",
            "labeled_at":     _now_iso(),
        },
    }


TASK_BUILDERS = {
    "content_quality": build_content_quality_record,
    "ad_performance":  build_ad_performance_record,
    "persona_segment": build_persona_segment_record,
    "market_research": build_market_research_record,
}


# ---------------------------------------------------------------------------
# CSV 読み込み・変換
# ---------------------------------------------------------------------------

def convert_csv(
    task: str,
    input_path: Path,
    mapping: dict[str, str],
    encoding: str = "utf-8-sig",
) -> list[dict]:
    builder = TASK_BUILDERS[task]
    records = []
    skipped = 0

    with open(input_path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rec = builder(row, mapping, i + 1)
            if rec is None:
                skipped += 1
            else:
                records.append(rec)

    if skipped:
        print(f"  [warn] {skipped} rows skipped (input_text が空)")
    return records


def load_mapping(mapping_path: Path) -> dict[str, str]:
    """YAML または JSON のマッピング設定を読み込む。"""
    text = mapping_path.read_text(encoding="utf-8")
    if mapping_path.suffix in (".yaml", ".yml") and _HAS_YAML:
        return yaml.safe_load(text)
    return json.loads(text)


def generate_mapping_template(
    task: str, headers: list[str], out_path: Path
):
    """CSV 列名を確認してマッピングテンプレートを生成する。"""
    auto = auto_detect_mapping(headers)
    template = {
        "_help": (
            "右辺に CSV の列名を入れてください。"
            "不要なフィールドは null または削除してください。"
        ),
        "_csv_columns": headers,
        "_auto_detected": auto,
    }
    # タスク別の必須フィールドを先頭に表示
    required = {
        "content_quality": ["input_text", "url", "page_title", "target_keyword",
                            "content_type", "word_count", "quality_score",
                            "relevance_score", "llmo_visibility", "improvement_tags"],
        "ad_performance":  ["input_text", "url", "campaign_id", "ad_format",
                            "platform", "target_segment", "budget_jpy",
                            "actual_ctr", "actual_cvr", "actual_roas", "performance_tier"],
        "persona_segment": ["input_text", "persona_segment", "confidence",
                            "device_type", "region", "age_group", "gender"],
        "market_research": ["input_text", "research_topic", "target_market",
                            "data_period", "sentiment", "trend_tags", "importance_score"],
    }
    for field in required.get(task, []):
        template[field] = auto.get(field, None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"マッピングテンプレートを生成しました: {out_path}")
    print("  → ファイルを開いて列名を確認・編集してから --mapping に指定してください")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="社内CSV → OpenMythos JSONL コンバータ"
    )
    parser.add_argument("--task", choices=list(TASK_BUILDERS.keys()),
                        help="変換するタスク種別")
    parser.add_argument("--input", help="入力CSVファイルパス")
    parser.add_argument("--output", help="出力JONSLファイルパス（省略時は自動命名）")
    parser.add_argument("--mapping", help="列マッピング設定ファイル（JSON/YAML）")
    parser.add_argument("--auto-map", action="store_true",
                        help="列名から自動推定してマッピング（設定ファイル不要）")
    parser.add_argument("--inspect", metavar="CSV",
                        help="CSVの列名を確認してマッピングテンプレートを生成するだけ")
    parser.add_argument("--encoding", default="utf-8-sig",
                        help="CSVの文字コード（デフォルト: utf-8-sig = BOM付きUTF-8、Excelで保存した場合に使用）")
    parser.add_argument("--out-dir", default="data/converted",
                        help="出力ディレクトリ（--output 省略時）")
    args = parser.parse_args()

    # --inspect: 列名確認＋テンプレート生成のみ
    if args.inspect:
        csv_path = Path(args.inspect)
        task = args.task or "content_quality"
        with open(csv_path, encoding=args.encoding, newline="") as f:
            headers = next(csv.reader(f))
        print(f"\nCSV列名 ({len(headers)}列):")
        for h in headers:
            print(f"  {h}")
        out = Path("configs") / f"mapping_{task}_{csv_path.stem}.json"
        generate_mapping_template(task, headers, out)
        return

    if not args.task or not args.input:
        parser.error("--task と --input は必須です（--inspect 以外の場合）")

    csv_path = Path(args.input)

    # マッピングの決定
    if args.mapping:
        mapping = load_mapping(Path(args.mapping))
        print(f"マッピング設定読み込み: {args.mapping}")
    elif args.auto_map:
        with open(csv_path, encoding=args.encoding, newline="") as f:
            headers = next(csv.reader(f))
        mapping = auto_detect_mapping(headers)
        print(f"自動推定マッピング: {mapping}")
    else:
        # 設定なし → 自動推定を試みてテンプレートも生成
        with open(csv_path, encoding=args.encoding, newline="") as f:
            headers = next(csv.reader(f))
        mapping = auto_detect_mapping(headers)
        tmpl_path = Path("configs") / f"mapping_{args.task}_{csv_path.stem}.json"
        generate_mapping_template(args.task, headers, tmpl_path)
        print(f"\n自動推定マッピングで変換を続行します。")
        print(f"結果を確認して {tmpl_path} を編集し --mapping で再実行することを推奨します。\n")

    # 変換
    print(f"変換中: {csv_path} ...")
    records = convert_csv(args.task, csv_path, mapping, args.encoding)
    print(f"  {len(records)} 件変換完了")

    # 出力
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.task}_{csv_path.stem}.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"出力完了: {out_path}")
    print(f"\n次のステップ:")
    print(f"  python scripts/preprocess.py --task {args.task} --input {out_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""ドライブ移管対応_進捗管理 xlsx を顧客別 JSONL に取り込む。

「案件別ファイル一覧」シート(Client / 分類 / URL / ファイル形式 ...)を
顧客(Client)単位に集約し、提案書・見積書・RFP 等のカテゴリ別に
ファイルをまとめて data/drive_migration/clients.jsonl に出力する。

Usage:
    python scripts/import_drive_migration.py [xlsx_path]
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import openpyxl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = REPO_ROOT / "ドライブ移管対応_進捗管理20260706.xlsx"
OUT_DIR = REPO_ROOT / "data" / "drive_migration"

# ファイル実体を持たないプレースホルダ分類
PLACEHOLDER_CATEGORIES = {"(該当ファイルなし)", "(URL未設定)"}


def load_files_sheet(xlsx_path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    ws = wb["案件別ファイル一覧"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h) if h is not None else "" for h in next(rows)]
    records = []
    for row in rows:
        rec = {header[i]: row[i] for i in range(min(len(header), len(row)))}
        if rec.get("Client"):
            records.append(rec)
    wb.close()
    return records


def build_clients(records: list[dict]) -> "OrderedDict[str, dict]":
    clients: "OrderedDict[str, dict]" = OrderedDict()
    for rec in records:
        name = str(rec["Client"]).strip()
        entry = clients.setdefault(
            name,
            {
                "client": name,
                "tantosha": rec.get("担当者"),
                "shozoku": rec.get("所属"),
                "source_sheet": rec.get("元シート"),
                "folder_url": rec.get("新規URL"),
                "files": [],
            },
        )
        category = rec.get("分類")
        if category is None or str(category).strip() in PLACEHOLDER_CATEGORIES:
            continue
        synced = rec.get("同期日時")
        entry["files"].append(
            {
                "category": str(category).strip(),
                "url": rec.get("URL"),
                "format": rec.get("ファイル形式"),
                "uploader": rec.get("格納者"),
                "synced_at": str(synced) if synced is not None else None,
            }
        )
    return clients


def main() -> None:
    xlsx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    if not xlsx_path.exists():
        sys.exit(f"xlsx が見つかりません: {xlsx_path}")

    records = load_files_sheet(xlsx_path)
    clients = build_clients(records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "clients.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for entry in clients.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_files = sum(len(c["files"]) for c in clients.values())
    by_category: dict[str, int] = {}
    for c in clients.values():
        for file in c["files"]:
            by_category[file["category"]] = by_category.get(file["category"], 0) + 1

    print(f"入力: {xlsx_path.name} ({len(records)} 行)")
    print(f"出力: {out_path.relative_to(REPO_ROOT)}")
    print(f"顧客数: {len(clients)}, ファイル数: {total_files}")
    for cat, n in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()

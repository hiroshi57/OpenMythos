# -*- coding: utf-8 -*-
"""ドライブ移管 進捗xlsx から残作業のアクションリストを生成する。

ユニット4シート(第1/第2/第3/アソシエイト)の Migration Status・エラー詳細を
横断集計し、対処方法別にグループ化した Markdown を
data/drive_migration/actions.md に出力する。

Usage:
    python scripts/drive_migration_actions.py [xlsx_path]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import openpyxl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = REPO_ROOT / "ドライブ移管対応_進捗管理20260706.xlsx"
OUT_PATH = REPO_ROOT / "data" / "drive_migration" / "actions.md"

UNIT_SHEETS = ["第1ユニット", "第2ユニット", "第3ユニット", "アソシエイト"]

# 列インデックス (ヘッダは各シート2行目)
COL = {
    "client": 0,
    "tantosha": 1,
    "shozoku": 2,
    "owners": 4,
    "source_url": 5,
    "owner_sheet_url": 6,
    "dest_url": 8,
    "discovery": 10,
    "bunrui": 11,
    "status": 12,
    "error_msg": 13,
    "error_detail": 16,
}

DOMAIN_POLICY_ERROR = "The domain administrator has not allowed writers"


def load_rows(xlsx_path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    rows = []
    for sheet in UNIT_SHEETS:
        ws = wb[sheet]
        for raw in ws.iter_rows(min_row=3, values_only=True):
            if not raw or raw[COL["client"]] is None:
                continue
            rec = {k: raw[i] if i < len(raw) else None for k, i in COL.items()}
            rec["sheet"] = sheet
            rows.append(rec)
    wb.close()
    return rows


def fmt_row(r: dict, extra: str = "") -> str:
    base = f"- **{r['client']}**(担当: {r['tantosha']} / {r['sheet']})"
    return base + (f"\n  {extra}" if extra else "")


def main() -> None:
    xlsx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    if not xlsx_path.exists():
        sys.exit(f"xlsx が見つかりません: {xlsx_path}")

    rows = load_rows(xlsx_path)
    failed = [r for r in rows if r["status"] == "FAILED"]

    domain_blocked = [r for r in failed if r["error_msg"] and DOMAIN_POLICY_ERROR in str(r["error_msg"])]
    perm_pending = [r for r in failed if r["error_msg"] and "sufficient permissions" in str(r["error_msg"])]
    not_found = [r for r in failed if r["error_msg"] and "not found" in str(r["error_msg"]).lower()]
    other_failed = [r for r in failed if r not in domain_blocked + perm_pending + not_found]

    create_folder = [r for r in rows if r["status"] == "CREATE_FOLDER"]
    kakunin = [r for r in rows if r["status"] and "確認" in str(r["status"])]
    kinshi = [r for r in rows if r["status"] and "作成禁止" in str(r["status"])]
    missing = [
        r for r in rows
        if (r["bunrui"] and "存在しない" in str(r["bunrui"]))
        or (r["discovery"] in ("INVALID_URL", "NOT_FOUND"))
    ]

    lines: list[str] = []
    add = lines.append
    add("# ドライブ移管 アクションリスト")
    add("")
    add(f"元データ: `{xlsx_path.name}` / 生成: `python scripts/drive_migration_actions.py`")
    add("※ 顧客情報を含むため公開リポジトリにコミットしないこと(data/ は gitignore 済み)。")
    add("")

    add(f"## A. ドメインポリシーでブロック中 — {len(domain_blocked)}件(一括解消可能)")
    add("")
    add("エラー: `The domain administrator has not allowed writers to move items into a shared drive.`")
    add("")
    add("個別の権限不足ではなく **Workspace の共有ドライブ設定** による一括ブロック。対処はどちらか:")
    add("1. **管理者に依頼**: 管理コンソール → ドライブとドキュメント → 共有設定 で「編集者による共有ドライブへの移動」を許可")
    add("2. **実行アカウントを移管先共有ドライブの「コンテンツ管理者」以上にする**(移動には対象ドライブのメンバーシップが必要)")
    add("")
    for r in domain_blocked:
        add(fmt_row(r, f"移管先: {r['dest_url']}"))
    add("")

    add(f"## B. オーナー権限移管が未完了 — {len(perm_pending)}件(オーナーへ依頼)")
    add("")
    add("エラー: `The user does not have sufficient permissions for this file.`")
    add("ファイルオーナーにオーナー変更用シートでの権限移管を依頼する。")
    add("")
    for r in perm_pending:
        owners = str(r["owners"] or "(オーナー一覧なし)").replace("\n", ", ")
        add(fmt_row(r, f"オーナー: {owners}\n  変更用シート: {r['owner_sheet_url']}"))
    add("")

    add(f"## C. 移管元フォルダが見つからない — {len(not_found)}件(URL確認)")
    add("")
    for r in not_found:
        add(fmt_row(r, f"移管元URL: {r['source_url']}\n  エラー: {r['error_msg']}"))
    add("")

    if other_failed:
        add(f"## C2. その他の FAILED — {len(other_failed)}件")
        add("")
        for r in other_failed:
            add(fmt_row(r, f"エラー: {r['error_msg']} / {r['error_detail']}"))
        add("")

    add(f"## D. CREATE_FOLDER 待ち — {len(create_folder)}件(ツール実行待ち)")
    add("")
    by_sheet: dict[str, int] = defaultdict(int)
    for r in create_folder:
        by_sheet[r["sheet"]] += 1
    for sheet, n in by_sheet.items():
        add(f"- {sheet}: {n}件")
    add("")

    add(f"## E. フォルダ所在不明(④存在しない / INVALID_URL / NOT_FOUND)— {len(missing)}件(担当者へ確認)")
    add("")
    by_tantosha: dict[str, list[dict]] = defaultdict(list)
    for r in missing:
        by_tantosha[str(r["tantosha"])].append(r)
    for tantosha, rs in sorted(by_tantosha.items(), key=lambda x: -len(x[1])):
        add(f"### {tantosha}({len(rs)}件)")
        for r in rs:
            note = str(r["bunrui"] or r["discovery"] or "")
            add(f"- {r['client']}({r['sheet']} / {note})")
        add("")

    add(f"## F. 判断待ち — 確認 {len(kakunin)}件 / 作成禁止(重複) {len(kinshi)}件")
    add("")
    for r in kakunin:
        add(fmt_row(r, f"ステータス: {r['status']}"))
    for r in kinshi:
        add(fmt_row(r, f"ステータス: {r['status']}"))
    add("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"出力: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"A ドメインポリシー: {len(domain_blocked)}")
    print(f"B オーナー移管未完了: {len(perm_pending)}")
    print(f"C 移管元不明: {len(not_found)}")
    print(f"C2 その他FAILED: {len(other_failed)}")
    print(f"D CREATE_FOLDER: {len(create_folder)}")
    print(f"E 所在不明: {len(missing)}")
    print(f"F 判断待ち: {len(kakunin) + len(kinshi)}")


if __name__ == "__main__":
    main()

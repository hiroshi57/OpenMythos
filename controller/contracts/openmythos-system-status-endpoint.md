---
task_id: openmythos-system-status-endpoint
project: OpenMythos
owner_controller: true
status: proposed
state_version: 0
state_commit: (none)
current_phase: scoping
verified_at: 2026-09-01T08:59:08+09:00
last_executor: (none)
next_executor: (unassigned)
---

## 目標
GET /v1/system/status (version/latest_sprint/changelog_recent/total_routes/uptime_s) と GET /v1/system/skills (タグ別ルート数の一覧) を serve/api.py に追加する。/health と同じ認証・レート制限方針を適用する（無条件バイパスの特例を新設しない）。あわせて Plans.md の『Sprint 75 候補テーマ』節を、実態（75A/B/C 全て実装済み）に合わせて Sprint 71-74 と同じ完了済み形式に修正する。

## 受入条件
- [ ] GET /v1/system/status が version/latest_sprint/changelog_recent/total_routes/uptime_s を返す
- [ ] GET /v1/system/skills がタグ別のルート数を返す（新規スキルモジュールは作らず app.routes の反射的な集計で実装する）
- [ ] 認証は /health と同じ扱い（API_KEY設定時はBearer必須のまま、無条件バイパスにしない）。レート制限は /health 同様に除外可
- [ ] tests/test_sprint*.py に新規エンドポイントのテストを追加
- [ ] Plans.md の Sprint 75 セクションを『詳細（完了）』形式に修正する（71-74と同じ体裁）
- [ ] CHANGELOG.mdの表を都度パースする実装を選ぶ場合は、その脆さ（過去に順序が崩れた実績あり）をコード上のコメントで明記する。静的サイドカーJSON方式を選ぶ場合はリリース手順に更新ステップを追記する

## スコープ記述
- 許可: serve/api.py, serve/routers/**, Plans.md, tests/test_sprint*.py, CHANGELOG.md
- 禁止: open_mythos/skills/env_sensor.py, open_mythos/skills/transfer_optimizer.py, open_mythos/skills/infra_dashboard.py

## 受入コミット
- state_commit: (none)

## 検証済みの判断
(まだなし)

## 失敗アプローチ（再試行させないためのメモ）
(まだなし)

## オープンブロッカー
(まだなし)

## 実行者への申し送り（読み取り専用ビュー）
これはまだ「提案」段階のコントラクトです。status が verified になるまで実行者は着手しないこと。
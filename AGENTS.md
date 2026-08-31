# AGENTS.md — OpenMythos

このリポジトリで作業するエージェント（Codex CLI等）への指示。

## タスクを再開するとき

second-brain-controller のコントラクト方式を使っている。渡された task-id で
`controller/contracts/<task-id>.md` を読み、同名の `controller/archive/<task-id>.briefing.md`
があれば読む。**これ以外の過去のやり取りを再現しようとしない** — このコントラクトが唯一の正準状態。

目標・受入条件・スコープ記述・検証済みの判断・失敗アプローチ・オープンブロッカーを確認してから着手し、
スコープ記述の「許可」範囲外のファイルは変更しない。

## 作業を終えるとき

コントラクトファイルを直接編集しない。以下のいずれかを人間に実行してもらう:

```powershell
./scripts/sbc/Handoff-Task.ps1 -TaskId <task-id> -To claude-code -TestCommand "<テストコマンド>"
./scripts/sbc/Block-Task.ps1 -TaskId <task-id> -Reason "<何を試して何が起きたか>" -Kind FailedApproach
./scripts/sbc/Complete-Task.ps1 -TaskId <task-id>
```

## 禁止事項

- `controller/contracts/*.md` を直接編集しない（`owner_controller: true`）
- `state_version` を手で書き換えない
- スコープ記述の「禁止」領域を変更しない

---
<!-- プロジェクト固有の指示はここから下に追記してください -->

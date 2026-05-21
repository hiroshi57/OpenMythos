# Plans.md エントリテンプレート
# Lead Agent が Plans.md に追記する際のフォーマット

## タスクテーブル行

```markdown
| task-id | 説明 | 担当 | 状態 | DoD |
|---------|------|------|------|-----|
| 1.1.1 | Worker エージェントに実装させる内容 | Worker | cc:TODO | (a) pytest pass (b) review pass |
```

## ステータスの意味

| マーカー | 意味 |
|---------|------|
| `cc:TODO` | 未着手 |
| `cc:WIP` | 実装中（Worker が動いている） |
| `cc:完了 [hash]` | 完了・マージ済み |
| `cc:不要` | スキップ・不要と判断 |

## ユースケース別テンプレート

### UC-05: テストスイート自動生成
```markdown
| 1.1.1 | open_mythos/main.py のユニットテスト生成 | Worker | cc:TODO | (a) カバレッジ80%以上 (b) 全テスト通過 |
| 1.1.2 | open_mythos/moda.py のユニットテスト生成 | Worker | cc:TODO | (a) カバレッジ80%以上 (b) 全テスト通過 |
```

### UC-06: API コントラクトファースト
```markdown
| 2.1.1 | /users エンドポイント実装 | Worker | cc:TODO | (a) spec 準拠 (b) テスト通過 |
| 2.1.2 | /orders エンドポイント実装 | Worker | cc:TODO | (a) spec 準拠 (b) テスト通過 |
| 2.1.3 | 統合テスト | Worker | cc:TODO | (a) E2E テスト通過 |
```

### UC-12: インシデント対応
```markdown
| 3.1.1 | ログ解析・根本原因特定 | Worker | cc:TODO | (a) 原因レポート作成 |
| 3.1.2 | 修正実装 | Worker | cc:TODO | (a) 再現テスト追加 (b) 修正 push |
| 3.1.3 | 事後報告書作成 | Worker | cc:TODO | (a) ポストモーテム記録 |
```

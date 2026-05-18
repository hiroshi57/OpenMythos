#!/usr/bin/env bash
# think-act-learn.sh — THINK-ACT-LEARN ループの 1 サイクルを実行する
# 使い方: bash framework/scripts/think-act-learn.sh <task-id>
#
# THINK: Plans.md のタスク定義を読み込む
# ACT:   Claude Code に実装を指示する
# LEARN: 完了後にメモリを更新する
set -euo pipefail

TASK_ID="${1:?task-id が必要です}"
CONTRACT_DIR=".claude/state/contracts"
MEMORY_DIR=".claude/agent-memory/claude-code-harness-worker"
LOG_DIR=".claude/state/logs"

mkdir -p "$LOG_DIR" "$MEMORY_DIR"

LOG_FILE="$LOG_DIR/${TASK_ID}-$(date +%Y%m%d-%H%M%S).log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  THINK-ACT-LEARN Loop: $TASK_ID"
echo "  $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THINK フェーズ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[ THINK ] タスク定義を確認中..."

CONTRACT_FILE="$CONTRACT_DIR/${TASK_ID}.sprint-contract.json"
if [ ! -f "$CONTRACT_FILE" ]; then
  echo "⚠️  Sprint Contract が見つかりません: $CONTRACT_FILE"
  echo "    先に create-contract.sh を実行してください"
  exit 1
fi

DESCRIPTION=$(python -c "import json,sys; d=json.load(open('$CONTRACT_FILE', encoding='utf-8')); print(d['description'])")
FILES=$(python -c "import json,sys; d=json.load(open('$CONTRACT_FILE', encoding='utf-8')); print('\n'.join(d.get('files',[])))")
VALIDATION=$(python -c "import json,sys; d=json.load(open('$CONTRACT_FILE', encoding='utf-8')); print('\n'.join(d.get('validation_commands',[])))")

echo "  タスク: $DESCRIPTION"
echo "  ファイル:"
echo "$FILES" | sed 's/^/    - /'
echo "  検証:"
echo "$VALIDATION" | sed 's/^/    $ /'

# MEMORY.md があれば参照
if [ -f "$MEMORY_DIR/MEMORY.md" ]; then
  echo ""
  echo "[ THINK ] 関連メモリを確認中..."
  grep -i "$(echo "$DESCRIPTION" | cut -c1-20)" "$MEMORY_DIR/MEMORY.md" || echo "  (関連メモリなし)"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ACT フェーズ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[ ACT ] 実装フェーズ開始..."
echo "  検証コマンドを順次実行します"
echo ""

# 検証コマンド実行
EXIT_CODE=0
while IFS= read -r cmd; do
  if [ -n "$cmd" ]; then
    echo "  \$ $cmd"
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
      echo "  ✅ PASS: $cmd"
    else
      echo "  ❌ FAIL: $cmd"
      EXIT_CODE=1
    fi
    echo ""
  fi
done <<< "$VALIDATION"

if [ $EXIT_CODE -ne 0 ]; then
  echo "[ ACT ] ❌ 検証失敗 — ログ: $LOG_FILE"
  exit 1
fi

echo "[ ACT ] ✅ 全検証通過"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEARN フェーズ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[ LEARN ] 学習記録を更新中..."

# 完了記録を append
LEARN_LOG="$MEMORY_DIR/completed-tasks.md"
if [ ! -f "$LEARN_LOG" ]; then
  cat > "$LEARN_LOG" << 'INITEOF'
---
name: completed-tasks
description: 完了したタスクの記録と学習ポイント
metadata:
  type: project
---

# 完了タスク記録

INITEOF
fi

cat >> "$LEARN_LOG" << EOF

## $TASK_ID — $(date +%Y-%m-%d)
- **説明**: $DESCRIPTION
- **結果**: PASS
- **ログ**: $LOG_FILE
EOF

echo "  ✅ 記録: $LEARN_LOG"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  COMPLETE: $TASK_ID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

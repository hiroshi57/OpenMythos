#!/usr/bin/env bash
# record-memory.sh — Memory Agent: タスク完了後に学習を記録する
# 使い方: bash framework/scripts/record-memory.sh <type> <slug> "<description>" "<body>"
#   type: feedback | project | reference | user
#   slug: ハイフン区切りの識別子 (例: db-pool-exhaustion)
set -euo pipefail

TYPE="${1:?type が必要 (feedback|project|reference|user)}"
SLUG="${2:?slug が必要}"
DESCRIPTION="${3:?description が必要}"
BODY="${4:?body が必要}"

MEMORY_DIR=".claude/agent-memory/claude-code-harness-worker"
mkdir -p "$MEMORY_DIR"

MEM_FILE="$MEMORY_DIR/${SLUG}.md"
INDEX_FILE="$MEMORY_DIR/MEMORY.md"

# メモリファイル生成
cat > "$MEM_FILE" << EOF
---
name: ${SLUG}
description: ${DESCRIPTION}
metadata:
  type: ${TYPE}
---

${BODY}
EOF

echo "✅ メモリ記録: $MEM_FILE"

# MEMORY.md インデックス更新
if [ ! -f "$INDEX_FILE" ]; then
  echo "# Memory Index" > "$INDEX_FILE"
  echo "" >> "$INDEX_FILE"
fi

# 既存エントリを更新 or 追加
ENTRY="- [${DESCRIPTION}](${SLUG}.md) — ${BODY:0:80}"
if grep -q "\[${SLUG}\]" "$INDEX_FILE" 2>/dev/null; then
  # 既存エントリを更新（Windows/Mac 両対応の sed）
  python -c "
import re, sys
with open('$INDEX_FILE', 'r') as f: content = f.read()
new_line = '${ENTRY}'
content = re.sub(r'- \[.*${SLUG}.*\].*\n', new_line + '\n', content)
with open('$INDEX_FILE', 'w') as f: f.write(content)
"
else
  echo "$ENTRY" >> "$INDEX_FILE"
fi

echo "✅ インデックス更新: $INDEX_FILE"
echo ""
echo "記録内容:"
cat "$MEM_FILE"

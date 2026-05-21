#!/usr/bin/env bash
# create-contract.sh — Lead Agent が Sprint Contract を生成するスクリプト
# 使い方: bash framework/scripts/create-contract.sh <task-id> "<説明>" [file1 file2 ...]
set -euo pipefail

TASK_ID="${1:?task-id が必要です}"
DESCRIPTION="${2:?description が必要です}"
shift 2
FILES=("$@")

CONTRACT_DIR=".claude/state/contracts"
mkdir -p "$CONTRACT_DIR"

CONTRACT_FILE="$CONTRACT_DIR/${TASK_ID}.sprint-contract.json"

# ファイルリストを JSON 配列に変換
FILES_JSON="[]"
if [ ${#FILES[@]} -gt 0 ]; then
  FILES_JSON=$(printf '"%s",' "${FILES[@]}" | sed 's/,$//' | awk '{print "["$0"]"}')
fi

# 検証コマンド自動検出
VALIDATION_CMDS="[]"
if [ -f "pyproject.toml" ] || [ -f "pytest.ini" ] || [ -f "setup.cfg" ]; then
  VALIDATION_CMDS='["python -m pytest --tb=short -q"]'
elif [ -f "package.json" ]; then
  if grep -q '"test"' package.json 2>/dev/null; then
    VALIDATION_CMDS='["npm test"]'
  fi
fi

cat > "$CONTRACT_FILE" << EOF
{
  "schema_version": "sprint-contract.v1",
  "task_id": "${TASK_ID}",
  "description": "${DESCRIPTION}",
  "files": ${FILES_JSON},
  "dod": [
    "(a) 検証コマンドが全て通過する",
    "(b) preflight 6 項目が全て verified: true",
    "(c) Reviewer が APPROVE を返す"
  ],
  "validation_commands": ${VALIDATION_CMDS},
  "flags": [],
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "✅ Sprint Contract 作成: $CONTRACT_FILE"
cat "$CONTRACT_FILE"

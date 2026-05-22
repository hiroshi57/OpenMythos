#!/usr/bin/env bash
# harness-mem-init.sh — harness-mem 初期化スクリプト
# session.json の harness_mem を healthy 状態に更新し、
# agent-memory ディレクトリの存在を保証する。
#
# 使い方:
#   bash scripts/harness-mem-init.sh
#
# セッション開始時に自動実行するには CLAUDE.md の "セッション開始時" コマンドに追加:
#   bash scripts/harness-mem-init.sh
set -euo pipefail

# ── パス解決 ──────────────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
STATE_DIR="${REPO_ROOT}/.claude/state"
SESSION_JSON="${STATE_DIR}/session.json"
MEMORY_DIR="${REPO_ROOT}/.claude/agent-memory/claude-code-harness-worker"
MEMORY_INDEX="${MEMORY_DIR}/MEMORY.md"

# ── ディレクトリ保証 ───────────────────────────────────────
mkdir -p "${STATE_DIR}"
mkdir -p "${MEMORY_DIR}"

# MEMORY.md が存在しない場合のみ雛形を生成
if [ ! -f "${MEMORY_INDEX}" ]; then
  cat > "${MEMORY_INDEX}" << 'EOF'
# Memory Index
> OpenMythos プロジェクト — 最終更新: (harness-mem-init により生成)

## project（プロジェクト状況）

## feedback（バグパターン・注意点）

## reference（外部リソース）
EOF
  echo "✅ MEMORY.md を初期化しました: ${MEMORY_INDEX}"
fi

# ── session.json の harness_mem を更新 ────────────────────
NOW="$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || python -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))")"

if [ -f "${SESSION_JSON}" ]; then
  # Python で JSON を安全に更新（jq 非依存）
  python - <<PYEOF
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

path = "${SESSION_JSON}"
now  = "${NOW}"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

data.setdefault("harness_mem", {})
data["harness_mem"]["healthy"]      = True
data["harness_mem"]["last_checked"] = now
data["harness_mem"]["last_error"]   = None
data["updated_at"]                  = now

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("session.json harness_mem -> healthy=true (" + now + ")")
PYEOF
else
  # session.json が無い場合は最小構成で生成
  python - <<PYEOF
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

path = "${SESSION_JSON}"
now  = "${NOW}"

data = {
  "harness_mem": {
    "healthy": True,
    "last_checked": now,
    "last_error": None
  },
  "state": "initialized",
  "updated_at": now
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("session.json created: " + path)
PYEOF
fi

# ── Plans.md 存在チェック ─────────────────────────────────
PLANS="${REPO_ROOT}/Plans.md"
if [ ! -f "${PLANS}" ]; then
  echo "⚠️  Plans.md が存在しません。/harness-plan でスプリント計画を作成してください。"
else
  TODO_COUNT=$(grep -c "cc:TODO" "${PLANS}" 2>/dev/null || echo 0)
  WIP_COUNT=$(grep -c "cc:WIP"  "${PLANS}" 2>/dev/null || echo 0)
  DONE_COUNT=$(grep -c "cc:完了" "${PLANS}" 2>/dev/null || echo 0)
  echo "📋 Plans.md: TODO=${TODO_COUNT} WIP=${WIP_COUNT} 完了=${DONE_COUNT}"
fi

echo ""
echo "✅ harness-mem 初期化完了"
echo "   memory dir : ${MEMORY_DIR}"
echo "   session    : ${SESSION_JSON}"

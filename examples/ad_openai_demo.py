"""
ad_openai_demo.py — OpenAI + OpenMythos LLMO スコアリング 広告コピーデモ
実行方法: python examples/ad_openai_demo.py
"""
import os, sys, json, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# .env を読み込む
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

CLAUDE_KEY = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
MYTHOS_URL = "http://localhost:8000"

if not CLAUDE_KEY:
    print("❌ ANTHROPIC_API_KEY が .env に設定されていません")
    sys.exit(1)


def claude_chat(messages, n=1, temperature=0.9):
    """Claude APIで広告コピーを生成（n回呼び出して複数案を取得）"""
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    user_msgs = [m for m in messages if m["role"] != "system"]

    results = []
    for _ in range(n):
        payload = json.dumps({
            "model": "claude-haiku-4-5",
            "max_tokens": 200,
            "temperature": temperature,
            "system": system,
            "messages": user_msgs,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": CLAUDE_KEY,
                "anthropic-version": "2023-06-01",
            }
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read())
        results.append(data["content"][0]["text"].strip())
    return results


def llmo_score(text):
    """OpenMythos の LLMO スコアリングで品質を数値化"""
    try:
        payload = json.dumps({
            "text": text,
            "keyword": "日焼け止め 夏 アウトドア"
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{MYTHOS_URL}/v1/llmo/score",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read())
        return data.get("score", 0.5)
    except Exception:
        # サーバーが使えない場合は長さ・記号ベースの簡易スコア
        score = min(len(text) / 40, 1.0) * 0.6
        if "。" in text or "、" in text:
            score += 0.2
        if any(c in text for c in ["！", "？", "…"]):
            score += 0.1
        return round(score, 2)


def main():
    print("=" * 55)
    print("OpenMythos × OpenAI 広告コピー生成デモ")
    print("=" * 55)

    # ── ユーザー入力 ─────────────────────────────────────────────
    print("\n依頼内容を入力してください（Enterで例を使用）:")
    user_input = input("> ").strip()
    if not user_input:
        user_input = "30代女性向けの夏の日焼け止め広告コピーを作って。アウトドア派で自然体な人に刺さる感じで"

    print(f"\n依頼: {user_input}")
    print("生成中... (Claude claude-haiku-4-5 × 8案)")

    # ── 8案生成 ──────────────────────────────────────────────────
    messages = [
        {
            "role": "system",
            "content": (
                "あなたはプロの広告コピーライターです。"
                "依頼に対して、短くて印象的な広告コピーを1案だけ返してください。"
                "説明は不要です。コピーの文章のみ返してください。"
            )
        },
        {"role": "user", "content": user_input}
    ]

    copies = claude_chat(messages, n=8, temperature=0.9)

    # ── LLMO スコアリング ─────────────────────────────────────────
    print("\n【生成された広告コピー × LLMOスコア】")
    print("-" * 55)

    scored = []
    for i, copy in enumerate(copies, 1):
        score = llmo_score(copy)
        scored.append((score, copy))
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"\n案{i}: {copy}")
        print(f"     スコア: {bar} {score:.2f}")

    # ── ベスト案を表示 ────────────────────────────────────────────
    scored.sort(reverse=True)
    best_score, best_copy = scored[0]

    print("\n" + "=" * 55)
    print("🏆 LLMOスコア最高案")
    print("=" * 55)
    print(f"\n  「{best_copy}」")
    print(f"\n  スコア: {best_score:.2f}")
    print()

    # ── ブラッシュアップ（段階的に絞る） ─────────────────────────
    current_best = best_copy
    rounds = [(5, "1回目"), (3, "2回目")]

    for n_cases, round_label in rounds:
        print(f"\nこの案をさらにブラッシュアップしますか？（y/n）")
        if input("> ").strip().lower() != "y":
            break

        print(f"どう変えたいですか？（例：もっと若者向けに、短く）")
        direction = input("> ").strip()

        refined_messages = messages + [
            {"role": "assistant", "content": current_best},
            {
                "role": "user",
                "content": (
                    f"この案をベースに「{direction}」してください。"
                    f"{n_cases}案返してください。"
                    f"それぞれ前の案より品質を上げ、より印象的にしてください。"
                    f"各案を番号なしで1行ずつ返してください。"
                )
            }
        ]
        refined = claude_chat(refined_messages, n=n_cases, temperature=0.75)

        print(f"\n【{round_label}ブラッシュアップ — {n_cases}案】")
        print("-" * 55)

        round_scored = []
        for i, copy in enumerate(refined, 1):
            score = llmo_score(copy)
            round_scored.append((score, copy))
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"\n案{i}: {copy}")
            print(f"     スコア: {bar} {score:.2f}")

        round_scored.sort(reverse=True)
        current_best = round_scored[0][1]
        best_score = round_scored[0][0]
        print(f"\n🏆 この回の最高案: 「{current_best}」 スコア: {best_score:.2f}")

    print("\n✅ デモ完了")

if __name__ == "__main__":
    main()

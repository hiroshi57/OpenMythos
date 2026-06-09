"""
広告コピー生成デモ
実行方法: python examples/ad_demo.py
"""
import urllib.request
import json

BASE_URL = "http://localhost:8000"

def generate(prompt, task="ad_performance", max_tokens=300):
    data = json.dumps({
        "prompt": prompt,
        "task": task,
        "max_new_tokens": max_tokens
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as res:
        result = json.loads(res.read())
    return result

print("=" * 50)
print("OpenMythos 広告コピー生成デモ")
print("=" * 50)

prompt = "30代女性向けの夏の日焼け止め広告コピーを5案作って。アウトドア派で自然体な人に刺さる感じで"
print(f"\n依頼: {prompt}\n")
print("生成中...")

result = generate(prompt)

print("\n【生成された広告コピー】")
print("-" * 50)
print(result["text"])
print("-" * 50)
print(f"\n処理時間: {result['latency_ms']:.0f}ms")
print(f"生成トークン数: {result['generated_tokens']}")

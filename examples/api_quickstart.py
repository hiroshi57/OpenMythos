"""
OpenMythos API — クイックスタートサンプル

使い方:
    # サーバーを起動してから実行
    uvicorn serve.api:app --port 8000 &
    python examples/api_quickstart.py
"""

import sys
import requests

BASE = "http://localhost:8000"
HDR  = {"Authorization": "Bearer dev", "Content-Type": "application/json"}


def check_health():
    r = requests.get(f"{BASE}/health")
    print("[health]", r.json())


def example_chat():
    print("\n=== チャット (OpenAI互換) ===")
    r = requests.post(f"{BASE}/v1/chat/completions", headers=HDR, json={
        "model": "openmythos",
        "messages": [
            {"role": "system", "content": "あなたは丁寧なアシスタントです。"},
            {"role": "user",   "content": "1+1は？"},
        ],
        "max_tokens": 64,
    })
    if r.status_code == 200:
        print(r.json()["choices"][0]["message"]["content"])
    else:
        print("ERROR:", r.status_code, r.text[:200])


def example_embeddings():
    print("\n=== 埋め込みベクトル生成 ===")
    r = requests.post(f"{BASE}/v1/embeddings", headers=HDR, json={
        "input": "OpenMythosは強力なLLMフレームワークです",
        "model": "openmythos",
    })
    if r.status_code == 200:
        vec = r.json()["data"][0]["embedding"]
        print(f"ベクトル次元: {len(vec)}, 先頭5値: {[round(v, 4) for v in vec[:5]]}")
    else:
        print("ERROR:", r.status_code)


def example_assistants():
    print("\n=== Assistants API (スレッド会話) ===")

    # アシスタント作成
    asst = requests.post(f"{BASE}/v1/assistants", headers=HDR, json={
        "name": "SampleBot",
        "instructions": "あなたは社内サポートデスクです。",
    }).json()
    print(f"アシスタント作成: {asst['id']}")

    # スレッド作成
    thread = requests.post(f"{BASE}/v1/threads", headers=HDR, json={}).json()
    print(f"スレッド作成: {thread['id']}")

    # メッセージ追加
    requests.post(f"{BASE}/v1/threads/{thread['id']}/messages", headers=HDR, json={
        "role": "user", "content": "こんにちは！",
    })

    # 実行
    run = requests.post(f"{BASE}/v1/threads/{thread['id']}/runs", headers=HDR, json={
        "assistant_id": asst["id"],
    }).json()
    print(f"実行ステータス: {run['status']}")

    # メッセージ一覧
    msgs = requests.get(f"{BASE}/v1/threads/{thread['id']}/messages", headers=HDR).json()
    for m in msgs["data"]:
        role = m["role"]
        text = m["content"][0]["text"]["value"] if m["content"] else ""
        print(f"  [{role}] {text[:80]}")


def example_rag():
    print("\n=== RAG (社内ドキュメント検索) ===")

    # ドキュメント登録
    requests.post(f"{BASE}/v1/rag/index", headers=HDR, json={
        "documents": [
            "有給休暇は年間20日付与されます。",
            "経費精算は月末締めです。領収書は3ヶ月以内に提出してください。",
            "社内Wikiはintranet.example.comで公開しています。",
        ],
        "session_id": "quickstart-demo",
    })
    print("ドキュメント登録完了")

    # 検索
    r = requests.post(f"{BASE}/v1/rag", headers=HDR, json={
        "query": "有給休暇は何日ありますか",
        "session_id": "quickstart-demo",
    })
    if r.status_code == 200:
        data = r.json()
        print("回答:", data.get("answer", str(data)[:200]))
    else:
        print("ERROR:", r.status_code)


def example_vector_store():
    print("\n=== ベクトルDB ===")

    docs = [
        "機械学習の基礎",
        "深層学習とニューラルネット",
        "自然言語処理の概要",
    ]
    for i, text in enumerate(docs):
        requests.post(f"{BASE}/v1/vector-store/upsert", headers=HDR, json={
            "id": f"qs-doc-{i}", "text": text, "metadata": {"idx": i},
        })
    print(f"{len(docs)}件登録完了")

    r = requests.post(f"{BASE}/v1/vector-store/query", headers=HDR, json={
        "query": "ニューラルネットワーク", "top_k": 2,
    })
    if r.status_code == 200:
        for item in r.json().get("results", []):
            score = item.get("score", 0)
            text  = item.get("text", "")[:60]
            print(f"  [{score:.3f}] {text}")
    else:
        print("ERROR:", r.status_code)


def example_security():
    print("\n=== セキュリティスキャン ===")
    r = requests.post(f"{BASE}/v1/security/scan", headers=HDR, json={
        "target_url": "http://example.com",
        "timeout": 5.0,
    })
    if r.status_code == 200:
        data = r.json()
        print(f"リスクスコア: {data['risk_score']:.1f}/10.0")
        print(f"検出件数: {len(data['findings'])}")
        for f in data["findings"][:3]:
            print(f"  [{f['severity']}] {f['title']}")
    else:
        print("ERROR:", r.status_code)


def example_oss_sbom():
    print("\n=== OSS依存関係分析 + SBOM生成 ===")
    r = requests.post(f"{BASE}/v1/security/oss/analyze", headers=HDR, json={
        "project_path": "."
    })
    if r.status_code == 200:
        data = r.json()
        print(f"総依存パッケージ数: {data['total_deps']}")
        print(f"脆弱性あり: {data['vulnerable_count']}")
    else:
        print("ERROR:", r.status_code)


def example_llmo():
    print("\n=== LLMOスコアリング (SEO/AIサーチ最適化) ===")
    r = requests.post(f"{BASE}/v1/llmo/score", headers=HDR, json={
        "text": (
            "デジタルマーケティングとは、GoogleやMetaなどのプラットフォームを活用して"
            "顧客を獲得する手法です。2024年の調査ではSEO投資は前年比32%増加しています。"
        ),
    })
    if r.status_code == 200:
        data = r.json()
        print(f"LLMOスコア: {data.get('llmo_total', '(確認してください)')}")
    else:
        print("ERROR:", r.status_code)


if __name__ == "__main__":
    print("OpenMythos API クイックスタート")
    print(f"接続先: {BASE}")
    print("-" * 50)

    try:
        check_health()
    except requests.ConnectionError:
        print("\nERROR: サーバーに接続できません。")
        print("  以下のコマンドでサーバーを起動してから再実行してください:")
        print("  uvicorn serve.api:app --port 8000")
        sys.exit(1)

    example_chat()
    example_embeddings()
    example_assistants()
    example_rag()
    example_vector_store()
    example_security()
    example_oss_sbom()
    example_llmo()

    print("\n=== 全サンプル完了 ===")
    print(f"APIドキュメント (Swagger UI): {BASE}/docs")

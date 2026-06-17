# OpenMythos

<p align="left">
  <img alt="Version" src="https://img.shields.io/badge/version-0.57.0-3670A0?style=for-the-badge">
  <img alt="Tests" src="https://img.shields.io/badge/tests-2862%20PASS-brightgreen?style=for-the-badge">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Implemented-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
</p>

> **Disclaimer:** OpenMythos is an independent, community-driven theoretical reconstruction based solely on publicly available research and speculation. It is not affiliated with, endorsed by, or connected to Anthropic or any of their proprietary systems.

---

## 概要 / What is OpenMythos?

OpenMythos は **Recurrent-Depth Transformer (RDT)** アーキテクチャを実装したオープンソース LLM フレームワークです。

**127本のREST APIエンドポイント**を備えたサーバーとして動作し、以下の機能を提供します:

| カテゴリ | 主な機能 |
|---------|---------|
| LLM推論 | OpenAI互換チャット・補完・埋め込み API |
| Assistants API | スレッド管理付き会話 (OpenAI Assistants互換) |
| RAG / 検索 | ドキュメントインデックス・ベクトルDB・セマンティック検索 |
| エージェント | ReAct・TDDサイクル・サブエージェント分解・デバッグ |
| マルチモーダル | LLaVA (画像理解) / Whisper (音声) / CLIP / Diffusion |
| ML訓練 | LoRA見積・SimPO・FSDP推定・LM評価 |
| セキュリティ | Webペネトレーションテスト・OSS脆弱性・SBOM生成 |
| DevOps | Docker管理・ファイル監視・クラウド実行 (Modal) |
| SEO / LLMO | AIサーチ最適化スコアリング・コンテンツ評価 |
| データ / 研究 | ArXiv検索・Web検索・ドメイン調査・コードWiki |

---

## クイックスタート / Quickstart

### 方法1: Docker Compose（推奨）

```bash
# 1. リポジトリをクローン
git clone <your-internal-repo-url> OpenMythos
cd OpenMythos

# 2. 環境変数を設定
cp .env.example .env
# .env を編集して API_KEY を設定してください

# 3. 起動
docker compose up --build -d

# 4. 動作確認
curl http://localhost:8000/health
# → {"status":"ok","version":"0.57.0"}

# 5. APIドキュメントをブラウザで確認
open http://localhost:8000/docs
```

### 方法2: ローカル直接起動

```bash
# 依存関係インストール
pip install -r requirements.txt
pip install fastapi uvicorn gunicorn

# サーバー起動（開発モード）
uvicorn serve.api:app --host 0.0.0.0 --port 8000 --reload

# サーバー起動（本番モード）
gunicorn serve.api:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 認証設定 / Authentication

```bash
# .env または環境変数で設定
API_KEY=your-secret-key-here

# リクエスト時は Bearer Token を付与
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model":"openmythos","messages":[{"role":"user","content":"こんにちは"}]}'
```

> **注意:** `API_KEY` を未設定のままだと認証なし (開発モード) で動作します。社内展開時は必ず設定してください。

---

## 主要APIの使い方

### チャット (OpenAI互換)

```python
import requests

BASE = "http://localhost:8000"
HDR  = {"Authorization": "Bearer your-key", "Content-Type": "application/json"}

resp = requests.post(f"{BASE}/v1/chat/completions", headers=HDR, json={
    "model": "openmythos",
    "messages": [
        {"role": "system", "content": "あなたは丁寧なアシスタントです。"},
        {"role": "user",   "content": "機械学習とは何ですか？"},
    ],
    "max_tokens": 256,
})
print(resp.json()["choices"][0]["message"]["content"])
```

### Assistants API (スレッド会話)

```python
# 1. アシスタント作成
asst = requests.post(f"{BASE}/v1/assistants", headers=HDR, json={
    "name": "SupportBot",
    "instructions": "あなたは社内サポートデスクです。丁寧に回答してください。"
}).json()

# 2. スレッド作成
thread = requests.post(f"{BASE}/v1/threads", headers=HDR, json={}).json()

# 3. ユーザーメッセージ追加
requests.post(f"{BASE}/v1/threads/{thread['id']}/messages", headers=HDR, json={
    "role": "user", "content": "有給休暇の申請方法を教えてください"
})

# 4. 実行
run = requests.post(f"{BASE}/v1/threads/{thread['id']}/runs", headers=HDR, json={
    "assistant_id": asst["id"]
}).json()

# 5. 返答を取得
msgs = requests.get(f"{BASE}/v1/threads/{thread['id']}/messages", headers=HDR).json()
for m in msgs["data"]:
    print(f"[{m['role']}] {m['content'][0]['text']['value']}")
```

### RAG (社内ドキュメント検索)

```python
# ドキュメントを登録
requests.post(f"{BASE}/v1/rag/index", headers=HDR, json={
    "documents": [
        "有給休暇は年間20日付与されます。申請はHRシステムから行ってください。",
        "経費精算は月末締めです。領収書は3ヶ月以内に提出してください。",
    ],
    "session_id": "company-handbook"
})

# 検索
result = requests.post(f"{BASE}/v1/rag", headers=HDR, json={
    "query": "有給休暇の申請方法",
    "session_id": "company-handbook"
}).json()
print(result["answer"])
```

### セキュリティスキャン

```python
# Webアプリ脆弱性スキャン
report = requests.post(f"{BASE}/v1/security/scan", headers=HDR, json={
    "target_url": "https://your-app.example.com",
    "timeout": 10.0
}).json()

print(f"リスクスコア: {report['risk_score']:.1f}/10.0")
print(f"検出件数: {len(report['findings'])}")

# Markdownレポート生成
md = requests.post(f"{BASE}/v1/security/report/md", headers=HDR, json={
    "target_url": "https://your-app.example.com"
}).json()["markdown"]
```

### ベクトルDB

```python
# 文書を登録
requests.post(f"{BASE}/v1/vector-store/upsert", headers=HDR, json={
    "id": "doc-001",
    "text": "OpenMythosはRDTアーキテクチャを採用しています",
    "metadata": {"source": "technical-docs", "team": "engineering"}
})

# 類似検索
results = requests.post(f"{BASE}/v1/vector-store/query", headers=HDR, json={
    "query": "アーキテクチャの特徴",
    "top_k": 5
}).json()
```

> 実行可能なサンプルコードは [`examples/api_quickstart.py`](examples/api_quickstart.py) を参照してください。

---

## 全APIエンドポイント一覧

### 基盤 (Core)
| Method | Path | 説明 |
|--------|------|------|
| GET | `/health` | ヘルスチェック |
| GET | `/metrics` | Prometheus メトリクス |
| POST | `/v1/chat/completions` | OpenAI互換チャット補完 |
| POST | `/v1/completions` | テキスト補完 |
| POST | `/v1/embeddings` | 埋め込みベクトル生成 |
| POST | `/v1/semantic-search` | セマンティック検索 |
| POST | `/v1/batch` | バッチ推論 |

### Assistants API (OpenAI互換)
| Method | Path | 説明 |
|--------|------|------|
| POST/GET | `/v1/assistants` | アシスタント作成・一覧 |
| GET/DELETE | `/v1/assistants/{id}` | アシスタント取得・削除 |
| POST/GET/DELETE | `/v1/threads/{id}` | スレッド操作 |
| POST/GET | `/v1/threads/{id}/messages` | メッセージ追加・一覧 |
| POST/GET | `/v1/threads/{id}/runs` | 実行作成・取得 |

### RAG / 検索
| Method | Path | 説明 |
|--------|------|------|
| POST | `/v1/rag/index` | ドキュメントインデックス登録 |
| POST | `/v1/rag` | RAG検索・回答生成 |
| POST | `/v1/vector-store/upsert` | ベクトルDB登録 |
| POST | `/v1/vector-store/query` | ベクトル類似検索 |
| POST | `/v1/search/searxng` | プライバシー重視Web検索 |
| POST | `/v1/search/web` | Web検索 |

### エージェント
| Method | Path | 説明 |
|--------|------|------|
| POST | `/v1/agent/run` | ReActエージェント実行 |
| POST | `/v1/agent/subagent/plan` | タスク分解計画 |
| POST | `/v1/agent/subagent/run` | サブエージェント実行 |
| POST | `/v1/agent/tdd/cycle` | TDDサイクル (コード生成+テスト) |
| POST | `/v1/agent/debug` | 自動デバッグ |
| POST | `/v1/agent/evolve` | 遺伝的アルゴリズム最適化 |

### セキュリティ
| Method | Path | 説明 |
|--------|------|------|
| POST | `/v1/security/scan` | Webアプリ脆弱性スキャン |
| POST | `/v1/security/report/md` | ペネトレーションテストレポート生成 |
| POST | `/v1/security/oss/analyze` | OSS依存関係・脆弱性分析 |
| POST | `/v1/security/oss/sbom` | SBOM (CycloneDX) 生成 |

### ML訓練 / 評価
| Method | Path | 説明 |
|--------|------|------|
| POST | `/v1/peft/estimate` | LoRAメモリ見積 |
| POST | `/v1/training/simpo/compute-loss` | SimPOロス計算 |
| POST | `/v1/training/fsdp/estimate` | FSDP分散学習見積 |
| POST | `/v1/lm-eval` | LMベンチマーク評価 |
| POST | `/v1/extract` | 構造化情報抽出 |

### マルチモーダル
| Method | Path | 説明 |
|--------|------|------|
| POST | `/v1/llava/chat` | 画像+テキスト理解 |
| POST | `/v1/whisper/transcribe` | 音声文字起こし |
| POST | `/v1/diffusion/generate` | 画像生成 |
| POST | `/v1/clip/encode/text` | CLIPテキスト埋め込み |

### DevOps / クラウド
| Method | Path | 説明 |
|--------|------|------|
| POST | `/v1/docker/containers` | Dockerコンテナ一覧 |
| POST | `/v1/docker/build` | Dockerイメージビルド |
| POST | `/v1/watch/config` | ファイル監視設定 |
| POST | `/v1/modal/run` | クラウド関数実行 (Modal) |

全エンドポイントの詳細は起動後に **http://localhost:8000/docs** で確認できます。

---

## Pythonライブラリとして使う

```python
# SEO / LLMOスコアリング
from open_mythos.llmo import LLMOScorer

scorer = LLMOScorer()
result = scorer.score("""
デジタルマーケティングとは、Google・Meta などのプラットフォームを活用して
顧客を獲得する手法です。2024年の調査では SEO 投資は前年比 32% 増加しています。
""")
print(f"LLMO スコア: {result.llmo_total:.3f}")
print(f"エンティティ密度: {result.entity_density:.3f}")

# 広告ROI計算
from open_mythos.tools_marketing import calculate_roi

roi = calculate_roi(ad_spend=500_000, revenue=1_900_000, cogs=400_000,
                    clicks=3_800, impressions=120_000)
print(f"ROI: {roi['roi_pct']:+.1f}%")
print(f"ROAS: {roi['roas']:.2f}x")

# ベクトルストア
from open_mythos.skills.vector_store import VectorStore

vs = VectorStore(dim=384)
vs.upsert("doc1", "重要なドキュメント", {"category": "tech"})
results = vs.query("ドキュメント", top_k=3)
```

---

## アーキテクチャ

OpenMythos は **Recurrent-Depth Transformer (RDT)** を実装しています。

```
Input
  ↓
[Prelude P]         — 標準Transformerレイヤー (1回実行)
  ↓
[Recurrent Block R] — T回ループ実行
  ↑_______↓          h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
  ↓
[Coda C]            — 標準Transformerレイヤー (1回実行)
  ↓
Output
```

**ループ数 = 推論深度**: ループを増やすほど深い推論が可能になります（推論時スケーリング）。

### モデルスケール

| バリアント | dim | パラメータ規模 | 推奨用途 |
|-----------|-----|------------|---------|
| `mythos_1b` | 2048 | ~1B | 開発・検証 |
| `mythos_7b` | 3584 | ~7B | 一般タスク |
| `mythos_100b` | 8192 | ~100B | 高精度タスク |

```python
from open_mythos import mythos_7b, OpenMythos
cfg = mythos_7b()
model = OpenMythos(cfg)
```

### Attention実装

| オプション | クラス | 説明 |
|-----------|--------|------|
| `"gqa"` | `GQAttention` | Grouped Query Attention — KVキャッシュ削減、Flash Attention 2対応 |
| `"mla"` | `MLAttention` | Multi-Latent Attention (DeepSeek-V2方式) — 圧縮KVラテント |

---

## 訓練

```bash
# シングルGPU
python training/3b_fine_web_edu.py

# マルチGPU (DDP)
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
  training/3b_fine_web_edu.py
```

| 設定 | 値 |
|------|---|
| オプティマイザ | AdamW |
| データセット | `HuggingFaceFW/fineweb-edu` |
| 精度 | bfloat16 (H100/A100) / float16 + GradScaler |
| スケジュール | Linear warmup (2000 steps) → cosine decay |

---

## テスト

```bash
# 全テスト実行
python -m pytest tests/ -q

# 特定スプリントのテスト
python -m pytest tests/test_sprint54.py -v

# カバレッジ確認
python -m pytest tests/ --tb=short -q
```

現在 **2862テスト PASS** (Sprint 54 / v0.57.0)

---

## ディレクトリ構成

```
OpenMythos/
├── open_mythos/           # コアライブラリ
│   ├── main.py            # RDTモデル本体
│   ├── assistant.py       # OpenAI Assistants API互換レイヤー
│   ├── skills/            # スキルプラグイン
│   │   ├── vector_store.py
│   │   ├── agent_framework.py
│   │   ├── security.py
│   │   ├── devops_cloud.py
│   │   └── ...
│   ├── llmo.py            # LLMOスコアリング
│   └── tools_marketing.py # マーケティングツール
├── serve/
│   ├── api.py             # FastAPI サーバー (127エンドポイント)
│   ├── auth.py            # Bearer Token認証 + レートリミット
│   ├── Dockerfile         # 本番イメージ
│   └── monitor.py         # 精度監視
├── tests/                 # テスト (2862 PASS)
├── examples/              # サンプルコード
│   └── api_quickstart.py  # API利用サンプル
├── docker-compose.yml     # ワンコマンド起動
├── .env.example           # 環境変数テンプレート
└── requirements.txt       # 依存関係
```

---

## 環境変数リファレンス

`.env.example` をコピーして `.env` を作成し、各値を設定してください。

```bash
cp .env.example .env
```

主要な環境変数:

| 変数 | デフォルト | 説明 |
|------|---------|------|
| `API_KEY` | (なし=認証無効) | Bearer Token認証キー |
| `DEVICE` | `cpu` | `cpu` または `cuda` |
| `DEFAULT_LOOPS` | `4` | デフォルトのループ回数 |
| `RATE_LIMIT_RPM` | `60` | 1分あたりのリクエスト上限 |
| `WORKERS` | `2` | Gunicornワーカー数 |
| `MODEL_CHECKPOINT` | (なし) | 学習済みチェックポイントのパス |

詳細は [`.env.example`](.env.example) を参照してください。

---

## Cloud Run へのデプロイ

```bash
cp serve/cloudrun.env.example serve/cloudrun.env
# serve/cloudrun.env を編集して GCP プロジェクト ID 等を設定

bash serve/deploy_cloudrun.sh
```

---

## 理論的背景

> ℹ️ 本節の「Claude Mythos」は、本プロジェクトが設定した**架空／仮説上の参照モデル**であり、Anthropic の実在製品ではない。OpenMythos はその想定アーキテクチャ（Recurrent-Depth Transformer）を学習目的で理論的に再構成した OSS であり、Anthropic とは無関係・非提携である。

OpenMythos が想定する「Claude Mythos」アーキテクチャは **Recurrent-Depth Transformer (RDT)** です。通常の Transformer が層を積み重ねるのに対し、RDT は一部の層を複数回再利用します。

- **ループ数 = 推論深度**: 推論時にループを増やすだけで深い推論が可能
- **LTI安定性**: 注入パラメータを `ρ(A) < 1` に制約することで訓練安定性を保証
- **Scaling Law**: ループ数とトークン数を同時にスケールする最適則が存在
- **MoE拡張**: FFNをMixture-of-Expertsに置換することで広範な知識を格納

詳細は [`docs/open_mythos.md`](docs/open_mythos.md) を参照してください。

---

## ライセンス

MIT License — Copyright (c) 2026 Kye Gomez

---

## 参考文献

- [Loop, Think, & Generalize — Implicit Reasoning in Recurrent Depth Transformers](https://arxiv.org/pdf/2604.07822)
- [Parcae — Scaling Laws for Stable Looped Language Models](https://arxiv.org/abs/2604.12946)
- [Reasoning with Latent Thoughts — On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416)
- [Relaxed Recursive Transformers — Effective Parameter Sharing with Layer-wise LoRA](https://arxiv.org/pdf/2410.20672)

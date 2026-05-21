# 社内運用ガイド — OpenMythos

## システム構成

```
                        Node.js / 既存サービス
                               │
                    ┌──────────▼──────────┐
                    │   SLA Router :8003  │  ← ループ数自動制御
                    │  serve/sla_router   │     fast / balanced / accurate
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐        │
    │  OpenMythos   │ │  A/B Router  │        │
    │  API  :8000   │ │    :8001     │        │
    │  serve/api.py │ │ serve/ab_    │        │
    └───────────────┘ │  router.py   │        │
                      └──────┬───────┘        │
                             │          ┌─────▼──────┐
                     既存MLモデル        │  Monitor   │
                       :9000            │   :8002    │
                                        │serve/monitor│
                                        └────────────┘
```

---

## 起動手順（開発・ステージング）

```bash
cd OpenMythos

# 1. OpenMythos 推論サーバー（必須）
uvicorn serve.api:app --port 8000 --reload

# 2. SLA ルーター（Node.js からはここを叩く）
uvicorn serve.sla_router:app --port 8003 --reload

# 3. A/B テストルーター（Phase 4-2 から使用）
uvicorn serve.ab_router:app --port 8001 --reload

# 4. 監視ダッシュボード
uvicorn serve.monitor:monitor_app --port 8002 --reload
```

### Docker Compose（一括起動）

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: { context: ., dockerfile: serve/Dockerfile }
    ports: ["8000:8000"]
    environment:
      MODEL_CHECKPOINT: /checkpoints/best.pt
      DEVICE: cpu   # GPU時は cuda
      DEFAULT_LOOPS: "4"
      MAX_LOOPS: "16"

  sla_router:
    build: { context: ., dockerfile: serve/Dockerfile }
    command: uvicorn serve.sla_router:app --host 0.0.0.0 --port 8003
    ports: ["8003:8003"]
    depends_on: [api]

  ab_router:
    build: { context: ., dockerfile: serve/Dockerfile }
    command: uvicorn serve.ab_router:app --host 0.0.0.0 --port 8001
    ports: ["8001:8001"]
    depends_on: [api]

  monitor:
    build: { context: ., dockerfile: serve/Dockerfile }
    command: uvicorn serve.monitor:monitor_app --host 0.0.0.0 --port 8002
    ports: ["8002:8002"]
    volumes: ["./data:/app/data"]
```

---

## Node.js からの呼び出し方

```typescript
// SLA ルーター経由（推奨）
const API = "http://localhost:8003";

// 広告審査（高速）
const res = await fetch(`${API}/infer`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "見出し: 今だけ50%OFF | 説明: 送料無料",
    task: "ad_performance",
    sla_mode: "fast",     // loops=2, budget=200ms
  }),
});

// 詐欺検知（高精度）
const res2 = await fetch(`${API}/infer`, {
  method: "POST",
  body: JSON.stringify({
    text: "深夜に新端末から高額送金",
    task: "fraud_detect",
    sla_mode: "accurate", // loops=16, budget=5000ms
  }),
});
```

---

## SLA 設定一覧（`serve/sla_router.py`）

| タスク | fast | balanced | accurate |
|---|---|---|---|
| `ad_performance` | loops=2, 200ms | loops=4, 500ms | loops=8, 1500ms |
| `content_quality` | loops=2, 300ms | loops=6, 800ms | loops=12, 3000ms |
| `fraud_detect` | loops=4, 600ms | loops=8, 1500ms | loops=16, 5000ms |
| `identity_verify` | loops=3, 400ms | loops=6, 1000ms | loops=10, 2500ms |

ランタイムで変更可能:
```bash
curl -X POST "http://localhost:8003/sla/config/ad_performance/fast?loops=3&budget_ms=250"
```

---

## 監視・アラート

```bash
# 全タスクのドリフト状況を確認
curl http://localhost:8002/monitor/dashboard | jq

# タスク別ドリフト
curl http://localhost:8002/monitor/drift/ad_performance

# 直近24時間メトリクス
curl http://localhost:8002/monitor/metrics?hours=24
```

### ベースライン登録（初期デプロイ時に1回）

```bash
curl -X POST http://localhost:8002/monitor/baseline \
  -H "Content-Type: application/json" \
  -d '{"task":"ad_performance","accuracy":0.82,"avg_score":0.76,"score_p25":0.55,"score_p75":0.91}'
```

---

## A/B テスト

```bash
# 現在の集計を確認
curl http://localhost:8001/ab/stats

# トラフィック比率の変更は serve/ab_router.py の OPENMYTHOS_TRAFFIC_PCT を編集
# 20% → 50% → 100% と段階的に増やす
```

---

## データパイプライン

```bash
# 1. 社内CSVを確認
python scripts/csv_to_jsonl.py --inspect 社内データ.csv --task ad_performance

# 2. 変換
python scripts/csv_to_jsonl.py --task ad_performance --input 社内データ.csv --auto-map

# 3. 前処理
python scripts/preprocess.py --task ad_performance --input data/converted/ad_performance_社内データ.jsonl

# 4. ファインチューニング（GPU 環境）
python scripts/finetune.py --task ad_performance --data-dir data/processed/ad_performance --device cuda
```

---

## Phase 2 GPU 訓練（GCP 準備後）

→ [docs/gcp_gpu_setup.md](gcp_gpu_setup.md) を参照

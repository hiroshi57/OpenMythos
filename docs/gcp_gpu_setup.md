# GCP GPU セットアップガイド — Phase 2 訓練環境

## 1. プロジェクト作成（一緒にやる作業）

GCPプロジェクトを作ったら、以下の情報を教えてください：
- **プロジェクト ID**（例: `openmythos-prod-123456`）

その後の手順はこのガイドに従って進めます。

---

## 2. 必要なAPIの有効化

```bash
gcloud config set project YOUR_PROJECT_ID

gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

---

## 3. GPU インスタンスの起動

### 推奨構成（コスト最適）

| 用途 | マシンタイプ | GPU | 月額目安 |
|---|---|---|---|
| 開発・小規模実験 | `n1-standard-8` | T4 × 1 | 約 ¥30,000 |
| 1B モデル訓練 | `n1-standard-16` | V100 × 1 | 約 ¥80,000 |
| 3B モデル訓練 | `a2-highgpu-1g` | A100 × 1 | 約 ¥200,000 |

```bash
# T4 インスタンス（開発・動作確認用）
gcloud compute instances create openmythos-train \
  --zone=asia-northeast1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --preemptible  # スポットで約70%割引
```

---

## 4. 環境セットアップ（インスタンス内）

```bash
# プロジェクトをコピー
git clone https://github.com/YOUR_ORG/OpenMythos.git
cd OpenMythos

# uv インストール
curl -Ls https://astral.sh/uv/install.sh | sh

# Python 環境
uv python install 3.12
uv venv .venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
uv pip install transformers datasets

# CUDA 確認
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 5. 訓練実行

### Phase 2-2: FineWeb-Edu プレトレイン（300M モデル）

```bash
python training/3b_fine_web_edu.py
```

### Phase 2-3: 社内データ ファインチューニング

```bash
# まずデータを前処理
python scripts/preprocess.py --task all --input data/samples/

# ファインチューニング（GPU使用）
python scripts/finetune.py \
  --task all \
  --data-dir data/processed/all \
  --device cuda \
  --epochs 6 \
  --n-loops 8 \
  --out-dir checkpoints/finetune
```

### perplexity × ループ数の検証

```bash
python scripts/eval_perplexity.py \
  --device cuda \
  --loops 1,2,4,8,16 \
  --out-dir results/
```

---

## 6. チェックポイントの保存（GCS）

```bash
# バケット作成
gsutil mb -l asia-northeast1 gs://YOUR_PROJECT_ID-checkpoints

# アップロード
gsutil cp checkpoints/finetune/best.pt gs://YOUR_PROJECT_ID-checkpoints/best.pt

# 推論サーバーへのダウンロード
gsutil cp gs://YOUR_PROJECT_ID-checkpoints/best.pt /app/best.pt
```

---

## 7. 訓練後の推論サーバー起動

```bash
# チェックポイントを指定して起動
MODEL_CHECKPOINT=/app/best.pt DEVICE=cuda \
  uvicorn serve.api:app --host 0.0.0.0 --port 8000
```

---

## 8. コスト管理

```bash
# 使い終わったら必ず停止
gcloud compute instances stop openmythos-train

# 削除する場合
gcloud compute instances delete openmythos-train
```

> **注意**: Preemptible（スポット）インスタンスは突然停止することがあります。
> `training/3b_fine_web_edu.py` はチェックポイント保存済みなので再開可能です。

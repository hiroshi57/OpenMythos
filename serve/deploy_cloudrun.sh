#!/bin/bash
# OpenMythos — GCP Cloud Run デプロイスクリプト
#
# ─────────────────────────────────────────────────────────────────
# PREREQUISITES (ユーザーが事前に実施):
#   1. GCP プロジェクト作成 & gcloud 認証:
#        gcloud auth login
#        gcloud config set project YOUR_PROJECT_ID
#   2. 必要な API の有効化:
#        gcloud services enable run.googleapis.com \
#            artifactregistry.googleapis.com \
#            cloudbuild.googleapis.com
#   3. Docker のインストール (ローカルビルドの場合)
#
# USAGE:
#   # 環境変数ファイルを使う場合
#   source serve/cloudrun.env && bash serve/deploy_cloudrun.sh
#
#   # 直接指定
#   GCP_PROJECT=my-proj GCP_REGION=asia-northeast1 bash serve/deploy_cloudrun.sh
#
#   # チェックポイント付き
#   GCP_PROJECT=my-proj MODEL_CHECKPOINT=/app/checkpoints/model.pt \
#       bash serve/deploy_cloudrun.sh
# ─────────────────────────────────────────────────────────────────

set -e

# ── 必須パラメータ確認 ────────────────────────────────────────
: "${GCP_PROJECT:?ERROR: GCP_PROJECT is not set. Run: export GCP_PROJECT=your-project-id}"
: "${GCP_REGION:=asia-northeast1}"

# ── 設定 ─────────────────────────────────────────────────────────
IMAGE_TAG="${IMAGE_TAG:-latest}"
SERVICE_NAME="${SERVICE_NAME:-openmythos-serve}"
REGISTRY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/openmythos"
IMAGE="${REGISTRY}/serve:${IMAGE_TAG}"

# モデル設定
DEVICE="${DEVICE:-cpu}"
DEFAULT_LOOPS="${DEFAULT_LOOPS:-4}"
MAX_LOOPS="${MAX_LOOPS:-16}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-}"

# Cloud Run リソース
MEMORY="${CLOUD_RUN_MEMORY:-4Gi}"
CPU="${CLOUD_RUN_CPU:-2}"
CONCURRENCY="${CLOUD_RUN_CONCURRENCY:-10}"
MIN_INST="${CLOUD_RUN_MIN_INSTANCES:-0}"
MAX_INST="${CLOUD_RUN_MAX_INSTANCES:-3}"

echo "=========================================="
echo " OpenMythos Cloud Run Deploy"
echo "  project    : ${GCP_PROJECT}"
echo "  region     : ${GCP_REGION}"
echo "  image      : ${IMAGE}"
echo "  service    : ${SERVICE_NAME}"
echo "  memory     : ${MEMORY}  cpu: ${CPU}"
echo "=========================================="

# ── Artifact Registry リポジトリ作成 (初回のみ) ──────────────
echo "[deploy] ensuring Artifact Registry repository exists…"
gcloud artifacts repositories create openmythos \
    --repository-format=docker \
    --location="${GCP_REGION}" \
    --project="${GCP_PROJECT}" \
    --quiet 2>/dev/null \
    || echo "[deploy] repository already exists — skipping"

# ── Docker 認証 ───────────────────────────────────────────────
echo "[deploy] configuring Docker credentials…"
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

# ── Docker ビルド & プッシュ ──────────────────────────────────
echo "[deploy] building Docker image: ${IMAGE}"
docker build \
    --platform linux/amd64 \
    -t "${IMAGE}" \
    -f serve/Dockerfile \
    .

echo "[deploy] pushing image…"
docker push "${IMAGE}"

# ── 環境変数の組み立て ────────────────────────────────────────
ENV_VARS="DEVICE=${DEVICE},DEFAULT_LOOPS=${DEFAULT_LOOPS},MAX_LOOPS=${MAX_LOOPS}"
if [ -n "${MODEL_CHECKPOINT}" ]; then
    ENV_VARS="${ENV_VARS},MODEL_CHECKPOINT=${MODEL_CHECKPOINT}"
fi

# ── Cloud Run デプロイ ────────────────────────────────────────
echo "[deploy] deploying to Cloud Run…"
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}" \
    --platform managed \
    --region "${GCP_REGION}" \
    --project "${GCP_PROJECT}" \
    --allow-unauthenticated \
    --memory "${MEMORY}" \
    --cpu "${CPU}" \
    --concurrency "${CONCURRENCY}" \
    --min-instances "${MIN_INST}" \
    --max-instances "${MAX_INST}" \
    --port 8000 \
    --set-env-vars "${ENV_VARS}"

# ── デプロイ後の URL 表示 ─────────────────────────────────────
echo ""
echo "=========================================="
echo " Deploy complete!"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${GCP_REGION}" \
    --project "${GCP_PROJECT}" \
    --format "value(status.url)" 2>/dev/null || echo "URL unavailable")
echo "  URL: ${SERVICE_URL}"
echo ""
echo "  Health check:"
echo "    curl ${SERVICE_URL}/health"
echo ""
echo "  Generate example:"
echo "    curl -X POST ${SERVICE_URL}/generate \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"prompt\": \"SEO article intro\", \"task\": \"seo_content\"}'"
echo "=========================================="

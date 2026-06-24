#!/bin/bash
# OpenMythos — SEO LoRA SFT runner for GCP T4 VM (Sprint 35 / task 35.2)
#
# Runs scripts/run_seo_ft.py on a CUDA GPU so the LoraTrainer takes the
# *real* training path (not the CPU simulation fallback) and we can verify
# the DoD: perplexity < 20.
#
# ─────────────────────────────────────────────────────────────────
# PREREQUISITES (performed by the user before running this script):
#
#   1. Create GCP project and enable Compute Engine API:
#        gcloud projects create YOUR_PROJECT_ID
#        gcloud config set project YOUR_PROJECT_ID
#        gcloud services enable compute.googleapis.com
#
#   2. Create a T4 VM (Deep Learning VM, PyTorch image recommended):
#        gcloud compute instances create openmythos-seo-ft \
#          --zone=us-central1-a \
#          --machine-type=n1-standard-8 \
#          --accelerator=type=nvidia-tesla-t4,count=1 \
#          --image-family=pytorch-latest-gpu \
#          --image-project=deeplearning-platform-release \
#          --boot-disk-size=100GB \
#          --maintenance-policy=TERMINATE
#
#   3. SSH into the VM and install the project:
#        gcloud compute ssh openmythos-seo-ft --zone=us-central1-a
#        git clone https://github.com/hiroshi57/OpenMythos.git
#        cd OpenMythos
#        pip install --upgrade pip
#        pip install -e .
#
#   4. Provide the training data. Either:
#        a) generate synthetic data on the VM:
#             python scripts/generate_seo_train.py --n 200 --output data/seo_train.jsonl
#        b) or copy your local file up:
#             gcloud compute scp data/seo_train.jsonl \
#               openmythos-seo-ft:~/OpenMythos/data/ --zone=us-central1-a
#
# ─────────────────────────────────────────────────────────────────
# USAGE (on the VM):
#   bash scripts/run_seo_ft_gcp.sh
#
#   # Custom rounds / steps
#   ROUNDS=5 MAX_STEPS=50 bash scripts/run_seo_ft_gcp.sh
#
#   # Custom input / output
#   INPUT=data/seo_train.jsonl OUT_DIR=checkpoints/seo_ft \
#     bash scripts/run_seo_ft_gcp.sh
#
# RESULT:
#   checkpoints/seo_ft/seo_ft_result.json  ← contains final_perplexity & target_ppl_met
#   Exit code 0 = DoD met (ppl < 20), 1 = not met.
# ─────────────────────────────────────────────────────────────────

set -e

# ── Configuration ─────────────────────────────────────────────────
INPUT="${INPUT:-data/seo_train.jsonl}"
ROUNDS="${ROUNDS:-3}"
MAX_STEPS="${MAX_STEPS:-20}"
LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_DIR="${OUT_DIR:-checkpoints/seo_ft}"
LOG_FILE="${LOG_FILE:-seo_ft.log}"

# ── Validate CUDA (this is the whole point — fail loudly if no GPU) ─
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    || { echo "ERROR: CUDA not available. Without a GPU run_seo_ft.py only simulates."; exit 1; }

# ── Ensure training data exists ───────────────────────────────────
if [ ! -f "${INPUT}" ]; then
    echo "[info] ${INPUT} not found — generating synthetic SEO data (200 records)."
    python scripts/generate_seo_train.py --n 200 --output "${INPUT}"
fi

echo "=========================================="
echo " OpenMythos SEO LoRA SFT (GCP T4)"
echo "  input      : ${INPUT}"
echo "  rounds     : ${ROUNDS}"
echo "  max_steps  : ${MAX_STEPS}"
echo "  lr         : ${LR}"
echo "  batch_size : ${BATCH_SIZE}"
echo "  out_dir    : ${OUT_DIR}"
echo "  log_file   : ${LOG_FILE}"
echo "=========================================="
python -c "import torch; print(' GPU        :', torch.cuda.get_device_name(0))"
echo "=========================================="

# ── Run SFT (tee to log, preserve run_seo_ft.py exit code) ─────────
set -o pipefail
python scripts/run_seo_ft.py \
    --input "${INPUT}" \
    --rounds "${ROUNDS}" \
    --max-steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --batch-size "${BATCH_SIZE}" \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"
RC=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo " Result summary: ${OUT_DIR}/seo_ft_result.json"
echo "=========================================="
cat "${OUT_DIR}/seo_ft_result.json" 2>/dev/null || true

if [ "${RC}" -eq 0 ]; then
    echo ""
    echo "[OK] DoD met — perplexity < 20."
else
    echo ""
    echo "[NG] perplexity >= 20 — increase --rounds/--max-steps or check data."
fi
exit "${RC}"

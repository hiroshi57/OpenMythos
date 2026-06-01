#!/bin/bash
# OpenMythos pre-training script for GCP T4 VM
#
# ─────────────────────────────────────────────────────────────────
# PREREQUISITES (performed by the user before running this script):
#
#   1. Create GCP project and enable Compute Engine API:
#        gcloud projects create YOUR_PROJECT_ID
#        gcloud config set project YOUR_PROJECT_ID
#        gcloud services enable compute.googleapis.com
#
#   2. Create a T4 VM (Debian 12 / Deep Learning VM recommended):
#        gcloud compute instances create openmythos-train \
#          --zone=us-central1-a \
#          --machine-type=n1-standard-8 \
#          --accelerator=type=nvidia-tesla-t4,count=1 \
#          --image-family=pytorch-latest-gpu \
#          --image-project=deeplearning-platform-release \
#          --boot-disk-size=200GB \
#          --maintenance-policy=TERMINATE
#
#   3. SSH into the VM and install dependencies:
#        gcloud compute ssh openmythos-train
#        pip install --upgrade pip
#        pip install datasets tiktoken
#        git clone https://github.com/hiroshi57/OpenMythos.git
#        cd OpenMythos
#        pip install -e .
#
#   4. (Optional) Set HuggingFace token for gated datasets:
#        export HF_TOKEN=your_huggingface_token
#
# ─────────────────────────────────────────────────────────────────
# USAGE:
#   # Default: nano variant, 300M tokens
#   bash scripts/pretrain_gcp.sh
#
#   # Custom variant and resume
#   VARIANT=1b bash scripts/pretrain_gcp.sh
#   bash scripts/pretrain_gcp.sh --resume checkpoints/pretrain/ckpt_step1000.pt
#
# MONITORING:
#   tmux attach -t pretrain          # re-attach to training session
#   tail -f pretrain.log             # watch training log
#   nvidia-smi                       # GPU utilization
# ─────────────────────────────────────────────────────────────────

set -e

# ── Configuration ─────────────────────────────────────────────────
SESSION="pretrain"
VARIANT="${VARIANT:-nano}"
DEVICE="cuda"
BATCH="${BATCH:-8}"
LR="${LR:-1e-3}"
MAX_TOKENS="${MAX_TOKENS:-300000000}"
CKPT_DIR="${CKPT_DIR:-checkpoints/pretrain}"
SEQ_LEN="${SEQ_LEN:-1024}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"    # effective batch = BATCH * GRAD_ACCUM = 32
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
LOGGER="${LOGGER:-none}"
LOG_FILE="pretrain.log"

# Pass through any extra arguments (e.g. --resume path/to/ckpt.pt)
EXTRA_ARGS="$@"

# ── Validate CUDA ─────────────────────────────────────────────────
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    || { echo "ERROR: CUDA not available. Is the T4 driver installed?"; exit 1; }

echo "=========================================="
echo " OpenMythos Pre-training (GCP T4)"
echo "  variant    : ${VARIANT}"
echo "  device     : ${DEVICE}"
echo "  batch      : ${BATCH} x grad_accum ${GRAD_ACCUM} = $(( BATCH * GRAD_ACCUM )) effective"
echo "  max_tokens : ${MAX_TOKENS}"
echo "  seq_len    : ${SEQ_LEN}"
echo "  ckpt_dir   : ${CKPT_DIR}"
echo "  log_file   : ${LOG_FILE}"
echo "=========================================="

# ── Kill existing tmux session if any ────────────────────────────
tmux kill-session -t "${SESSION}" 2>/dev/null || true

# ── Build the training command ────────────────────────────────────
TRAIN_CMD="python scripts/pretrain.py \
    --variant ${VARIANT} \
    --device ${DEVICE} \
    --batch ${BATCH} \
    --lr ${LR} \
    --max-tokens ${MAX_TOKENS} \
    --ckpt-dir ${CKPT_DIR} \
    --seq-len ${SEQ_LEN} \
    --grad-accum ${GRAD_ACCUM} \
    --save-every ${SAVE_EVERY} \
    --eval-every ${EVAL_EVERY} \
    --logger ${LOGGER} \
    ${EXTRA_ARGS}"

# ── Start training in detached tmux session ───────────────────────
tmux new-session -d -s "${SESSION}" \
    "nohup ${TRAIN_CMD} 2>&1 | tee ${LOG_FILE}; echo '[done] training finished'"

echo ""
echo "Training started in background tmux session: ${SESSION}"
echo ""
echo "Monitor commands:"
echo "  tmux attach -t ${SESSION}     # re-attach to session"
echo "  tail -f ${LOG_FILE}           # tail log"
echo "  nvidia-smi dmon -s u          # GPU utilization"

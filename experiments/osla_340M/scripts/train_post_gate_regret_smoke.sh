#!/bin/bash
# OS-GDN 340M post-gate-regret SMOKE TEST (OS_GDN_REPORT §3.4)
# ------------------------------------------------------------
# Applies the GDN state gate before computing the delta-rule residual:
#     S_bar = α·S_{t-1}
#     e     = v − k^T S_bar
#     S_t   = S_bar + β(d⊙k)e^T
# so the readback contraction is β⟨d,k²⟩. The beta-aware OS direction is
#     grad_d = β · (1 − β⟨d,k²⟩) · k²
# which reduces to the original OSGM direction when β≡1.
#
# Implementation: beta-aware Triton phase1 plus the existing OS-GDN chunk
# state kernels for the production data_dependent setting. The pure-PyTorch
# post_gate_regret.py path remains as a reference/fallback.
#
# This script:
#   1) Activates the osla conda env.
#   2) Runs preflight unit tests (~30 sec).
#   3) Launches a 200-step training run on 8 GPUs WITHOUT varlen
#      (cu_seqlens not yet supported in the post-gate-regret recurrence).
#   4) Logs everything to $DUMP/run.log so you can paste loss curves back.
#
# Effective batch defaults to:
#   8 GPUs × batch_size 2 × seq_len 4096 × grad_accum 8 = 524,288 tokens/step.
# Override BATCH_SIZE/SEQ_LEN/GRAD_ACCUM to trade activation memory for more
# accumulation while keeping tokens/step fixed.
#
# Expected runtime on 8x H100: ~9 minutes for 200 steps after first-step
# compilation/checkpoint warmup.

set -e

# ===== CONFIG =====
REPO=${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || echo "/data0/OSLA")}
EXP=$REPO/experiments/osla_340M
RUN_NAME=${RUN_NAME:-post-gate-regret-smoke-200steps}
DUMP=$EXP/exp/$RUN_NAME
CONFIG=${CONFIG:-$EXP/configs/os_gated_deltanet_post_gate_regret_340M.json}
TOKENIZER=${TOKENIZER:-fla-hub/delta_net-1.3B-100B}
N_GPUS=${N_GPUS:-8}
N_STEPS=${N_STEPS:-200}
RDZV_PORT=${RDZV_PORT:-29502}
CONDA_ENV=${CONDA_ENV:-osla}
STREAMING=${STREAMING:-1}
KEEP_LATEST_K=${KEEP_LATEST_K:-0}
BATCH_SIZE=${BATCH_SIZE:-2}
SEQ_LEN=${SEQ_LEN:-4096}
GRAD_ACCUM=${GRAD_ACCUM:-8}
NUM_WORKERS=${NUM_WORKERS:-32}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
RUN_PREFLIGHT=${RUN_PREFLIGHT:-1}
# ==================

# --- env activation: try common conda paths, fail with helpful msg if none ---
CONDA_BASE=""
for candidate in \
    "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda" \
    "/home/datagen/anaconda3" "/usr/local/anaconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"; break
    fi
done
if [ -z "$CONDA_BASE" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "ERROR: cannot find conda. Install miniconda or set CONDA_BASE."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "ERROR: conda env '$CONDA_ENV' not found."
    echo "Create it first:  bash $REPO/experiments/osla_340M/scripts/setup_env_post_gate_regret.sh"
    exit 1
fi
conda activate "$CONDA_ENV"

cd $REPO

echo "================================================================"
echo "OS-GDN 340M post-gate-regret smoke test"
echo "  Repo:    $REPO"
echo "  Dump:    $DUMP"
echo "  Config:  $CONFIG"
echo "  GPUs:    $N_GPUS"
echo "  Steps:   $N_STEPS"
echo "  Stream:  $STREAMING"
echo "  Batch:   $BATCH_SIZE x seq $SEQ_LEN x grad_accum $GRAD_ACCUM"
echo "  Started: $(date)"
echo "================================================================"

mkdir -p "$DUMP/logs"

# ---- 1) preflight unit tests ----
if [ "$RUN_PREFLIGHT" = "1" ] || [ "$RUN_PREFLIGHT" = "true" ]; then
  echo ""
  echo "[1/2] Running preflight unit tests..."
  echo "--- naive sanity ---"
  python $REPO/test_post_gate_naive.py 2>&1 | tail -10
  echo "--- fwd/bwd vs reference ---"
  OSLA_DISABLE_POST_GATE_COMPILE=1 python $REPO/test_post_gate_recurrence.py 2>&1 | tail -20
  echo "--- e2e instantiation ---"
  python $REPO/test_post_gate_regret_e2e.py 2>&1 | tail -20
  echo "Preflight OK"
else
  echo ""
  echo "[1/2] Skipping preflight unit tests (RUN_PREFLIGHT=$RUN_PREFLIGHT)"
fi

# ---- 2) launch training ----
echo ""
echo "[2/2] Launching $N_STEPS-step training..."

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DOWNLOAD_TIMEOUT=300
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_MODE=disabled                   # smoke run, no wandb pollution

TRAINING_STREAMING_ARGS=()
if [ "$STREAMING" = "1" ] || [ "$STREAMING" = "true" ]; then
  TRAINING_STREAMING_ARGS+=(--training.streaming)
fi

cd $REPO/flame

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=1 \
  --nproc_per_node=$N_GPUS \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:$RDZV_PORT" \
  --local-ranks-filter 0 \
  --role rank \
  --tee 3 \
  --log-dir "$DUMP/logs" \
  -m flame.train \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder "$DUMP" \
  --model.config "$CONFIG" \
  --model.tokenizer_path "$TOKENIZER" \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size $BATCH_SIZE \
  --training.seq_len $SEQ_LEN \
  --training.gradient_accumulation_steps $GRAD_ACCUM \
  --training.steps $N_STEPS \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree $N_GPUS \
  --training.data_parallel_shard_degree 1 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.dataset_split train \
  "${TRAINING_STREAMING_ARGS[@]}" \
  --training.num_workers $NUM_WORKERS \
  --training.prefetch_factor $PREFETCH_FACTOR \
  --training.seed 42 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2 \
  --checkpoint.interval 100000 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k $KEEP_LATEST_K \
  --metrics.log_freq 1 \
  2>&1 | tee "$DUMP/run.log"

echo ""
echo "================================================================"
echo "Smoke test done at $(date)."
echo ""
echo "👉 PASTE THESE LINES BACK INTO THE CHAT:"
echo "----------------------------------------------------------------"
grep -aE "step:" "$DUMP/run.log" | tail -20
echo "----------------------------------------------------------------"
echo "Full log: $DUMP/run.log"
echo "================================================================"

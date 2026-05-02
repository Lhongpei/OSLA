#!/bin/bash
# OS-GDN 340M post-gate-regret SMOKE TEST (OS_GDN_REPORT §3.4)
# ------------------------------------------------------------
# Replaces OSGM's surrogate `f(S_t)−f(S_{t−1})` regret with the gate-corrected
# `f(S_t)−f(α·S_{t−1})`, giving hypergradient
#     grad_d = ⟨ẽ, e'⟩ / ‖e'‖² − ⟨d, k²⟩
# which removes the state-gate contamination identified as the structural
# cause of the ~1-nat OS-GDN gap. Reduces to (1 − ⟨d, k²⟩) when α≡1, so
# un-gated OSDN is unaffected.
#
# Implementation: pure-pytorch chunked recurrence with checkpointing
# (fla/ops/os_gated_delta_rule/post_gate_regret.py). Slower than the chunk
# triton kernel but autograd-correct; if smoke succeeds, port to triton.
#
# This script:
#   1) Activates the osla conda env.
#   2) Runs preflight unit tests (~30 sec).
#   3) Launches a 200-step training run on 8 GPUs WITHOUT varlen
#      (cu_seqlens not yet supported in the post-gate-regret recurrence).
#   4) Logs everything to $DUMP/run.log so you can paste loss curves back.
#
# Effective batch: 8 GPUs × batch_size 2 × seq_len 4096 × grad_accum 8
#                = 524,288 tokens/step  (matches all 340M runs in the report).
#
# Expected runtime: ~30-60 min for 200 steps. The pytorch path is roughly
# 2-3× slower than the chunk kernel, but parallelism is the same.

set -e

# ===== CONFIG =====
REPO=/data0/OSLA                           # adjust if your clone lives elsewhere
EXP=$REPO/experiments/osla_340M
RUN_NAME=post-gate-regret-smoke-200steps
DUMP=$EXP/exp/$RUN_NAME
CONFIG=$EXP/configs/os_gated_deltanet_post_gate_regret_340M.json
TOKENIZER=fla-hub/delta_net-1.3B-100B
N_GPUS=${N_GPUS:-8}
N_STEPS=${N_STEPS:-200}
RDZV_PORT=${RDZV_PORT:-29502}
# ==================

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

cd $REPO

echo "================================================================"
echo "OS-GDN 340M post-gate-regret smoke test"
echo "  Repo:    $REPO"
echo "  Dump:    $DUMP"
echo "  Config:  $CONFIG"
echo "  GPUs:    $N_GPUS"
echo "  Steps:   $N_STEPS"
echo "  Started: $(date)"
echo "================================================================"

mkdir -p "$DUMP/logs"

# ---- 1) preflight unit tests ----
echo ""
echo "[1/2] Running preflight unit tests..."
echo "--- naive sanity ---"
python $REPO/test_post_gate_naive.py 2>&1 | tail -10
echo "--- fwd/bwd vs reference ---"
python $REPO/test_post_gate_recurrence.py 2>&1 | tail -20
echo "--- e2e instantiation ---"
python $REPO/test_post_gate_regret_e2e.py 2>&1 | tail -20
echo "Preflight OK"

# ---- 2) launch training ----
echo ""
echo "[2/2] Launching $N_STEPS-step training..."

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DOWNLOAD_TIMEOUT=300
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_DISABLED=true                   # smoke run, no wandb pollution

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
  --training.batch_size 2 \
  --training.seq_len 4096 \
  --training.gradient_accumulation_steps 8 \
  --training.steps $N_STEPS \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree $N_GPUS \
  --training.data_parallel_shard_degree 1 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2 \
  --checkpoint.interval 100000 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 1 \
  --metrics.log_freq 1 \
  2>&1 | tee "$DUMP/run.log"

echo ""
echo "================================================================"
echo "Smoke test done at $(date)."
echo ""
echo "👉 PASTE THESE LINES BACK INTO THE CHAT:"
echo "----------------------------------------------------------------"
grep -E "step:\s+[0-9]+\s+loss:" "$DUMP/run.log" | tail -20
echo "----------------------------------------------------------------"
echo "Full log: $DUMP/run.log"
echo "================================================================"

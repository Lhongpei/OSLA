#!/bin/bash
# Option-B ablation: freeze OSGM params to isolate "learning dynamics" from
# "graph structure" as the cause of the ~1 nat loss gap.
#
# Config: os_gated_deltanet_dd_decay_frozen_340M.json
#   - dd_decay mode (same as main os-gdn run)
#   - osgm_freeze = true:
#       * initial_scale is a buffer of ones (no gradient)
#       * osgm_a_proj.weight/bias frozen (no gradient, constant output = 6.9)
#   - d STILL evolves per-token via phase1 grad·k² accumulation with
#     σ(6.9) ≈ 0.999 constant decay — just no learning feedback
#
# Interpretation at step 2048:
#   loss ≈ 2.99 (baseline) → structure is fine; OSGM learning dynamics broken
#   loss ≈ 4.0 (same as dd_decay learnable) → graph structure itself is off
#
# 2048 steps, 8xH100, same data/optim/schedule as baseline. ~40 min.

set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_MODE=disabled

CONFIG=/data0/OSLA/experiments/osla_340M/configs/os_gated_deltanet_dd_decay_frozen_340M.json
DUMP=/data0/OSLA/experiments/osla_340M/exp/os-gated-deltanet-340M-ablation-B-frozen
TOKENIZER=fla-hub/delta_net-1.3B-100B

mkdir -p "$DUMP/logs"

cd /data0/OSLA/flame

echo "=============================================="
echo "Option-B ablation (OSGM params frozen)"
echo "Dump:    $DUMP"
echo "Started: $(date)"
echo "=============================================="

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:29530" \
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
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.varlen \
  --training.gradient_accumulation_steps 1 \
  --training.steps 2048 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree 8 \
  --training.data_parallel_shard_degree 1 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2 \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 64

echo ""
echo "=============================================="
echo "Ablation B done at $(date)"
echo "=============================================="

#!/bin/bash
# Pure control: d ≡ 1 throughout training (osgm_eta=0, d_min=d_max=1).
# This keeps OS-GDN's dispatch path (chunk_os_gated_delta_rule) active but
# makes d deterministically constant — kw = k · d = k everywhere.
#
# Math of kw=k through OS-GDN kernel should reduce exactly to baseline
# GatedDeltaNet (see chunk.py:190-216). So:
#   loss ≈ 2.99 (baseline) → all 1 nat gap is hypergradient dynamics
#   loss ≠ 2.99            → structural bug in OS-GDN chunk kernel itself
#
# 4-GPU variant (GPUs 4-7; cyzhou has GPUs 0-3) with grad_accum=2 to preserve
# effective global batch = 1 × 4 × 2 = 8 = baseline's 1 × 8 × 1.

set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_MODE=disabled

CONFIG=/data0/OSLA/experiments/osla_340M/configs/os_gated_deltanet_d_constant_340M.json
DUMP=/data0/OSLA/experiments/osla_340M/exp/os-gated-deltanet-340M-ablation-d-constant
TOKENIZER=fla-hub/delta_net-1.3B-100B

mkdir -p "$DUMP/logs"

cd /data0/OSLA/flame

echo "=============================================="
echo "d≡1 control test (2048 steps, GPUs 4-7 + grad_accum=2)"
echo "Dump:    $DUMP"
echo "Started: $(date)"
echo "=============================================="

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:29561" \
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
  --training.gradient_accumulation_steps 2 \
  --training.steps 2048 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree 4 \
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

echo "Done at $(date)"

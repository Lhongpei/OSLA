#!/bin/bash
# Variant A: post-gate residual on OSGM phase1 hypergradient.
#
# Phase1 weights k² by exp(chunk_local_cumsum(g_gdn)), so d's hypergradient
# 1 - <d, k_eff²> is computed against keys attenuated by how much they will
# persist in the chunk-local state. Tests H1 from OS_GDN_REPORT §3.4: the
# (1-α)·v contamination is the dominant cause of OS-GDN's ~1 nat gap, NOT
# time-scale mismatch (which variant B already falsified).
#
# Math (per token, per head, per K-dim):
#   k_sq_eff = k² · exp(g_cum_t),   g_cum = chunk_local_cumsum(g_gdn)
#   term_A   = 1 - <d, k_sq_eff>
#   d_next   = exp(g_decay) · d + η · term_A · k_sq_eff
#
# Implementation: layer computes g_cum in pure-torch under autograd, passes
# as a separate `g_phase1_residual` to the dispatch. Reuses the existing
# gate-aware kernel (the kernel is generic in its 3rd g arg). bwd through
# the cumsum is handled by torch autograd at the layer level.
#
# Comparison targets at step 2048:
#   gated_deltanet-340M-baseline-v2          : 2.99   (golden)
#   os-gated-deltanet-340M-dd-decay (d0=ones): 4.00
#   os-gated-deltanet-340M-ablation-gate-aware: 3.91   (current best OS-GDN)
#   os-gated-deltanet-340M-ablation-d-state-gate: 4.20 (variant B, falsified H2)
#
# 2048 steps, 8xH100, ~40 min.

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

CONFIG=/data0/OSLA/experiments/osla_340M/configs/os_gated_deltanet_post_gate_340M.json
DUMP=/data0/OSLA/experiments/osla_340M/exp/os-gated-deltanet-340M-ablation-post-gate-residual
TOKENIZER=fla-hub/delta_net-1.3B-100B

mkdir -p "$DUMP/logs"

cd /data0/OSLA/flame

echo "=============================================="
echo "Variant A: post-gate residual (k² weighted by exp(g_cum))"
echo "Dump:    $DUMP"
echo "Started: $(date)"
echo "=============================================="

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:29571" \
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

echo "Done at $(date)"

#!/bin/bash
# DeltaNet 340M baseline on 4xH100, matched to the 8-GPU recipe by doubling
# gradient accumulation: 4 GPUs * 2 accum * 1 sample/GPU * 65536 tokens equals
# the original 8 GPUs * 1 accum * 1 sample/GPU * 65536 tokens per optimizer step.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHON=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python
export PATH=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin:$PATH

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/DATA/disk1/cyzhou/hf_home
export HF_DATASETS_CACHE=/DATA/disk1/cyzhou/hf_home/datasets
export HF_HUB_CACHE=/DATA/disk1/cyzhou/hf_home/hub
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export http_proxy=172.30.2.40:3128
export https_proxy=172.30.2.40:3128
export HTTP_PROXY=172.30.2.40:3128
export HTTPS_PROXY=172.30.2.40:3128
export TMPDIR=/DATA/disk1/cyzhou/tmp
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_PROJECT=osla_340M
export WANDB_NAME=deltanet-340M-baseline-4gpu-fair

ROOT=/DATA/disk1/cyzhou/OSLA
DUMP=$ROOT/experiments/osla_340M/exp/deltanet-340M-baseline-4gpu-fair
CONFIG=$ROOT/flame/configs/delta_net_340M.json
TOKENIZER=fla-hub/delta_net-1.3B-100B
FINEWEB_SNAPSHOT=$HF_HUB_CACHE/datasets--HuggingFaceFW--fineweb-edu/snapshots/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
FINEWEB_DATA_FILES=$FINEWEB_SNAPSHOT/sample/10BT/*.parquet

mkdir -p "$DUMP/logs" "$TMPDIR"

cd "$ROOT/flame"

echo "=============================================="
echo "Training DeltaNet 340M baseline on GPUs ${CUDA_VISIBLE_DEVICES}"
echo "Dump: $DUMP"
echo "Started: $(date)"
echo "Fairness: global tokens/step = 4 * 2 * 1 * 65536 = 524288"
echo "=============================================="

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"$PYTHON" -m torch.distributed.run --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:29511" \
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
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree 4 \
  --training.data_parallel_shard_degree 1 \
  --training.dataset parquet \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.data_files "$FINEWEB_DATA_FILES" \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2 \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1

echo "DeltaNet 340M baseline 4-GPU fair training finished at $(date)"

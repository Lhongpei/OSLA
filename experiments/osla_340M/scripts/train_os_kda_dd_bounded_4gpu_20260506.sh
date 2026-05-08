#!/bin/bash
# OS-KDA 340M bounded DD/APF candidate on 4xH100.
# Fairness: 4 GPUs with gradient_accumulation_steps=2 preserves the original
# 8-GPU effective global batch: 4 * 1 * 2 == 8 * 1 * 1, so tokens/step and
# total tokens at 20480 steps match KDA / OS-KDA full runs.

set -euo pipefail

ROOT="/DATA/disk1/cyzhou/OSLA"
PYTHON="${PYTHON:-/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python}"
TORCHRUN="${TORCHRUN:-/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/torchrun}"
ENV_BIN="$(dirname "$PYTHON")"
export PATH="$ENV_BIN:$PATH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-osla_340M}"
export WANDB_NAME="${WANDB_NAME:-os-kda-340M-dd-eta0p003-dmin0p667-dmax1p5-4gpu-full-20260506}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT:$ROOT/flame:${PYTHONPATH:-}"

# Sparse diagnostics: effective KDA write step beta * <d, k^2>.
export OSKDA_LOG_S_EFF="${OSKDA_LOG_S_EFF:-1}"
export OSKDA_LOG_S_EFF_EVERY="${OSKDA_LOG_S_EFF_EVERY:-4600}"
export OSKDA_LOG_S_EFF_LIMIT="${OSKDA_LOG_S_EFF_LIMIT:-50000}"

DUMP="${DUMP:-$ROOT/experiments/osla_340M/exp/os-kda-340M-dd-eta0p003-dmin0p667-dmax1p5-4gpu-full-20260506}"
CONFIG="${CONFIG:-$ROOT/experiments/osla_340M/configs/os-kda-340M-dd-eta0p003-dmin0p667-dmax1p5-ba1.json}"
TOKENIZER="${TOKENIZER:-fla-hub/delta_net-1.3B-100B}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29516}"

mkdir -p "$DUMP/logs"
echo $$ > "$DUMP/train.pid"
{
  echo "started=$(date -Is)"
  echo "host=$(hostname)"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "dump=$DUMP"
  echo "config=$CONFIG"
  echo "wandb_name=$WANDB_NAME"
  echo "fair_batch=4 GPUs * batch 1 * grad_accum 2 = 8 sequences/step"
} | tee "$DUMP/launch.info"

cd "$ROOT/flame"
"$TORCHRUN" --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend c10d \
  --rdzv_endpoint "$RDZV_ENDPOINT" \
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
  --metrics.log_freq 1

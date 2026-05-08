#!/bin/bash
# OS-KDA 340M on 8xH100.
# Mirrors the KDA 340M baseline schedule; only the model config changes from
# kda_340M.json to os_kda_340M.json.

set -euo pipefail

ROOT="$(git -C "$(dirname "$0")/../../.." rev-parse --show-toplevel)"
PYTHON="${PYTHON:-/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python}"
TORCHRUN="${TORCHRUN:-/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/torchrun}"
ENV_BIN="$(dirname "$PYTHON")"
export PATH="$ENV_BIN:$PATH"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-osla_340M}"
export WANDB_NAME="${WANDB_NAME:-os-kda-340M-vs-kda-baseline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DUMP="${DUMP:-$ROOT/experiments/osla_340M/exp/os-kda-340M-vs-kda-baseline}"
CONFIG="${CONFIG:-$ROOT/experiments/osla_340M/configs/os_kda_340M.json}"
TOKENIZER="${TOKENIZER:-fla-hub/delta_net-1.3B-100B}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29507}"

mkdir -p "$DUMP/logs"

cd "$ROOT/flame"

"$TORCHRUN" --nnodes=1 \
  --nproc_per_node=8 \
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
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
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
  --metrics.log_freq 1

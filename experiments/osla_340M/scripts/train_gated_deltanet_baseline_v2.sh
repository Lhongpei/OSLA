#!/bin/bash
# Gated DeltaNet 340M Baseline v2 (chunk mode) on 8xH100
# Follows GatedDeltaNet guide: num_heads*head_dim = 0.75*hidden_size
# head_dim=128, num_heads=6, expand_v=2, 21 layers, ~380M params

set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_PROJECT=osla_340M
export WANDB_NAME=gated-deltanet-340M-baseline-v2

DUMP=/data0/OSLA/experiments/osla_340M/exp/gated-deltanet-340M-baseline-v2
CONFIG=/data0/OSLA/experiments/osla_340M/configs/gated_deltanet_340M_v2.json
TOKENIZER=fla-hub/delta_net-1.3B-100B

mkdir -p $DUMP/logs

cd /data0/OSLA/flame

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:29500" \
  --local-ranks-filter 0 \
  --role rank \
  --tee 3 \
  --log-dir $DUMP/logs \
  -m flame.train \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder $DUMP \
  --model.config $CONFIG \
  --model.tokenizer_path $TOKENIZER \
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

#!/bin/bash
# DeltaNet 1.3B Baseline on 8xH100, 100B tokens (fineweb-edu sample-100BT)
#
# Tokens per step: 8 GPUs × 65536 seq_len = 524,288
# Total steps: 100B / 524,288 ≈ 190,735

set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_PROJECT=deltanet_1.3B_100B
export WANDB_NAME=deltanet-1.3B-baseline

DUMP=/data0/OSLA/experiments/deltanet_1.3B_100B/exp/deltanet-1.3B-baseline
CONFIG=/data0/OSLA/flame/configs/delta_net_1B.json
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
  --lr_scheduler.warmup_steps 2048 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.varlen \
  --training.gradient_accumulation_steps 1 \
  --training.steps 190735 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree 1 \
  --training.data_parallel_shard_degree 8 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-100BT \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --activation_checkpoint.mode full \
  --checkpoint.interval 10000 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 5 \
  --metrics.log_freq 1

echo "DeltaNet 1.3B baseline (100B tokens) training finished!"

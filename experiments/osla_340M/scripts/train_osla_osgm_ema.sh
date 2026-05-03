#!/bin/bash
# EMA variant (unnormalized): d_i = 1/EMA(k_i^2)
# Tests the paper's claim that EMA fixed point d∝1/E[k²] is worse than OSGM's d∝E[k²]/Var(k²)
#
# Fair comparison: 7 GPUs, total tokens matched to 8-GPU baseline
#   original: 8 GPUs × 20480 steps = 163840 global batches
#   adjusted: 7 GPUs × 23406 steps = 163842 global batches (≈ same tokens)
#   warmup:   1024 × 8/7 ≈ 1170 steps (same warmup tokens)
#   ckpt:     2048 × 8/7 ≈ 2341 steps (same ckpt token interval)

set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_PROJECT=osla_340M
export WANDB_NAME=deltanet-340M-osla-osgm-ema

DUMP=/home/datagen/OSLA/experiments/osla_340M/exp/deltanet-340M-osla-osgm-ema
CONFIG=/home/datagen/OSLA/experiments/osla_340M/configs/osla_osgm_ema.json
TOKENIZER=fla-hub/delta_net-1.3B-100B

mkdir -p $DUMP/logs

cd /home/datagen/OSLA/flame

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
/home/datagen/anaconda3/envs/osla/bin/torchrun --nnodes=1 \
  --nproc_per_node=7 \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:29504" \
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
  --lr_scheduler.warmup_steps 1170 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.varlen \
  --training.gradient_accumulation_steps 1 \
  --training.steps 23406 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree 7 \
  --training.data_parallel_shard_degree 1 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2 \
  --checkpoint.interval 2341 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1

echo "EMA (unnormalized) training finished!"

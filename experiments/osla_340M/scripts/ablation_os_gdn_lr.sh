#!/bin/bash
# LR ablation for OS-GDN: fix dd_decay + initial_scale=ones (code edit already
# in place), sweep optimizer.lr only. Tests the hypothesis that the ~1 nat loss
# gap + mid-training spikes are optimization instability, not mechanism error.
#
# Baseline GDN uses lr=1e-3 and converges smoothly.
# All 3 OS-GDN decay modes (dd/none/constant) at lr=1e-3 cluster 4.0-4.5 @step2048
# with visible spikes to ~6 during warmup peak. Suggests lr too high for OSGM.
#
# Each run: 2048 steps, 8xH100, same data/optim/schedule otherwise.

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

CONFIG=/data0/OSLA/experiments/osla_340M/configs/os_gated_deltanet_dd_decay_340M.json
EXP_ROOT=/data0/OSLA/experiments/osla_340M/exp
TOKENIZER=fla-hub/delta_net-1.3B-100B

# (suffix, lr, rdzv_port)
run_one() {
    local SUFFIX="$1"
    local LR="$2"
    local PORT="$3"

    local DUMP="$EXP_ROOT/os-gated-deltanet-340M-ablation-$SUFFIX"
    mkdir -p "$DUMP/logs"

    echo ""
    echo "=============================================="
    echo "LR Ablation: $SUFFIX  (lr=$LR)"
    echo "Dump:    $DUMP"
    echo "Started: $(date)"
    echo "=============================================="

    cd /data0/OSLA/flame

    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    torchrun --nnodes=1 \
      --nproc_per_node=8 \
      --rdzv_backend c10d \
      --rdzv_endpoint "localhost:$PORT" \
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
      --optimizer.lr "$LR" \
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

    echo "[$SUFFIX] finished at $(date)"
}

run_one "L1-lr5e-4" "5e-4" 29520
run_one "L2-lr3e-4" "3e-4" 29521

echo ""
echo "=============================================="
echo "All LR ablations done at $(date)"
echo "=============================================="

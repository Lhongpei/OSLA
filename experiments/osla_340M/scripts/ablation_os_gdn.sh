#!/bin/bash
# Ablation runs (2048 steps each) to identify whether OSGM × GatedDeltaNet
# loss gap is driven by (R1) state-decay timescale mismatch, (R2) d-decay
# timescale too slow, or both.
#
# All experiments share:
#   - 8xH100, same effective batch (524,288 tokens/step)
#   - Same data / optimizer / schedule as the gated_deltanet-340M baseline
#   - `initial_scale = ones` fix already applied in the layer code
#   - Only 2048 training steps (≈40 min each on 8xH100)
#
# Comparison targets at step 2048:
#   gated_deltanet-340M-baseline-v2 : 2.99  (golden)
#   os-gated-deltanet-340M-dd-decay :      4.54  (before initial_scale fix)
#   os-gated-deltanet-340M-dd-decay-d0ones: 4.00  (initial_scale=1 fix only)
#
# Ablations:
#   E1 (decay_mode="none"):     d accumulates only (grows unbounded, clipped at d_max)
#   E3 (decay_mode="constant"): d_{t+1} = 0.9·d_t + ... (matches GDN gate timescale)

set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_MODE=disabled   # don't pollute wandb with short ablations

CONFIG_DIR=/data0/OSLA/experiments/osla_340M/configs
EXP_ROOT=/data0/OSLA/experiments/osla_340M/exp
TOKENIZER=fla-hub/delta_net-1.3B-100B

# (suffix, config_file, rdzv_port)
run_one() {
    local SUFFIX="$1"
    local CFG="$2"
    local PORT="$3"

    local DUMP="$EXP_ROOT/os-gated-deltanet-340M-ablation-$SUFFIX"
    mkdir -p "$DUMP/logs"

    echo ""
    echo "=============================================="
    echo "Ablation: $SUFFIX"
    echo "Config:  $CFG"
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
      --model.config "$CFG" \
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

    echo "[$SUFFIX] finished at $(date)"
}

run_one "E1-none-decay"     "$CONFIG_DIR/os_gated_deltanet_none_decay_340M.json"     29510
run_one "E3-constant-decay" "$CONFIG_DIR/os_gated_deltanet_constant_decay_340M.json" 29511

echo ""
echo "=============================================="
echo "All ablations done at $(date)"
echo "=============================================="

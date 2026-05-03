#!/bin/bash
# OS-GatedDeltaNet 340M with data-dependent d-decay on GPUs 0-3 (4xH100).
# GPUs 4-7 are occupied by another user's job, so we use 4 GPUs +
# gradient_accumulation_steps=2 to preserve the same effective global batch
# as the original 8-GPU recipe (1 * 4 * 2 = 8 = 1 * 8 * 1), keeping
# tokens-per-step (524,288) and total-tokens (~1.07e10) identical to the
# other 340M runs for fair comparison.
#
# On successful training completion, runs the full eval suite on cuda:0.

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_PROJECT=osla_340M
export WANDB_NAME=os-gated-deltanet-340M-dd-decay

DUMP=/data0/OSLA/experiments/osla_340M/exp/os-gated-deltanet-340M-dd-decay
CONFIG=/data0/OSLA/experiments/osla_340M/configs/os_gated_deltanet_dd_decay_340M.json
TOKENIZER=fla-hub/delta_net-1.3B-100B
EVAL_OUT=/data0/OSLA/experiments/osla_340M/eval_results/os-gated-deltanet-340M-dd-decay

mkdir -p "$DUMP/logs" "$EVAL_OUT"

cd /data0/OSLA/flame

echo "=============================================="
echo "Training OS-GDN dd-decay on GPUs 0-3"
echo "Dump: $DUMP"
echo "Started: $(date)"
echo "=============================================="

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:29501" \
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

echo ""
echo "=============================================="
echo "Training complete at $(date). Converting DCP checkpoint to HF format..."
echo "=============================================="

cd /data0/OSLA/flame
python -m flame.utils.convert_dcp_to_hf \
    --path "$DUMP" \
    --step 20480 \
    --config "$CONFIG" \
    --tokenizer "$TOKENIZER"

echo ""
echo "=============================================="
echo "Conversion done. Starting eval_full.sh on cuda:0..."
echo "=============================================="

# Evaluation uses a single device; cuda:0 is the first of our visible set.
bash /data0/OSLA/experiments/osla_340M/scripts/eval_full.sh \
    "$DUMP" "$EVAL_OUT" cuda:0

echo ""
echo "=============================================="
echo "All done at $(date)"
echo "=============================================="

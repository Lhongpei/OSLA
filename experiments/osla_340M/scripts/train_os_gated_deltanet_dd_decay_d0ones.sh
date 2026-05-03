#!/bin/bash
# OS-GatedDeltaNet 340M with data-dependent d-decay, 8xH100, **d0=ones fix**.
#
# Root cause of the previous os-gated-deltanet-340M-dd-decay run (loss stuck
# at ~3.41 vs baseline 2.45): OSGM kernel uses `kw = k * d` everywhere in the
# state/output path; with the OSGM preconditioner d_0 initialized to zeros AND
# the GatedDeltaNet state-forget gate simultaneously compressing h_t, the
# recurrent state was starved for thousands of steps. os_delta_net does not
# see this pathology because plain DeltaNet has no state-forget gate.
#
# Fix applied in /data0/OSLA/fla/layers/gated_deltanet.py (zeros → ones).
# Other config identical to the previous 8-GPU recipe.

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
export WANDB_NAME=os-gated-deltanet-340M-dd-decay-d0ones

DUMP=/data0/OSLA/experiments/osla_340M/exp/os-gated-deltanet-340M-dd-decay-d0ones
CONFIG=/data0/OSLA/experiments/osla_340M/configs/os_gated_deltanet_dd_decay_340M.json
TOKENIZER=fla-hub/delta_net-1.3B-100B
EVAL_OUT=/data0/OSLA/experiments/osla_340M/eval_results/os-gated-deltanet-340M-dd-decay-d0ones

mkdir -p "$DUMP/logs" "$EVAL_OUT"

cd /data0/OSLA/flame

echo "=============================================="
echo "Training OS-GDN dd-decay (d0=ones) on 8xH100"
echo "Dump: $DUMP"
echo "Started: $(date)"
echo "=============================================="

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:29502" \
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

echo ""
echo "=============================================="
echo "Training complete at $(date). Converting DCP → HF..."
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

bash /data0/OSLA/experiments/osla_340M/scripts/eval_full.sh \
    "$DUMP" "$EVAL_OUT" cuda:0

echo ""
echo "=============================================="
echo "All done at $(date)"
echo "=============================================="

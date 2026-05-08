#!/usr/bin/env bash
# OSDN / OSDN-APF 340M on GPU-5, fair 4-GPU recipe.
# Fairness: 4 GPUs * grad_accum 2 * batch 1 * 65536 tokens equals the
# original 8-GPU recipe's 524288 tokens per optimizer step.

set -euo pipefail

ROOT=/DATA/disk1/cyzhou/OSLA
PYTHON=${PYTHON:-/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python}
TORCHRUN=${TORCHRUN:-/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/torchrun}
TOKENIZER=fla-hub/delta_net-1.3B-100B
EXP_ROOT=$ROOT/experiments/osla_340M
TMPDIR=${TMPDIR:-/DATA/disk1/cyzhou/tmp}
GPU_SET=${GPU_SET:-0,1,2,3}

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT:$ROOT/flame:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_SET"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/DATA/disk1/cyzhou/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-300}
export http_proxy=${http_proxy:-172.30.2.40:3128}
export https_proxy=${https_proxy:-172.30.2.40:3128}
export HTTP_PROXY=${HTTP_PROXY:-172.30.2.40:3128}
export HTTPS_PROXY=${HTTPS_PROXY:-172.30.2.40:3128}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_PROJECT=${WANDB_PROJECT:-osla_340M}
export TMPDIR

FINEWEB_SNAPSHOT=$HF_HUB_CACHE/datasets--HuggingFaceFW--fineweb-edu/snapshots/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
FINEWEB_DATA_FILES=$FINEWEB_SNAPSHOT/sample/10BT/*.parquet

mkdir -p "$EXP_ROOT/exp" "$EXP_ROOT/migration_logs" "$TMPDIR"

gpu_set_idle() {
  local gpu pids
  IFS=',' read -ra gpus <<< "$GPU_SET"
  for gpu in "${gpus[@]}"; do
    pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')
    if [[ -n "$pids" ]]; then
      return 1
    fi
  done
  return 0
}

wait_for_gpu_set() {
  echo "[$(date -Is)] Waiting for GPU-5 GPUs $GPU_SET to become idle."
  until gpu_set_idle; do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits
    sleep 300
  done
}

run_train() {
  local name="$1"
  local config="$2"
  local port="$3"
  local dump="$EXP_ROOT/exp/$name"
  local log="$dump/run.log"
  local status="$dump/status.json"

  mkdir -p "$dump/logs"
  {
    echo "started=$(date -Is)"
    echo "host=$(hostname)"
    echo "gpu_set=$GPU_SET"
    echo "dump=$dump"
    echo "config=$config"
    echo "wandb_name=$name"
    echo "fair_batch=4 GPUs * batch 1 * grad_accum 2 = 8 sequences/step"
    echo "tokens_per_step=524288"
  } | tee "$dump/launch.info"

  if [[ -d "$dump/checkpoint/step-20480" ]]; then
    printf '{"status":"already_complete","step":20480,"time":"%s"}\n' "$(date -Is)" > "$status"
    echo "[$(date -Is)] $name already has step-20480; skipping."
    return 0
  fi

  wait_for_gpu_set
  export WANDB_NAME="$name"

  echo "[$(date -Is)] START train $name" | tee -a "$log"
  cd "$ROOT/flame"
  "$TORCHRUN" --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:${port}" \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    --log-dir "$dump/logs" \
    -m flame.train \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder "$dump" \
    --model.config "$config" \
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
    --metrics.log_freq 1 2>&1 | tee -a "$log"

  echo "[$(date -Is)] SUCCESS train $name" | tee -a "$log"
  printf '{"status":"trained","step":20480,"time":"%s"}\n' "$(date -Is)" > "$status"

  cd "$ROOT/flame"
  "$PYTHON" -m flame.utils.convert_dcp_to_hf \
    --path "$dump" \
    --step 20480 \
    --config "$config" \
    --tokenizer "$TOKENIZER" 2>&1 | tee -a "$log"
  printf '{"status":"converted","step":20480,"time":"%s"}\n' "$(date -Is)" > "$status"
}

NO_APF=osdn-340M-eta0p003-dmin0p667-dmax1p5-d0ones-4gpu-fair-20260506
CONFIG_NO_APF=$EXP_ROOT/configs/osdn_340M_eta0p003_dmin0p667_dmax1p5_d0ones.json
APF=osdn-apf-340M-eta0p003-dmin0p667-dmax1p5-d0ones-4gpu-fair-20260506
CONFIG_APF=$EXP_ROOT/configs/osdn_apf_340M_eta0p003_dmin0p667_dmax1p5_d0ones.json

echo "[$(date -Is)] queued OSDN/OSDN-APF eta=0.003 d_min=0.6666667 d_max=1.5 d0=ones on GPU set $GPU_SET"
run_train "$NO_APF" "$CONFIG_NO_APF" 29561
run_train "$APF" "$CONFIG_APF" 29562
echo "[$(date -Is)] OSDN/OSDN-APF queue done."

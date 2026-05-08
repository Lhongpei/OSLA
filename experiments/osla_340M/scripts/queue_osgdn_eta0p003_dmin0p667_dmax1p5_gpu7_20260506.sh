#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cyzhou/OSLA
PYTHON=/home/cyzhou/miniconda3/envs/osla/bin/python
TOKENIZER=fla-hub/delta_net-1.3B-100B
EXP_ROOT=$ROOT/experiments/osla_340M
TMPDIR=${TMPDIR:-/home/cyzhou/tmp}

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT:$ROOT/flame:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/cyzhou/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_TRUST_REMOTE_CODE=1
export http_proxy=172.30.2.40:3128
export https_proxy=172.30.2.40:3128
export HTTP_PROXY=172.30.2.40:3128
export HTTPS_PROXY=172.30.2.40:3128
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_PROJECT=osla_340M
export TMPDIR

mkdir -p "$EXP_ROOT/exp" "$TMPDIR"

gpu_busy_count() {
  local count=0 pids
  for i in 0 1 2 3 4 5 6 7; do
    pids=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')
    if [[ -n "$pids" ]]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

wait_for_all_gpus() {
  echo "[$(date -Is)] Waiting for GPU7 machine GPUs 0-7 to become idle."
  while [[ "$(gpu_busy_count)" != "0" ]]; do
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

  mkdir -p "$dump/logs"
  echo "$config" > "$dump/config_path.txt"
  echo "$name" > "$dump/run_name.txt"

  if [[ -d "$dump/checkpoint/step-20480" ]]; then
    echo "[$(date -Is)] $name already has step-20480; skipping training."
    return 0
  fi

  wait_for_all_gpus
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export WANDB_NAME="$name"

  echo "[$(date -Is)] START train $name" | tee -a "$log"
  cd "$ROOT/flame"
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "$PYTHON" -m torch.distributed.run --nnodes=1 \
    --nproc_per_node=8 \
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
    --metrics.log_freq 1 2>&1 | tee -a "$log"
  echo "[$(date -Is)] SUCCESS train $name" | tee -a "$log"
}

convert_model() {
  local name="$1"
  local config="$2"
  local dump="$EXP_ROOT/exp/$name"
  if [[ -s "$dump/model.safetensors" && -s "$dump/config.json" ]]; then
    echo "[$(date -Is)] $name already converted."
    return 0
  fi
  cd "$ROOT/flame"
  "$PYTHON" -m flame.utils.convert_dcp_to_hf \
    --path "$dump" \
    --step 20480 \
    --config "$config" \
    --tokenizer "$TOKENIZER"
}

run_candidate() {
  local name="$1"
  local config="$2"
  local port="$3"
  local status="$EXP_ROOT/exp/$name/status.json"

  mkdir -p "$EXP_ROOT/exp/$name"
  if run_train "$name" "$config" "$port"; then
    convert_model "$name" "$config"
    printf '{"status":"converted","time":"%s"}\n' "$(date -Is)" > "$status"
  else
    local code=$?
    printf '{"status":"failed","exit_code":%s,"time":"%s"}\n' "$code" "$(date -Is)" > "$status"
    return "$code"
  fi
}

NO_DD=os-gdn-post-gate-regret-eta0p003-dmin0p6667-dmax1p5-no-dd-340m-8gpu-65k-fair-20260506
CONFIG_NO_DD=$EXP_ROOT/configs/os_gated_deltanet_post_gate_regret_340M_eta0p003_dmin0p667_dmax1p5.json
APF=os-gdn-post-gate-regret-eta0p003-dmin0p6667-dmax1p5-apf-340m-8gpu-65k-fair-20260506
CONFIG_APF=$EXP_ROOT/configs/os_gated_deltanet_post_gate_regret_340M_eta0p003_dmin0p667_dmax1p5_dd_gdn_decay.json

echo "[$(date -Is)] queued eta=0.003 d_min=0.6666667 d_max=1.5 OS-GDN + OS-GDN-APF training only"
run_candidate "$NO_DD" "$CONFIG_NO_DD" 29551 || true
run_candidate "$APF" "$CONFIG_APF" 29552 || true

echo "[$(date -Is)] eta0p003 dmin0p6667 dmax1p5 train-only queue done."

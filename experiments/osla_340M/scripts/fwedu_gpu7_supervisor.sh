#!/usr/bin/env bash
set -euo pipefail

BASE=/home/cyzhou/OSLA/experiments/osla_340M
EXP_DIR="$BASE/exp"
RESULT_DIR="$BASE/eval_results"
LOG_DIR="$BASE/migration_logs"
PY=/home/cyzhou/miniconda3/envs/osla/bin/python
HF=/home/cyzhou/miniconda3/envs/osla/bin/hf
SCRIPT="$BASE/scripts/eval_fwedu_val.py"

MODELS=(
  gated-deltanet-340M-baseline
  gated-deltanet-340M-baseline-v2
  deltanet-340M-osla-osgm-chunk-run2
  deltanet-340M-osla-osgm-dd-decay
)

GPUS=(0 1 2 3)

mkdir -p "$EXP_DIR" "$RESULT_DIR" "$LOG_DIR"

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=172.30.2.40:3128
export https_proxy=172.30.2.40:3128
export HTTP_PROXY=172.30.2.40:3128
export HTTPS_PROXY=172.30.2.40:3128

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_DIR/fwedu_gpu7_supervisor.log"
}

log "start"

for model in "${MODELS[@]}"; do
  if [[ ! -f "$EXP_DIR/$model/model.safetensors" ]]; then
    log "download $model"
    "$HF" download Chenyu-Zhou/osla-340m-models \
      --include "$model/*" \
      --local-dir "$EXP_DIR" \
      --max-workers 8
  fi
done

pids=()
for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  gpu="${GPUS[$i]}"
  out="$RESULT_DIR/$model/fwedu_val_10m.json"
  log_file="$LOG_DIR/fwedu_gpu7_${model}.log"
  if [[ -f "$out" ]]; then
    log "skip existing $model"
    continue
  fi
  log "launch gpu=$gpu model=$model"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export HF_HOME=/home/cyzhou/.cache/huggingface
    export HF_DATASETS_CACHE=/home/cyzhou/.cache/huggingface/datasets
    export HF_HUB_CACHE=/home/cyzhou/.cache/huggingface/hub
    export HUGGINGFACE_HUB_CACHE=/home/cyzhou/.cache/huggingface/hub
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export TMPDIR=/tmp
    "$PY" "$SCRIPT" \
      --model_path "$EXP_DIR/$model" \
      --output "$out" \
      --batch_size 16 \
      --device cuda:0
  ) >"$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

log "done status=$status"
exit "$status"

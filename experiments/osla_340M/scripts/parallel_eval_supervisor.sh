#!/usr/bin/env bash
set -euo pipefail

BASE=/DATA/disk1/cyzhou/OSLA/experiments/osla_340M
EXP_DIR="$BASE/exp"
RESULT_DIR="$BASE/eval_results"
LOG_DIR="$BASE/migration_logs"
PY=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python

FW_SCRIPT="$BASE/scripts/eval_fwedu_val.py"
RULER_SCRIPT="$BASE/scripts/eval_ruler_sniah.py"

RULER_N=${RULER_N:-50}
RULER_BATCH_SIZE=${RULER_BATCH_SIZE:-4}
FW_BATCH_SIZE=${FW_BATCH_SIZE:-16}
SLEEP_SEC=${SLEEP_SEC:-20}

GPUS=(0 1 2 3 6 7)
MODELS=(
  kda-340M-baseline
  gated-deltanet-340M-baseline
  gated-deltanet-340M-baseline-v2
  deltanet-340M-osla-osgm-chunk-run2
  deltanet-340M-osla-osgm-dd-decay
)
TASKS=(niah_single_1 niah_single_2 niah_single_3)

declare -A EXPECTED_SIZE=(
  [kda-340M-baseline]=1494453832
  [gated-deltanet-340M-baseline]=1598151416
  [gated-deltanet-340M-baseline-v2]=1519593376
  [deltanet-340M-osla-osgm-chunk-run2]=1497586288
  [deltanet-340M-osla-osgm-dd-decay]=1498378720
)

mkdir -p "$RESULT_DIR" "$LOG_DIR"
mkdir -p /DATA/disk1/cyzhou/tmp /DATA/disk1/cyzhou/hf_home/{hub,datasets,transformers}

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_DIR/parallel_eval_supervisor.log"
}

model_ready() {
  local model=$1
  local file="$EXP_DIR/$model/model.safetensors"
  [[ -f "$file" ]] || return 1
  local size
  size=$(stat -c%s "$file")
  [[ "$size" == "${EXPECTED_SIZE[$model]}" ]]
}

gpu_free() {
  local gpu=$1
  for used in "${ACTIVE_GPU[@]:-}"; do
    [[ "$used" == "$gpu" ]] && return 1
  done
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
  [[ "${mem:-999999}" -lt 500 ]]
}

job_running() {
  local output=$1
  pgrep -f -- "$output" >/dev/null 2>&1
}

job_busy_or_done() {
  local output=$1
  [[ -f "$output" ]] && return 0
  job_running "$output"
}

can_start_key() {
  local key=$1
  local output=$2
  [[ -z "${STARTED[$key]:-}" ]] || return 1
  job_busy_or_done "$output" && return 1
  return 0
}

reap_finished() {
  local new_pids=()
  local new_gpus=()
  local new_labels=()
  local i pid
  for i in "${!ACTIVE_PID[@]}"; do
    pid="${ACTIVE_PID[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      new_gpus+=("${ACTIVE_GPU[$i]}")
      new_labels+=("${ACTIVE_LABEL[$i]}")
    else
      wait "$pid" || true
      log "finished gpu=${ACTIVE_GPU[$i]} label=${ACTIVE_LABEL[$i]}"
    fi
  done
  ACTIVE_PID=("${new_pids[@]}")
  ACTIVE_GPU=("${new_gpus[@]}")
  ACTIVE_LABEL=("${new_labels[@]}")
}

launch_job() {
  local gpu=$1
  local key=$2
  local label=$3
  local log_file=$4
  shift 4
  log "launch gpu=$gpu label=$label log=$log_file"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export NLTK_DATA=/DATA/disk1/cyzhou/nltk_data
    export OSLA_RULER_NUM_SAMPLES="$RULER_N"
    export TMPDIR=/DATA/disk1/cyzhou/tmp
    export HF_ENDPOINT=https://hf-mirror.com
    export http_proxy=172.30.2.40:3128
    export https_proxy=172.30.2.40:3128
    export HTTP_PROXY=172.30.2.40:3128
    export HTTPS_PROXY=172.30.2.40:3128
    export HF_HOME=/DATA/disk1/cyzhou/hf_home
    export HF_HUB_CACHE=/DATA/disk1/cyzhou/hf_home/hub
    export HUGGINGFACE_HUB_CACHE=/DATA/disk1/cyzhou/hf_home/hub
    export HF_DATASETS_CACHE=/DATA/disk1/cyzhou/hf_home/datasets
    export TRANSFORMERS_CACHE=/DATA/disk1/cyzhou/hf_home/transformers
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    "$@"
  ) >"$log_file" 2>&1 &
  ACTIVE_PID+=("$!")
  ACTIVE_GPU+=("$gpu")
  ACTIVE_LABEL+=("$label")
  STARTED["$key"]=1
}

next_gpu() {
  local gpu
  for gpu in "${GPUS[@]}"; do
    if gpu_free "$gpu"; then
      printf '%s\n' "$gpu"
      return 0
    fi
  done
  return 1
}

all_done() {
  local model task
  for model in "${MODELS[@]}"; do
    model_ready "$model" || return 1
    [[ -f "$RESULT_DIR/$model/fwedu_val_10m.json" ]] || return 1
    for task in "${TASKS[@]}"; do
      [[ -f "$RESULT_DIR/$model/ruler_sniah_${task}_2k4k8k_n${RULER_N}.json" ]] || return 1
    done
  done
  [[ "${#ACTIVE_PID[@]}" -eq 0 ]]
}

declare -A STARTED=()
ACTIVE_PID=()
ACTIVE_GPU=()
ACTIVE_LABEL=()

log "start RULER_N=$RULER_N RULER_BATCH_SIZE=$RULER_BATCH_SIZE FW_BATCH_SIZE=$FW_BATCH_SIZE"

while true; do
  reap_finished

  launched=0
  for model in "${MODELS[@]}"; do
    if ! model_ready "$model"; then
      continue
    fi

    out_dir="$RESULT_DIR/$model"
    mkdir -p "$out_dir"

    key="fwedu|$model"
    if can_start_key "$key" "$out_dir/fwedu_val_10m.json"; then
      if gpu=$(next_gpu); then
        launch_job "$gpu" "$key" "fwedu:$model" "$LOG_DIR/fwedu_${model}.log" \
          "$PY" "$FW_SCRIPT" \
            --model_path "$EXP_DIR/$model" \
            --output "$out_dir/fwedu_val_10m.json" \
            --batch_size "$FW_BATCH_SIZE" \
            --device cuda:0
        launched=1
      fi
    fi

    for task in "${TASKS[@]}"; do
      key="ruler|$model|$task"
      out="$out_dir/ruler_sniah_${task}_2k4k8k_n${RULER_N}.json"
      if can_start_key "$key" "$out"; then
        if gpu=$(next_gpu); then
          launch_job "$gpu" "$key" "ruler:$model:$task" "$LOG_DIR/ruler_${model}_${task}_n${RULER_N}.log" \
            "$PY" "$RULER_SCRIPT" \
              --model_path "$EXP_DIR/$model" \
              --output "$out" \
              --tasks "$task" \
              --lengths 2048,4096,8192 \
              --batch_size "$RULER_BATCH_SIZE" \
              --device cuda:0
          launched=1
        fi
      fi
    done
  done

  if all_done; then
    log "all requested FW-Edu and RULER jobs done"
    break
  fi

  if [[ "$launched" -eq 0 ]]; then
    sleep "$SLEEP_SEC"
  fi
done

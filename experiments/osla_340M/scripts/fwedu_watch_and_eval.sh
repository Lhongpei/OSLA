#!/usr/bin/env bash
set -euo pipefail

ROOT=/DATA/disk1/cyzhou/OSLA
PYTHON=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python
HF=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/hf
REPO=Chenyu-Zhou/osla-340m-models
EXP_DIR=$ROOT/experiments/osla_340M/exp
OUT_ROOT=$ROOT/experiments/osla_340M/eval_results
LOG_DIR=$ROOT/migration_logs
SCRIPT=$ROOT/experiments/osla_340M/scripts/eval_fwedu_val.py

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT:$ROOT/flame:${PYTHONPATH:-}"
export HF_HOME=/DATA/disk1/cyzhou/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=/DATA/disk1/cyzhou/tmp
export TORCHINDUCTOR_CACHE_DIR=/DATA/disk1/cyzhou/torchinductor_cache
export http_proxy=172.30.2.40:3128
export https_proxy=172.30.2.40:3128
export HTTP_PROXY=172.30.2.40:3128
export HTTPS_PROXY=172.30.2.40:3128
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$EXP_DIR" "$OUT_ROOT" "$LOG_DIR"

models=(
  kda-340M-baseline
  gated-deltanet-340M-baseline
  gated-deltanet-340M-baseline-v2
  deltanet-340M-osla-osgm-chunk-run2
  deltanet-340M-osla-osgm-dd-decay
)

echo "FW-Edu watcher started at $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

while true; do
  pending=0

  for model in "${models[@]}"; do
    model_dir="$EXP_DIR/$model"
    out_dir="$OUT_ROOT/$model"
    json="$out_dir/fwedu_val_10m.json"
    log="$out_dir/fwedu_val_10m.log"

    if [[ -s "$json" ]]; then
      echo "[$(date -Is)] SKIP eval exists: $model"
      continue
    fi

    if [[ ! -s "$model_dir/model.safetensors" ]]; then
      pending=1
      echo "[$(date -Is)] download attempt: $model"
      if "$HF" download "$REPO" --include "$model/*" --local-dir "$EXP_DIR"; then
        echo "[$(date -Is)] download ok: $model"
      else
        echo "[$(date -Is)] download not ready or failed: $model"
        sleep 120
        continue
      fi
    fi

    if [[ ! -s "$model_dir/model.safetensors" ]]; then
      pending=1
      echo "[$(date -Is)] model still missing after download attempt: $model"
      continue
    fi

    mkdir -p "$out_dir"
    echo "[$(date -Is)] eval start: $model"
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON" "$SCRIPT" \
      --model_path "$model_dir" \
      --output "$json" \
      --device cuda:0 \
      --batch_size 16 \
      --block_size 4096 \
      --max_eval_tokens 10000000 \
      2>&1 | tee "$log"
    echo "[$(date -Is)] eval done: $model"
  done

  if [[ "$pending" -eq 0 ]]; then
    echo "[$(date -Is)] all available models evaluated"
    if [[ -s "$OUT_ROOT/kda-340M-baseline/fwedu_val_10m.json" ]] &&
       [[ -s "$OUT_ROOT/gated-deltanet-340M-baseline/fwedu_val_10m.json" ]] &&
       [[ -s "$OUT_ROOT/gated-deltanet-340M-baseline-v2/fwedu_val_10m.json" ]] &&
       [[ -s "$OUT_ROOT/deltanet-340M-osla-osgm-chunk-run2/fwedu_val_10m.json" ]] &&
       [[ -s "$OUT_ROOT/deltanet-340M-osla-osgm-dd-decay/fwedu_val_10m.json" ]]; then
      echo "[$(date -Is)] all target FW-Edu evals complete"
      exit 0
    fi
  fi

  sleep 180
done

#!/usr/bin/env bash
set -euo pipefail

ROOT=/DATA/disk1/cyzhou/OSLA
PYTHON=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python
EXP_DIR=$ROOT/experiments/osla_340M/exp
OUT_ROOT=$ROOT/experiments/osla_340M/eval_results
SCRIPT=$ROOT/experiments/osla_340M/scripts/eval_ruler_sniah.py

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT:$ROOT/flame:${PYTHONPATH:-}"
export HF_HOME=/DATA/disk1/cyzhou/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=/DATA/disk1/cyzhou/tmp
export NLTK_DATA=/DATA/disk1/cyzhou/nltk_data
export TORCHINDUCTOR_CACHE_DIR=/DATA/disk1/cyzhou/torchinductor_cache
export http_proxy=172.30.2.40:3128
export https_proxy=172.30.2.40:3128
export HTTP_PROXY=172.30.2.40:3128
export HTTPS_PROXY=172.30.2.40:3128
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

models=(
  kda-340M-baseline
  gated-deltanet-340M-baseline
  gated-deltanet-340M-baseline-v2
  deltanet-340M-osla-osgm-chunk-run2
  deltanet-340M-osla-osgm-dd-decay
)

echo "RULER S-NIAH watcher started at $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

while true; do
  pending=0
  for model in "${models[@]}"; do
    model_dir="$EXP_DIR/$model"
    out_dir="$OUT_ROOT/$model"
    json="$out_dir/ruler_sniah_2k4k8k.json"
    log="$out_dir/ruler_sniah_2k4k8k.log"

    if [[ -s "$json" ]]; then
      echo "[$(date -Is)] SKIP RULER exists: $model"
      continue
    fi
    if [[ ! -s "$model_dir/model.safetensors" ]]; then
      pending=1
      echo "[$(date -Is)] WAIT model missing: $model"
      continue
    fi

    mkdir -p "$out_dir"
    echo "[$(date -Is)] RULER start: $model"
    "$PYTHON" "$SCRIPT" \
      --model_path "$model_dir" \
      --output "$json" \
      --lengths 2048,4096,8192 \
      --tasks niah_single_1,niah_single_2,niah_single_3 \
      --batch_size 1 \
      --device cuda:0 \
      2>&1 | tee "$log"
    echo "[$(date -Is)] RULER done: $model"
  done

  if [[ "$pending" -eq 0 ]]; then
    echo "[$(date -Is)] all RULER S-NIAH evals complete"
    exit 0
  fi
  sleep 180
done

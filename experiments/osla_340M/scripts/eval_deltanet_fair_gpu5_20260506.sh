#!/usr/bin/env bash
set -euo pipefail

BASE=/DATA/disk1/cyzhou/OSLA/experiments/osla_340M
ROOT=/DATA/disk1/cyzhou/OSLA
PY=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python
MODEL=$BASE/exp/deltanet-340M-baseline-4gpu-fair
OUT=$BASE/eval_results/deltanet-340M-baseline-4gpu-fair
LOG_DIR=$BASE/migration_logs/deltanet_fair_eval_20260506

mkdir -p "$OUT" "$LOG_DIR" /DATA/disk1/cyzhou/tmp

export PATH=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin:$PATH
export PYTHONPATH=$ROOT:$ROOT/flame:${PYTHONPATH:-}
export NLTK_DATA=/DATA/disk1/cyzhou/nltk_data
export TMPDIR=/DATA/disk1/cyzhou/tmp
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=172.30.2.40:3128
export https_proxy=172.30.2.40:3128
export HTTP_PROXY=172.30.2.40:3128
export HTTPS_PROXY=172.30.2.40:3128
export HF_HOME=/DATA/disk1/cyzhou/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_TRUST_REMOTE_CODE=1
export OSLA_RULER_NUM_SAMPLES=${OSLA_RULER_NUM_SAMPLES:-50}

echo "start $(date -Is)"
echo "model=$MODEL"
echo "out=$OUT"

if [[ ! -s "$MODEL/model.safetensors" || ! -s "$MODEL/config.json" ]]; then
  echo "missing converted HF checkpoint under $MODEL" >&2
  exit 1
fi

run_ruler() {
  local gpu=$1
  local task=$2
  local out=$OUT/ruler_sniah_${task}_2k4k8k_n${OSLA_RULER_NUM_SAMPLES}.json
  if [[ -s "$out" ]]; then
    echo "skip existing $out"
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$BASE/scripts/eval_ruler_sniah.py" \
      --model_path "$MODEL" \
      --output "$out" \
      --tasks "$task" \
      --lengths 2048,4096,8192 \
      --batch_size 4 \
      --device cuda:0
  ) > "$LOG_DIR/ruler_${task}.log" 2>&1
}

run_retrieval() {
  local gpu=$1
  local out=$OUT/retrieval_lmeval.json
  if [[ -s "$out" ]]; then
    echo "skip existing $out"
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$BASE/scripts/eval.py" \
      --model_path "$MODEL" \
      --output "$out" \
      --tasks "triviaqa,drop,nq_open" \
      --batch_size 64 \
      --device cuda:0
  ) > "$LOG_DIR/retrieval_lmeval.log" 2>&1
}

run_ruler 0 niah_single_1 &
pid_s1=$!
run_ruler 1 niah_single_2 &
pid_s2=$!
run_ruler 2 niah_single_3 &
pid_s3=$!
run_retrieval 6 &
pid_ret=$!

status=0
for pid in "$pid_s1" "$pid_s2" "$pid_s3" "$pid_ret"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "done $(date -Is) status=$status"
exit "$status"

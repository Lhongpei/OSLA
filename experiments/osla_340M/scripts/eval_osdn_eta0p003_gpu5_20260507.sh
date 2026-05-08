#!/usr/bin/env bash
set -euo pipefail

BASE=/DATA/disk1/cyzhou/OSLA/experiments/osla_340M
ROOT=/DATA/disk1/cyzhou/OSLA
PY=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python

MODEL_NAME=osdn-340M-eta0p003-dmin0p667-dmax1p5-d0ones-4gpu-fair-20260506
MODEL=$BASE/exp/$MODEL_NAME
OUT=$BASE/eval_results/$MODEL_NAME
LOG_DIR=$BASE/migration_logs/${MODEL_NAME}_eval_20260507
DATA_FILES=/DATA/disk1/cyzhou/hf_home/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/*.parquet

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
export HF_HOME=/DATA/disk1/cyzhou/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_TRUST_REMOTE_CODE=1
export OSLA_RULER_NUM_SAMPLES=${OSLA_RULER_NUM_SAMPLES:-50}

wait_for_gpu() {
  local gpu=$1
  while true; do
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [[ "${mem:-999999}" -lt 500 ]]; then
      return 0
    fi
    echo "[$(date -Is)] wait gpu=$gpu memory.used=${mem}MiB"
    sleep 30
  done
}

run_fwedu() {
  local gpu=$1
  local out=$OUT/fwedu_val_10m.json
  if [[ -s "$out" ]]; then
    echo "skip existing $out"
    return 0
  fi
  wait_for_gpu "$gpu"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$BASE/scripts/eval_fwedu_val_local_parquet_20260506.py" \
      --model_path "$MODEL" \
      --output "$out" \
      --data_files "$DATA_FILES" \
      --batch_size 16 \
      --max_eval_tokens 10000000 \
      --device cuda:0
  ) > "$LOG_DIR/fwedu_val_10m.log" 2>&1
}

run_ruler() {
  local gpu=$1
  local task=$2
  local out=$OUT/ruler_sniah_${task}_2k4k8k_n${OSLA_RULER_NUM_SAMPLES}.json
  if [[ -s "$out" ]]; then
    echo "skip existing $out"
    return 0
  fi
  wait_for_gpu "$gpu"
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

run_retrieval_task() {
  local gpu=$1
  local task=$2
  local out=$OUT/retrieval_lmeval_${task}.json
  if [[ -s "$out" ]]; then
    echo "skip existing $out"
    return 0
  fi
  wait_for_gpu "$gpu"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$BASE/scripts/eval.py" \
      --model_path "$MODEL" \
      --output "$out" \
      --tasks "$task" \
      --batch_size 48 \
      --device cuda:0
  ) > "$LOG_DIR/retrieval_lmeval_${task}.log" 2>&1
}

merge_retrieval() {
  "$PY" - <<'PY'
import json
from pathlib import Path

out_dir = Path("/DATA/disk1/cyzhou/OSLA/experiments/osla_340M/eval_results/osdn-340M-eta0p003-dmin0p667-dmax1p5-d0ones-4gpu-fair-20260506")
parts = [
    out_dir / "retrieval_lmeval_triviaqa.json",
    out_dir / "retrieval_lmeval_drop.json",
    out_dir / "retrieval_lmeval_nq_open.json",
]
if not all(p.exists() and p.stat().st_size > 0 for p in parts):
    raise SystemExit("missing retrieval part")
merged = {"model_path": None, "results": {}, "config": {}}
for path in parts:
    data = json.loads(path.read_text())
    merged["model_path"] = merged["model_path"] or data.get("model_path")
    merged["results"].update(data.get("results", {}))
    if not merged["config"]:
        merged["config"] = data.get("config", {})
(out_dir / "retrieval_lmeval.json").write_text(json.dumps(merged, indent=2) + "\n")
print(json.dumps(merged["results"], indent=2))
PY
}

run_commonsense() {
  local gpu=$1
  local out=$OUT/commonsense_lm.json
  if [[ -s "$out" ]]; then
    echo "skip existing $out"
    return 0
  fi
  wait_for_gpu "$gpu"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$BASE/scripts/eval.py" \
      --model_path "$MODEL" \
      --output "$out" \
      --tasks "piqa,hellaswag,winogrande,arc_easy,arc_challenge,social_iqa,boolq,wikitext,lambada_openai" \
      --batch_size 48 \
      --device cuda:0
  ) > "$LOG_DIR/commonsense_lm.log" 2>&1
}

echo "start $(date -Is)"
echo "model=$MODEL"
echo "out=$OUT"

if [[ ! -s "$MODEL/model.safetensors" || ! -s "$MODEL/config.json" ]]; then
  echo "missing converted HF checkpoint under $MODEL" >&2
  exit 1
fi

status=0

run_fwedu 6 &
pid_fwedu=$!
run_ruler 7 niah_single_1 &
pid_s1=$!
for pid in "$pid_fwedu" "$pid_s1"; do
  if ! wait "$pid"; then status=1; fi
done

run_ruler 6 niah_single_2 &
pid_s2=$!
run_ruler 7 niah_single_3 &
pid_s3=$!
for pid in "$pid_s2" "$pid_s3"; do
  if ! wait "$pid"; then status=1; fi
done

run_retrieval_task 6 nq_open &
pid_nq=$!
run_retrieval_task 7 drop &
pid_drop=$!
for pid in "$pid_nq" "$pid_drop"; do
  if ! wait "$pid"; then status=1; fi
done

run_retrieval_task 6 triviaqa &
pid_trivia=$!
run_commonsense 7 &
pid_common=$!
for pid in "$pid_trivia" "$pid_common"; do
  if ! wait "$pid"; then status=1; fi
done

if [[ "$status" -eq 0 ]]; then
  merge_retrieval > "$LOG_DIR/retrieval_lmeval_merge.log" 2>&1
fi

echo "done $(date -Is) status=$status"
exit "$status"

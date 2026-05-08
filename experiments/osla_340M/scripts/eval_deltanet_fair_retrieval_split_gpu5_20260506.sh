#!/usr/bin/env bash
set -euo pipefail

BASE=/DATA/disk1/cyzhou/OSLA/experiments/osla_340M
ROOT=/DATA/disk1/cyzhou/OSLA
PY=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python
MODEL=$BASE/exp/deltanet-340M-baseline-4gpu-fair
OUT=$BASE/eval_results/deltanet-340M-baseline-4gpu-fair
LOG_DIR=$BASE/migration_logs/deltanet_fair_eval_20260506

mkdir -p "$OUT" "$LOG_DIR"

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

run_task() {
  local gpu=$1
  local task=$2
  local out=$OUT/retrieval_lmeval_${task}.json
  if [[ -s "$out" ]]; then
    echo "skip existing $out"
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$BASE/scripts/eval.py" \
      --model_path "$MODEL" \
      --output "$out" \
      --tasks "$task" \
      --batch_size 64 \
      --device cuda:0
  ) > "$LOG_DIR/retrieval_lmeval_${task}.log" 2>&1
}

echo "start split retrieval $(date -Is)"
run_task 0 nq_open &
pid_nq=$!
run_task 1 drop &
pid_drop=$!
run_task 2 triviaqa &
pid_trivia=$!

status=0
for pid in "$pid_nq" "$pid_drop" "$pid_trivia"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  "$PY" - <<'PY'
import json
from pathlib import Path

out_dir = Path("/DATA/disk1/cyzhou/OSLA/experiments/osla_340M/eval_results/deltanet-340M-baseline-4gpu-fair")
parts = [
    out_dir / "retrieval_lmeval_triviaqa.json",
    out_dir / "retrieval_lmeval_drop.json",
    out_dir / "retrieval_lmeval_nq_open.json",
]
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
fi

echo "done split retrieval $(date -Is) status=$status"
exit "$status"

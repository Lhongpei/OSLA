#!/usr/bin/env bash
set -euo pipefail

ROOT=/DATA/disk1/cyzhou/OSLA
EXP=$ROOT/experiments/osla_340M
PYTHON=/DATA/disk1/cyzhou/miniconda3/envs/osla/bin/python
EVAL_PY=$EXP/scripts/eval.py
OUT_ROOT=$EXP/eval_results/commonsense_repeats_20260506
TASKS=piqa,hellaswag,winogrande,arc_easy,arc_challenge,social_iqa,boolq,wikitext,lambada_openai

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT:$ROOT/flame:${PYTHONPATH:-}"
export TMPDIR=/DATA/disk1/cyzhou/tmp
export HF_HOME=/DATA/disk1/cyzhou/.cache/huggingface
export HF_DATASETS_CACHE=/DATA/disk1/cyzhou/.cache/huggingface/datasets
export HF_HUB_CACHE=/DATA/disk1/cyzhou/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/DATA/disk1/cyzhou/.cache/huggingface/transformers
export TORCHINDUCTOR_CACHE_DIR=/DATA/disk1/cyzhou/torchinductor_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_TRUST_REMOTE_CODE=1
export http_proxy=172.30.2.40:3128
export https_proxy=172.30.2.40:3128
export HTTP_PROXY=172.30.2.40:3128
export HTTPS_PROXY=172.30.2.40:3128

mkdir -p "$OUT_ROOT" "$TMPDIR"

run_variant() {
  local name="$1"
  local model_path="$2"
  local physical_gpu="$3"

  for run in 1 2 3 4 5; do
    local run_dir="$OUT_ROOT/$name/run$(printf '%02d' "$run")"
    mkdir -p "$run_dir"
    echo "[$(date '+%F %T')] start $name run $run on physical GPU $physical_gpu"
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYTHON" "$EVAL_PY" \
      --model_path "$model_path" \
      --output "$run_dir/commonsense_lm.json" \
      --tasks "$TASKS" \
      --batch_size 64 \
      --device cuda:0 \
      2>&1 | tee "$run_dir/eval.log"
    echo "[$(date '+%F %T')] done $name run $run"
  done
}

run_variant \
  osdn_no_dd \
  "$EXP/exp/deltanet-340M-osla-osgm-chunk-run2" \
  6 &
pid_a=$!

run_variant \
  osdn_apf_dd \
  "$EXP/exp/deltanet-340M-osla-osgm-dd-decay" \
  7 &
pid_b=$!

echo "$pid_a $pid_b" > "$OUT_ROOT/worker_pids.txt"
wait "$pid_a"
wait "$pid_b"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

out_root = Path("/DATA/disk1/cyzhou/OSLA/experiments/osla_340M/eval_results/commonsense_repeats_20260506")
tasks = ["piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "social_iqa", "boolq", "lambada_openai"]
metric_order = {
    "piqa": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "social_iqa": "acc,none",
    "boolq": "acc,none",
    "lambada_openai": "acc,none",
}

summary = {}
for variant_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
    rows = []
    for result_path in sorted(variant_dir.glob("run*/commonsense_lm.json")):
        data = json.loads(result_path.read_text())
        values = {}
        for task in tasks:
            values[task] = float(data["results"][task][metric_order[task]])
        values["avg"] = sum(values[t] for t in tasks) / len(tasks)
        values["run"] = result_path.parent.name
        values["path"] = str(result_path)
        rows.append(values)
    if rows:
        best = max(rows, key=lambda row: row["avg"])
        summary[variant_dir.name] = {"runs": rows, "best": best}

(out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
for variant, payload in summary.items():
    best = payload["best"]
    print(variant, best["run"], f"avg={best['avg']:.6f}", " ".join(f"{task}={best[task]:.6f}" for task in tasks))
PY

echo "All commonsense repeats complete. Summary: $OUT_ROOT/summary.json"

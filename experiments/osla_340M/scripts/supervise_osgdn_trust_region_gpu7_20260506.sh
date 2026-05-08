#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cyzhou/OSLA
PYTHON=/home/cyzhou/miniconda3/envs/osla/bin/python
TOKENIZER=fla-hub/delta_net-1.3B-100B
EXP_ROOT=$ROOT/experiments/osla_340M
OUT_ROOT=$EXP_ROOT/eval_results
TMPDIR=${TMPDIR:-/home/cyzhou/tmp}

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT:$ROOT/flame:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/cyzhou/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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

mkdir -p "$EXP_ROOT/exp" "$OUT_ROOT" "$TMPDIR"

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

quick_eval() {
  local name="$1"
  local model="$EXP_ROOT/exp/$name"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  export CUDA_VISIBLE_DEVICES=0

  "$PYTHON" "$EXP_ROOT/scripts/eval.py" \
    --model_path "$model" \
    --output "$out/commonsense_lm.json" \
    --tasks "piqa,hellaswag,winogrande,arc_easy,arc_challenge,social_iqa,boolq,wikitext,lambada_openai" \
    --batch_size 64 \
    --device cuda:0 2>&1 | tee "$out/quick_commonsense.log"

  "$PYTHON" "$EXP_ROOT/scripts/eval_fwedu_val.py" \
    --model_path "$model" \
    --output "$out/fwedu_val_10m.json" \
    --block_size 4096 \
    --batch_size 16 \
    --max_eval_tokens 10000000 \
    --device cuda:0 2>&1 | tee "$out/quick_fwedu.log"

  "$PYTHON" "$EXP_ROOT/scripts/eval_jrt.py" \
    --model_path "$model" \
    --output "$out/retrieval_jrt.json" \
    --tasks "based_fda,based_fda_twice,based_swde,based_swde_twice,based_squad,based_squad_twice" \
    --batch_size 64 \
    --device cuda:0 \
    --context_length 2000 \
    --answer_length 50 2>&1 | tee "$out/quick_jrt.log"

  "$PYTHON" - "$out" > "$out/quick_summary.json" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
common = json.loads((out / "commonsense_lm.json").read_text())
res = common.get("results", common)
vals = []
for task, key in [
    ("piqa", "acc_norm,none"),
    ("hellaswag", "acc_norm,none"),
    ("winogrande", "acc,none"),
    ("arc_easy", "acc_norm,none"),
    ("arc_challenge", "acc_norm,none"),
    ("social_iqa", "acc,none"),
    ("boolq", "acc,none"),
]:
    vals.append(float(res[task][key]))
summary = {
    "common_avg": sum(vals) / len(vals),
    "wikitext_word_ppl": float(res["wikitext"]["word_perplexity,none"]),
    "lambada_ppl": float(res["lambada_openai"]["perplexity,none"]),
    "lambada_acc": float(res["lambada_openai"]["acc,none"]),
}
for fname, key in [("fwedu_val_10m.json", "fwedu_val_10m_ppl"), ("retrieval_jrt.json", "jrt_avg")]:
    p = out / fname
    if p.exists():
        data = json.loads(p.read_text())
        if fname == "fwedu_val_10m.json":
            summary[key] = float(data.get("perplexity", data.get("ppl")))
        else:
            vals = []
            for v in data.values():
                if isinstance(v, dict):
                    for kk in ("contains", "score", "acc"):
                        if kk in v:
                            vals.append(float(v[kk]))
                            break
                elif isinstance(v, (float, int)):
                    vals.append(float(v))
            if vals:
                summary[key] = sum(vals) / len(vals)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  cat "$out/quick_summary.json"
}

passes_baseline_gate() {
  local summary="$1"
  "$PYTHON" - "$summary" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
ok = (
    s.get("common_avg", 0.0) >= 0.463198 and
    s.get("wikitext_word_ppl", 1e9) <= 27.4271
)
raise SystemExit(0 if ok else 1)
PY
}

run_candidate() {
  local name="$1"
  local config="$2"
  local port="$3"
  run_train "$name" "$config" "$port"
  convert_model "$name" "$config"
  quick_eval "$name"
}

CANDIDATE_A=os-gdn-post-gate-regret-eta0p3-dmin0p5-dmax2-340m-8gpu-65k-fair-20260506
CONFIG_A=$EXP_ROOT/configs/os_gated_deltanet_post_gate_regret_340M_eta0p3_dmin0p5_dmax2.json
CANDIDATE_B=os-gdn-post-gate-regret-eta0p1-dmin0-dmax1e9-340m-8gpu-65k-fair-20260506
CONFIG_B=$EXP_ROOT/configs/os_gated_deltanet_post_gate_regret_340M_eta0p1_dmin0_dmax1e9.json

run_candidate "$CANDIDATE_A" "$CONFIG_A" 29541
if passes_baseline_gate "$OUT_ROOT/$CANDIDATE_A/quick_summary.json"; then
  echo "[$(date -Is)] Candidate A passes baseline gate; not launching fallback."
else
  echo "[$(date -Is)] Candidate A misses baseline gate; launching lower-eta high-cap fallback."
  run_candidate "$CANDIDATE_B" "$CONFIG_B" 29542
fi

echo "[$(date -Is)] supervisor done."

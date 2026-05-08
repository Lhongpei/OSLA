#!/usr/bin/env bash
set -eo pipefail

ROOT=/home/cyzhou/OSLA
PYTHON=/home/cyzhou/miniconda3/envs/osla/bin/python
EXP_DIR=$ROOT/experiments/osla_340M/exp
OUT_ROOT=$ROOT/experiments/osla_340M/eval_results
SCRIPT_DIR=$ROOT/experiments/osla_340M/scripts
PPL_PY=$ROOT/evals/ppl.py

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT:$ROOT/flame:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HOME=/home/cyzhou/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=/home/cyzhou/tmp
export NLTK_DATA=/home/cyzhou/nltk_data
export TORCHINDUCTOR_CACHE_DIR=/home/cyzhou/torchinductor_cache
export http_proxy=172.30.2.40:3128
export https_proxy=172.30.2.40:3128
export HTTP_PROXY=172.30.2.40:3128
export HTTPS_PROXY=172.30.2.40:3128

mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$OUT_ROOT"

mark_failed() {
  local out="$1"
  local kind="$2"
  local code="$3"
  printf '{"exit_code":%s,"time":"%s"}\n' "$code" "$(date -Is)" > "$out/FAILED.$kind.json"
}

launch_main() {
  local name="$1"
  local model="$2"
  local device="$3"
  local out="$OUT_ROOT/$name"
  local log="$out/eval_full_gpu7.log"
  mkdir -p "$out"
  rm -f "$out/FAILED.main.json" "$out/SUCCESS.main.json"
  (
    set -eo pipefail
    trap 'code=$?; mark_failed "$out" main "$code"; exit "$code"' ERR
    echo "START main $name $(date -Is) device=$device model=$model"
    "$PYTHON" "$SCRIPT_DIR/eval.py" \
      --model_path "$model" \
      --output "$out/commonsense_lm.json" \
      --tasks "piqa,hellaswag,winogrande,arc_easy,arc_challenge,social_iqa,boolq,wikitext,lambada_openai" \
      --batch_size 64 \
      --device "$device"
    "$PYTHON" "$SCRIPT_DIR/eval_jrt.py" \
      --model_path "$model" \
      --output "$out/retrieval_jrt.json" \
      --tasks "based_fda,based_fda_twice,based_swde,based_swde_twice,based_squad,based_squad_twice" \
      --batch_size 64 \
      --device "$device" \
      --context_length 2000 \
      --answer_length 50
    "$PYTHON" "$SCRIPT_DIR/eval.py" \
      --model_path "$model" \
      --output "$out/retrieval_lmeval.json" \
      --tasks "triviaqa,drop,nq_open" \
      --batch_size 64 \
      --device "$device"
    "$PYTHON" "$SCRIPT_DIR/eval.py" \
      --model_path "$model" \
      --output "$out/longbench.json" \
      --tasks "longbench_narrativeqa,longbench_qasper,longbench_multifieldqa_en,longbench_hotpotqa,longbench_2wikimqa,longbench_musique,longbench_gov_report,longbench_qmsum,longbench_multi_news,longbench_trec,longbench_triviaqa,longbench_samsum,longbench_passage_count,longbench_passage_retrieval_en" \
      --batch_size 16 \
      --device "$device"
    "$PYTHON" "$PPL_PY" \
      -p "$model" \
      -d fla-hub/pg19 \
      -s train \
      --block_size 20480 \
      --bucket_size 2048 \
      --batch_size 8 \
      --device "$device" \
      2>&1 | tee "$out/ppl_pg19_20k.log"
    echo "SUCCESS main $name $(date -Is)"
    printf '{"time":"%s"}\n' "$(date -Is)" > "$out/SUCCESS.main.json"
  ) > "$log" 2>&1 &
  echo $! > "$out/eval_full_gpu7.pid"
  echo "[$name] main PID $(cat "$out/eval_full_gpu7.pid") -> $log"
}

launch_fwedu() {
  local name="$1"
  local model="$2"
  local device="$3"
  local out="$OUT_ROOT/$name"
  local log="$out/fwedu_val_10m.log"
  mkdir -p "$out"
  rm -f "$out/FAILED.fwedu.json" "$out/SUCCESS.fwedu.json"
  (
    set -eo pipefail
    trap 'code=$?; mark_failed "$out" fwedu "$code"; exit "$code"' ERR
    echo "START fwedu $name $(date -Is) device=$device model=$model"
    "$PYTHON" "$SCRIPT_DIR/eval_fwedu_val.py" \
      --model_path "$model" \
      --output "$out/fwedu_val_10m.json" \
      --block_size 4096 \
      --batch_size 16 \
      --max_eval_tokens 10000000 \
      --device "$device"
    echo "SUCCESS fwedu $name $(date -Is)"
    printf '{"time":"%s"}\n' "$(date -Is)" > "$out/SUCCESS.fwedu.json"
  ) > "$log" 2>&1 &
  echo $! > "$out/fwedu_val_10m.pid"
  echo "[$name] fwedu PID $(cat "$out/fwedu_val_10m.pid") -> $log"
}

launch_ruler() {
  local name="$1"
  local model="$2"
  local device="$3"
  local out="$OUT_ROOT/$name"
  local log="$out/ruler_sniah_2k4k8k_n50.log"
  mkdir -p "$out"
  rm -f "$out/FAILED.ruler.json" "$out/SUCCESS.ruler.json"
  (
    set -eo pipefail
    trap 'code=$?; mark_failed "$out" ruler "$code"; exit "$code"' ERR
    echo "START ruler $name $(date -Is) device=$device model=$model"
    for task in niah_single_1 niah_single_2 niah_single_3; do
      "$PYTHON" "$SCRIPT_DIR/eval_ruler_sniah.py" \
        --model_path "$model" \
        --output "$out/ruler_sniah_${task}_2k4k8k_n50.json" \
        --tasks "$task" \
        --lengths 2048,4096,8192 \
        --batch_size 4 \
        --device "$device" \
        --limit 50
    done
    echo "SUCCESS ruler $name $(date -Is)"
    printf '{"time":"%s"}\n' "$(date -Is)" > "$out/SUCCESS.ruler.json"
  ) > "$log" 2>&1 &
  echo $! > "$out/ruler_sniah_2k4k8k_n50.pid"
  echo "[$name] ruler PID $(cat "$out/ruler_sniah_2k4k8k_n50.pid") -> $log"
}

N1=os-gdn-post-gate-no-dd-wdfix-340m-8gpu-65k-fair-20260505-175937
M1=$EXP_DIR/$N1
N2=os-gdn-post-gate-dd-gdn-decay-wdfix-340m-8gpu-65k-fair-20260505-224851
M2=$EXP_DIR/$N2

test -s "$M1/model.safetensors"
test -s "$M2/model.safetensors"
test -x "$SCRIPT_DIR/eval_ruler_sniah.py"

launch_main "$N1" "$M1" cuda:0
launch_main "$N2" "$M2" cuda:1
launch_fwedu "$N1" "$M1" cuda:2
launch_fwedu "$N2" "$M2" cuda:3
launch_ruler "$N1" "$M1" cuda:4
launch_ruler "$N2" "$M2" cuda:5

echo "LAUNCHED_ALL $(date -Is)"

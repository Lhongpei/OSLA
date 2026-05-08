#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/datagen/OSLA}
PYTHON=${PYTHON:-/home/datagen/anaconda3/envs/osla/bin/python}
EXP_ROOT=$ROOT/experiments/osla_340M
TOKENIZER=${TOKENIZER:-fla-hub/delta_net-1.3B-100B}
NAME=${NAME:-os-gdn-post-gate-regret-sep-apf-eta0p003-dmin0p6667-dmax1p5-340m-8gpu-65k-datagen-20260507}
CONFIG=${CONFIG:-$EXP_ROOT/configs/os_gated_deltanet_post_gate_regret_340M_eta0p003_dmin0p667_dmax1p5_sep_apf.json}
DUMP=$EXP_ROOT/exp/$NAME
LOG=$DUMP/run.log
PORT=${PORT:-29567}

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT:$ROOT/flame:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/data0/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=1
if [[ "${OSLA_USE_PROXY:-0}" == "1" ]]; then
  export http_proxy=${http_proxy:-172.30.2.40:3128}
  export https_proxy=${https_proxy:-172.30.2.40:3128}
  export HTTP_PROXY=${HTTP_PROXY:-172.30.2.40:3128}
  export HTTPS_PROXY=${HTTPS_PROXY:-172.30.2.40:3128}
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
fi
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export WANDB_PROJECT=${WANDB_PROJECT:-osla_340M}
export WANDB_NAME=$NAME

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
  echo "[$(date -Is)] Waiting for datagen GPUs 0-7 to become idle."
  while [[ "$(gpu_busy_count)" != "0" ]]; do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits
    sleep 300
  done
}

mkdir -p "$DUMP/logs"
echo "$CONFIG" > "$DUMP/config_path.txt"
echo "$NAME" > "$DUMP/run_name.txt"

wait_for_all_gpus
cd "$ROOT/flame"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "[$(date -Is)] START train $NAME" | tee -a "$LOG"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"$PYTHON" -m torch.distributed.run --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:$PORT" \
  --local-ranks-filter 0 \
  --role rank \
  --tee 3 \
  --log-dir "$DUMP/logs" \
  -m flame.train \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder "$DUMP" \
  --model.config "$CONFIG" \
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
  --metrics.log_freq 1 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] SUCCESS train $NAME" | tee -a "$LOG"
cd "$ROOT/flame"
"$PYTHON" -m flame.utils.convert_dcp_to_hf \
  --path "$DUMP" \
  --step 20480 \
  --config "$CONFIG" \
  --tokenizer "$TOKENIZER"
printf '{"status":"converted","time":"%s"}\n' "$(date -Is)" > "$DUMP/status.json"

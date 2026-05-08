#!/bin/bash
# Wait for GPUs 0-3 to become free, then launch the bounded no-APF OS-KDA run.

set -euo pipefail

ROOT="/DATA/disk1/cyzhou/OSLA"
DUMP="$ROOT/experiments/osla_340M/exp/os-kda-340M-noapf-eta0p003-dmin0p667-dmax1p5-4gpu-full-20260506"
TRAIN="$ROOT/experiments/osla_340M/scripts/train_os_kda_noapf_bounded_4gpu_20260506.sh"

mkdir -p "$DUMP"
echo "waiter_started=$(date -Is)" | tee -a "$DUMP/wait_launch.log"

while true; do
  if pgrep -af "os-kda-340M-noapf-eta0p003-dmin0p667-dmax1p5|train_os_kda_noapf_bounded_4gpu_20260506" \
    | grep -v pgrep \
    | grep -v wait_launch >/dev/null; then
    echo "already_running=$(date -Is)" | tee -a "$DUMP/wait_launch.log"
    exit 0
  fi

  max_mem=$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, '$1 + 0 < 4 {gsub(/ /, "", $2); if ($2 > m) m = $2} END {print m + 0}'
  )
  echo "$(date -Is) max_mem_0_3=${max_mem}MiB" | tee -a "$DUMP/wait_launch.log"

  if [ "$max_mem" -lt 5000 ]; then
    break
  fi
  sleep 60
done

echo "launching=$(date -Is)" | tee -a "$DUMP/wait_launch.log"
cd "$ROOT"
nohup bash "$TRAIN" > "$DUMP/run.log" 2>&1 &
echo "launch_pid=$!" | tee -a "$DUMP/wait_launch.log"

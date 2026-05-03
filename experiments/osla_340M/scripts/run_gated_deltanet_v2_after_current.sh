#!/bin/bash
# Queue: wait for current training → run v2 → run KDA → run full eval on all models

echo "[$(date)] Waiting for gated-deltanet-340M-baseline to finish..."
while pgrep -f "dump_folder.*gated-deltanet-340M-baseline[^-]" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] Starting gated-deltanet-340M-baseline-v2..."
bash /data0/OSLA/experiments/osla_340M/scripts/train_gated_deltanet_baseline_v2.sh
echo "[$(date)] gated-deltanet-340M-baseline-v2 finished."

echo "[$(date)] Starting kda-340M-baseline..."
bash /data0/OSLA/experiments/osla_340M/scripts/train_kda_baseline.sh
echo "[$(date)] kda-340M-baseline finished."

echo "[$(date)] All training done. Starting full evaluation on all models..."
bash /data0/OSLA/experiments/osla_340M/scripts/eval_all_models.sh cuda:0
echo "[$(date)] All evaluations done."

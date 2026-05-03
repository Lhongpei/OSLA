#!/bin/bash
# Run all three variant A stability rescues sequentially.
# Each is 2048 steps × 8×H100 ≈ 40 min. Total ≈ 2 hours.

set -e
cd /data0/OSLA

EXP_DIR=/data0/OSLA/experiments/osla_340M
SCRIPT_DIR=$EXP_DIR/scripts
LOG_DIR=$EXP_DIR/exp

echo "###### [$(date)] Run #1: cumulative + LR=5e-4 ######"
bash "$SCRIPT_DIR/ablation_post_gate_residual_lr5e4.sh" \
    > "$LOG_DIR/ablation_post_gate_lr5e4.log" 2>&1 || echo "Run #1 exited non-zero"

echo "###### [$(date)] Run #2: cumulative + clamp(-5) + LR=1e-3 ######"
bash "$SCRIPT_DIR/ablation_post_gate_residual_clamp.sh" \
    > "$LOG_DIR/ablation_post_gate_clamp.log" 2>&1 || echo "Run #2 exited non-zero"

echo "###### [$(date)] Run #3: remaining decay + LR=1e-3 ######"
bash "$SCRIPT_DIR/ablation_post_gate_residual_remaining.sh" \
    > "$LOG_DIR/ablation_post_gate_remaining.log" 2>&1 || echo "Run #3 exited non-zero"

echo "###### [$(date)] All three runs completed ######"

# Print step-2048 loss for each
echo
echo "=== Summary ==="
for tag in lr5e4 clamp remaining; do
    log="$LOG_DIR/ablation_post_gate_${tag}.log"
    if [ -f "$log" ]; then
        line=$(grep "step: 2048" "$log" | tail -1)
        echo "[${tag}] $line"
    fi
done

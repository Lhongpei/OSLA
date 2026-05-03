#!/bin/bash
# Parallel evaluation: 6 models on 6 GPUs simultaneously
set -e

EXP_DIR="/data0/OSLA/experiments/osla_340M"
EVAL_SCRIPT="$EXP_DIR/scripts/eval_full.sh"

# Model name -> (model_path, gpu_id)
declare -a NAMES=(
    "1_deltanet_baseline"
    "2_osgm"
    "3_osgm_learnable_d0"
    "4_deltanet_gated"
    "5_gated_deltanet_v2"
    "6_kda"
)
declare -a PATHS=(
    "$EXP_DIR/exp/deltanet-340M-baseline"
    "$EXP_DIR/exp/deltanet-340M-osla-osgm-chunk-run2"
    "$EXP_DIR/exp/deltanet-340M-osla-osgm-chunk-learnable-d0"
    "$EXP_DIR/exp/gated-deltanet-340M-baseline"
    "$EXP_DIR/exp/gated-deltanet-340M-baseline-v2"
    "$EXP_DIR/exp/kda-340M-baseline"
)
declare -a GPUS=(0 1 2 3 6 7)

PIDS=()

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    model_path="${PATHS[$i]}"
    gpu="${GPUS[$i]}"
    output_dir="$EXP_DIR/eval_results/$name"

    if [ ! -f "$model_path/config.json" ]; then
        echo "[$name] SKIP: no config.json at $model_path"
        continue
    fi

    mkdir -p "$output_dir"
    echo "[$name] Starting on cuda:$gpu -> $output_dir"

    nohup bash "$EVAL_SCRIPT" "$model_path" "$output_dir" "cuda:$gpu" \
        > "$output_dir/eval.log" 2>&1 &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
done

echo ""
echo "=============================================="
echo "All ${#PIDS[@]} evaluations launched in parallel!"
echo "PIDs: ${PIDS[*]}"
echo "Monitor: tail -f $EXP_DIR/eval_results/*/eval.log"
echo "=============================================="

# Wait for all to finish
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "PID $pid finished (exit code: $?)"
done

echo ""
echo "All evaluations complete!"

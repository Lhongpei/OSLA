#!/bin/bash
# Evaluate all trained models with the full evaluation suite
# Runs sequentially on a single GPU after all training completes
#
# Models:
#   1. DeltaNet Baseline (374M)
#   2. OSGM (chunk, run2)
#   3. OSGM + Learnable D0
#   4. DeltaNet + gate (~400M)
#   5. GatedDeltaNet v2 (~380M)
#   6. KDA (~374M)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="/data0/OSLA/experiments/osla_340M"
EVAL_SCRIPT="$SCRIPT_DIR/eval_full.sh"
DEVICE="${1:-cuda:0}"

echo "=============================================="
echo "Batch evaluation of all models"
echo "Device: $DEVICE"
echo "=============================================="

declare -A MODELS
MODELS=(
    ["1_deltanet_baseline"]="$EXP_DIR/exp/deltanet-340M-baseline"
    ["2_osgm"]="$EXP_DIR/exp/deltanet-340M-osla-osgm-chunk-run2"
    ["3_osgm_learnable_d0"]="$EXP_DIR/exp/deltanet-340M-osla-osgm-chunk-learnable-d0"
    ["4_deltanet_gated"]="$EXP_DIR/exp/gated-deltanet-340M-baseline"
    ["5_gated_deltanet_v2"]="$EXP_DIR/exp/gated-deltanet-340M-baseline-v2"
    ["6_kda"]="$EXP_DIR/exp/kda-340M-baseline"
)

# Sort by key to ensure consistent order
for key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    model_path="${MODELS[$key]}"
    output_dir="$EXP_DIR/eval_results/$key"

    echo ""
    echo "======================================================"
    echo "Evaluating: $key"
    echo "Model path: $model_path"
    echo "Output dir: $output_dir"
    echo "======================================================"

    if [ ! -d "$model_path" ]; then
        echo "  WARNING: Model directory not found, skipping: $model_path"
        continue
    fi

    # Check if model has a config.json (i.e., training completed and saved)
    if [ ! -f "$model_path/config.json" ]; then
        echo "  WARNING: No config.json found (training may not have completed), skipping."
        continue
    fi

    if [ -f "$output_dir/commonsense_lm.json" ] && \
       [ -f "$output_dir/longbench.json" ]; then
        echo "  Already evaluated (commonsense_lm.json + longbench.json exist), skipping."
        echo "  To re-evaluate, delete: $output_dir/"
        continue
    fi

    bash "$EVAL_SCRIPT" "$model_path" "$output_dir" "$DEVICE"

    echo ""
    echo "  => $key evaluation complete."
done

echo ""
echo "=============================================="
echo "All model evaluations complete!"
echo "Results in: $EXP_DIR/eval_results/"
echo "=============================================="

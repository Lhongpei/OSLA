#!/bin/bash
# Convert DCP checkpoints to HF format, then run full evaluation on all models
set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300

EXP_DIR="/data0/OSLA/experiments/osla_340M"
FLAME_DIR="/data0/OSLA/flame"
TOKENIZER="fla-hub/delta_net-1.3B-100B"

cd "$FLAME_DIR"

# -----------------------------------------------
# Convert new models from DCP to HF format
# -----------------------------------------------
echo "=============================================="
echo "Converting DCP checkpoints to HF format"
echo "=============================================="

declare -A CONVERT_MODELS
CONVERT_MODELS=(
    ["gated-deltanet-340M-baseline"]="$EXP_DIR/configs/gated_deltanet_340M.json"
    ["gated-deltanet-340M-baseline-v2"]="$EXP_DIR/configs/gated_deltanet_340M_v2.json"
    ["kda-340M-baseline"]="$EXP_DIR/configs/kda_340M.json"
)

for model_name in "${!CONVERT_MODELS[@]}"; do
    model_path="$EXP_DIR/exp/$model_name"
    config_path="${CONVERT_MODELS[$model_name]}"

    if [ -f "$model_path/config.json" ] && [ -f "$model_path/model.safetensors" ]; then
        echo "[$model_name] Already converted, skipping."
        continue
    fi

    if [ ! -d "$model_path/checkpoint/step-20480" ]; then
        echo "[$model_name] WARNING: No step-20480 checkpoint found, skipping."
        continue
    fi

    echo ""
    echo "[$model_name] Converting checkpoint..."
    python -m flame.utils.convert_dcp_to_hf \
        --path "$model_path" \
        --step 20480 \
        --config "$config_path" \
        --tokenizer "$TOKENIZER"
    echo "[$model_name] Conversion done."
done

echo ""
echo "=============================================="
echo "Starting full evaluation on all models"
echo "=============================================="

bash "$EXP_DIR/scripts/eval_all_models.sh" cuda:0

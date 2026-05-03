#!/bin/bash
# Wait for both decay trainings to finish, convert checkpoints, then evaluate
set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_TRUST_REMOTE_CODE=1

EXP_DIR="/data0/OSLA/experiments/osla_340M"

# ==========================================
# Model 8: OSGM + Constant Decay (gamma=0.999)
# ==========================================
MODEL_NAME_1="deltanet-340M-osla-osgm-constant-decay"
MODEL_PATH_1="$EXP_DIR/exp/$MODEL_NAME_1"
OUTPUT_DIR_1="$EXP_DIR/eval_results/8_osgm_constant_decay"
CONFIG_1="$EXP_DIR/configs/osla_osgm_constant_decay.json"

echo "[$(date)] Waiting for constant decay training to finish..."
while pgrep -f "$MODEL_NAME_1" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] Constant decay training done."

# Convert DCP to HF format
echo "[$(date)] Converting constant decay checkpoint..."
cd /data0/OSLA/flame
python -m flame.utils.convert_dcp_to_hf \
    --path "$MODEL_PATH_1" \
    --step 20480 \
    --config "$CONFIG_1" \
    --tokenizer fla-hub/delta_net-1.3B-100B
echo "[$(date)] Conversion done."

# Run full evaluation
echo "[$(date)] Starting constant decay evaluation..."
bash "$EXP_DIR/scripts/eval_full.sh" "$MODEL_PATH_1" "$OUTPUT_DIR_1" cuda:0
echo "[$(date)] Constant decay evaluation done!"

# ==========================================
# Model 9: OSGM + Data-Dependent Decay
# ==========================================
MODEL_NAME_2="deltanet-340M-osla-osgm-dd-decay"
MODEL_PATH_2="$EXP_DIR/exp/$MODEL_NAME_2"
OUTPUT_DIR_2="$EXP_DIR/eval_results/9_osgm_dd_decay"
CONFIG_2="$EXP_DIR/configs/osla_osgm_dd_decay.json"

echo "[$(date)] Waiting for dd decay training to finish..."
while pgrep -f "$MODEL_NAME_2" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] DD decay training done."

# Convert DCP to HF format
echo "[$(date)] Converting dd decay checkpoint..."
cd /data0/OSLA/flame
python -m flame.utils.convert_dcp_to_hf \
    --path "$MODEL_PATH_2" \
    --step 20480 \
    --config "$CONFIG_2" \
    --tokenizer fla-hub/delta_net-1.3B-100B
echo "[$(date)] Conversion done."

# Run full evaluation
echo "[$(date)] Starting dd decay evaluation..."
bash "$EXP_DIR/scripts/eval_full.sh" "$MODEL_PATH_2" "$OUTPUT_DIR_2" cuda:0
echo "[$(date)] DD decay evaluation done!"

echo ""
echo "[$(date)] All done! Results in:"
echo "  $OUTPUT_DIR_1"
echo "  $OUTPUT_DIR_2"

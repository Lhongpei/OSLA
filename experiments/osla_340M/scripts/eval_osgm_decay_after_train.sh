#!/bin/bash
# Wait for OSGM-decay training to finish, convert checkpoint, then evaluate
set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_TRUST_REMOTE_CODE=1

EXP_DIR="/data0/OSLA/experiments/osla_340M"
MODEL_NAME="deltanet-340M-osla-osgm-decay-chunk"
MODEL_PATH="$EXP_DIR/exp/$MODEL_NAME"
OUTPUT_DIR="$EXP_DIR/eval_results/7_osgm_decay"

echo "[$(date)] Waiting for OSGM-decay training to finish..."
while pgrep -f "$MODEL_NAME" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] Training done."

# Convert DCP to HF format
echo "[$(date)] Converting checkpoint..."
cd /data0/OSLA/flame
python -m flame.utils.convert_dcp_to_hf \
    --path "$MODEL_PATH" \
    --step 20480 \
    --config "$EXP_DIR/configs/osla_osgm_decay_chunk.json" \
    --tokenizer fla-hub/delta_net-1.3B-100B
echo "[$(date)] Conversion done."

# Run full evaluation
echo "[$(date)] Starting evaluation..."
bash "$EXP_DIR/scripts/eval_full.sh" "$MODEL_PATH" "$OUTPUT_DIR" cuda:0
echo "[$(date)] All done!"

#!/bin/bash
# Run remaining evaluation phases in parallel, skipping completed ones
# Uses larger batch sizes for faster evaluation

set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_TRUST_REMOTE_CODE=1

EXP_DIR="/data0/OSLA/experiments/osla_340M"
SCRIPT_DIR="$EXP_DIR/scripts"
EVAL_PY="$SCRIPT_DIR/eval.py"
EVAL_JRT_PY="$SCRIPT_DIR/eval_jrt.py"
PPL_PY="/data0/OSLA/evals/ppl.py"

eval_model() {
    local name="$1"
    local model_path="$2"
    local device="$3"
    local output_dir="$EXP_DIR/eval_results/$name"

    mkdir -p "$output_dir"
    echo "[$name @ $device] Starting remaining phases..."

    # Phase 1: Commonsense/LM
    if [ ! -f "$output_dir/commonsense_lm.json" ]; then
        echo "  [$name] [1/5] Commonsense/LM..."
        python "$EVAL_PY" \
            --model_path "$model_path" \
            --output "$output_dir/commonsense_lm.json" \
            --tasks "piqa,hellaswag,winogrande,arc_easy,arc_challenge,social_iqa,boolq,wikitext,lambada_openai" \
            --batch_size 64 --device "$device"
    else
        echo "  [$name] [1/5] Commonsense/LM: SKIP (done)"
    fi

    # Phase 2: JRT Retrieval
    if [ ! -f "$output_dir/retrieval_jrt.json" ]; then
        echo "  [$name] [2/5] JRT Retrieval..."
        python "$EVAL_JRT_PY" \
            --model_path "$model_path" \
            --output "$output_dir/retrieval_jrt.json" \
            --tasks "based_fda,based_fda_twice,based_swde,based_swde_twice,based_squad,based_squad_twice" \
            --batch_size 64 --device "$device" \
            --context_length 2000 --answer_length 50
    else
        echo "  [$name] [2/5] JRT Retrieval: SKIP (done)"
    fi

    # Phase 3: Retrieval lm_eval
    if [ ! -f "$output_dir/retrieval_lmeval.json" ]; then
        echo "  [$name] [3/5] Retrieval lm_eval..."
        python "$EVAL_PY" \
            --model_path "$model_path" \
            --output "$output_dir/retrieval_lmeval.json" \
            --tasks "triviaqa,drop,nq_open" \
            --batch_size 64 --device "$device"
    else
        echo "  [$name] [3/5] Retrieval lm_eval: SKIP (done)"
    fi

    # Phase 4: LongBench
    if [ ! -f "$output_dir/longbench.json" ]; then
        echo "  [$name] [4/5] LongBench..."
        python "$EVAL_PY" \
            --model_path "$model_path" \
            --output "$output_dir/longbench.json" \
            --tasks "longbench_narrativeqa,longbench_qasper,longbench_multifieldqa_en,longbench_hotpotqa,longbench_2wikimqa,longbench_musique,longbench_gov_report,longbench_qmsum,longbench_multi_news,longbench_trec,longbench_triviaqa,longbench_samsum,longbench_passage_count,longbench_passage_retrieval_en" \
            --batch_size 16 --device "$device"
    else
        echo "  [$name] [4/5] LongBench: SKIP (done)"
    fi

    # Phase 5: PPL PG19
    if [ ! -f "$output_dir/ppl_pg19_20k.log" ]; then
        echo "  [$name] [5/5] PPL PG19..."
        python "$PPL_PY" \
            -p "$model_path" -d fla-hub/pg19 -s train \
            --block_size 20480 --bucket_size 2048 --batch_size 8 \
            --device "$device" 2>&1 | tee "$output_dir/ppl_pg19_20k.log"
    else
        echo "  [$name] [5/5] PPL PG19: SKIP (done)"
    fi

    echo "[$name @ $device] ALL DONE."
}

# Launch all 6 models in parallel on available GPUs
declare -a NAMES=(1_deltanet_baseline 2_osgm 3_osgm_learnable_d0 4_deltanet_gated 5_gated_deltanet_v2 6_kda)
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
    eval_model "${NAMES[$i]}" "${PATHS[$i]}" "cuda:${GPUS[$i]}" \
        > "$EXP_DIR/eval_results/${NAMES[$i]}/eval.log" 2>&1 &
    PIDS+=($!)
    echo "Launched ${NAMES[$i]} on cuda:${GPUS[$i]} (PID: ${PIDS[-1]})"
done

echo ""
echo "All 6 launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."

for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "PID $pid done (exit: $?)"
done

echo "ALL EVALUATIONS COMPLETE!"

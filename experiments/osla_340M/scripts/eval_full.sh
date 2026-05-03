#!/bin/bash
# Full evaluation suite aligned with Gated DeltaNet paper
# Covers: Commonsense/LM, Real Retrieval (cloze), LongBench 14 tasks
#
# Usage: bash eval_full.sh <model_path> <output_dir> [device]
# Example: bash eval_full.sh exp/deltanet-340M-baseline results/deltanet-baseline cuda:0

set -e

source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate osla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL_PATH="${1:?Usage: eval_full.sh <model_path> <output_dir> [device]}"
OUTPUT_DIR="${2:?Usage: eval_full.sh <model_path> <output_dir> [device]}"
DEVICE="${3:-cuda:0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY="$SCRIPT_DIR/eval.py"
EVAL_JRT_PY="$SCRIPT_DIR/eval_jrt.py"
PPL_PY="/data0/OSLA/evals/ppl.py"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Full Evaluation Suite"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=============================================="

# -----------------------------------------------
# 1. Commonsense / LM benchmarks
# -----------------------------------------------
echo ""
echo "[1/5] Commonsense / LM benchmarks..."
echo "  Tasks: piqa, hellaswag, winogrande, arc_easy, arc_challenge, social_iqa, boolq, wikitext, lambada_openai"

python "$EVAL_PY" \
    --model_path "$MODEL_PATH" \
    --output "$OUTPUT_DIR/commonsense_lm.json" \
    --tasks "piqa,hellaswag,winogrande,arc_easy,arc_challenge,social_iqa,boolq,wikitext,lambada_openai" \
    --batch_size 64 \
    --device "$DEVICE"

echo "  => Saved to $OUTPUT_DIR/commonsense_lm.json"

# -----------------------------------------------
# 2. Real Retrieval (JRT cloze completion, 2K tokens)
# -----------------------------------------------
echo ""
echo "[2/5] Real Retrieval (JRT cloze, context=2K)..."
echo "  Tasks: FDA, SWDE, SQuAD (single + twice variants)"

python "$EVAL_JRT_PY" \
    --model_path "$MODEL_PATH" \
    --output "$OUTPUT_DIR/retrieval_jrt.json" \
    --tasks "based_fda,based_fda_twice,based_swde,based_swde_twice,based_squad,based_squad_twice" \
    --batch_size 64 \
    --device "$DEVICE" \
    --context_length 2000 \
    --answer_length 50

echo "  => Saved to $OUTPUT_DIR/retrieval_jrt.json"

# -----------------------------------------------
# 3. Real Retrieval via lm_eval (TriviaQA, DROP, NQ)
# -----------------------------------------------
echo ""
echo "[3/5] Real Retrieval (lm_eval: TriviaQA, DROP, NQ)..."

python "$EVAL_PY" \
    --model_path "$MODEL_PATH" \
    --output "$OUTPUT_DIR/retrieval_lmeval.json" \
    --tasks "triviaqa,drop,nq_open" \
    --batch_size 64 \
    --device "$DEVICE"

echo "  => Saved to $OUTPUT_DIR/retrieval_lmeval.json"

# -----------------------------------------------
# 4. LongBench 14 tasks
# -----------------------------------------------
echo ""
echo "[4/5] LongBench (14 English single-doc + multi-doc + summarization tasks)..."
echo "  Tasks: narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique,"
echo "         gov_report, qmsum, multi_news, trec, triviaqa, samsum, passage_count, passage_retrieval_en"

python "$EVAL_PY" \
    --model_path "$MODEL_PATH" \
    --output "$OUTPUT_DIR/longbench.json" \
    --tasks "longbench_narrativeqa,longbench_qasper,longbench_multifieldqa_en,longbench_hotpotqa,longbench_2wikimqa,longbench_musique,longbench_gov_report,longbench_qmsum,longbench_multi_news,longbench_trec,longbench_triviaqa,longbench_samsum,longbench_passage_count,longbench_passage_retrieval_en" \
    --batch_size 16 \
    --device "$DEVICE"

echo "  => Saved to $OUTPUT_DIR/longbench.json"

# -----------------------------------------------
# 5. Length Extrapolation PPL (PG19, up to 20K)
# -----------------------------------------------
echo ""
echo "[5/5] Length Extrapolation PPL (PG19, block=20480, bucket=2048)..."

python "$PPL_PY" \
    -p "$MODEL_PATH" \
    -d fla-hub/pg19 \
    -s train \
    --block_size 20480 \
    --bucket_size 2048 \
    --batch_size 8 \
    --device "$DEVICE" \
    2>&1 | tee "$OUTPUT_DIR/ppl_pg19_20k.log"

echo "  => Saved to $OUTPUT_DIR/ppl_pg19_20k.log"

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Results in: $OUTPUT_DIR/"
echo "=============================================="

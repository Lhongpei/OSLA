# OSLA vs DeltaNet 340M Pretraining Experiment

## Overview

This experiment compares **OSLA (Online Scaled Linear Attention)** against the standard **DeltaNet** (delta rule linear attention) on a 340M-parameter language model pretraining task.

OSLA extends DeltaNet by introducing an **adaptive per-dimension preconditioner** to the state update rule:

- **DeltaNet**: `S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k_t) * k_t^T`
- **OSLA**:    `S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k_t) * k_t^T / (D_t + 1)`

where `D_t = D_{t-1} + k_t^2` is a cumulative per-dimension preconditioner (AdaGrad-style), controlling the magnitude of state updates based on key statistics.

## Setup

### Model Architecture

Both models share identical architecture (374.4M parameters):

| Hyperparameter | Value |
|----------------|-------|
| Hidden size    | 1024  |
| Num layers     | 24    |
| Num heads      | 8     |
| Head dim (K=V) | 128   |
| FFN ratio      | 4x (SwiGLU) |
| Vocab size     | 32000 |
| QK activation  | SiLU  |
| QK norm        | L2    |
| Short conv     | Yes (kernel=4) |

Config files: [`configs/baseline.json`](configs/baseline.json) (DeltaNet), [`configs/osla.json`](configs/osla.json) (OSLA)

### Training

| Setting | Value |
|---------|-------|
| Hardware | 8x NVIDIA H100 80GB |
| Dataset | FineWeb-Edu sample-10BT |
| Tokenizer | fla-hub/delta_net-1.3B-100B (32K vocab) |
| Total steps | 20,480 |
| Tokens seen | ~10.7B |
| Sequence length | 65,536 (varlen packing, context_len=4096) |
| Batch size | 1 per GPU x 8 GPUs = 524,288 tokens/step |
| Optimizer | AdamW (lr=1e-3, eps=1e-15) |
| LR schedule | Cosine decay (warmup 1024 steps, min_lr=0.1x) |
| Grad clipping | max_norm=1.0, skip NaN/Inf |
| Precision | BF16 (AMP) |
| Activation ckpt | Selective (every 2 layers) |

Key difference: DeltaNet uses **chunk** mode (parallelizable, O(T) training); OSLA uses **fused_recurrent** mode (sequential, required by the preconditioner).

### Training Speed

| Model | Throughput (tok/s) | Wall Time | MFU |
|-------|--------------------|-----------|-----|
| DeltaNet (chunk)         | ~110,000 | ~3.5h  | ~38% |
| OSLA (fused_recurrent)   | ~30,000  | ~13h   | ~10% |

OSLA is ~3.6x slower due to the sequential recurrence (fused_recurrent cannot parallelize across time steps like the chunk algorithm).

### Training Loss Trajectory

| Step | DeltaNet | OSLA |
|------|----------|------|
| 1    | 10.58    | 10.58 |
| 100  | 7.35     | 7.09  |
| 500  | 4.31     | 4.28  |
| 1,000  | 3.41   | 3.47  |
| 2,000  | 3.00   | 3.05  |
| 5,000  | 2.76   | 2.81  |
| 10,000 | 2.61   | 2.65  |
| 15,000 | 2.54   | 2.58  |
| 20,000 | 2.48   | 2.51  |
| **20,480** | **2.49** | **2.53** |

Both models converge to similar final loss, with DeltaNet slightly lower (~0.04 gap).

## Evaluation Results

### Perplexity

| Benchmark | DeltaNet (baseline) | OSLA | Delta |
|-----------|--------------------:|-----:|------:|
| **WikiText word-ppl** | **28.50** | 30.02 | +1.52 |
| **PG19 block 0 ppl** | 27.33 | **26.80** | **-0.53** |
| **PG19 block 1 ppl** | 22.95 | **22.18** | **-0.77** |
| **PG19 block 2 ppl** | 22.17 | **21.62** | **-0.55** |
| **PG19 block 3 ppl** | 22.62 | **22.12** | **-0.50** |

- Lower is better. Bold indicates the winner.
- PG19 evaluates on long-form book text (block_size=2048, 4 blocks per sample).
- WikiText evaluates on shorter Wikipedia articles.

### Downstream Tasks (DeltaNet baseline only)

OSLA's fused_recurrent mode is too slow for loglikelihood-based evaluation (sequential inference at batch_size=1). Downstream scores are reported for the baseline only as a reference:

| Task | Accuracy | Acc (norm) |
|------|----------|------------|
| lambada_openai | 0.321 | - |
| piqa | 0.662 | 0.650 |
| hellaswag | 0.333 | 0.396 |
| winogrande | 0.509 | - |
| arc_easy | 0.571 | 0.508 |
| arc_challenge | 0.241 | 0.270 |

## Analysis

### OSLA outperforms DeltaNet on long-context evaluation

On PG19 (long book text), OSLA consistently beats DeltaNet by **0.5-0.8 ppl** across all blocks. This aligns with the theoretical motivation: the adaptive preconditioner `D_t = sum(k_t^2)` accumulates key statistics over the sequence, allowing the model to modulate state update magnitudes based on what dimensions have been frequently written to. This is particularly beneficial for long documents where the state accumulates over thousands of tokens.

### DeltaNet is slightly better on WikiText

On WikiText (shorter articles), DeltaNet wins by 1.5 ppl. With shorter sequences, the preconditioner has less time to accumulate meaningful statistics, and the additional `/(D_t + 1)` denominator may over-dampen early updates.

### Training efficiency trade-off

OSLA's fused_recurrent mode is 3.6x slower than DeltaNet's chunk mode. This is a fundamental limitation: the preconditioner `D_t` depends on all previous `k_t`, making the recurrence inherently sequential. A chunk-based OSLA implementation (in progress under `fla/ops/osla_delta_rule/chunk.py`) would be needed to close this gap.

## Bug Fixes Applied During This Experiment

### 1. Backward pass gradient errors (original kernel)

The original OSLA kernel (`fla/ops/delta_rule/fused_recurrent_osla.py`) had incorrect gradients:
- `dq` was computed using un-normalized state `S_t` (missing the `/(D_t+1)` denominator)
- `dk` ignored the gradient through the preconditioner `D_t`

We switched to the corrected kernel at `fla/ops/osla_delta_rule/fused_recurrent.py` (commits `48db7c5`..`5f9a015`).

### 2. Numerical instability (eps too small)

The corrected kernel used `eps=1e-5` in the denominator `(D_t + eps)`. With L2-normalized keys, individual dimensions can have very small `k[i]^2`, leading to:
- Forward: up to 1e5x amplification of state updates
- Backward: `O(1/eps^2) = O(1e10)` gradient magnitudes

This caused NaN at ~800 training steps. Fixed by changing eps from `1e-5` to `1.0`, which:
- Caps maximum amplification to 1.0x (no amplification when `D=0`)
- Matches the intended behavior: `1/(D+1)` starts at 1 and decays as D grows
- Eliminates gradient explosion in the backward pass

### 3. Model type registration

Added OSLA model type (`osla_delta_net`) to `fla/models/__init__.py` so that `AutoModelForCausalLM` can resolve it.

## Reproducing

### Train baseline
```bash
bash experiments/osla_340M/scripts/train_baseline.sh
```

### Train OSLA
```bash
bash experiments/osla_340M/scripts/train_osla.sh
```

### Evaluate
```bash
# Convert DCP checkpoint to HuggingFace format (from flame/ directory)
python -m flame.utils.convert_dcp_to_hf \
  --path <exp_dir> --step 20480 \
  --tokenizer fla-hub/delta_net-1.3B-100B \
  --config <config.json>

# WikiText perplexity
python experiments/osla_340M/scripts/eval.py \
  --model_path <model_dir> \
  --output <output.json> \
  --tasks wikitext --batch_size 4

# PG19 perplexity
python evals/ppl.py \
  --path <model_dir> \
  --block_size 2048 --bucket_size 512 --batch_size 4
```

## File Structure

```
experiments/osla_340M/
├── README.md                       # This report
├── configs/
│   ├── baseline.json               # DeltaNet (chunk mode)
│   └── osla.json                   # OSLA (fused_recurrent mode)
├── scripts/
│   ├── train_baseline.sh           # 8xH100 training script
│   ├── train_osla.sh               # 8xH100 training script
│   └── eval.py                     # lm_eval wrapper
├── baseline_eval_results.json      # Full baseline eval metrics
└── osla_wikitext_eval_results.json # OSLA WikiText metrics
```

Trained model weights (safetensors) are stored locally under `exp/` (gitignored due to size).

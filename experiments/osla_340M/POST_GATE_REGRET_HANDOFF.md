# OS-GDN Post-Gate Regret — Hand-off Instructions

## What this is

Implementation of the §3.4 hypergradient from `OS_GDN_REPORT.md`:

```
grad_d = ⟨ẽ, e'⟩ / ‖e'‖²  −  ⟨d, k²⟩
```

This targets the structural ~1-nat gap by removing state-gate contamination
from OSGM's regret. When α≡1 it reduces to `1 − ⟨d, k²⟩` (the un-modified
dd_decay rule), so un-gated tests still pass.

## What's already validated locally

- 4× pytorch reference sanity checks (α=1 reduction, cold start, finiteness, ratio≈1)
- 4× fwd/bwd parity checks against autograd-through-loop reference (fp32 noise level)
- E2E forward+backward on a small OSGDN model: all 45/45 params get gradients,
  `osgm_a_proj.bias` stays at 6.9 end-to-end
- Single-layer 340M-shape speed: pytorch path is **faster** than the gate_aware
  chunk variant in the report (3.5 s vs 6.0 s fwd+bwd at T=2048)

## Why we need a second machine

The local 8-GPU box is fully booked by the running 1.3B job
(`osla-osgm-dd-decay-1.3B`, ~4d 15h remaining). All implementation work is
done; we just need GPUs to run the smoke training.

## Steps on the second machine

### Option A — fresh machine (no env yet)

```bash
# 1. Get the code (clone or update existing repo)
git clone git@github.com:Lhongpei/OSLA.git
cd OSLA
git fetch origin
git checkout post-gate-regret-smoke

# 2. One-time env setup (creates `osla` conda env with pinned versions:
#    torch 2.6.0, triton 3.2.0, transformers 4.51.3, fla -e, flame -e).
#    Idempotent — safe to re-run; only installs what's missing.
bash experiments/osla_340M/scripts/setup_env_post_gate_regret.sh

# 3. Run the smoke (env is auto-detected and activated by the script)
bash experiments/osla_340M/scripts/train_post_gate_regret_smoke.sh
```

### Option B — env already exists

```bash
cd /path/to/your/OSLA   # any path; script uses git toplevel
git fetch origin
git checkout post-gate-regret-smoke
bash experiments/osla_340M/scripts/train_post_gate_regret_smoke.sh
```

The smoke script auto-detects conda from common locations (`~/anaconda3`,
`~/miniconda3`, `/opt/conda`, `$(conda info --base)`); override with
`CONDA_BASE=...` or `CONDA_ENV=...` env vars if needed.

The script:
1. Runs three preflight pytest-style sanity scripts (~30 s)
2. Launches 200 steps of training on 8 GPUs
3. Tee's logs to `experiments/osla_340M/exp/post-gate-regret-smoke-200steps/run.log`
4. Prints the last 20 step log lines at the end

Total runtime: **~30-60 min** for 200 steps.

## What I need back

Just paste the lines marked `👉 PASTE THESE LINES BACK INTO THE CHAT`
(the grep'd `step: ... loss: ...` lines from the run log).

If anything blows up, paste the last 50 lines of the run log and I'll
diagnose.

## Config knobs you might want to change

In the script (top-level vars):
- `N_STEPS` (default 200) — bump to 2048 if you want a "step 2k" loss number
  comparable to the report table
- `N_GPUS` (default 8) — drop to 4 if some are still in use

In the json config (`os_gated_deltanet_post_gate_regret_340M.json`):
- `osgm_post_gate_regret_chunk_size` — checkpoint chunk size for the
  recurrence. 64 (default) matches the chunk kernel; larger = less memory
  overhead but more recompute; smaller = the opposite.

## Known limitation

The pytorch implementation does **not** support `cu_seqlens` (varlen / packed
sequences). The smoke script disables `--training.varlen` and uses fixed
seq_len=4096, batch=2 per GPU, grad_accum=8 to keep tokens-per-step at the
same 524,288 used in the report. If we proceed to a long run, this becomes
a triton port (~1 day extra work).

## Decision tree after smoke results

| Step-2k loss vs report baselines | Verdict | Next |
|---|---|---|
| ≤ 3.0 (matches/beats baseline 2.99) | §3.4 was right | Triton port + 20480-step run + downstream evals |
| 3.0–3.5 | partial fix | Try eta/d_min ablation, or check if extra contamination remains |
| 3.5–3.9 | weaker than gate_aware | Likely a bug in my impl — debug |
| > 4.0 | broken | Probably a real bug — diff against test outputs |

Reference numbers from the report at step 2k: gate_aware=3.91, dd_decay=4.00,
baseline GDN=2.99.

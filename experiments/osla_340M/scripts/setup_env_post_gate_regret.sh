#!/bin/bash
# One-time environment setup for the post-gate-regret smoke test.
# ----------------------------------------------------------------
# Creates a conda env named `osla` (or whatever $CONDA_ENV is set to)
# with the exact deps verified to work on the primary box. Idempotent:
# safe to re-run; will only install what's missing.
#
# Versions pinned from the working machine:
#   python      3.11
#   torch       2.6.0
#   triton      3.2.0  (comes bundled with torch)
#   transformers 4.51.3
#   fla         (installed -e from this repo)
#   flame       (installed -e from flame/ submodule)
#
# Usage:
#   bash experiments/osla_340M/scripts/setup_env_post_gate_regret.sh
#
# Override env name:
#   CONDA_ENV=foo bash experiments/osla_340M/scripts/setup_env_post_gate_regret.sh

set -e

CONDA_ENV=${CONDA_ENV:-osla}
PY_VERSION=${PY_VERSION:-3.11}
REPO=${REPO:-$(git rev-parse --show-toplevel)}

echo "================================================================"
echo "Setting up env '$CONDA_ENV' for OS-GDN post-gate-regret smoke"
echo "  Repo:    $REPO"
echo "  Python:  $PY_VERSION"
echo "================================================================"

# --- 0) locate conda ---
CONDA_BASE=""
for candidate in \
    "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda" \
    "/home/datagen/anaconda3" "/usr/local/anaconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done
if [ -z "$CONDA_BASE" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base 2>/dev/null)
    fi
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "ERROR: cannot find conda. Install miniconda first:"
    echo "  https://docs.conda.io/projects/miniconda/en/latest/"
    exit 1
fi
echo "  Using conda at $CONDA_BASE"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- 1) ensure submodule (flame) is checked out ---
if [ -f "$REPO/.gitmodules" ]; then
    echo ""
    echo "[1/4] Ensuring flame submodule is initialized..."
    (cd "$REPO" && git submodule update --init --recursive)
fi

# --- 2) create env if missing ---
echo ""
echo "[2/4] Checking conda env '$CONDA_ENV'..."
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "  env '$CONDA_ENV' already exists — using it."
else
    echo "  creating fresh env '$CONDA_ENV' with python=$PY_VERSION..."
    conda create -n "$CONDA_ENV" "python=$PY_VERSION" -y
fi
conda activate "$CONDA_ENV"
echo "  active python: $(which python)"

# --- 3) install deps ---
echo ""
echo "[3/4] Installing python deps..."
# Torch first — pip will pick the right CUDA wheel based on the host.
# We pin torch==2.6.0 because that's what the primary box runs and what
# the chunk kernels were tested against; bumping torch independently
# also bumps triton, which can introduce kernel ABI changes.
python -m pip install --upgrade pip
python -m pip install \
    "torch==2.6.0" \
    "transformers==4.51.3" \
    "datasets>=3.3.0" \
    "einops" \
    "ninja" \
    "wandb" \
    "tiktoken" \
    "tensorboard" \
    "torchdata"

# fla (this repo) and flame (submodule) — editable installs.
echo "  installing fla (this repo) -e ..."
python -m pip install -e "$REPO"
echo "  installing flame -e ..."
python -m pip install -e "$REPO/flame"

# --- 4) verify ---
echo ""
echo "[4/4] Verifying imports..."
python - <<'PY'
import torch, triton, transformers, fla
print(f"torch        {torch.__version__}")
print(f"triton       {triton.__version__}")
print(f"transformers {transformers.__version__}")
print(f"fla          {getattr(fla, '__version__', 'local-editable')}")
print(f"cuda         available={torch.cuda.is_available()}, "
      f"device_count={torch.cuda.device_count()}")
# Critical: import the post-gate-regret module
from fla.ops.os_gated_delta_rule.post_gate_regret import post_gate_regret_recurrence
print("post_gate_regret_recurrence: importable ✓")
# And the OSGDN model class
from fla.models.os_gated_deltanet import OSGDNConfig, OSGDNForCausalLM
print("OSGDNForCausalLM: importable ✓")
PY

echo ""
echo "================================================================"
echo "Setup complete."
echo "  Env:      $CONDA_ENV"
echo "  Activate: source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $CONDA_ENV"
echo ""
echo "Next: run the smoke test:"
echo "  bash $REPO/experiments/osla_340M/scripts/train_post_gate_regret_smoke.sh"
echo "================================================================"

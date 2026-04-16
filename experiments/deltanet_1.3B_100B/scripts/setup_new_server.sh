#!/bin/bash
# Setup script for running OSLA-OSGM dd-decay 1.3B / 100B training on a fresh server.
#
# Assumptions about the fresh server:
#   - 8x H100 (or equivalent) GPUs, CUDA driver already installed
#   - Anaconda/Miniconda installed at $HOME/anaconda3 (override with CONDA_HOME)
#   - Network access to GitHub and to https://hf-mirror.com
#   - You will run training under /data0/OSLA so the absolute paths inside the
#     train script just work. If your fresh server cannot use /data0, edit
#     OSLA_ROOT below AND update the absolute paths in
#     train_osla_osgm_dd_decay_1.3B.sh accordingly.
#
# What this script does:
#   1. Clones the OSLA repo (branch osla-osgm-dd-decay-1.3B) into $OSLA_ROOT
#   2. Clones the matching flame commit into $OSLA_ROOT/flame
#   3. Creates the `osla` conda env with python 3.11 + pytorch + deps
#   4. Pip-installs fla (this repo) and flame in editable mode
#   5. Pre-downloads the tokenizer and the fineweb-edu sample-100BT dataset
#      via the hf-mirror endpoint (you should run this on the data node before
#      kicking off training to avoid stalling at step 0)
#   6. Reminds you to `wandb login` before training
#
# Usage:
#   bash setup_new_server.sh             # full setup
#   bash setup_new_server.sh --skip-data # skip dataset prefetch (faster)

set -e

OSLA_ROOT=${OSLA_ROOT:-/data0/OSLA}
CONDA_HOME=${CONDA_HOME:-$HOME/anaconda3}
OSLA_REPO=${OSLA_REPO:-git@github.com:Lhongpei/OSLA.git}
OSLA_BRANCH=${OSLA_BRANCH:-main}
FLAME_REPO=${FLAME_REPO:-https://github.com/fla-org/flame.git}
FLAME_COMMIT=${FLAME_COMMIT:-e11e7be75b9e45e84dbecbe8f0efa27d6af7d101}
ENV_NAME=${ENV_NAME:-osla}

SKIP_DATA=0
for arg in "$@"; do
  case "$arg" in
    --skip-data) SKIP_DATA=1 ;;
  esac
done

echo "==> OSLA_ROOT=$OSLA_ROOT  branch=$OSLA_BRANCH  env=$ENV_NAME"

# ---------------------------------------------------------------------------
# 1. Clone OSLA
# ---------------------------------------------------------------------------
if [ ! -d "$OSLA_ROOT/.git" ]; then
  echo "==> Cloning OSLA into $OSLA_ROOT"
  mkdir -p "$(dirname "$OSLA_ROOT")"
  git clone "$OSLA_REPO" "$OSLA_ROOT"
  cd "$OSLA_ROOT"
  git checkout "$OSLA_BRANCH"
else
  echo "==> $OSLA_ROOT already exists, pulling latest on $OSLA_BRANCH"
  cd "$OSLA_ROOT"
  git fetch origin "$OSLA_BRANCH"
  git checkout "$OSLA_BRANCH"
  git pull origin "$OSLA_BRANCH"
fi

# ---------------------------------------------------------------------------
# 2. Clone flame at the known-good commit
# ---------------------------------------------------------------------------
if [ ! -d "$OSLA_ROOT/flame/.git" ]; then
  echo "==> Cloning flame into $OSLA_ROOT/flame at $FLAME_COMMIT"
  rm -rf "$OSLA_ROOT/flame"
  git clone "$FLAME_REPO" "$OSLA_ROOT/flame"
  cd "$OSLA_ROOT/flame"
  git checkout "$FLAME_COMMIT"
else
  echo "==> flame already present, leaving as-is"
fi

# ---------------------------------------------------------------------------
# 3. Create conda environment
# ---------------------------------------------------------------------------
source "$CONDA_HOME/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Creating conda env $ENV_NAME (python 3.11)"
  conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

echo "==> Installing PyTorch 2.5 (cu124) + base deps"
pip install --upgrade pip
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install \
  "transformers>=4.45.0" \
  "datasets>=3.3.0" \
  "triton>=3.0" \
  torchdata einops ninja wandb tiktoken tensorboard \
  huggingface_hub hf_transfer

# ---------------------------------------------------------------------------
# 4. Editable installs for fla (this repo) and flame
# ---------------------------------------------------------------------------
echo "==> pip install -e $OSLA_ROOT (fla)"
cd "$OSLA_ROOT"
pip install -e . --no-build-isolation

echo "==> pip install -e $OSLA_ROOT/flame"
cd "$OSLA_ROOT/flame"
pip install -e . --no-build-isolation

# ---------------------------------------------------------------------------
# 5. Pre-download tokenizer + dataset via hf-mirror
# ---------------------------------------------------------------------------
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "==> Pre-downloading tokenizer fla-hub/delta_net-1.3B-100B"
python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("fla-hub/delta_net-1.3B-100B", trust_remote_code=True)
print("tokenizer cached OK")
PY

if [ "$SKIP_DATA" -eq 0 ]; then
  echo "==> Pre-downloading HuggingFaceFW/fineweb-edu sample-100BT (this is large; ~hundreds of GB)"
  python - <<'PY'
from datasets import load_dataset
# Streaming would skip the cache; we want it on disk so step 0 is fast.
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-100BT",
    split="train",
)
print("dataset cached:", ds)
PY
else
  echo "==> --skip-data set; skipping dataset prefetch (training will stream-download on first use)"
fi

# ---------------------------------------------------------------------------
# 6. Final reminders
# ---------------------------------------------------------------------------
cat <<'EOF'

==================================================================
Setup complete. Before launching training:

  conda activate osla
  wandb login            # paste your W&B API key

Then kick off training:

  bash $OSLA_ROOT/experiments/deltanet_1.3B_100B/scripts/train_osla_osgm_dd_decay_1.3B.sh

Logs / checkpoints will land in:
  $OSLA_ROOT/experiments/deltanet_1.3B_100B/exp/osla-osgm-dd-decay-1.3B
==================================================================
EOF

#!/usr/bin/env bash
# Pansori fine-tune -- the server run script. Run it stage by stage inside tmux.
#   bash pansori/server/run.sh setup | binarize | train-ac | train-var
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ENV_NAME="${ENV_NAME:-dsp}"          # a fresh env for openvpi (Python >= 3.10)
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LOGDIR="$REPO/logs"; mkdir -p "$LOGDIR"
# The root .gitignore is upstream and is not touched. Let the directory ignore itself.
[ -e "$LOGDIR/.gitignore" ] || printf '*\n!.gitignore\n' > "$LOGDIR/.gitignore"

# Release assets. Both URLs verified as of 2026-08-18.
RMVPE_URL="${RMVPE_URL:-https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip}"
HNSEP_URL="${HNSEP_URL:-https://github.com/yxlllc/vocal-remover/releases/download/hnsep_240512/hnsep_240512.zip}"
VOCODER_URL="${VOCODER_URL:-https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip}"

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

activate() {
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  export PYTHONPATH="$REPO"
  export CUDA_VISIBLE_DEVICES="$GPU"
  # ~/.local/lib/pythonX.Y/site-packages cuts in front of the conda env. An old
  # tensorflow there kills the tensorboard import at protobuf, and a numpy there makes
  # pip misjudge requirements as "already satisfied" and never install them into the env.
  export PYTHONNOUSERSITE=1
}

cmd_setup() {
  log "conda env '$ENV_NAME' (Python 3.10)"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.10
  conda activate "$ENV_NAME"
  export PYTHONNOUSERSITE=1          # see the comment in activate() above
  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  echo 'export PYTHONNOUSERSITE=1' > "$CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh"

  log "PyTorch"
  python -c "import torch" 2>/dev/null || \
    pip install torch --index-url https://download.pytorch.org/whl/cu124
  # requirements.txt has no torchaudio (an upstream omission). RMVPE uses Resample.
  python -c "import torchaudio" 2>/dev/null || \
    pip install torchaudio --index-url https://download.pytorch.org/whl/cu124
  log "dependencies"
  pip install -r requirements.txt
  # lightning 2.3 imports pkg_resources, which setuptools 81+ removed. Hence the pin.
  pip install "setuptools<81"

  mkdir -p checkpoints/rmvpe
  if [ -n "$RMVPE_URL" ]; then
    log "RMVPE"
    curl -L "$RMVPE_URL" -o /tmp/rmvpe.zip && unzip -o /tmp/rmvpe.zip -d checkpoints/rmvpe
  else
    echo "⚠ RMVPE_URL unset — put checkpoints/rmvpe/model.pt in place yourself (binarize needs it)"
  fi
  # acoustic binarize builds a DecomposedWaveform even with breathiness and the rest off,
  # and the default hnsep is 'vr', so without this checkpoint it stops there.
  if [ -n "$HNSEP_URL" ]; then
    log "HN-SEP (vr)"
    curl -L "$HNSEP_URL" -o /tmp/hnsep.zip && unzip -o /tmp/hnsep.zip -d checkpoints/
  else
    echo "⚠ HNSEP_URL unset — put checkpoints/vr/{model.pt,config.yaml} in place yourself"
  fi
  if [ -n "$VOCODER_URL" ]; then
    log "NSF-HiFiGAN"
    curl -L "$VOCODER_URL" -o /tmp/voc.zip && unzip -o /tmp/voc.zip -d checkpoints/
  else
    echo "⚠ VOCODER_URL unset — put it at the vocoder_ckpt path in configs/acoustic.yaml"
  fi

  log "verifying"
  python -c "import torch, lightning; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
}

cmd_binarize() {
  activate
  log "re-validating labels"
  ( cd pansori/preprocess && python 07_validate.py --set full ) || {
      echo "07_validate FAIL — aborting binarize"; exit 1; }

  for kind in acoustic variance; do
    log "binarize $kind"
    python scripts/binarize.py --config "configs/pansori_${kind}.yaml" \
      2>&1 | tee "$LOGDIR/binarize_${kind}.log"
  done
  log "next: measure spec_min/spec_max with compute_spec_stats.py and put them in the config"
}

cmd_train() {
  local kind="$1" exp="$2"
  activate
  nvidia-smi --query-gpu=index,name,memory.used --format=csv
  log "train $kind → checkpoints/$exp"
  python scripts/train.py --config "configs/pansori_${kind}.yaml" \
    --exp_name "$exp" --reset 2>&1 | tee "$LOGDIR/train_${exp}.log"
}

case "${1:-}" in
  setup)     cmd_setup ;;
  binarize)  cmd_binarize ;;
  train-ac)  cmd_train acoustic "${2:-pansori_ac_v1}" ;;
  train-var) cmd_train variance "${2:-pansori_var_v1}" ;;
  *) echo "usage: $0 {setup|binarize|train-ac|train-var} [exp_name]"; exit 1 ;;
esac

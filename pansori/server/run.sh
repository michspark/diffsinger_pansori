#!/usr/bin/env bash
# 판소리 fine-tune — 서버 실행 스크립트. tmux 안에서 단계별로 돌린다.
#   bash pansori/server/run.sh setup | binarize | train-ac | train-var
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ENV_NAME="${ENV_NAME:-dsp}"          # openvpi 용 새 env (Python >= 3.10)
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LOGDIR="$REPO/logs"; mkdir -p "$LOGDIR"
# 루트 .gitignore 는 upstream 이라 건드리지 않는다. 디렉토리가 자기를 무시하게 둔다.
[ -e "$LOGDIR/.gitignore" ] || printf '*\n!.gitignore\n' > "$LOGDIR/.gitignore"

# 릴리스 자산. 2026-08-18 기준 두 URL 모두 확인됨.
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
  # ~/.local/lib/pythonX.Y/site-packages 가 conda env 앞에 끼어든다. 거기 낡은
  # tensorflow 가 있으면 tensorboard import 가 protobuf 에서 죽고, 거기 numpy 가
  # 있으면 pip 이 "이미 설치됨"으로 오판해 env 안에 안 넣는다.
  export PYTHONNOUSERSITE=1
}

cmd_setup() {
  log "conda env '$ENV_NAME' (Python 3.10)"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.10
  conda activate "$ENV_NAME"
  export PYTHONNOUSERSITE=1          # 위 activate() 의 주석 참고
  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  echo 'export PYTHONNOUSERSITE=1' > "$CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh"

  log "PyTorch"
  python -c "import torch" 2>/dev/null || \
    pip install torch --index-url https://download.pytorch.org/whl/cu124
  # requirements.txt 에 torchaudio 가 없다 (upstream 누락). RMVPE 가 Resample 을 쓴다.
  python -c "import torchaudio" 2>/dev/null || \
    pip install torchaudio --index-url https://download.pytorch.org/whl/cu124
  log "의존성"
  pip install -r requirements.txt
  # lightning 2.3 은 pkg_resources 를 import 한다. setuptools 81+ 에서 빠졌으므로 고정.
  pip install "setuptools<81"

  mkdir -p checkpoints/rmvpe
  if [ -n "$RMVPE_URL" ]; then
    log "RMVPE"
    curl -L "$RMVPE_URL" -o /tmp/rmvpe.zip && unzip -o /tmp/rmvpe.zip -d checkpoints/rmvpe
  else
    echo "⚠ RMVPE_URL 미설정 — checkpoints/rmvpe/model.pt 를 직접 넣을 것 (binarize 필수)"
  fi
  # acoustic binarize 는 breathiness 등을 안 써도 DecomposedWaveform 을 만들고,
  # 기본 hnsep 이 'vr' 이라 이 체크포인트가 없으면 거기서 멈춘다.
  if [ -n "$HNSEP_URL" ]; then
    log "HN-SEP (vr)"
    curl -L "$HNSEP_URL" -o /tmp/hnsep.zip && unzip -o /tmp/hnsep.zip -d checkpoints/
  else
    echo "⚠ HNSEP_URL 미설정 — checkpoints/vr/{model.pt,config.yaml} 을 직접 넣을 것"
  fi
  if [ -n "$VOCODER_URL" ]; then
    log "NSF-HiFiGAN"
    curl -L "$VOCODER_URL" -o /tmp/voc.zip && unzip -o /tmp/voc.zip -d checkpoints/
  else
    echo "⚠ VOCODER_URL 미설정 — configs/acoustic.yaml 의 vocoder_ckpt 경로에 넣을 것"
  fi

  log "확인"
  python -c "import torch, lightning; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
}

cmd_binarize() {
  activate
  log "라벨 재검증"
  ( cd pansori/preprocess && python 07_validate.py --set full ) || {
      echo "07_validate FAIL — binarize 중단"; exit 1; }

  for kind in acoustic variance; do
    log "binarize $kind"
    python scripts/binarize.py --config "configs/pansori_${kind}.yaml" \
      2>&1 | tee "$LOGDIR/binarize_${kind}.log"
  done
  log "다음: compute_spec_stats.py 로 spec_min/spec_max 실측 후 config 반영"
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

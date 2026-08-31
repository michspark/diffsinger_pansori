#!/usr/bin/env bash
# 판소리 추론 예시 일괄 생성 — 파이프라인의 각 단계를 하나씩 벗겨 본다.
#
#   bash pansori/examples/run_examples.sh [AC_CKPT] [DUR_CKPT] [PITCH_CKPT]
#
# 어느 모델을 어느 코퍼스로 돌릴지는 환경변수로 정한다. 기본값은 whisper 라벨로
# 학습한 v1 이고, 가사 A/B 의 다른 쪽은 이렇게 부른다:
#
#   AC_EXP=pansori_ac_snap VAR_EXP=pansori_var_snap DATA=data/pansori_snap \
#   DS_DIR=pansori/examples/ds_snap OUT_DIR=pansori/examples/out_snap \
#     bash pansori/examples/run_examples.sh
#
#   AC_EXP / VAR_EXP  체크포인트 디렉토리 이름
#   DATA              라벨 코퍼스 (ph_dur·노트가 여기서 온다). 오디오는 어느 쪽을
#                     가리켜도 같은 파일이다 — 08_export.py --wavs_from 이 링크한다
#   DS_DIR / OUT_DIR  중간 .ds 와 결과 wav. 두 변종이 서로를 덮어쓰지 않게 나눈다
#
# 인수를 주지 않으면 검증 손실이 최저인 체크포인트를 고른다 (마지막이 아니다).
# variance 는 헤드마다 최적 시점이 다르므로 duration 과 pitch 를 따로 고르고
# 두 번에 나눠 돌린다 — 자세한 근거는 아래 주석 참고.
#
# 만드는 것 (홀드아웃 발화마다 4단계 사다리):
#   1_orig     원본 녹음                       — 상한
#   2_resynth  실제 ph_dur + 실제 f0 → acoustic — acoustic 단독 성능
#   3_pitch    실제 ph_dur, f0 는 variance      — 피치 예측만 추가
#   4_full     노트+가사만, dur/f0 둘 다 variance — 전체 추론
# 그리고 악보에서:
#   5_score    녹음이 없는 새 가사+멜로디 → 화자별 합성
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

EX="pansori/examples"
AC_EXP="${AC_EXP:-pansori_ac_v1}"
VAR_EXP="${VAR_EXP:-pansori_var_v1}"
DATA="${DATA:-data/pansori}"
DS="${DS_DIR:-$EX/ds}"
OUT="${OUT_DIR:-$EX/out}"
mkdir -p "$DS" "$OUT"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-dsp}"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1

# num_ckpt_keep 은 오래된 체크포인트를 지운다. checkpoints/best/ 에 따로 복사해 둔 것이
# 실험 디렉토리에서 사라졌으면 되돌려 놓는다 — 그러지 않으면 아래 선택기가 "진짜 최저는
# 삭제됐다"고 알리고 차선을 고르게 된다.
for saved in checkpoints/best/*_step*.ckpt; do
  [ -e "$saved" ] || continue
  base=$(basename "$saved"); exp=${base%%_step*}; step=${base##*_step}; step=${step%.ckpt}
  dest="checkpoints/$exp/model_ckpt_steps_${step}.ckpt"
  if [ -d "checkpoints/$exp" ] && [ ! -e "$dest" ]; then
    echo "복원: $dest ← $saved"
    cp "$saved" "$dest"
  fi
done

# 가장 최근이 아니라 검증 손실이 최저인 체크포인트를 쓴다 (pansori/docs/GOTCHAS.md).
#
# variance 의 두 헤드는 같은 시점에 최적이 아니다. 실측: dur_loss 는 step 16k 에서
# 최저를 찍고 이후 계속 오르는데 pitch_loss 는 32k 까지도 내려간다. total_loss 는
# dur 이 지배해서 20k 를 고르는데, 그 체크포인트로 피치를 뽑으면 32k 보다 f0 오차가
# 25% 크다. 그래서 헤드마다 따로 고르고 variance 를 두 번 돌린다 —
# 1차는 --predict dur, 2차는 그 산출물에 --predict pitch (ph_dur 는 manual 로 읽힌다).
AC_CKPT="${1:-$(python pansori/examples/best_ckpt.py "$AC_EXP")}"
DUR_CKPT="${2:-$(python pansori/examples/best_ckpt.py "$VAR_EXP" --metric validation/dur_loss)}"
PITCH_CKPT="${3:-$(python pansori/examples/best_ckpt.py "$VAR_EXP" --metric validation/pitch_loss)}"
: "${AC_CKPT:?acoustic 체크포인트가 없다}"
: "${DUR_CKPT:?variance 체크포인트가 없다}"
: "${PITCH_CKPT:?variance 체크포인트가 없다}"

# reflow 는 확률적 샘플러다. 시드를 고정하지 않으면 같은 입력이 실행마다 다른
# 결과를 내고 표의 숫자가 흔들린다 (실측: f0 RMSE 가 회당 40~90 센트 움직인다).
SEED="${SEED:-20260818}"

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
log "acoustic step $AC_CKPT · variance dur $DUR_CKPT / pitch $PITCH_CKPT"
echo "$AC_CKPT $DUR_CKPT $PITCH_CKPT" > "$OUT/CKPT_STEPS"

quiet() { grep -viE "pkg_resources|^\[0;33m|it/s\]|^ *$" || true; }

# 학습에 쓰이지 않은 홀드아웃 곡에서 고른 발화. 남창/여창을 섞었다 —
# 남성 통성은 f0_min 55 Hz 가 없으면 통째로 무성 처리되는 구간이다.
ITEMS=(
  "d207244a_0085"   # 김소희  여창
  "8b4d0824_0078"   # 조상현  남창
  "e719dced_0197"   # 박동진  남창
)

for item in "${ITEMS[@]}"; do
  log "$item"

  # --- 1. 원본 --------------------------------------------------------------
  # -L: 가사 A/B 의 두 번째 코퍼스는 wavs/ 가 첫 번째를 가리키는 심볼릭 링크다
  # (08_export.py --wavs_from). find 는 기본적으로 링크된 디렉토리로 내려가지 않는다.
  wav=$(find -L "$DATA" -name "$item.wav" | head -1)
  cp "$wav" "$OUT/${item}_1_orig.wav"

  # --- 2. 재합성: 실제 dur + 실제 f0 ---------------------------------------
  python pansori/make_ds.py from-label "$item" --data "$DATA" --with-dur --with-f0 \
    --out "$DS/${item}_gt.ds" 2>&1 | quiet
  python scripts/infer.py acoustic "$DS/${item}_gt.ds" \
    --exp "$AC_EXP" --ckpt "$AC_CKPT" --seed "$SEED" --out "$OUT" \
    --title "${item}_2_resynth" 2>&1 | quiet

  # --- 3. 실제 dur, f0 만 예측 ----------------------------------------------
  python pansori/make_ds.py from-label "$item" --data "$DATA" --with-dur \
    --out "$DS/${item}_dur.ds" 2>&1 | quiet
  python scripts/infer.py variance "$DS/${item}_dur.ds" \
    --exp "$VAR_EXP" --ckpt "$PITCH_CKPT" --predict pitch \
    --seed "$SEED" --out "$DS" --title "${item}_dur_var" 2>&1 | quiet
  python scripts/infer.py acoustic "$DS/${item}_dur_var.ds" \
    --exp "$AC_EXP" --ckpt "$AC_CKPT" --seed "$SEED" --out "$OUT" \
    --title "${item}_3_pitch" 2>&1 | quiet

  # --- 4. 전체 추론: 노트와 가사만 ------------------------------------------
  python pansori/make_ds.py from-label "$item" --data "$DATA" \
    --out "$DS/${item}_notes.ds" 2>&1 | quiet
  python scripts/infer.py variance "$DS/${item}_notes.ds" \
    --exp "$VAR_EXP" --ckpt "$DUR_CKPT" --predict dur \
    --seed "$SEED" --out "$DS" --title "${item}_notes_dur" 2>&1 | quiet
  python scripts/infer.py variance "$DS/${item}_notes_dur.ds" \
    --exp "$VAR_EXP" --ckpt "$PITCH_CKPT" --predict pitch \
    --seed "$SEED" --out "$DS" --title "${item}_notes_var" 2>&1 | quiet
  python scripts/infer.py acoustic "$DS/${item}_notes_var.ds" \
    --exp "$AC_EXP" --ckpt "$AC_CKPT" --seed "$SEED" --out "$OUT" \
    --title "${item}_4_full" 2>&1 | quiet
done

# --- 5. 녹음이 없는 새 악보 -------------------------------------------------
# 가사→음소를 사전이 맡는 경로. 구 파이프라인에는 이게 없었다.
for score in "$EX"/scores/*.txt; do
  name=$(basename "$score" .txt)
  for spk in 김소희 조상현 박동진; do
    log "$name — $spk"
    python pansori/make_ds.py from-score "$score" --spk "$spk" \
      --out "$DS/${name}_${spk}.ds" 2>&1 | quiet
    python scripts/infer.py variance "$DS/${name}_${spk}.ds" \
      --exp "$VAR_EXP" --ckpt "$DUR_CKPT" --predict dur \
      --seed "$SEED" --out "$DS" --title "${name}_${spk}_dur" 2>&1 | quiet
    python scripts/infer.py variance "$DS/${name}_${spk}_dur.ds" \
      --exp "$VAR_EXP" --ckpt "$PITCH_CKPT" --predict pitch \
      --seed "$SEED" --out "$DS" --title "${name}_${spk}_var" 2>&1 | quiet
    python scripts/infer.py acoustic "$DS/${name}_${spk}_var.ds" \
      --exp "$AC_EXP" --ckpt "$AC_CKPT" --seed "$SEED" --out "$OUT" \
      --title "5_score_${name}_${spk}" 2>&1 | quiet
  done
done

# 사다리 4단의 mel 을 한 장으로 — 점수가 "왜" 나쁜지는 그림에서만 보인다.
log "스펙트로그램"
python pansori/examples/spectrograms.py "${ITEMS[1]}" --wav-dir "$OUT" 2>&1 | quiet

log "완료 — $OUT"
ls -la "$OUT"/*.wav

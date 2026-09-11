#!/usr/bin/env bash
# Generate the pansori inference examples in one go -- peeling the pipeline back a stage at a time.
#
#   bash pansori/examples/run_examples.sh [AC_CKPT] [DUR_CKPT] [PITCH_CKPT]
#
# Which model runs against which corpus is set by environment variables. The default is
# v1, trained on the whisper labels; the other arm of the lyrics A/B is invoked like this:
#
#   AC_EXP=pansori_ac_snap VAR_EXP=pansori_var_snap DATA=data/pansori_snap \
#   DS_DIR=pansori/examples/ds_snap OUT_DIR=pansori/examples/out_snap \
#     bash pansori/examples/run_examples.sh
#
#   AC_EXP / VAR_EXP  checkpoint directory names
#   DATA              the label corpus (ph_dur and notes come from here). The audio is
#                     the same file whichever you point at -- 08_export.py --wavs_from links it
#   DS_DIR / OUT_DIR  the intermediate .ds and the output wavs. Kept apart so the two
#                     variants do not overwrite each other
#
# With no arguments it picks the lowest-validation-loss checkpoint (not the last one).
# variance's heads peak at different steps, so duration and pitch are selected separately
# and run in two passes -- the reasoning is in the comments below.
#
# What it produces (a 4-rung ladder per hold-out utterance):
#   1_orig     the original recording              -- the ceiling
#   2_resynth  real ph_dur + real f0 -> acoustic   -- acoustic alone
#   3_pitch    real ph_dur, f0 from variance       -- pitch prediction added
#   4_full     notes + lyrics only, dur and f0 both from variance -- full inference
# And from the scores:
#   5_score    new lyrics + melody with no recording -> synthesised per speaker
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

# num_ckpt_keep deletes older checkpoints. If one saved separately in checkpoints/best/
# has disappeared from the experiment directory, restore it -- otherwise the selector below
# reports "the true minimum was deleted" and settles for the runner-up.
for saved in checkpoints/best/*_step*.ckpt; do
  [ -e "$saved" ] || continue
  base=$(basename "$saved"); exp=${base%%_step*}; step=${base##*_step}; step=${step%.ckpt}
  dest="checkpoints/$exp/model_ckpt_steps_${step}.ckpt"
  if [ -d "checkpoints/$exp" ] && [ ! -e "$dest" ]; then
    echo "restored: $dest <- $saved"
    cp "$saved" "$dest"
  fi
done

# Use the lowest-validation-loss checkpoint, not the most recent (pansori/docs/GOTCHAS.md).
#
# variance's two heads do not peak at the same step. Measured: dur_loss bottoms out at
# step 16k and rises steadily after, while pitch_loss keeps falling to 32k. total_loss is
# dominated by dur and picks 20k -- and pulling pitch from that checkpoint gives 25% more
# f0 error than 32k. So each head is selected separately and variance runs twice: first
# --predict dur, then --predict pitch over that output (ph_dur is read as manual).
AC_CKPT="${1:-$(python pansori/examples/best_ckpt.py "$AC_EXP")}"
DUR_CKPT="${2:-$(python pansori/examples/best_ckpt.py "$VAR_EXP" --metric validation/dur_loss)}"
PITCH_CKPT="${3:-$(python pansori/examples/best_ckpt.py "$VAR_EXP" --metric validation/pitch_loss)}"
: "${AC_CKPT:?no acoustic checkpoint}"
: "${DUR_CKPT:?no variance checkpoint}"
: "${PITCH_CKPT:?no variance checkpoint}"

# reflow is a stochastic sampler. Without a fixed seed the same input gives a different
# result every run and the numbers in the table wobble (measured: f0 RMSE moves 40-90
# cents between runs).
SEED="${SEED:-20260818}"

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
log "acoustic step $AC_CKPT · variance dur $DUR_CKPT / pitch $PITCH_CKPT"
echo "$AC_CKPT $DUR_CKPT $PITCH_CKPT" > "$OUT/CKPT_STEPS"

quiet() { grep -viE "pkg_resources|^\[0;33m|it/s\]|^ *$" || true; }

# Utterances picked from hold-out songs that were not used in training. Male and female
# voices are mixed -- male tongseong has spans that go entirely unvoiced without
# f0_min 55 Hz.
ITEMS=(
  "d207244a_0085"   # Kim So-hee        female voice
  "8b4d0824_0078"   # Cho Sang-hyeon    male voice
  "e719dced_0197"   # Park Dong-jin     male voice
)

for item in "${ITEMS[@]}"; do
  log "$item"

  # --- 1. the original ------------------------------------------------------
  # -L: in the lyrics A/B's second corpus, wavs/ is a symlink to the first's
  # (08_export.py --wavs_from). find does not descend into linked directories by default.
  wav=$(find -L "$DATA" -name "$item.wav" | head -1)
  cp "$wav" "$OUT/${item}_1_orig.wav"

  # --- 2. resynthesis: real dur + real f0 -----------------------------------
  python pansori/make_ds.py from-label "$item" --data "$DATA" --with-dur --with-f0 \
    --out "$DS/${item}_gt.ds" 2>&1 | quiet
  python scripts/infer.py acoustic "$DS/${item}_gt.ds" \
    --exp "$AC_EXP" --ckpt "$AC_CKPT" --seed "$SEED" --out "$OUT" \
    --title "${item}_2_resynth" 2>&1 | quiet

  # --- 3. real dur, only f0 predicted ---------------------------------------
  python pansori/make_ds.py from-label "$item" --data "$DATA" --with-dur \
    --out "$DS/${item}_dur.ds" 2>&1 | quiet
  python scripts/infer.py variance "$DS/${item}_dur.ds" \
    --exp "$VAR_EXP" --ckpt "$PITCH_CKPT" --predict pitch \
    --seed "$SEED" --out "$DS" --title "${item}_dur_var" 2>&1 | quiet
  python scripts/infer.py acoustic "$DS/${item}_dur_var.ds" \
    --exp "$AC_EXP" --ckpt "$AC_CKPT" --seed "$SEED" --out "$OUT" \
    --title "${item}_3_pitch" 2>&1 | quiet

  # --- 4. full inference: notes and lyrics only -----------------------------
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

# --- 5. a new score with no recording --------------------------------------
# The path where the dictionary handles lyrics->phonemes. The old pipeline had none.
for score in "$EX"/scores/*.txt; do
  name=$(basename "$score" .txt)
  for spk in 김소희 조상현 박동진; do   # speaker directory names
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

# The 4 rungs' mels on one sheet -- *why* a score is bad is only visible in the picture.
log "spectrograms"
python pansori/examples/spectrograms.py "${ITEMS[1]}" --wav-dir "$OUT" 2>&1 | quiet

log "done — $OUT"
ls -la "$OUT"/*.wav

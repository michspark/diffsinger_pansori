#!/usr/bin/env python3
"""Frozen constants for the old-vs-new comparison.

Everything here is a constant with no CLI override, on purpose. The most common
way a comparison like this quietly becomes meaningless is that the analysis grid
drifts between a pilot run and the final run, so the grid is not a parameter.

The 10 ms grid is not an arbitrary round number. RMVPE runs natively at 100 Hz
(modules/pe/rmvpe/inference.py:74 resamples from 0.01 s to the requested hop), so
asking for hop = round(0.01 * sr) makes resample_align_curve the identity: 441 at
44.1 kHz and 240 at 24 kHz, both exact. Any other grid interpolates one side and
not the other, which biases every motion metric in the direction of whichever
system got smoothed.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OLD_REPO = Path.home() / 'Desktop' / 'DiffSinger_moonInRiver'

COMPARE = REPO / 'pansori' / 'compare'
OUT = COMPARE / 'out'
CACHE = COMPARE / 'cache'

# ---------------------------------------------------------------- analysis grid

TIMESTEP = 0.01          # seconds; RMVPE's native rate
ANALYSIS_HZ = 100.0

# Common band-limited spectral basis. 24 kHz because that is all the old vocoder
# ever produced -- upsampling it instead would manufacture a band it never made
# and flatter it. fmax 11 kHz keeps clear of both Nyquist and the old mel's
# 12 kHz filterbank edge.
SPEC = dict(sr=24000, n_fft=1024, hop=240, n_mels=80, fmin=40, fmax=11000)
MFCC_N = 13              # c0 dropped -- it is gain, confounded by normalisation

# ---------------------------------------------------------------- systems

# Per-head best checkpoints. run_examples.sh restores these from checkpoints/best/
# because num_ckpt_keep: 8 already deleted the acoustic optimum once.
CKPT_ACOUSTIC = 28000
CKPT_DUR = 16000
CKPT_PITCH = 32000

SEEDS = (1, 2, 3, 4, 5)  # never -1; the sampler is stochastic and this is the floor

OLD_LADDER = {
    'aux': OLD_REPO / 'infer_out' / 'ladder_pansori_aux',
    'ds3': OLD_REPO / 'infer_out' / 'ladder_pansori_ds3',
}
# pansori_aux is a deterministic FastSpeech2 regressor, so its single render IS
# the model's answer rather than a draw -- that is why the seed treatment is
# asymmetric and still legitimate. pansori_ds3 is a shallow-diffusion sampler,
# so it is spectral-only and carries a single-draw label everywhere it appears.
OLD_PRIMARY = 'aux'

# Rung mapping, settled by reading DiffSinger_moonInRiver/pansori/eval/resynth_ladder.py
#   :189  wavs['2_gt_crepe'] = vocoder.spec2wav(gt_np, f0=crepe)   <- GROUND TRUTH mel
#   :196  wavs['5_pred_own'] = vocoder.spec2wav(pred_np, f0=own)
#   --pred_dur defaults False, so 5_pred_own used ground-truth durations.
RUNG_PAIRS = [
    # (label, old rung, new rung, what it isolates)
    ('보코더 천장', '2_gt_crepe', '0_vocoder', 'GT mel + GT f0 -- vocoder and extractor only'),
    ('모델 출력', '5_pred_own', '3_pitch', 'GT dur + own f0 + own mel -- the A/B pair'),
]
NEW_RUNGS = ('0_vocoder', '2_resynth', '3_pitch', '4_full')

# 4_full has no old counterpart (the old ladder never ran --pred_dur) and its
# frame alignment is broken by predicted durations, so it is excluded from every
# frame-aligned metric and from correlation with ground truth.
FRAME_ALIGNED = ('0_vocoder', '2_resynth', '3_pitch')

# ---------------------------------------------------------------- 시김새 params

NOTE_MIN_VOICED = 5      # notes with fewer voiced frames are skipped
NOTE_TRIM = 0.2          # core mask trims max(1, round(0.2*L)) frames each end
TRANSITION_MS = 120      # +-120 ms around a note boundary
SLOPE_FLOOR = 200.0      # cents/s; below this a boundary is too flat to have a direction
KKEOKGI_CENTS = 80.0     # excursion against the note step that counts as a 꺾기
KKEOKGI_RETURN_MS = 100
WOBBLE_MEDIAN_MS = 150   # detrending baseline for ornamental wobble

# ---------------------------------------------------------------- statistics

# Pre-registered primary endpoints. Fixed before any number is looked at;
# everything else in the report is labelled exploratory. Without this, 30-odd
# metrics across two rungs is a p-hacking machine.
PRIMARY_ENDPOINTS = (
    'within_std_ratio',
    'within_corr',
    'rpa50',
    'transition_dir_agree',
)
BOOTSTRAP_N = 10000
SEED_IQR_FACTOR = 2.0    # effect must exceed 2x the median per-item seed IQR

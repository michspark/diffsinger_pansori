# Phase 2 result — pansori *yoseong* is not a sine wave

`pansori/eval/vibrato_stats.py`, 300 training utterances → 746 notes, against CREPE.
The R² gate of the Transition-Sustain plan (Bonada & Blaauw, arXiv 1803.04030).

**Verdict: there is no evidence supporting the sine model. Phases 4–6 (sustain) are
recommended for cancellation.** Phase 3 (transition + note-position features) is unaffected
by this conclusion, and is in fact strengthened by it.

> **Note on reproducing this.** The measurement was made in the MoonInTheRiver repository,
> before the port. `pansori/eval/vibrato_stats.py` was **not** carried across to this branch
> and the command in §7 refers to the old repository's layout. The numbers stand as a
> record; re-running them would mean re-porting the script first.
>
> *yoseong* (요성) is the wide vibrato of pansori — see *sigimsae* in the README glossary.

---

## 1. The surface numbers look like a pass

| | Median | Mean | 25% | 75% |
|---|---|---|---|---|
| Sine only (as the plan defines it) | **0.752** | 0.694 | 0.543 | 0.890 |
| Quadratic trend only | 0.364 | 0.408 | 0.137 | 0.656 |
| Trend + sine | 0.935 | 0.874 | 0.836 | 0.978 |
| What the sine adds on top of the trend | 0.454 | 0.466 | 0.213 | 0.716 |

The plan's criterion is `R² > 0.7 → proceed with everything`. **But these numbers are an
artefact.**

## 2. The control group shows it

Computing R² searches ω (3 multipliers) × α,β (a 4×4 grid) and solves A·θ analytically by
least squares. The median note is 64 frames (0.34 s) long and many of those are **around a
single period**. One arm of a sine fits any gentle curve.

So the **same model was fitted to residuals from a *different* note of the same length**.
That is the only way to separate "it fitted because there is yoseong" from "it fitted
because the model is flexible".

| Periods | n | R² measured | R² control | Difference | Length (frames) |
|---|---|---|---|---|---|
| < 1.5 | 323 | 0.776 | 0.863 | **−0.087** | 54 |
| 1.5–3 | 254 | 0.824 | 0.821 | +0.003 | 62 |
| 3–5 | 52 | 0.625 | 0.688 | **−0.064** | 112 |
| ≥ 5 | 16 | 0.382 | 0.340 | +0.042 | 189 |
| **All** | **645** | **0.782** | **0.829** | **−0.047** | |

**The sine fits residuals from unrelated notes as well as real ones — slightly better, in
fact.** In no bucket does the difference exceed ±0.09. Even restricted to the identifiable
range (≥3 periods): measured 0.615 vs control 0.603, a difference of 0.012.

## 3. Three corroborating circumstances

- **`has_vibrato` (prominence > 3) holds for only 29 of 746 notes, 3.9%.** Most pansori
  notes have no clean periodic component at all.
- **Those 29 fit *worse*** — R²(sine) 0.358 vs 0.752 overall. If the hypothesis were right,
  the opposite should happen.
- **ω is not constant.** For `has_vibrato` notes its IQR is 3.45 Hz (median 4.41 Hz), which
  is wide. That does not match the premise of a fixed-rate sine.

Phase θ has circular concentration R = 0.081 (expected 0.032 under uniformity, Rayleigh
p = 0.007). With n = 746 that is statistically significant but effectively uniform in
magnitude — **the plan's judgement to leave θ a free parameter was itself correct.** There
is just nothing to apply it to.

## 4. What the figure says

`pansori/review/vibrato_long_notes.png` — the residuals of the 12 longest notes with their
best-fit sines.

- Of the 12, **one is textbook yoseong** (451 frames, 4.6 Hz, R² = 0.62). Even there the
  amplitude grows, and a constant-A sine cannot follow that.
- The rest are irregular oscillation, monotone glide, or large jumps of ±600–1500 cents.
- Some of those jumps look like **CREPE octave errors** (a 1200-cent step shape).

## 5. What this implies for the plan

The plan wrote down this possibility itself:

> The paper's premise is Western singing, where vibrato is 50–100 cents. Pansori yoseong is
> much wider at 135–160 cents, so that premise may not hold. R² is the test of it.

**It does not hold.** Pansori's 135 cents of within-note movement is not "slow sinusoidal
oscillation" — its principal component is **irregular, aperiodic movement**.

- **Phases 4–6 (global note IDs, sustain model, ω·θ bootstrap) — recommended for
  cancellation.** Building 800,000 note embeddings and a phase-initialisation pipeline for
  roughly 4% of notes does not pay for itself, and on the other 96% the learned A would
  produce meaningless oscillation. This is the same failure pattern already seen twice in
  Stage B and §5.5: right in magnitude, unrelated in content.
- **Phase 3 (transition + note-position features) — valid, and strengthened.** If the
  principal component of the movement is aperiodic and monotone, then the transition is
  exactly where it belongs, and note-position information is exactly what is missing there.
  That is true independently of the sine hypothesis.

## 6. Data issues this turned up along the way

| | Value | Note |
|---|---|---|
| Note length | median 0.34 s, max **5.31 s** | 6.8% over 1 s, 2.4% over 2 s |
| Within-note std > 300 cents | **2.1%** | Not yoseong — pitch movement, or a CREPE octave error |

A five-second "note" means `(pitch_midi, midi_dur)` stayed constant across that whole span.
It is worth checking whether ROSVOT or `06_merge.py` is fusing long spans into one. Had
Phase 5 survived, it would have carried one (θ, α, β) set per note and so would not have
held up on a five-second note anyway.

## 7. Reproduction

Historical — this is the command as run in the old repository (see the note at the top).

```sh
conda activate diffsinger && export PYTHONPATH=.
python pansori/eval/vibrato_stats.py \
    --config usr/configs/midi/cascade/pansori/aux.yaml --exp_name pansori_aux \
    --set train --n 300 --plot_notes 12
```

Outputs: `infer_out/note_vibrato_init.npz`,
`pansori/review/{vibrato_freq_hist,vibrato_phase_hist,r2_hist,r2_gain_hist,vibrato_long_notes}.png`

## 8. Limits

- 300 training utterances / 746 notes. The ≥5-period bucket is thin at n = 16.
- The control group **needs a donor of the same length**, so it could not be computed for
  the longest notes (86.5% of all notes got one, but only 34.7% of the longest decile). The
  conclusions about long notes are weaker in proportion.
- `--min_frames 40` and the 90%-voiced condition bias the sample towards short, stable
  notes.

# pansori/compare — quantitative comparison, old model vs new

An evaluation package that measures `~/Desktop/DiffSinger_moonInRiver` (MoonInTheRiver
cascade, 24 kHz) and this repository's `pansori_ac_v1` + `pansori_var_v1` (openvpi reflow,
44.1 kHz) **on the same held-out utterances with the same rulers**. Report: `report.html`.

## The conclusion, in one line

**The new model moves like pansori, but it does not move together with the original.**
Within-note variation ratio 0.29 → 0.92 (Holm p = 0.0005, clearing both the seed-noise floor
and the vocoder-ceiling gap); correlation with the original 0.035 → 0.037 (unchanged).

## Running it

```sh
conda activate dsp && cd ~/Desktop/DiffSinger && export PYTHONPATH=. PYTHONNOUSERSITE=1

python pansori/compare/smoke.py                              # 5 gates + null controls
python pansori/compare/render_new.py --items fair --seeds 5  # 19 items × 5 seeds, ~40 s
python pansori/compare/collect.py                            # → out/results.jsonl
python pansori/compare/stats.py                              # → out/summary.json + console table
python pansori/compare/report.py                             # → report.html
```

Hold-out extension (absolute numbers + a representativeness test):

```sh
python pansori/compare/render_new.py --items sample:200 --seeds 3
python pansori/compare/collect.py --items sample:200 --seeds 3 --only-new --no-spectral \
       --out pansori/compare/out/results_ext.jsonl
python pansori/compare/stats.py     # picks up results_ext.jsonl automatically if present
```

**If `smoke.py` fails, every number after it is void.** In particular, if G1 (sample-rate
invariance) breaks, a 24 kHz system and a 44.1 kHz system cannot go in the same table.

## The five traps this code knows about

1. **Jeong Gwon-jin's 5 items under `a580300d` are training data for the new model.** The
   old model held them out, but under the new configuration that one song is the speaker's
   entire output, so it was released automatically. Dropping those 5 from the 24 leaves the
   **19 fair pairs** that are the headline. `items.assert_held_out()` enforces this on every
   run by reading `test_prefixes` out of `configs/exp/ac_v1.yaml` — it is not a comment.
2. **`|Δf0| per frame` is a frame-rate artefact.** The old pipeline ran at 5.33 ms, the new
   one at 11.61 ms, so the same curve comes out ~2.2× larger on the new side. Everything
   here is measured in **cents/second on a common 10 ms grid**. The per-frame ratios in
   `docs/NEXT.md §C5` cannot be used to compare systems.
3. **RMVPE accepts `f0_min`/`f0_max` but does not use them**
   (`modules/pe/rmvpe/inference.py:63-78`). It drops an octave on male *tongseong* — median
   4.4% of the 19 items, worst case 28%. Uncorrected, the **original's** within-note
   variation inflates to 650 cents, which contradicts the published figure of 161. So note
   residuals are wrapped into (−600, 600] (`sigimsae.octave_snap`) and that is applied
   **identically to the reference and to every system**. The correction is used only in
   Tier 2 — a genuine octave error should be penalised by the pitch-accuracy metrics.
4. **The old `2_gt_crepe` is GT mel** (the vocoder ceiling). That is not the same thing as
   the new `2_resynth` (which passes through the acoustic model), so a new `0_vocoder` rung
   was added. The only true correspondence is old `5_pred_own` ↔ new `3_pitch`.
5. **The two repositories' note grids are identical.** `cumsum(note_dur)` reproduces the old
   `notes.jsonl` onsets for all 19 items to within 0.000 ms (gate G2). So one staircase
   measures both sides — there is no need to port the old `note_seg.py`.

## Design decisions

- **One reference f0 for everything: RMVPE.** But the old model was trained against CREPE
  targets, so swapping the reference could disadvantage it; the report therefore carries the
  difference between the two references (162 cents; the original's within-note variation is
  133 under CREPE vs 160 under RMVPE). The amplitude effect is larger than that difference;
  the pitch-accuracy side is not, and was never claimed as an improvement.
- **Seeds**: the new model uses a reflow sampler, so 5-seed medians. The old `pansori_aux`
  is a deterministic regression, so a single render *is* the model's answer — that is what
  justifies the asymmetry. `pansori_ds3` is a sampler, so it is marked spectral-only and
  single-draw.
- **Claim rule (pre-registered)**: exceeds 2× the seed IQR **∧** Holm p < 0.05 **∧** exceeds
  the vocoder-ceiling gap. All three values are printed in the table.
- **No sine-vibrato fitting.** `pansori/docs/VIBRATO_VERDICT.md` refuted it on this corpus —
  the same sinusoid fitted residuals from unrelated notes just as well (control 0.829 vs
  measured 0.782), and only 3.9% of notes have a clean periodic component.
- **No PESQ/STOI either.** They are not installed, and they are built for narrowband
  telephone-speech intelligibility, so on pansori they produce authoritative-looking
  nonsense.
- **`pansori/eval/` is never imported** (it cannot run — `note_seg.py` is missing, and
  `docs/NEXT.md §C4` recommends retiring it). Only the definitions were carried across.
  `f0_metrics.gross_error_rate` is not used as-is: its denominator is the set of commonly
  voiced frames, which means **marking hard frames unvoiced improves the score**. RPA-50
  uses the reference's voiced frames as the denominator instead (both versions are reported
  side by side).

## Files

| File | Responsibility |
|---|---|
| `config.py` | 10 ms analysis grid, common spectral basis, seeds, checkpoints, **4 pre-registered endpoints**. No CLI overrides |
| `items.py` | Hold-out enforcement, the fair 19 pairs, stratified sampling |
| `notes.py` | Note grid, staircase, core mask, boundaries |
| `f0.py` | 10 ms RMVPE (disk-cached) + the old CREPE loader |
| `sigimsae.py` | Residual decomposition, octave snap, motion (cents/s), wobble, transitions |
| `pitch_metrics.py` | RPA / RCA / voicing / octave jumps |
| `spectral.py` | MCD/LSD on a common 24 kHz basis, band energy |
| `render_new.py` | In-process batch renderer (prep → variance → acoustic, 3 processes) |
| `collect.py` | → `out/results.jsonl` (long format) |
| `stats.py` | Wilcoxon · bootstrap CI · Holm · seed floor · extension representativeness |
| `plots.py` | Inline SVG (theme-aware) |
| `report.py` | Self-contained HTML |
| `smoke.py` | Gates G1–G5 + null controls |

`render_new.py` loads the model once per process. Spawning `scripts/infer.py` per render
takes ~2.6 hours for the same work, nearly all of it model loading — measured: 304 renders
in 19 seconds.

## Known limits

- `4_full` (durations predicted too) has no counterpart, because the old ladder never built
  that rung. It also breaks frame alignment, so it is excluded from frame-based metrics and
  only gets distribution metrics.
- For some hold-out items `note_slur` cannot be recovered (more words than notes), so no
  `4_full` input can be built. Those items lose just that rung and stay in the rest of the
  comparison.
- The old model's noise floor could not be measured (only one render exists).
- n = 19 is small. It is fixed at that by the old model having only 24 renders, and adding
  new items does not increase paired power. That is why the 200-utterance extension is used
  only for absolute numbers and for the **representativeness test**.

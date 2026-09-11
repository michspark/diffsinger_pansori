# Remaining work — paste this file into the next session to pick up where this left off

Branch `feat/pansori`. The history is in `pansori/docs/PORTING.md`; the runbook is
`pansori/README.md`.

## Done

The porting session (`8212d64`..`f6ba635`):

- Committed 53 uncommitted files of work from the original repository
  (`~/Desktop/DiffSinger_moonInRiver`) as `341e927`
- Ported all of `pansori/` and generated `dictionaries/korean-coda.txt` (11,172 syllables →
  34 Coda phonemes)
- `04_align.py` now also records MFA's words tier, so `ph_num` can be derived without
  re-running MFA
- Rewrote `06_merge.py` / `07_validate.py` against the openvpi CSV schema
- `08_export.py` re-cuts to 44.1 kHz mono and splits across the 23 speakers
- Generated `configs/pansori/{acoustic,variance}.yaml` (via `make_configs.py`)

The bring-up session:

- **env `dsp`** (Python 3.10 / torch 2.6.0+cu124 / lightning 2.3.3). `pip check` passes
- **Three checkpoints** downloaded: `rmvpe`,
  `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`, `vr` (HN-SEP)
- **New `check_dict_agreement.py`** — the harness that decides a mapping change by scoring
  it against the dictionary
- **Three label defects fixed** → 1-syllable agreement **87.92% → 89.59%**, 2-syllable
  75.01% → 76.43%
- Labels regenerated (04 `--skip_align` → 06 → 07 PASS → 08). **3,109 of 26,955 items
  (11.5%)** changed
- **binarize complete** — acoustic 4.5 GB / variance 381 MB, train 27.8 h · valid 1.08 h,
  all 23 speakers have training data
- **Spec statistics measured and fed into the config.** The ceiling really did exceed the
  default (see below)
- **acoustic + variance trained** — `checkpoints/pansori_{ac,var}_v1`, both from scratch
- **Inference path opened** — `pansori/make_ds.py` (label/score → `.ds`) plus
  `examples/run_examples.sh`. lyrics → phonemes → variance → acoustic → wav runs end to end
- **The f0-flattening question has an answer** — systematic flattening does not reproduce
  (§D below)
- **Report published** — the whole pipeline explained, with 15 synthesis examples
  (`build_report.py`)

## Training results

Both models finished training. **Neither one's most recent checkpoint is its best.**

| | Steps run | Lowest validation | Last | Status |
|---|---|---|---|---|
| acoustic | 100,000 (ran to the configured end) | **0.17800 @ 28,000** | 0.18834 | After 28k it only oscillated between 0.178 and 0.203 for 72k steps |
| variance | 32,000 (stopped) | **0.08761 @ 20,000** | 0.10423 | Stopped after `dur_loss` rose three times in a row |

acoustic ran 3.5× longer than it needed to, and by the time it finished **the 28,000-step
checkpoint had already been deleted by `num_ckpt_keep: 8`.** Without the copy in
`checkpoints/best/` it would have quietly settled for the lowest survivor, 76,000 (0.18335).
`run_examples.sh` restores it before running.

### Checkpoints are chosen per head

| Metric | Lowest step | Note |
|---|---|---|
| acoustic `validation/total_loss` | **28,000** | 0.17800 |
| variance `validation/dur_loss` | **16,000** | 0.07797, rises continuously after |
| variance `validation/pitch_loss` | **32,000** | 0.00861, still falling when stopped |
| variance `validation/total_loss` | 20,000 | A compromise dominated by dur |

Pulling pitch from the 20k that `total_loss` picks gives **25% more f0 error** than 32k. So
`run_examples.sh` selects per head and runs variance twice — first `--predict dur`, then
`--predict pitch` over that output (`ph_dur` is read as manual).

Since `num_ckpt_keep: 8` deletes older files, the checkpoints above were copied to
**`checkpoints/best/`** as `{exp}_step{N}.ckpt` (1.6 GB; gitignored, so not in the
repository).

```sh
bash pansori/examples/run_examples.sh    # regenerate the examples (restore + best-pick are automatic)
tensorboard --logdir checkpoints
```

## To do

### A. Next time training runs
1. **Reduce `max_updates` or add early stopping.** acoustic was already at its best by 28k
   and the remaining 72k steps just burned three GPU-hours. Raise `num_ckpt_keep` at the
   same time — at the current value (8) the best checkpoint does not survive
2. Only 6.1 GB of the 24 GB of VRAM was used. `max_batch_frames` in `make_configs.py` could
   go up, but the learning rate is set for the current batch and would have to move with it,
   and comparability against upstream defaults would be lost
3. **Augmentation is still off.** `use_key_shift_embed` / `use_speed_embed` +
   `augmentation_args`. The speaker skew is severe (the top 3 are 36%), so it could help the
   minority speakers

### B. Inference — open
3. `bash pansori/examples/run_examples.sh` produces the 4-rung ladder (original / resynth /
   pitch-predicted / fully inferred) and the score syntheses. Scores go in
   `pansori/examples/scores/*.txt` as `syllable note duration_seconds`, where `-` is sigimsae
   and `_` is a rest. Drop in a new score and it is synthesised for three speakers
   automatically.
   - **`--spk` cannot take a Hangul speaker name** (`utils/infer_utils.py:56` matches
     `[0-9A-Za-z_-]+`). `make_ds.py` works around it by writing `spk_mix` into the `.ds`
     directly — upstream is not touched
   - **`note_slur` is required.** It is what binds notes to words when inferring without
     `ph_dur`, and it is not in the labels, so it is recovered by DP from the measured
     timings. In a score it is written directly as `-`

### C. Evaluation
4. `pansori/eval/resynth_ladder.py` and `f0_metrics.py` are still written against 24 kHz /
   80 mel / the old inference API. But **`pansori/examples/` largely takes over their
   role** — the ladder (`run_examples.sh`), the scores (`measure.py`), flattening
   (`f0_report.py --seeds N`), and sampling variance (`seed_spread.py`). Retiring the old
   scripts is probably more correct than adapting them.
5. **f0 flattening did not reproduce — this item is closed.** Amplitude ratios, measured as
   the median over 5 seeds: resynth 0.86–0.98, f0-predicted 0.72–0.85, dur+f0-predicted
   0.82–1.07. Four of the six predicted tracks have ranges that straddle 1.0. **2–14% of the
   attenuation already happens in the acoustic model and the vocoder, so it is not the pitch
   predictor's fault.** At least by this measurement, there is little basis for attaching a
   contour tokenizer.
   *Note: re-measure once acoustic training finishes.*

### D. Outstanding label issues (measured only)
Re-measure any time with `check_dict_agreement.py --syllables N --focus <symbol>`. Fixing
one costs 2 minutes for 04 (`--skip_align`) → 06 → 07, plus 20 minutes for the 08 re-cut.
**No re-alignment is needed.**

6. **ㅈ/ㅉ/ㅊ + glide** (96+): 죠 is [tɕo], so MFA says `j o` while the dictionary says
   `j y o`. Since 죠 and 조 are homophones in Korean, this is a **dictionary-side** problem —
   adding `y` in `ipa_to_coda.csv` would turn the far more common 조 into `j y o` too.
   Fixing it means touching `build_dictionary.py`
7. **요 → `y eo` vs dictionary `y o`** (156): a vowel-side issue. Not looked at yet
8. **ㄴ coming out as `d`** (내/나/니/너, 700+ in total): MFA misrecognition, not a mapping
   problem. The mapping cannot fix it
9. **Silence-absorption threshold**: a 50 ms overlap absorbs the entire silence. 6.2%
   (3,772) of absorbed spans have under 50% coverage
10. Most of the remaining disagreement is **normal liaison** (은 → `eu n`, 좋 → `j o`). A
    context-free syllable dictionary genuinely cannot reproduce that — 100% agreement is not
    the goal

### E. Cleanup / decisions
11. Whether to push `~/Desktop/DiffSinger_moonInRiver` to its origin (it only has local
    commits)
12. Candidates for deletion: the old 24 kHz `wavs/` (5.5 GB) and the old
    `data/binary/pansori*` (16 GB) — after verification
13. ~~**Breath (AP) labelling**~~ — there is a path now. Set `breath_ap.enabled: true` in
    `preprocess/align.yaml` and FoxBreatheLabeler overlays AP onto the SP spans after SOFA
    alignment, with `make_configs.py` dropping `merged_phoneme_groups` automatically
    (README §8). There is no longer any need to put energy-based detection into 06 directly.
    What remains is to actually run it and measure

## Things to watch

- **No `git merge` / `git remote add`.** The two repositories share no common ancestor
- **The three old checkpoints cannot be reused.** 24 kHz / 80 mel / FastSpeech2 — not a
  single tensor matches
- **Omitting `f0_min: 55` / `f0_max: 750`** makes the whole low register of male *tongseong*
  come out unvoiced
- **`spec_max` must use the measured value.** openvpi's default is `[0]`, but this corpus's
  measured ceiling is **+2.838**. Leave it and the diffusion target escapes [-1, 1] after
  normalisation. The floor is safe the other way — `nvSTFT` compresses with
  `log(clamp(x, 1e-5))`, so nothing below -11.513 exists
- **The env is broken without `PYTHONNOUSERSITE=1`.** An old tensorflow in
  `~/.local/lib/python3.10/site-packages` kills the `tensorboard` import at protobuf, and a
  numpy there makes pip misjudge requirements as "already satisfied". It is set in the `dsp`
  env's `activate.d`
- **Preprocessing cannot run under the `base` env** (numba vs numpy 2.5 conflict). Use the
  `pansori` env (3.10)
- **Do not judge a model from a single render.** Both models use a reflow sampler, so
  changing nothing but the seed moves f0 RMSE by ±17% and mel L1 by ±8%
  (`seed_spread.py`). To compare tracks, fix the seed or render several times and use the
  median
- **The most recent checkpoint is not the best — the tooling now guards this.**
  `best_ckpt.py` picks the lowest-validation-loss step and `run_examples.sh` restores
  deleted ones from `checkpoints/best/`. It actually fired this time: acoustic's optimum
  (28k) had already been deleted when training ended
- **Jeong Gwon-jin (226 utterances) was released from the hold-out automatically** — one
  song was that speaker's entire output, which would have left zero training data

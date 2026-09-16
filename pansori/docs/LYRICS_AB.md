# Lyrics A/B — with and without changbon correction

Two of these have run. §1–6 are the first, **whisper vs snapped** (03b), concluded
2026-08-29: the correction did not help. §7 is the second and current one,
**whisper vs hybrid** (03c), where each arm is also aligned by the SOFA trained on its own
lyrics. Read §6 before §7 — the reading discipline is the same and is not repeated.

The **procedure and result** of training on two label sets and comparing the synthesised
audio. The reasoning behind the verdict, and its limits, are in
[`LYRICBOOK_VERDICT.md`](LYRICBOOK_VERDICT.md); the command summary is in
[`../README.md`](../README.md).

---

`03b_snap_lyrics.py` corrects the Whisper transcription against the *changbon* — the
published libretto (`lyricbook/corpus.jsonl`).
[`LYRICBOOK_VERDICT.md`](LYRICBOOK_VERDICT.md) judged that **by MFA alignment metrics** this
correction makes things worse, but §4 of that same document states the limit of the verdict:
MFA structurally favours text derived from the audio, so it cannot answer "which lyrics are
better **for training**". Answering that means training on both label sets and comparing the
synthesised audio. This section is that procedure.

**Exactly one thing changes: the labels.** Audio, notes, hold-out, spec statistics, configs
and training settings must all be identical, or the difference cannot be attributed to the
lyrics.

## 1. Building the snapped corpus

```sh
conda activate pansori
python pansori/preprocess/01b_recut_wavs.py --set full        # restore stage 04's input (README §8)

conda activate aligner && cd pansori/preprocess
python 04_align.py --set full --lyrics lyrics_snapped.jsonl --suffix _snap   # several hours

# 06 and 07 read fixed filenames out of data/{set}/. Assembling a new --set out of
# symlinks lets them run unchanged.
cd ../data && mkdir -p full_snap && cd full_snap
ln -s ../full/{segments.jsonl,notes.jsonl,spk_map.json,wavs} .
ln -s ../full/phones_snap.jsonl phones.jsonl                  # this is the only difference

conda activate pansori && cd ../../preprocess
python 06_merge.py    --set full_snap
python 07_validate.py --set full_snap                         # do not proceed without PASS
python 08_export.py   --set full_snap --out data/pansori_snap --wavs_from data/pansori
```

`--wavs_from` symlinks each speaker's `wavs/` to the existing corpus. The audio is identical
byte for byte across the two variants, so there is no reason to re-cut 14 GB — and pointing
at the same files is itself what guarantees "everything but the labels is the same".

## 2. Configs and binarize

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/make_configs.py --data data/pansori_snap --name snap
python scripts/binarize.py --config configs/exp/ac_snap.yaml
python scripts/binarize.py --config configs/exp/var_snap.yaml
```

**Do not re-run `compute_spec_stats.py`.** Mel statistics come only from the audio, and the
audio is identical. Reusing `configs/pansori/spec_stats.yaml` as-is ties the conditions
together rather than loosening them.

## 3. Training — both runs in one group

```sh
python pansori/train.py --config configs/exp/ac_snap.yaml \
  --exp_name pansori_ac_snap --reset --lyrics-source snapped --wandb-group lyrics-ab

python pansori/train.py --config configs/exp/var_snap.yaml \
  --exp_name pansori_var_snap --reset --lyrics-source snapped --wandb-group lyrics-ab
```

Step counts and `num_ckpt_keep` live inside the experiment files
(`configs/exp/{ac,var}_snap.yaml`). The first time these two ran they were passed via
`--hparams`, which leaves the conditions only in shell history.

- **The step counts are matched to the baseline.** `pansori_var_v1` actually stopped at
  **32k**, not the 80k in its config (both the checkpoints and the tfevents agree). Running
  longer would train only the snapped arm's pitch head further and tilt the comparison. For
  acoustic, the baseline went to 100k but its optimum was 28k, so 50k is plenty.
- **`num_ckpt_keep` is raised.** At the defaults (acoustic 8 / variance 5) the early
  checkpoints get deleted — and the measured optima are all early: acoustic 28k, dur 16k,
  pitch 32k.

## 4. Evaluation — the same ruler, twice

`run_examples.sh` takes the model and corpus from environment variables (README §11).

```sh
bash pansori/examples/run_examples.sh                          # whisper
python pansori/examples/measure.py --json pansori/examples/out/metrics.json

AC_EXP=pansori_ac_snap VAR_EXP=pansori_var_snap DATA=data/pansori_snap \
DS_DIR=pansori/examples/ds_snap OUT_DIR=pansori/examples/out_snap \
  bash pansori/examples/run_examples.sh
python pansori/examples/measure.py --out pansori/examples/out_snap \
  --json pansori/examples/out_snap/metrics.json
```

Keep `SEED` the same across both runs. Reflow is a stochastic sampler; changing nothing but
the seed moves f0 RMSE by ±17% — **if the gap between two models is smaller than
`seed_spread.py`'s variation, there is no gap.**

## 5. Result (run of 2026-08-29)

**The correction did not help.** Measuring 60 held-out pairs × 3 seeds at the `2_resynth`
rung, exactly one metric separates, and it does so **only on utterances where the lyrics
actually changed**: voiced/unvoiced agreement, whisper winning 21–6 with 3 ties
(p = 0.006), against 10–12 with 8 ties in the control group (30 pairs whose phoneme
sequences are identical, p = 0.83). The full numbers and their interpretation are in
[`LYRICBOOK_VERDICT.md`](LYRICBOOK_VERDICT.md) §9.

```sh
python pansori/examples/ab_ladder.py --n 60 --seeds 3      # both arms at once, ~5 min
```

`ab_ladder.py` loads each arm's model **once** and renders all 60 items
(`run_examples.sh` spawns a fresh process per render, costing 25 s each). Half the items are
a **control group** whose phoneme sequences do not differ by a single character across the
two corpora — since both are measured with the same two models, any incidental difference
between the training runs shows up equally in both groups, and only a difference that
appears in the treatment group alone can be attributed to the lyrics.

## 6. Reading this carefully

- **Do not compare the two runs' validation losses as higher/lower.** Each is computed
  against its own labels. They sit side by side in wandb to show convergence shape and when
  overfitting starts; which arm is better is decided only by the synthesis metrics in §4.
- The set of items passing 07 is not exactly identical across the two variants. Count the
  difference and state it.
- There is still no listening verification. This experiment only answers "which label set
  produces better synthesis".


---

## 7. Second A/B — whisper vs hybrid, each with its own aligner (2026-09-16)

03b failed on a mechanism §10 of [`LYRICBOOK_VERDICT.md`](LYRICBOOK_VERDICT.md) names: below
`match_score` 0.55 the changbon span runs p90 **2.00×** Whisper's syllable count, and a label
with twice the syllables of the singing is wrong however correct its spelling. `03c_hybrid_lyrics.py`
keeps Whisper's syllable count one for one and borrows changbon syllables only for stretches
of equal length pinned by exact matches. This is the A/B for that arm.

### 7.1 What each arm is, and why the aligner moves with it

| | arm A | arm B |
|---|---|---|
| lyrics | `lyrics.jsonl` (Whisper) | `lyrics_hybrid.jsonl` (03c) |
| aligner | SOFA `korean_pansori` | SOFA `korean_pansori_hyb` |
| that SOFA was bootstrapped from | `~/src/SOFA/data_whisper` | `~/src/SOFA/data` |
| alignment | `phones_sofa.jsonl` | `phones_hyb2.jsonl` |
| corpus | `data/pansori_sofa` | `data/pansori_hyb2` |
| run | `pansori_ac_sofa` · `pansori_var_sofa` | `pansori_ac_hyb2` · `pansori_var_hyb2` |

**Two things change between the arms at once, deliberately.** SOFA has to be trained before it
can align (`preprocess/SKILL` and `align.yaml`), and its training data is bootstrapped from an
alignment of some lyric source, so "the lyrics" and "the aligner fitted to the lyrics" are not
separable in a pipeline that has to start somewhere. The question this answers is therefore
**"which lyric source, together with an aligner fitted to it, trains a better model"** — a
whole-pipeline comparison, not a single-variable one. §1–5's A/B did hold the aligner fixed and
could make the narrower claim; this one cannot, and should not be read as if it did.

The control group is what keeps it honest: its utterances have **identical lyrics and differing
aligners**, so whatever the aligner alone is worth shows up there. Only a difference the
treatment group shows *over* the control can be attributed to the lyrics.

### 7.2 The pairing, measured

- **26,955 utterances in each corpus, set difference 0 in both directions.**
- `configs/pansori/data_hyb2.yaml` is identical to `data_sofa.yaml` apart from the corpus
  name — same 23 speakers, same `spk_id`s, same hold-out songs.
- Of 117 resolved acoustic hparams, **3 differ**: `base_config`, `binary_data_dir`, and
  `datasets[].raw_data_dir`. All three are the corpus pointer. Variance: 3 of 116, the same three.
- The audio is not merely equal, it is **the same files** — `08_export.py` ran with
  `--wavs_from data/pansori_sofa`, so arm B's wavs are symlinks into arm A's corpus.
- `max_updates: 40000` and `num_ckpt_keep: 12` are copied from `ac_sofa`/`var_sofa` unchanged.
- **`compute_spec_stats.py` was not re-run.** Mel statistics come only from the audio.

### 7.3 Splitting treatment from control

`ab_ladder.py` used to split on `ph_seq`, which is right only while both arms share an aligner.
Here they do not, and it shows: of the shared 26,955 utterances **22,305 differ in `ph_seq`
(82.7%)** but only **7,274 (27.0%)** differ by a single Hangul character. The aligner's judgement
about where silence goes moves nearly everything.

So this pair splits on the stage-03 **text**, Hangul only (03c can respace a line it did not
otherwise touch). On the hold-out that is **245 treatment against 764 control**, so a 60-item
run draws 30 and 30.

### 7.4 Which checkpoint aligned which — recovered, and now recorded

`phones_sofa2.jsonl` and `phones_hyb2.jsonl` were both produced on 2026-09-13, and **nothing on
disk said which SOFA checkpoint made either.** `align.yaml` records the settings in force now;
it had been edited and reverted. Recovering it took two measurements:

1. On the 20,388 utterances whose phone sequence is identical across a round's two arms, the
   boundaries agree **to the bit** (mean |Δ| 0.00 ms) — for `sofa`/`hyb` and for `sofa2`/`hyb2`
   alike. **Within a round the checkpoint is a constant**, so neither pair gives each lyric
   source an aligner of its own.
2. The `.lab` files handed to SOFA are **byte-identical between rounds** for both arms
   (`diff -rq`: 0 differing entries over 27,720 files) and the phone inventory is the same 36
   symbols, so dictionary, `lab_unit` and G2P did not move. What moved is the aligner's own
   output: sil 30,254 → 19,845, AP 48,352 → 37,234.

Round 1 therefore ran `korean_pansori` (trained 09-06 from `data_whisper`) and round 2 ran
`korean_pansori_hyb` (trained 09-13 13:44 from `data`); no SOFA training ran after 13:51. Which
makes **`phones_sofa.jsonl` + `phones_hyb2.jsonl`** the pair where each arm carries its own
aligner, and `phones_sofa2.jsonl` — Whisper's lyrics aligned by the hybrid SOFA — a cell of the
grid this A/B does not use.

Stage 04 now writes `phones{suffix}.meta.json` beside the jsonl and takes `--sofa-ckpt`, so this
does not have to be reconstructed again.

### 7.5 Running it

```sh
conda activate pansori && cd pansori/preprocess
python 07_validate.py --set full_hyb2                               # PASS is mandatory
python 08_export.py   --set full_hyb2 --out data/pansori_hyb2 --wavs_from data/pansori_sofa

conda activate dsp && cd ../.. && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/make_configs.py --data data/pansori_hyb2 --name hyb2
python scripts/binarize.py --config configs/exp/ac_hyb2.yaml
python scripts/binarize.py --config configs/exp/var_hyb2.yaml

python pansori/train.py --config configs/exp/ac_hyb2.yaml  --exp_name pansori_ac_hyb2 \
  --reset --lyrics-source hybrid --wandb-group lyrics-ab-hybrid
python pansori/train.py --config configs/exp/var_hyb2.yaml --exp_name pansori_var_hyb2 \
  --reset --lyrics-source hybrid --wandb-group lyrics-ab-hybrid
```

Arm A needs no training — `pansori_ac_sofa` and `pansori_var_sofa` already ran to 40k on this
very corpus (verified: all 2,690 of 조상현's `ph_dur` rows in `data/pansori_sofa` match
`full_sofa`, and none match `full_sofa2`).

```sh
python pansori/examples/ab_ladder.py --arms sofa hybrid \
  --ckpt sofa=28000 --ckpt hybrid=N --n 60 --seeds 3
python pansori/examples/build_listening_test.py --n 40
```

### 7.6 Result

Not yet recorded.

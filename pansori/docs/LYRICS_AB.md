# Lyrics A/B — with and without changbon correction

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

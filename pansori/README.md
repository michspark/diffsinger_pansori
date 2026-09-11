# Pansori — fine-tuning DiffSinger on Korean traditional narrative singing

This directory holds everything used to turn a collection of raw *pansori* recordings into
a singing-voice model: the preprocessing pipeline, the training and inference wrappers, and
the measurement tools that decide whether a change actually helped.

**New here? Read sections 1–4.** They explain what the project is and why the pipeline has
the shape it has. Sections 5 onward are the runbook.

| | |
|---|---|
| Base model | [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger) — acoustic + variance, reflow sampler, 44.1 kHz |
| Corpus | 394 pansori recordings → **26,955 utterances / 28.9 h**, 23 speakers |
| Language pipeline | Whisper transcription → forced alignment → Coda-34 phoneme set |
| Branch | `feat/pansori` |

---

## 1. What is pansori, and what does this do?

**Pansori (판소리)** is a Korean traditional performing art: a single singer (*sorikkun*)
and a single drummer perform an hours-long story, alternating between sung passages and
spoken narration. The singing voice is nothing like pop or classical vocal — it is rough
and strained by design, with very long sustained tones decorated by *sigimsae*, a dense
vibrato-and-slide ornamentation that can wander a whole semitone or more around the nominal
note.

This project fine-tunes DiffSinger on that material, so that the model can sing **new**
lyrics and **new** melodies in a pansori voice. Concretely:

- **In**: 28 GB of source audio with note MIDI and section labels, but **no lyrics text**.
- **Out**: two trained checkpoints (an *acoustic* model and a *variance* model) plus the
  tooling to synthesise from a score and to measure how good the result is.

Feed it a score like this and it sings:

```
_   rest  0.30      # '_' is a rest  -> SP
춘  A4    0.45
향  G4    0.40
-   F4    0.55      # '-' continues the previous syllable (melisma / sigimsae)
```

## 2. Why it is hard

Three measured facts drive almost every design decision in this directory. If something in
the pipeline looks over-engineered, it is usually one of these:

**a. The corpus has no lyrics.** There is audio, there is note MIDI, there are section
labels — but no text. So stage 03 transcribes with `whisper-large-v3`, and every downstream
label inherits that transcription's errors. An alternative arm snaps the transcription to a
published libretto; see §8.

**b. Forced aligners are built for speech, and pansori is not speech.** Run a
speech-trained aligner over a four-second sustained vowel and it does not know what to do
with it. Measured on this corpus: **34.03%** of phonemes in the aligner's own TextGrid sit
exactly on its 0.03 s minimum-duration floor — that is the aligner declining to place a
boundary, not a real 30 ms phoneme.
→ [`docs/SEGMENTATION_VERDICT.md`](docs/SEGMENTATION_VERDICT.md),
[`docs/LYRICBOOK_VERDICT.md`](docs/LYRICBOOK_VERDICT.md)

**c. Long silences that carry a note.** **52.2%** of corpus time is "no phoneme here, but
the MIDI says a note is sounding" — the aligner gave up, but the singer did not stop
singing. Stage 06 absorbs those spans into the preceding vowel rather than discarding them.

The same story underlies all three: *the alignment, not the model, is the bottleneck.* That
is why there is a switchable aligner seam (§8) and three separate rulers for measuring
labels (§10) rather than just a loss curve.

## 3. Ground rule: upstream is never modified

**Not one line of an upstream openvpi file has been changed.** Everything this project adds
lives in exactly four places:

```
pansori/                        this directory
configs/pansori/                generated config layers
configs/exp/                    hand-written experiment configs
dictionaries/korean-coda.txt    generated pronunciation dictionary
```

Keep it that way. Rebasing onto upstream openvpi stays a no-op as long as this holds, and
it has held for 36 commits. Where upstream behaviour needed changing, it was done by
subclassing (`variance_task.py`), by config (`task_cls:`), or by wrapping
(`train.py` wraps `scripts/train.py`) — never by editing.

## 4. Glossary

The corpus, the docs and some comments use Korean domain terms. This is all of them:

| Term | Meaning |
|---|---|
| **pansori** (판소리) | The art form: one singer + one drummer, hours-long sung storytelling |
| **changbon** (창본) | The published libretto — the printed text of a pansori piece |
| **myeongchang** (명창) | A master singer. 22 of the 23 speakers are one; the 23rd, `unknown`, pools every recording whose singer is unidentified |
| **sigimsae** (시김새) | Ornamentation — slides, wide vibrato, pitch inflections on a held tone |
| **jangdan** (장단) | The rhythmic cycle a passage is set in |
| **aniri** (아니리) | Spoken narration between sung passages — **excluded** at stage 01 |
| **changjo** (창조) | A recitative-like style, also excluded at stage 01 |
| **tongseong** (통성) | Chest/full voice. Its low register is why `f0_min: 55`, not the default 65 |
| **eojeol / eumjeol** (어절 / 음절) | Word / syllable. Stage 06 re-splits words into syllables for `ph_num` |

---

## 5. The pipeline end to end

```
~/Desktop/Pansori_Data/{Audio, labels_2.json, rosvot_midi}     394 songs, 28 GB, not in the repo
  │
  ├─ 01 segment ──────────────────────────► segments.jsonl + wavs/
  ├─ 02 speaker map ──────────────────────► spk_map.json
  ├─ 03 transcribe (whisper-large-v3) ────► lyrics.jsonl
  │    └─ 03b snap to changbon (optional) ► lyrics_snapped.jsonl        [lyrics arm B]
  ├─ 04 align (MFA | SOFA) ───────────────► phones.jsonl + aligned/*.TextGrid
  ├─ 05 notes (slice ROSVOT MIDI) ────────► notes.jsonl
  ▼
 06 merge ──────────────────────────────► transcriptions.csv
 07 validate ──────────────────────────► PASS / FAIL   ← gate: never binarize without PASS
 08 export (re-cut at 44.1 kHz) ───────► data/pansori/{speaker}/
  ▼
 build_dictionary → make_configs → binarize → compute_spec_stats (bootstrap)
  ▼
 train acoustic ┐
 train variance ┴──► checkpoints/ ──► best_ckpt (per head) ──► make_ds ──► synthesis
  ▼
 measure · seed_spread · f0_report · build_report · ab_ladder · label_report · intelligibility
```

Stages are joined by **jsonl files on disk**, not by function calls, because their re-run
costs differ by two orders of magnitude: forced alignment takes hours, re-parsing its output
takes two minutes. When the phoneme mapping was fixed, stage 04 re-parsed without
re-aligning.

Every stage works inside `pansori/data/{set}/` and is selected with `--set` (in practice
`--set full`).

### Where the data goes

| Point | Remaining | Why it dropped |
|---|---|---|
| Source audio | 394 songs | — |
| Intersected with section labels | **387 songs** | No label, or no audio |
| 01 segmentation | **31,529 utterances / 33.48 h** | *aniri* and *changjo* excluded; fragments under 2 s discarded |
| 04 alignment succeeded | **26,955 utterances** | Fewer than 2 Hangul characters, or alignment failed (−4,574) |
| 06 merge | 26,955 | Nothing lost |
| 08 final corpus | **26,955 utterances / 28.93 h / 23 speakers** | — |

## 6. Directory map

```
pansori/
├── preprocess/           Stages 01–08. jsonl between stages, so each re-runs alone
│   ├── pansori_lib.py      Shared: source path, IPA→Coda rules, phone classes, jsonl IO
│   ├── aligners.py         The MFA/SOFA backends behind stage 04 (§8)
│   └── align.yaml          ★ The aligner seam — one line picks the backend
├── alignment/            IPA→Coda mapping table and its scorer
│   ├── ipa_to_coda.csv     korean_mfa IPA (108) → Coda (34)
│   └── check_dict_agreement.py  ★ Run before AND after touching the mapping
├── examples/             ★ Inference. After training, start here (§11)
│   ├── run_examples.sh     Generates the 4-rung ladder + score synthesis in one go
│   ├── scores/*.txt        Scores. Drop a file in and it gets sung
│   ├── best_ckpt.py        Picks the lowest-validation-loss checkpoint, per head
│   ├── measure.py          f0 RMSE · mel L1 · voiced/unvoiced agreement
│   ├── seed_spread.py      Sampling noise floor — read this before trusting a number
│   ├── f0_report.py        Measures sigimsae flattening
│   ├── ab_ladder.py        Two models over N held-out pairs + sign test
│   ├── build_listening_test.py   Blind listening page
│   └── build_report.py     Explanation + audio, rendered as one page
├── eval/                 ★ Label and pronunciation rulers (§12)
│   ├── label_report.py     Diagnoses ph_dur in the labels — the MFA-vs-SOFA verdict table
│   ├── intelligibility.py  5-rung ladder × {Whisper CER, MFA re-alignment}
│   └── compare_lyrics_alignment.py   Paired MFA comparison of two lyric sources
├── metrics/              Per-phone-class duration metrics (torchmetrics)
├── compare/              Quantitative comparison against the previous model
├── lyricbook/            changbon → corpus.jsonl (regenerate with build_corpus.py)
├── experiments/          Segmentation simulations — the evidence behind stage 01
├── server/run.sh         setup / binarize / train runner, tmux-oriented
├── docs/                 See the index in §14
├── train.py              scripts/train.py + wandb (upstream untouched)
├── variance_task.py      VarianceTask subclass — registers the phone-duration metrics
├── sofa_train.py         Wraps SOFA's trainer: adds wandb + best-checkpoint saving
├── tb_to_wandb.py        Uploads a finished run's TensorBoard logs to wandb
├── build_dictionary.py   All 11,172 Hangul syllables → the Coda-34 phoneme set
├── make_configs.py       Reads the speaker list on disk and renders configs/pansori/
├── make_ds.py            Label or score → .ds (the inference input format)
└── compute_spec_stats.py Measures this corpus's mel envelope → spec_min / spec_max
```

Our configs live in two directories under the repository root:

```
configs/
├── pansori/          generated by make_configs.py — do not hand-edit
│   ├── acoustic.yaml · variance.yaml       "kind" layer (knows nothing about the corpus)
│   ├── data_pansori.yaml · data_snap.yaml · data_sofa.yaml    corpus layer
│   └── spec_stats.yaml                     generated by compute_spec_stats.py
└── exp/              ★ one experiment = one hand-written file (§9)
    └── ac_v1.yaml · var_v1.yaml · ac_snap.yaml · var_snap.yaml · ac_sofa.yaml · var_sofa.yaml
```

### Legacy — kept as a record, not runnable

These predate the port to openvpi and import modules that were never brought across
(`note_seg.py`, `warm_start.py`, `contour_tokens.py`, `preprocess/08_f0.py`). They are left
in place because their conclusions are cited elsewhere, but **they will not run**, and
`compare/` deliberately does not import from `eval/`:

- `eval/midi_baseline.py`, `eval/resynth_ladder.py`, `eval/f0_metrics.py` — superseded by
  `examples/` (ladder, scoring, flattening, seed spread).
- `alignment/{analyze_alignment,build_review,build_overlay,build_transcriptions}.py` —
  pilot-era, hardcoded to `data/pilot` and the 7-field schema that `06_merge.py` replaced.
  `build_transcriptions.py` still writes `alignment/FIELD_CHECK.md`.
- `experiments/*` — one-off simulations pinned to five pilot UUIDs. Their conclusion is
  already baked into `01_segment.py`.

### Where the data lives

None of it is in the repository — it is all gitignored.

| What | Where |
|---|---|
| Source audio / labels / notes | `~/Desktop/Pansori_Data/` (`Audio/`, `labels_2.json`, `rosvot_midi/`) |
| Preprocessing intermediates | `pansori/data/full/` |
| Training corpus (44.1 kHz) | `data/pansori/{speaker}/` (8.7 GB) |
| Binarized | `data/pansori/binary_{acoustic,variance}/` |

The source path is hardcoded in exactly one place: `PANSORI_DATA` in
`preprocess/pansori_lib.py`.

## 7. Environments

| env | Python | Used by |
|---|---|---|
| `pansori` | 3.10 | Preprocessing 01–03, 06–08 |
| `aligner` | 3.14 | MFA 3.4.1 + `korean_mfa` + mecab-ko → 04–05 |
| `dsp` | 3.10 | openvpi itself (binarize / train / infer / eval) |
| `sofa` | 3.10 | The SOFA aligner (optional — §8) |
| `fbl` | 3.10 | FoxBreatheLabeler, for breath (AP) labels (optional — §8) |

`bash pansori/server/run.sh setup` creates the first three and downloads the checkpoints.
`sofa` / `fbl` are only needed when switching aligners (install commands in §8).

Two traps bite every one of these environments. Guard against them at creation time:

- **`PYTHONNOUSERSITE=1` is mandatory.** Without it, stale packages in `~/.local` make pip
  skip requirements as "already satisfied", leaving the env silently half-built, and an old
  tensorflow there kills the `tensorboard` import at protobuf. It is set in each env's
  `activate.d`.
- **`setuptools<81`.** `librosa<0.10` uses `pkg_resources`, which setuptools 81 removed.
  `dsp` already pins it via lightning.

Also: `base` and the old `diffsinger` (3.8) env cannot be used — numpy 2.5 conflict and too
old a Python. And `requirements.txt` is missing `torchaudio` (an upstream omission); RMVPE
needs it.

---

## 8. Preprocessing

| # | Script | in → out | env |
|---|---|---|---|
| 1 | `01_segment.py` | labels + Audio + MIDI → `segments.jsonl` + `wavs/` | base |
| 1b | `01b_recut_wavs.py` | segments → `wavs/` restored | pansori |
| 2 | `02_speaker_map.py` | segments → `spk_map.json` | base |
| 3 | `03_transcribe.py` | wavs → `lyrics.jsonl` (whisper-large-v3) | base |
| 3b | `03b_snap_lyrics.py` | lyrics + changbon → `lyrics_snapped.jsonl` (optional) | pansori |
| 4 | `04_align.py` | lyrics + wavs → `phones.jsonl` (MFA or SOFA — below) | aligner |
| 4b | `04b_sofa_data.py` | phones → SOFA training data (optional) | pansori |
| 5 | `05_notes.py` | ROSVOT MIDI → `notes.jsonl` | aligner |
| 6 | `06_merge.py` | phones + notes → `transcriptions.csv` | pansori |
| 7 | `07_validate.py` | transcriptions → PASS/FAIL | pansori |
| 8 | `08_export.py` | re-cut at 44.1 kHz + fan out per speaker | pansori |

```sh
cd pansori/preprocess
conda activate aligner && python 04_align.py --set full --skip_align  # re-parse only, ~2 min
conda activate pansori
python 06_merge.py --set full && python 07_validate.py --set full     # PASS is mandatory
python 08_export.py --set full --out data/pansori                     # re-cut audio, ~20 min
```

**Never re-run `01_segment.py`.** It is not reproducible — run under a different env,
`librosa.effects.split` moves boundaries by 3–11 ms and every phoneme in the existing
`phones.jsonl` goes out of sync. If you need `wavs/`, rebuild them from the manifest:

```sh
python pansori/preprocess/01b_recut_wavs.py --set full   # 31,529 items, 4,574 actually re-cut
```

**Never binarize without a `07_validate.py` PASS.** A failed binarizer assert aborts the
run, it does not skip the item. If you touched `ipa_to_coda.csv` or `convert_phone`, run
`alignment/check_dict_agreement.py` before and after and decide on the number
(currently 89.59% at 1 syllable).

### Switching aligner: MFA ↔ SOFA

Stage 4's aligner is chosen by **one line** in `preprocess/align.yaml` (override with
`--aligner mfa|sofa`). Either backend writes `aligned{suffix}/*.TextGrid`, so nothing from
stage 06 onward changes. **The checked-in value is `aligner: sofa`.**

| | MFA | SOFA |
|---|---|---|
| Designed for | Speech (HMM-GMM) | Singing |
| Phonemes | IPA → `ipa_to_coda.csv` + `convert_phone` (89.59% agreement) | Reads `korean-coda.txt` directly, so **Coda comes out natively** |
| Fine-tuning | Effectively impossible | Trainable on this corpus (below) |
| Breath (AP) | None | Labelled via FoxBreatheLabeler (`breath_ap.enabled`) |
| env | `aligner` | `sofa` (shelled out to via `conda run`) |

Why switch: the two numbers in §2 — 34.03% of phonemes pinned to the 0.03 s floor, and
52.2% of corpus time being note-without-phoneme — are the same story told twice. The
aligner cannot handle pansori's sustained tones. The other arm, correcting the *lyrics*
against the changbon, was tried and measured; its verdict was
*"there is far more headroom in fixing the alignment than in changing the lyrics"*
([`docs/LYRICBOOK_VERDICT.md`](docs/LYRICBOOK_VERDICT.md)).

#### Installing SOFA (once)

Both envs hit the same two traps as `dsp`. Guard them immediately after creation:

```sh
git clone https://github.com/qiuqiao/SOFA ~/src/SOFA
conda create -y -n sofa python=3.10
echo 'export PYTHONNOUSERSITE=1' \
  > ~/miniconda3/envs/sofa/etc/conda/activate.d/no_user_site.sh   # mkdir -p first if needed
conda activate sofa
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r ~/src/SOFA/requirements.txt "setuptools<81"
cp dictionaries/korean-coda.txt ~/src/SOFA/dictionary/
python infer.py --help          # must get this far

# Only if you also want breath (AP) labels.
# FoxBreatheLabeler ships no requirements.txt — the list below is what
# textgrid_add_ap.py actually imports (trainCLS uses both lightning and pytorch_lightning).
git clone https://github.com/autumn-DL/FoxBreatheLabeler ~/src/FoxBreatheLabeler
conda create -y -n fbl python=3.10
echo 'export PYTHONNOUSERSITE=1' \
  > ~/miniconda3/envs/fbl/etc/conda/activate.d/no_user_site.sh
conda activate fbl
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install click "einops==0.6.1" "librosa<0.10.0" "matplotlib~=3.7.3" "numpy~=1.24.1" \
            "PyYAML~=6.0.1" textgrid tqdm "lightning>=2.0.0" pytorch-lightning \
            numba chardet "setuptools<81"
cd ~/src/FoxBreatheLabeler && python textgrid_add_ap.py --help    # must get this far
# Checkpoint (config.yaml + model_ckpt_steps_7000.ckpt):
#   https://github.com/autumn-DL/FoxBreatheLabeler/releases/latest
```

Then point `align.yaml`'s `sofa.root` / `sofa.ckpt` / `breath_ap.*` at where you put them.

#### Fine-tuning SOFA

There is no public Korean SOFA checkpoint (the published one is Mandarin
`opencpop-extension`). Bootstrap the training data from the existing MFA alignment, but
**do not hand it over wholesale** — given everything, SOFA simply learns MFA's 0.03 s floor.
Split by per-utterance floor rate and throw away the bad timings:

```sh
conda activate pansori
python pansori/preprocess/01b_recut_wavs.py --set full          # restore data/full/wavs
python pansori/preprocess/04b_sofa_data.py --set full --out ~/src/SOFA/data
#   floor_rate <= 0.15 → full_label (timings trusted too)
#   above             → weak_label (phoneme sequence only, timings discarded)
#   no alignment      → no_label   (audio only)
#   Tune with --max_floor_rate. It also prints the utterance count per threshold.
```

Measured over 26,955 utterances: median floor rate **0.35**. At `0.15` the split is
full 4,501 / weak 22,454 / no 4,574.

```sh
conda activate sofa && cd ~/src/SOFA
# point configs/binarize_config.yaml's data_folder at the --out above
python binarize.py

# Pre-trained: pretrained_multilingual_singing.zip from releases (v1.0.1, 359 MB)
# set model_name to korean_pansori in configs/train_config.yaml
python train.py -p ckpt/pretrained_multilingual_singing.ckpt
tensorboard --logdir=ckpt/
```

**Use `-p` even though the phoneme set differs.** `load_pretrained` in
`modules/task/forced_alignment.py:190-197` takes the backbone as-is and **re-initialises
only the head** when the vocab size differs. So the acoustic backbone is inherited from the
Mandarin checkpoint and only the Korean phoneme classifier is learned from scratch — there
is no reason to train from zero.

**Zero-shot inference is impossible.** The vocab is baked into the checkpoint
(`forced_alignment.py:309` looks up `self.vocab[ph]` directly) and the Mandarin vocab has no
Coda phonemes, so `--aligner sofa` before training dies with `KeyError`. Train first,
always.

#### Aligning the whole corpus with SOFA, then A/B

Put `aligner: sofa` and the trained `sofa.ckpt` in `align.yaml`, then:

```sh
conda activate aligner && cd pansori/preprocess
python 04_align.py --set full --aligner sofa --suffix _sofa

conda activate pansori && cd ../data && mkdir -p full_sofa && cd full_sofa
ln -s ../full/{segments.jsonl,notes.jsonl,spk_map.json,wavs} .
ln -s ../full/phones_sofa.jsonl phones.jsonl          # this is the only difference
cd ../../preprocess
python 06_merge.py --set full_sofa && python 07_validate.py --set full_sofa
python 08_export.py --set full_sofa --out data/pansori_sofa --wavs_from data/pansori

conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/make_configs.py --data data/pansori_sofa --name sofa
python pansori/eval/label_report.py --corpus data/pansori data/pansori_sofa \
       --html pansori/eval/out/label_report.html      # ★ this table is the verdict
```

For breath labels too, set `breath_ap.enabled: true` in `align.yaml` and run the same
sequence. Once `AP` appears in `ph_seq`, `make_configs.py` drops `merged_phoneme_groups`
automatically.

`alignment/check_dict_agreement.py` scores the IPA→Coda **conversion**, so it does not apply
to the SOFA arm (there is no conversion step). `eval/intelligibility.py` uses MFA as a
*measuring instrument*, so it stays on MFA regardless of this setting — that keeps it
comparable with earlier numbers.

## 9. Training

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0

python pansori/build_dictionary.py         # → dictionaries/korean-coda.txt
python pansori/make_configs.py             # → the generated layer in configs/pansori/

python scripts/binarize.py --config configs/exp/ac_v1.yaml
python scripts/binarize.py --config configs/exp/var_v1.yaml
python pansori/compute_spec_stats.py       # → configs/pansori/spec_stats.yaml

python pansori/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
python pansori/train.py --config configs/exp/var_v1.yaml --exp_name pansori_var_v1 --reset
```

**Spec statistics need a bootstrap** — binarize once with the defaults, then measure.
`spec_min`/`spec_max` are only read at training time, and `configs/pansori/acoustic.yaml`
layers `spec_stats.yaml` in as a base, so **neither re-binarizing nor regenerating configs
is needed** afterwards.

### Adding an experiment

Configs are split along three axes and reassembled by upstream's own `base_config` chain:

```
configs/pansori/acoustic.yaml      kind    — model, batch, phoneme set. Knows no corpus
configs/pansori/data_pansori.yaml  corpus  — datasets (23 speakers) + num_spk
configs/exp/ac_v1.yaml             experiment — hand-written
```

The first two are generated by `make_configs.py`; do not hand-edit them. Put what you want
to change **in the experiment file** — that way a condition that would otherwise live only
in a `--hparams` shell argument is recorded in git:

```yaml
# configs/exp/ac_big.yaml
base_config:
  - configs/pansori/acoustic.yaml
  - configs/pansori/data_snap.yaml     # this line changes the data

binary_data_dir: data/pansori_snap/binary_acoustic

max_updates: 50000                     # training conditions
num_ckpt_keep: 12
hidden_size: 512                       # model structure
```

**Later wins** (`utils/hparams.py:64-82`) — bases are applied in order, and the experiment
file's own keys win last. For a new corpus, only that one layer needs generating:

```sh
python pansori/make_configs.py --data data/pansori_snap --name snap
```

`pansori/train.py` is `scripts/train.py` plus wandb; with `--no-wandb` it behaves
identically. Upload a finished run's TensorBoard logs with `tb_to_wandb.py` (do **not** use
`wandb sync --sync-tensorboard` — it prefixes metric names with the directory, so runs never
overlay on the same chart):

```sh
pip install wandb        # not in the dsp env. Invoke the CLI as python -m wandb
python pansori/tb_to_wandb.py --exp pansori_ac_v1 --run-id pansori_ac_v1_tb
```

Variance runs also emit per-phone-class duration metrics (`dur_acc_onset`,
`floor_rate_pred`, and the `phone_dur_acc` figure) — how to read them is in
[`docs/EVAL.md`](docs/EVAL.md) §2.

## 10. Checkpoint selection

**The latest checkpoint is not the best one, and this bit is not optional.** Measured on
`pansori_ac_v1` / `pansori_var_v1`:

| Metric | Best step | Value |
|---|---|---|
| acoustic `validation/total_loss` | **28,000** | 0.17800 (last step, 100k, was 0.18834) |
| variance `validation/dur_loss` | **16,000** | 0.07797, rises thereafter |
| variance `validation/pitch_loss` | **32,000** | 0.00861, still falling when stopped |
| variance `validation/total_loss` | 20,000 | a compromise dominated by dur |

The heads disagree, so they are selected separately: pulling pitch from the 20k checkpoint
that `total_loss` would pick gives **25% worse f0 error** than 32k. That is why
`run_examples.sh` picks per head and runs variance twice — first `--predict dur`, then
`--predict pitch` over that output.

`num_ckpt_keep: 8` deletes older checkpoints, and acoustic's best (28k) had *already been
deleted* by the time the 100k run finished. The kept ones are copied to
`checkpoints/best/{exp}_step{N}.ckpt`; `best_ckpt.py` selects and `run_examples.sh` restores
them before running.

## 11. Inference

```sh
bash pansori/examples/run_examples.sh          # ladder + score synthesis → examples/out/
python pansori/examples/measure.py             # scores against the original
python pansori/examples/seed_spread.py         # seed-to-seed variation
python pansori/examples/build_report.py        # explanation + audio, one page
```

`run_examples.sh` builds a **4-rung ladder** for each held-out utterance. Each rung leaves
the model more to invent, so when the output is bad you can tell which stage is at fault.

| | Given | Model produced |
|---|---|---|
| `1_orig` | — | — (the original recording) |
| `2_resynth` | `ph_dur` + `f0` | mel |
| `3_pitch` | `ph_dur` | `f0` + mel |
| `4_full` | notes + lyrics | `ph_dur` + `f0` + mel |

To add a song, drop one file into `examples/scores/` in the format shown in §1.

**Both models use a reflow sampler.** Changing nothing but the seed moves f0 RMSE by ±17%
and mel L1 by ±8%. If the gap between two tracks is smaller than `seed_spread.py`'s
variation, there is no gap.

## 12. Evaluation

Whether the words are intelligible cannot be measured by `compare/`'s pitch and spectral
metrics. Three rulers fill that hole — details in [`docs/EVAL.md`](docs/EVAL.md).

```sh
python pansori/eval/label_report.py                  # C. label diagnosis, seconds
python pansori/eval/intelligibility.py --stage all   # B. intelligibility, 40 samples ~15 min
                                                     # A. training-time metrics go to wandb
```

**Start with C.** Knowing what the model was shown is faster than suspecting the model —
22.3% of this corpus's phonemes sit on MFA's 0.03 s floor.

For old-vs-new model comparison, see [`compare/README.md`](compare/README.md).

---

## 13. Current state

Both models are trained; inference and evaluation paths are open end to end.

- acoustic and variance trained from scratch → `checkpoints/pansori_{ac,var}_v1`
- Corpus: 26,955 items, train 27.8 h / valid 1.08 h, all 23 speakers have training data
- 1-syllable dictionary agreement improved 87.92% → 89.59% by fixing three label defects
- Spec statistics measured; the ceiling (**+2.838**) really does exceed openvpi's default
- The f0-flattening question was answered and **closed** — systematic flattening does not
  reproduce across seeds
- SOFA is trained and wired behind the stage-4 seam; the A/B against MFA is the open item

Open work, in priority order, is in [`docs/NEXT.md`](docs/NEXT.md). The headline items:
cut `max_updates` (acoustic wasted 72k steps past its optimum), turn on augmentation
(speaker skew is severe — the top 3 are 36% of the data), and run the aligner A/B to its
verdict table.

## 14. Documentation index

| Doc | What it holds |
|---|---|
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | Source audio → synthesised voice, the full stage-by-stage record |
| [`docs/TRAINING.md`](docs/TRAINING.md) | What happens inside one training step |
| [`docs/EVAL.md`](docs/EVAL.md) | Phoneme and pronunciation measurement — the three rulers |
| [`docs/GOTCHAS.md`](docs/GOTCHAS.md) | Config and data traps. **Start here when stuck** |
| [`docs/LYRICS_AB.md`](docs/LYRICS_AB.md) | The lyrics A/B experiment: procedure and result |
| [`docs/PORTING.md`](docs/PORTING.md) | How this came across from MoonInTheRiver to openvpi |
| [`docs/NEXT.md`](docs/NEXT.md) | What is left to do |
| [`docs/SEGMENTATION_VERDICT.md`](docs/SEGMENTATION_VERDICT.md) | Why 6 seconds — read before changing segmentation |
| [`docs/LYRICBOOK_VERDICT.md`](docs/LYRICBOOK_VERDICT.md) | Verdict on correcting lyrics against the changbon |
| [`docs/SIGIMSAE_PLAN.md`](docs/SIGIMSAE_PLAN.md) | The sigimsae modelling plan, including its abandoned stages |
| [`docs/VIBRATO_VERDICT.md`](docs/VIBRATO_VERDICT.md) | Why the sine-vibrato model was rejected |
| [`compare/README.md`](compare/README.md) | Quantitative comparison against the previous model |
| [`server/README.md`](server/README.md) | The `run.sh` stage runner |
| [`alignment/FIELD_CHECK.md`](alignment/FIELD_CHECK.md) | Generated field-validation report (legacy schema) |

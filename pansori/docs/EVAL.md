# Phoneme and pronunciation measurement — the three rulers

The tools that measure whether the synthesis is **intelligible**. The full pipeline record
is in [`PIPELINE.md`](PIPELINE.md); the command summary is in [`../README.md`](../README.md).

---

When the synthesis pronounced things badly, **there was no ruler in this repository that
could measure it.** All 22 of `compare/`'s metrics are pitch or spectrum
(`compare/stats.py:40-60`), and even the closest one, `voiced_agree`, only asks "should this
frame make sound at all" — so dropping a consonant entirely does not move it (0.95–0.97 on
the hold-out). Every verdict reached so far, [lyrics A/B](LYRICS_AB.md) included, was
reached without being able to see pronunciation. The three tools below fill that hole.

| Ruler | Question it answers | Where it shows | Cost |
|---|---|---|---|
| **A** `variance_task.py` | Does the model predict phoneme durations accurately? | wandb, every validation | Free (automatic) |
| **B** `eval/intelligibility.py` | Is the rendered audio actually intelligible? | Console, wandb | 40 samples, ~15 min |
| **C** `eval/label_report.py` | How broken are the training labels themselves? | Console, HTML | Seconds |

**Start with C.** Knowing what the model was shown is faster than suspecting the model.

## 1. C — label diagnosis

```sh
python pansori/eval/label_report.py                                    # data/pansori
python pansori/eval/label_report.py --corpus data/pansori data/pansori_snap \
       --html pansori/eval/out/label_report.html

# Aligner A/B (MFA vs SOFA) — this table is the verdict. See README §8
python pansori/eval/label_report.py --corpus data/pansori data/pansori_sofa \
       --html pansori/eval/out/label_report.html
```

Needs no model and no GPU. It reads only `ph_dur` out of `transcriptions.csv`.

**Currently known values (compare against these next time you run it):**

| | `data/pansori` | `data/pansori_snap` |
|---|---:|---:|
| Phonemes at exactly `0.030000` s | **22.3%** | 23.7% |
| ≤30 ms | 26.4% | 27.9% |
| ≤3 frames (the same ruler the training metrics use) | 30.7% | 32.2% |
| Onsets ≤30 ms | 34.8% | 36.9% |
| Codas ≤30 ms | 12.0% | 13.3% |
| Fraction of time that is silence | 21.7% | 24.0% |

22.3% piling up at `0.030000` is not a distribution, it is a **spike** — 3.2× the
next-most-common value (0.04). 0.03 is MFA's minimum phoneme duration, so a phoneme sitting
there was not *measured* to be that long; **the aligner failed to find acoustic evidence and
pinned it to the minimum**. And they cluster on onset plosives (`pp` 48% · `tt` 46% ·
`b` 42% · `g` 42%). At hop 512, 30 ms is 2.6 frames, so the model gets two or three frames
for those consonants — and even those may be in the wrong place.

`LYRICBOOK_VERDICT.md` §7-4 already pointed at the same thing —
*"there is far more headroom in fixing the alignment itself than in changing the lyrics."*

## 2. A — training-time wandb metrics

These attach automatically when variance is trained with `pansori/train.py` (README §9).
The config key `task_cls: pansori.variance_task.PansoriVarianceTask` does it —
**upstream is left alone and only the subclass is swapped in.**

wandb charts (the `metrics/` namespace):

| Key | How to read it |
|---|---|
| `dur_acc_onset` / `dur_acc_coda` / `dur_acc_vowel` / `dur_acc_glide` / `dur_acc_sp` | Per-class duration accuracy (±20%). **If only onset is low, consonants are being smeared** |
| `dur_mae_onset_ms` / `dur_mae_vowel_ms` | Per-class absolute error. Onsets are mostly 2–4 frames, so their accuracy can sit pinned at the floor and not move; this still moves underneath it |
| `floor_rate_pred` | Fraction of predicted durations that are ≤3 frames |
| `floor_rate_gt` | Fraction of the **labels** that are ≤3 frames. A constant, independent of the model — the baseline, and a cross-check |
| `floor_rate_pred_onset` | Predicted floor rate, onsets only |
| `dur_acc_macro` | Macro average of the 35 phonemes' accuracies. Read next to upstream's `ph_dur_acc` (micro) to see whether only the common phonemes are being hit |
| `phone_dur_acc` (figure) | Per-phoneme accuracy and floor-rate bars, grouped by class |

**Overlay `floor_rate_pred` on `floor_rate_gt`.** If the prediction tracks the labels, the
model has learned the labels' disease as-is; if it is much lower, it is behaving differently
from the labels. Which of those is correct is not something this metric answers — that is
§3's job.

**Two traps.**

- **Different scope.** Upstream's `ph_dur_acc` / `rhythm_corr` sit behind the
  `num_valid_plots` (10) gate, so they score only the **first 10 utterances**
  (`variance_task.py:253,260`). The new `dur_acc_*` use `num_valid_metric_items: 0` and see
  the **whole validation set**. Do not subtract one from the other. The original two are
  deliberately left as they are, to keep the comparison line with earlier runs.
- **The unit is frames.** The model only sees `ph_dur` after binarize quantises it to mel
  frames, so this measures in frames too. The floor is 3 frames (34.8 ms), which means these
  line up with §1's **`≤3 frames` row, not its `≤30 ms` row**. Cross-check, measured: the
  validation labels' ≤3-frame rate is 30.6899%, and training recorded `floor_rate_gt` at
  30.6974% — a difference of 1 item in 13,379 (a rounding tie).

## 3. B — the intelligibility evaluation

Measures each rung of the ladder to isolate **where it collapses**. It is split into stages
because it needs three different environments.

```sh
conda activate dsp     && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/eval/intelligibility.py --stage render --n 40 --seeds 3

conda activate pansori && export PYTHONPATH=.
python pansori/eval/intelligibility.py --stage asr    --n 40 --seeds 3

conda activate aligner && export PYTHONPATH=.          # the mfa CLI must be on PATH
python pansori/eval/intelligibility.py --stage mfa    --n 40 --seeds 3

conda activate dsp     && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/eval/intelligibility.py --stage score  --n 40 --seeds 3 \
       --wandb --wandb-group intelligibility
```

Each stage skips rows it has already produced, so it is safe to interrupt and restart.
Rendering is handed straight to `compare/render_new.py` (there is not a second copy of the
render logic). The sample is `compare/items.py`'s stratified sample, and
`assert_held_out()` prevents training-data leakage.

**Four traps — miss these and you will read the table backwards.**

1. **On the whisper side, `1_orig` is a null control, not a ceiling.** The CER reference
   text is `data/full/lyrics.jsonl`, which is itself Whisper's transcription of these very
   recordings (`03_transcribe.py`). Re-transcribing the original compares it against itself,
   so the CER comes out near zero. **That does not mean Whisper understands pansori well.**
   If it is *not* near zero, Whisper's decoding is not reproducing, and every number after
   it is void. The real ceiling for synthesis is **`0_vocoder`** (GT mel → vocoder). On the
   MFA side it is the other way round: `1_orig` is a proper ceiling, because the aligner
   receives the audio and the text independently.
2. **`spn` is not used for comparing rungs.** It went in on the theory that "MFA emits spn
   when there is no consonant in the sound", but measured, **its value was identical across
   all five rungs of the same utterance.** `spn` is emitted for out-of-vocabulary words, so
   it is a property of the **lyrics**, not the audio. It is printed once as a reference
   figure for how much OOV the sample contains. `phone_match` is near-constant for the same
   reason — the phoneme sequence is fixed by the lyrics we supplied.
3. **The two metrics that matter on the MFA side are `floor_rate` and `boundary_mae_ms`.**
   `floor_rate` applies **the same ruler §1 applies to the labels** to the synthesis, so the
   two tables can be read side by side.
4. **Whisper hallucinates sounds that are not there.** That is why both backends are read
   together. If Whisper produces a plausible sentence but MFA's `floor_rate` is high, that
   sentence is imagined.

**Reading table — which rung falls, and what is to blame.**

| Symptom | Culprit | What to look at next |
|---|---|---|
| `1_orig` CER is not near zero | The measurement itself is unstable | Stop here |
| `0_vocoder` is already bad | Vocoder or extractor | No need to read the rungs below it |
| It already collapses at `2_resynth` | **Labels (MFA alignment) or the acoustic model** | §1's onset floor rate |
| `2_resynth` is fine but `4_full` collapses | **Duration prediction** | §2's `dur_acc_onset` |
| Every rung is uniformly bad | Phoneme set or dictionary | `dictionaries/korean-coda.txt` |

**First observation (n = 3, one seed — preliminary, direction only).** Null control passed
at 0.000. CER goes `0_vocoder` 0.33 → `2_resynth` 1.00 → `4_full` 1.69, and the MFA floor
rate goes `1_orig` 30% → `2_resynth` 40% → `4_full` 65%. So **it already collapses at the
`2_resynth` rung** — and that is with GT `ph_dur` and GT f0 both supplied. By the reading
table, that points at the labels and the acoustic model, not at duration prediction. With
only 3 samples, re-measure at `--n 40 --seeds 3` before calling it.

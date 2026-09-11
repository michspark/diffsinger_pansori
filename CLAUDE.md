# CLAUDE.md

Pansori (판소리) fine-tuning of openvpi/DiffSinger. Branch `feat/pansori`.

This file is a map and a set of rules, not an explanation. The explanations already
exist and are current — **read the relevant doc before acting on its subject.**

## Read first

| Question | Doc |
|---|---|
| How do I run anything? | `pansori/README.md` |
| What happens at each stage, and why? | `pansori/docs/PIPELINE.md` |
| What happens inside one training step? | `pansori/docs/TRAINING.md` |
| How is pronunciation measured? | `pansori/docs/EVAL.md` |
| **Something broke.** | `pansori/docs/GOTCHAS.md` — start here |
| What is in flight? | `pansori/docs/NEXT.md` |
| How did this get ported from MoonInTheRiver? | `pansori/docs/PORTING.md` |
| Old vs new model, quantitatively | `pansori/compare/README.md` |

**Settled questions — read the verdict before re-proposing the idea:**
`pansori/docs/LYRICBOOK_VERDICT.md` (snapping lyrics to the 창본 makes alignment *worse*;
❌ dead end) · `pansori/docs/VIBRATO_VERDICT.md` (the sine vibrato model fits unrelated
notes as well as real ones; ❌ dead end, Phases 4–6 halted) · `pansori/docs/SEGMENTATION_VERDICT.md`
(MFA is usable, but only at ≤6 s utterances) · `pansori/docs/NEXT.md` §C5 (f0 flattening
did not reproduce; item closed).

## Hard rules

1. **Never modify an upstream file.** Fork code lives only in `pansori/`,
   `configs/pansori/`, `configs/exp/`, `dictionaries/korean-coda.txt`. This is what
   keeps rebasing onto openvpi affordable. Add behaviour from outside instead — the
   three patterns already in use:
   - swap `task_cls` to a subclass (`pansori.variance_task.PansoriVarianceTask`)
   - wrap the entry point (`pansori/train.py` is `scripts/train.py` + wandb)
   - patch at a seam (`pansori/train.py` patches `SummaryWriter`, because the logger is
     hard-wired at `basics/base_task.py:427`)
2. **`configs/pansori/*.yaml` is generated** by `pansori/make_configs.py`. Fix the
   generator, never its output. `merged_phoneme_groups` in particular is decided from
   `transcriptions.csv` — never add or remove it by hand.
3. **Never re-run `01_segment.py`.** It does not reproduce: in a different env
   `librosa.effects.split` moves boundaries 3–11 ms and invalidates every existing
   `phones.jsonl`. To get the wavs back: `python pansori/preprocess/01b_recut_wavs.py --set full`.
4. **Never binarize without a `07_validate.py` PASS.** The binarizer asserts inside
   worker processes — it kills a 40-minute job midway, it does not skip the item.
5. **Never push to `upstream`** — it is MoonInTheRiver. Ours is `pansori`
   (michspark/diffsinger_pansori); ask before pushing there. `origin` points at
   michspark/DiffSinger, **which does not exist** — a push to it just fails.
   **Never `git merge` or `git remote add` the moonInRiver repo**: the two histories
   have no common ancestor.

Rules 1–5 are enforced by hooks in `.claude/hooks/`. A refusal will name the alternative.

## Environments

Never use `base`, and never the old `diffsinger` env (torch 2.0.1, pytorch_lightning 0.7.1).

| env | Python | used for |
|---|---|---|
| `pansori` | 3.10 | preprocess 01–03, 06–08 |
| `aligner` | 3.14 | 04–05 (MFA 3.4.1 + `korean_mfa`) |
| `dsp` | 3.10 | **openvpi proper** — binarize / train / infer / eval |
| `sofa`, `fbl` | 3.10 | optional aligner swap; shelled out to via `conda run` |

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
```

`PYTHONNOUSERSITE=1` is **mandatory**, not hygiene: a stale tensorflow in `~/.local`
kills the tensorboard import inside protobuf, and its numpy makes pip report
requirements as already satisfied so they never install. `setuptools<81` is pinned
because `librosa<0.10` still needs `pkg_resources`. `torchaudio` is missing from
`requirements.txt` (upstream omission) and RMVPE needs it.

## Commands

```sh
python pansori/build_dictionary.py                            # → dictionaries/korean-coda.txt
python pansori/make_configs.py [--data DIR --name NAME]       # → configs/pansori/
python scripts/binarize.py --config configs/exp/ac_v1.yaml
python pansori/compute_spec_stats.py                          # → configs/pansori/spec_stats.yaml
python pansori/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
tensorboard --logdir checkpoints
python pansori/examples/best_ckpt.py pansori_var_v1 --metric validation/pitch_loss
bash pansori/examples/run_examples.sh                         # 4-rung ladder + score synthesis
```

`pansori/train.py` is `scripts/train.py` plus wandb; `--no-wandb` makes them identical.
Spec stats bootstrap: binarize once with defaults, then measure. `spec_min`/`spec_max`
are used only at training time, so **no re-binarize and no config regeneration** —
`configs/pansori/acoustic.yaml` already carries `spec_stats.yaml` as a base layer.

## Config layering

Three axes, reassembled by upstream's own `base_config` chain (`utils/hparams.py:64-82`,
**later wins**):

```
configs/pansori/{acoustic,variance}.yaml   kind    — generated; knows nothing of a corpus
configs/pansori/data_{name}.yaml           corpus  — generated; datasets + num_spk
configs/exp/*.yaml                         run     — handwritten, 9–14 lines
```

Put a run's conditions in its experiment file, **not in `--hparams`**, so they live in
git rather than in shell history. `binary_data_dir` is the one key depending on both
axes, so the experiment file always states it.

Two traps in upstream's loader:
- **`--reset` resets the config only.** Without it, the saved `checkpoints/<exp>/config.yaml`
  overrides your edited file (`utils/hparams.py:96-97`). *With* it, `trainer.fit` still
  resumes from the newest checkpoint — to truly restart, empty the directory.
- `--hparams k=v` is top-level only, splits on `,` and `=`, and `eval()`s unknown keys.

## Measurement discipline

- **Never judge a model from a single draw.** Both models are reflow samplers; seed alone
  moves f0 RMSE ±17% and mel L1 ±8%. A difference smaller than `pansori/examples/seed_spread.py`'s
  spread is not a difference.
- **The latest checkpoint is not the best.** `DsModelCheckpoint` is built with
  `monitor='step', mode='max'` (`basics/base_task.py:411-418`), so it keeps the N most recent
  by step — validation loss has no influence on retention. Acoustic's best was step 28,000 of
  100,000 and `num_ckpt_keep: 8` had already deleted it; preserved copies live in
  `checkpoints/best/`. Heads differ too: variance `dur_loss` bottoms at 16k, `pitch_loss` at
  32k, so **never select on `total_loss`** when a model has several heads.
- Look at the **labels before the model** (`pansori/eval/label_report.py`): 22.3% of phones in
  this corpus sit exactly on MFA's 0.03 s floor. Knowing what the model was shown is faster
  than suspecting the model.
- `pansori/compare/smoke.py` gates everything downstream of it. If it fails, every number
  after it is void.

## Working agreements

- **Commit each completed unit of work without being asked.** Subject: capitalized
  imperative, no conventional-commit prefix, no PR number, stating intent or effect rather
  than the mechanical change — `Split the config along the axes that actually vary`. Body:
  prose explaining *why*, with measured numbers and `file:line` citations where they apply.
- **Ask before pushing.** `pansori` is the only remote that is both ours and real.
- Docs and code comments in `pansori/` are English. The Korean that remains is data —
  speaker names, the dictionary's syllables, score lyrics, and the orthography examples the
  phoneme rules are argued from — plus the prose rendered into the generated reports
  (`compare/report.py`, `compare/plots.py`, `examples/build_report.py` and the two HTML
  templates). Leave those alone; write everything else in English.
- Cite measured numbers, never estimates. Reports re-read their numbers from artifacts
  rather than transcribing them.

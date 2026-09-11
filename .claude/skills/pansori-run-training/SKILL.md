---
name: pansori-run-training
description: Use when starting, resuming, monitoring or debugging a DiffSinger training run in this fork — binarize, train, TensorBoard/wandb, choosing a checkpoint, or triaging a crash, OOM, KeyError or a loss that stopped improving.
---

# Running training

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
```

`PYTHONNOUSERSITE=1` is mandatory — a stale tensorflow in `~/.local` kills the tensorboard
import inside protobuf.

## Order

```sh
python pansori/build_dictionary.py                     # → dictionaries/korean-coda.txt
python pansori/make_configs.py                         # → configs/pansori/
python scripts/binarize.py --config configs/exp/ac_v1.yaml
python scripts/binarize.py --config configs/exp/var_v1.yaml
python pansori/compute_spec_stats.py                   # → configs/pansori/spec_stats.yaml
python pansori/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
```

**Spec stats bootstrap.** Binarize once with defaults, then measure. `spec_min`/`spec_max` are
used only at training time and `configs/pansori/acoustic.yaml` already carries
`spec_stats.yaml` as a base layer, so **no re-binarize and no config regeneration**.

**Prerequisites.** `07_validate.py` must PASS first (the binarizer asserts inside workers — it
kills the job, it does not skip). Three downloads must exist: `checkpoints/rmvpe/`,
`checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/`, `checkpoints/vr/`. `vr` (HN-SEP) is
needed **even with breathiness, voicing and tension all off** — acoustic binarize constructs a
`DecomposedWaveform` unconditionally. `bash pansori/server/run.sh setup` fetches all three.

`pansori/train.py` is `scripts/train.py` plus wandb; `--no-wandb` makes them identical. The
phone-class duration metrics come from `task_cls: pansori.variance_task.PansoriVarianceTask`,
which `make_configs.py` renders, and you read them in wandb.

## The two `--reset` traps

- **Without `--reset`**, the saved `checkpoints/<exp>/config.yaml` overrides the config file you
  just edited (`utils/hparams.py:96-97`). Your edit does nothing.
- **With `--reset`**, the config is rebuilt but `trainer.fit` **still resumes from the newest
  checkpoint** (`basics/base_task.py:462`). `--reset` resets the config, not the run. To truly
  start over, the directory has to be empty — and ask before emptying one.

## Monitoring

```sh
tensorboard --logdir checkpoints          # add --reload_multifile=true under DDP
python pansori/tb_to_wandb.py --exp pansori_ac_v1 --run-id pansori_ac_v1_tb
```

Do **not** use `wandb sync --sync-tensorboard`: it namespaces every metric by the directory
the tfevents file sits in, so a backfilled `latest/validation/total_loss` never lands on the
same chart as a live run's `validation/total_loss`. wandb also will not reuse a deleted run id
— pass a fresh `--run-id` and keep the display name via `--exp`.

## Choosing a checkpoint — the part that is usually got wrong

`DsModelCheckpoint` is built with `monitor='step', mode='max'` (`basics/base_task.py:411-418`),
so **it keeps the N most recent, not the N best.** Validation loss has no influence on
retention whatsoever.

- Acoustic's best was **28,000 of 100,000** steps; `num_ckpt_keep: 8` had already deleted it by
  the time the run finished.
- **Heads bottom out at different steps.** Variance `dur_loss` at 16k, `pitch_loss` at 32k.
  Selecting on `total_loss` (20k) gives **25% larger f0 error** than picking pitch at 32k.
  `run_examples.sh` therefore runs variance twice: `--predict dur`, then `--predict pitch` on
  that output.
- Copy anything good to `checkpoints/best/` *before* the run ends.

```sh
python pansori/examples/best_ckpt.py pansori_var_v1 --metric validation/pitch_loss
```

**Stopping rule:** stop when validation fails to set a new minimum three times running
(= 12,000 steps at `val_check_interval: 4000`). Do not trust `max_updates`; acoustic burned
3 GPU-hours past its optimum. Raise `num_ckpt_keep` at the same time you lower `max_updates`.

One notch is noise — validation redraws `t` and the noise each time, so ±0.005 means nothing
(28,000 at 0.17800 and 64,000 at 0.17988 are indistinguishable).

## Triage

| Symptom | Cause |
|---|---|
| `KeyError` on a config key at startup | key read unconditionally but absent from the base chain; e.g. `permanent_ckpt_start` is in `acoustic.yaml`/`variance.yaml` but not `base.yaml` |
| Binarize dies on a phoneme assert | phone set vs labels disagree — check `merged_phoneme_groups` against `transcriptions.csv`; never hand-edit it |
| Binarize dies before any progress | missing `checkpoints/vr/` (HN-SEP), or no `07_validate.py` PASS |
| Training hard-fails at fit start | stale or missing `binary_data_dir` — `basics/base_task.py:444-462` copies `spk_map.json`, `lang_map.json` and the dictionaries out of it |
| Config edit had no effect | missing `--reset` |
| OOM | `max_batch_frames` in `make_configs.py` (50,000 acoustic / 80,000 variance). Raising it means retuning lr, which costs comparability with upstream defaults |
| Low register is silent | `f0_min` is not 55 |
| Loss diverges / target out of range | `spec_max` left at upstream's `[0]`; measured max here is +2.838 |

Never delete a checkpoint directory to "start clean" without asking. `checkpoints/best/` holds
the only surviving copies of several optima.

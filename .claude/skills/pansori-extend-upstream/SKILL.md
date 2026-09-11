---
name: pansori-extend-upstream
description: Use when adding a model component, metric, logger, callback or CLI behaviour to DiffSinger in this fork — especially when the obvious implementation would mean editing a file under modules/, training/, basics/, utils/, preprocessing/, inference/ or scripts/.
---

# Extending without touching upstream

The rule: **not one line of an upstream file is modified.** Fork code lives only in
`pansori/`, `configs/pansori/`, `configs/exp/`, `dictionaries/korean-coda.txt`. The reason is
economic — it is what keeps rebasing onto openvpi cheap — and it holds even when editing
upstream would be two lines and the workaround is twenty.

A hook refuses these edits. If you are reading this because one fired, pick a pattern below.

## Three patterns, in order of preference

### 1. Subclass, and select it by config string

Upstream imports classes from dotted strings (`task_cls`, `binarizer_cls`, `vocoder`,
`optimizer_cls`), so a subclass is reachable with no upstream change at all.

`pansori/variance_task.py` adds phone-class duration metrics this way:

```yaml
task_cls: pansori.variance_task.PansoriVarianceTask
num_valid_metric_items: 0      # 0 = whole validation set
```

Rendered by `make_configs.py`. Note what it *does not* do: upstream's `rhythm_corr` and
`ph_dur_acc` are left in place deliberately, so the comparison line to earlier runs survives.
Adding a metric is not a reason to remove one.

### 2. Wrap the entry point

`pansori/train.py` is `scripts/train.py` plus wandb. It re-creates the twelve lines of
`sys.path` and env setup rather than importing them, and `--no-wandb` makes it byte-for-byte
equivalent in behaviour. `set_hparams()` parses with `parse_known_args` (`utils/hparams.py:47`),
so extra flags pass straight through.

### 3. Patch at a seam

When neither works, patch the object upstream reaches for, not the file that reaches for it.

`pansori/train.py` wanted validation audio and mel figures in wandb. Subclassing the logger
would not work: the trainer's logger is hard-wired to `DsTensorBoardLogger`
(`basics/base_task.py:427`), and the audio and figures never pass through the Lightning logger
at all — they go straight to the `SummaryWriter`
(`training/acoustic_task.py:215-242`, `training/variance_task.py:314-335`). So a logger
subclass would capture the scalars and silently drop everything worth looking at. Instead
`wandb.init(sync_tensorboard=True)` patches the `SummaryWriter` **class**; Lightning builds its
writer lazily on first log, long after that runs.

The same discipline is applied to third-party trees: `pansori/sofa_train.py` intercepts SOFA's
`pl.Trainer` constructor to add a `ModelCheckpoint` rather than editing SOFA's files, "because
intercepting the constructor is what survives a rebase".

Other worked examples: `--spk` cannot take a Hangul speaker name
(`utils/infer_utils.py:56` parses `[0-9A-Za-z_-]+`), so `make_ds.py` writes the `.ds` file's
`spk_mix` directly — the same dict `--spk` would have filled.

## Checklist for a new module

- [ ] Does the **twin pipeline** need it too? Acoustic and variance are parallel and almost
      every file has a counterpart. Say explicitly if only one arm is in scope.
- [ ] Does `deployment/modules/` need to stay in step? ONNX export uses traceable
      re-implementations; a change that only lands in the training path will pass tests that
      do not exist and fail at export.
- [ ] Is it reachable by config string, so no upstream file names it?
- [ ] Does the docstring say **why**, citing upstream `file:line`? Every module in `pansori/`
      does. That is what makes the next rebase reviewable.
- [ ] Smoke test: construct it standalone, populate `hparams` by hand, run one forward pass.
      There is no test suite in this repo — verification is empirical.

## When there is genuinely no seam

Say so, name the upstream lines that would have to change, and ask. Do not edit quietly and do
not build something elaborate to avoid a conversation. A deliberate, documented exception is
recoverable at rebase time; an undocumented one is what makes rebases expensive.

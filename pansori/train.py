#!/usr/bin/env python3
"""scripts/train.py + wandb, without touching a single upstream line.

The trainer's logger is hard-wired to DsTensorBoardLogger (basics/base_task.py:427)
and the validation audio and mel figures do not go through the Lightning logger at
all -- they are written straight to the SummaryWriter (training/acoustic_task.py:215-242,
training/variance_task.py:314-335). So subclassing the logger would capture the
scalars and silently drop everything that is actually worth looking at.

wandb.init(sync_tensorboard=True) patches the SummaryWriter class instead. Lightning
builds its writer lazily on the first log, which is long after this module runs, so
the patch catches scalars, audio and figures alike and no upstream file changes.

Everything else mirrors scripts/train.py exactly. set_hparams() parses with
parse_known_args (utils/hparams.py:47), so the --wandb-* flags below pass through it
untouched.

    PYTHONPATH=. python pansori/train.py --config configs/exp/ac_snap.yaml \
      --exp_name pansori_ac_snap --reset \
      --hparams max_updates=50000,num_ckpt_keep=12 \
      --lyrics-source snapped --wandb-group lyrics-ab

--no-wandb makes this identical to scripts/train.py.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision

from utils.hparams import set_hparams, hparams  # noqa: E402


def parse_wandb_args():
    """Our own flags. Unknown args (--config, --exp_name, ...) are left for set_hparams."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--wandb-project', default='pansori')
    parser.add_argument('--wandb-group', default=None,
                        help='group two runs as one experiment (e.g. lyrics-ab)')
    parser.add_argument('--wandb-tags', default='',
                        help='comma-separated')
    parser.add_argument('--wandb-name', default=None,
                        help='defaults to --exp_name')
    parser.add_argument('--lyrics-source', default=None,
                        choices=['whisper', 'snapped'],
                        help='the variable that distinguishes this run; recorded in the wandb config to filter on')
    parser.add_argument('--no-wandb', action='store_true')
    args, _ = parser.parse_known_args()
    return args


def start_wandb(args):
    """Init wandb before any SummaryWriter exists. Returns the run, or None."""
    if args.no_wandb:
        return None

    import wandb

    exp_name = hparams['exp_name']
    work_dir = root_dir / 'checkpoints' / exp_name
    work_dir.mkdir(parents=True, exist_ok=True)

    config = dict(hparams)
    if args.lyrics_source:
        config['lyrics_source'] = args.lyrics_source

    tags = [t for t in args.wandb_tags.split(',') if t]
    if args.lyrics_source:
        tags.append(f'lyrics:{args.lyrics_source}')

    # id = exp_name and resume='allow' because training resumes from the highest-step
    # checkpoint on its own (basics/base_task.py:471). A re-run after a crash has to
    # land in the same wandb run or the curve breaks in two.
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or exp_name,
        id=exp_name,
        resume='allow',
        group=args.wandb_group,
        tags=tags or None,
        dir=str(work_dir),
        config=config,
        sync_tensorboard=True,
    )
    print(f'| wandb: {run.url}')
    return run


def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)

    task_cls.start()


def main():
    wandb_args = parse_wandb_args()

    set_hparams()
    if not hparams['nccl_p2p']:
        print("Disabling NCCL P2P")
        os.environ['NCCL_P2P_DISABLE'] = '1'

    run = None
    try:
        run = start_wandb(wandb_args)
    except Exception as exc:            # wandb must never be the reason a run dies
        print(f'⚠ wandb init failed; logging to TensorBoard only: {exc}')

    try:
        run_task()
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    main()

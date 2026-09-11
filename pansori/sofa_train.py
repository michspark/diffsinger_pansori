#!/usr/bin/env python3
"""SOFA training + wandb + keeping the best checkpoint. Not one line of the SOFA
repository is edited.

SOFA's train.py is missing two things:

  1. **It specifies no logger.** lightning's default TensorBoardLogger attaches,
     so patching the SummaryWriter class via `wandb.init(sync_tensorboard=True)`
     carries scalars and figures across unchanged -- the same trick
     pansori/train.py uses on upstream.
  2. **There is no ModelCheckpoint callback.** lightning's default callback keeps
     only the last one, so once the validation minimum passes it is gone. That is
     the trap docs/NEXT.md A1 hit on the previous training run (acoustic's best
     was 28k and num_ckpt_keep deleted it).

So pl.Trainer is wrapped to slot in a ModelCheckpoint watching
`valid/total_loss`. Intercepting the constructor rather than editing SOFA's files
is what keeps rebasing affordable.

    conda activate sofa
    PYTHONNOUSERSITE=1 python ~/Desktop/DiffSinger/pansori/sofa_train.py \
        -p ckpt/pretrained_multilingual_singing/v1.0.0_multilingual_singing.ckpt

With --no-wandb it is equivalent to calling SOFA's train.py directly.
"""

import argparse
import os
import pathlib
import sys

SOFA_ROOT = pathlib.Path(os.environ.get("SOFA_ROOT", "~/src/SOFA")).expanduser()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--pretrained_model_path", "-p", default=None)
parser.add_argument("--config_path", "-c", default="configs/train_config.yaml")
parser.add_argument("--data_folder", "-d", default="data")
parser.add_argument("--resume", "-r", action="store_true")
parser.add_argument("--wandb-project", default="pansori")
parser.add_argument("--wandb-group", default="aligner-ab")
parser.add_argument("--wandb-name", default="sofa_korean_pansori")
parser.add_argument("--save-top-k", type=int, default=5)
parser.add_argument("--no-wandb", action="store_true")
args, _ = parser.parse_known_args()

os.chdir(SOFA_ROOT)                 # SOFA's relative modules.* imports depend on cwd
sys.path.insert(0, str(SOFA_ROOT))

import lightning as pl              # noqa: E402
import yaml                         # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402

config = yaml.safe_load(open(args.config_path, encoding="utf-8"))

if not args.no_wandb:
    import wandb
    wandb.init(
        project=args.wandb_project, group=args.wandb_group, name=args.wandb_name,
        job_type="aligner-train", sync_tensorboard=True,
        config={"stage": "sofa-align", "model_name": config["model_name"],
                "pretrained": args.pretrained_model_path,
                "total_steps": config["optimizer_config"]["total_steps"]},
    )

# Keeps the validation minimum alive. save_last is kept too so --resume is not broken.
_Trainer = pl.Trainer


class TrainerWithBestCkpt(_Trainer):
    def __init__(self, *a, **kw):
        ckpt_dir = pathlib.Path("ckpt") / config["model_name"] / "best"
        kw["callbacks"] = list(kw.get("callbacks") or []) + [ModelCheckpoint(
            dirpath=str(ckpt_dir), monitor="valid/total_loss", mode="min",
            save_top_k=args.save_top_k, save_last=True,
            filename="{step}-{valid/total_loss:.4f}", auto_insert_metric_name=False,
        )]
        print(f"[sofa_train] best checkpoints -> {ckpt_dir} (top {args.save_top_k} by valid/total_loss)")
        super().__init__(*a, **kw)


pl.Trainer = TrainerWithBestCkpt

import train as sofa_train           # noqa: E402  (must be imported after the patch)

sofa_train.main.callback(
    config_path=args.config_path, data_folder=args.data_folder,
    pretrained_model_path=args.pretrained_model_path, resume=args.resume,
)

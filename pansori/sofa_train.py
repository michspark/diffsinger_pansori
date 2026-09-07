#!/usr/bin/env python3
"""SOFA 학습 + wandb + 최적 체크포인트 보존. SOFA 저장소는 한 줄도 고치지 않는다.

SOFA 의 train.py 는 두 가지가 없다:

  1. **로거를 지정하지 않는다.** lightning 기본값인 TensorBoardLogger 가 붙으므로
     `wandb.init(sync_tensorboard=True)` 로 SummaryWriter 클래스를 패치하면 스칼라도
     그림도 그대로 따라온다 — pansori/train.py 가 upstream 에 쓰는 것과 같은 수법이다.
  2. **ModelCheckpoint 콜백이 없다.** lightning 기본 콜백은 마지막 것만 남기므로
     검증 최저점이 지나가면 사라진다. docs/NEXT.md A1 이 지난 학습에서 겪은 그 함정이다
     (acoustic 은 28k 가 최선이었는데 num_ckpt_keep 이 그걸 지웠다).

그래서 pl.Trainer 를 감싸 `valid/total_loss` 를 감시하는 ModelCheckpoint 를 끼워 넣는다.
SOFA 파일을 고치는 대신 생성자를 가로채는 쪽이 rebase 를 감당할 수 있다.

    conda activate sofa
    PYTHONNOUSERSITE=1 python ~/Desktop/DiffSinger/pansori/sofa_train.py \
        -p ckpt/pretrained_multilingual_singing/v1.0.0_multilingual_singing.ckpt

--no-wandb 를 주면 SOFA 의 train.py 를 그대로 부르는 것과 같다.
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

os.chdir(SOFA_ROOT)                 # SOFA 의 modules.* 상대 임포트가 cwd 에 걸려 있다
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

# 검증 최저점을 살려 둔다. save_last 도 함께 남겨 --resume 이 끊기지 않게 한다.
_Trainer = pl.Trainer


class TrainerWithBestCkpt(_Trainer):
    def __init__(self, *a, **kw):
        ckpt_dir = pathlib.Path("ckpt") / config["model_name"] / "best"
        kw["callbacks"] = list(kw.get("callbacks") or []) + [ModelCheckpoint(
            dirpath=str(ckpt_dir), monitor="valid/total_loss", mode="min",
            save_top_k=args.save_top_k, save_last=True,
            filename="{step}-{valid/total_loss:.4f}", auto_insert_metric_name=False,
        )]
        print(f"[sofa_train] 최적 체크포인트 → {ckpt_dir} (valid/total_loss 기준 상위 {args.save_top_k})")
        super().__init__(*a, **kw)


pl.Trainer = TrainerWithBestCkpt

import train as sofa_train           # noqa: E402  (patch 후에 임포트해야 한다)

sofa_train.main.callback(
    config_path=args.config_path, data_folder=args.data_folder,
    pretrained_model_path=args.pretrained_model_path, resume=args.resume,
)

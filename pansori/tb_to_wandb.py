#!/usr/bin/env python3
"""Replay a finished run's tensorboard log into wandb under the live runs' key names.

`wandb sync --sync-tensorboard` works but namespaces every metric by the directory
the tfevents file sits in, so `checkpoints/{exp}/lightning_logs/latest` arrives as
`latest/validation/total_loss` while a live `pansori/train.py` run writes
`validation/total_loss`. The two never land on the same chart, which defeats the
point of backfilling a baseline. Pointing the sync at a flat copy just moves the
problem -- wandb walks up to the parent and namespaces by that instead.

So this reads the events itself and logs each scalar under the name the task
actually used. Images and audio come along when they are in the log; they are the
validation mel plots and vocoded samples (training/acoustic_task.py:215-242).

    python pansori/tb_to_wandb.py --exp pansori_var_v1 \
      --group lyrics-ab --lyrics-source whisper

Idempotent per run id: it resumes the run whose id is the exp name, so a second
invocation appends rather than creating a duplicate.
"""

import argparse
import io
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp', required=True, help='checkpoints/{exp}')
    parser.add_argument('--logdir', default=None,
                        help='defaults to checkpoints/{exp}/lightning_logs/latest')
    parser.add_argument('--project', default='pansori')
    parser.add_argument('--run-id', default=None,
                        help='defaults to --exp. wandb will not reuse an id it has deleted, '
                             'so on a re-upload give a fresh id and keep the name via --exp')
    parser.add_argument('--group', default=None)
    parser.add_argument('--tags', default='')
    parser.add_argument('--lyrics-source', default=None,
                        choices=['whisper', 'snapped'])
    parser.add_argument('--no-media', action='store_true',
                        help='upload scalars only (figures and audio are large)')
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    import wandb
    from PIL import Image
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logdir = Path(args.logdir) if args.logdir else \
        REPO / 'checkpoints' / args.exp / 'lightning_logs' / 'latest'
    guidance = {'scalars': 0, 'images': 0, 'audio': 0}
    accumulator = EventAccumulator(str(logdir), size_guidance=guidance)
    accumulator.Reload()
    tags = accumulator.Tags()

    # step -> {key: value}. Every tag is keyed by step so one wandb.log call per
    # step reproduces the shape a live run writes, instead of one row per tag.
    by_step = defaultdict(dict)
    for tag in tags['scalars']:
        for event in accumulator.Scalars(tag):
            by_step[event.step][tag] = event.value

    media_count = 0
    if not args.no_media:
        for tag in tags.get('images', []):
            for event in accumulator.Images(tag):
                by_step[event.step][tag] = wandb.Image(
                    Image.open(io.BytesIO(event.encoded_image_string)))
                media_count += 1
        for tag in tags.get('audio', []):
            for event in accumulator.Audio(tag):
                # tensorboard stores audio as an encoded wav; wandb wants samples
                samples, rate = sf.read(io.BytesIO(event.encoded_audio_string),
                                        dtype='float32')
                by_step[event.step][tag] = wandb.Audio(
                    np.asarray(samples), sample_rate=int(rate))
                media_count += 1

    print(f'{logdir}\n  {len(tags["scalars"])} scalar tags · {len(by_step):,} steps · '
          f'{media_count:,} media')

    config = {'backfilled_from': str(logdir)}
    if args.lyrics_source:
        config['lyrics_source'] = args.lyrics_source
    tag_list = [t for t in args.tags.split(',') if t] + ['backfilled-from-tensorboard']
    if args.lyrics_source:
        tag_list.append(f'lyrics:{args.lyrics_source}')

    run = wandb.init(project=args.project, name=args.exp,
                     id=args.run_id or args.exp,
                     resume='allow', group=args.group, tags=tag_list,
                     config=config, dir=str(logdir.parents[1]))
    for step in sorted(by_step):
        run.log(by_step[step], step=step)
    run.finish()
    print(f'done: {run.url}')


if __name__ == '__main__':
    main()

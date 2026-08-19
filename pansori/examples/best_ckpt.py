#!/usr/bin/env python3
"""Print the checkpoint step with the lowest validation loss, not the latest one.

pansori/README.md's standing rule is that the most recent checkpoint is not the
best one -- the old pipeline started overfitting around 56k steps. Picking by
hand means reading TensorBoard every time, so the example runner asks this
instead.

Only steps whose .ckpt still exists are eligible: `num_ckpt_keep` prunes old
ones, so the true minimum can already be gone, and returning a step with no
file behind it would fail later and further away from the cause. When that
happens it says so on stderr rather than silently returning second best.

Usage:
  python pansori/examples/best_ckpt.py pansori_ac_v1
  python pansori/examples/best_ckpt.py pansori_var_v1 --metric validation/pitch_loss
"""

import argparse
import glob
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description='검증 손실 최저 체크포인트 찾기')
    parser.add_argument('exp')
    parser.add_argument('--metric', default='validation/total_loss')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    exp_dir = REPO / 'checkpoints' / args.exp
    available = {int(m.group(1)) for p in exp_dir.glob('model_ckpt_steps_*.ckpt')
                 if (m := re.search(r'steps_(\d+)\.ckpt', p.name))}
    if not available:
        sys.exit(f'{exp_dir}: 체크포인트 없음')

    events = sorted(glob.glob(str(exp_dir / 'lightning_logs' / '*' / 'events.out.tfevents.*')))
    if not events:
        print(max(available))
        return

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    scalars = {}
    for path in events:
        ea = EventAccumulator(path, size_guidance={'scalars': 0})
        ea.Reload()
        if args.metric in ea.Tags()['scalars']:
            for s in ea.Scalars(args.metric):
                scalars[s.step] = s.value
    # step 0 is the untrained sanity check, never a candidate
    scalars.pop(0, None)
    if not scalars:
        print(max(available))
        return

    best_step = min(scalars, key=scalars.get)
    kept = {s: v for s, v in scalars.items() if s in available}
    if not kept:
        sys.exit(f'{args.exp}: 검증된 step 중 남아 있는 체크포인트가 없다')
    pick = min(kept, key=kept.get)

    if not args.quiet:
        note = '' if pick == best_step else \
            f' (전체 최저는 step {best_step:,} 이지만 num_ckpt_keep 으로 삭제됨)'
        print(f'{args.exp}: {args.metric} 최저 {kept[pick]:.5f} @ step {pick:,}'
              f' / 마지막 step {max(scalars):,} 은 {scalars[max(scalars)]:.5f}{note}',
              file=sys.stderr)
    print(pick)


if __name__ == '__main__':
    main()

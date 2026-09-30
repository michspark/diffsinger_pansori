#!/usr/bin/env python3
"""Which utterances may be compared, and why five of them may not.

The old repo held out six songs, one per myeongchang. The new one holds out five:
configs/pansori/data_pansori.yaml gives 정권진 `test_prefixes: []` because that
singer's only song IS the holdout song, so holding it out would leave the
speaker with zero training data (pansori/docs/NEXT.md, last line).

So a580300d is test data for the old model and TRAINING data for the new one.
Comparing on those five items would be the single most damaging flaw in the
report, and the asymmetry runs both ways -- 정권진 also had no training data at
all for the old model.

assert_held_out() therefore reads the live config rather than trusting a list in
this file. If someone re-runs make_configs.py and the holdout changes, the eval
fails loudly instead of quietly comparing training data.
"""

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402

DATA = C.REPO / 'data' / 'pansori'


def _load_datasets():
    """(speaker, test_prefixes) for each dataset entry in the corpus config.

    Parsed by hand rather than with yaml.safe_load so this works before anything
    else in the repo is importable, and so the failure mode is a clear exception
    rather than a silently empty list.
    """
    text = (C.REPO / 'configs' / 'pansori' / 'data_pansori.yaml').read_text(encoding='utf-8')
    block = text.split('\ndatasets:', 1)
    if len(block) != 2:
        raise RuntimeError('configs/pansori/data_pansori.yaml has no datasets: block')
    entries, speaker, prefixes, in_prefixes = [], None, [], False
    for line in block[1].splitlines():
        if re.match(r'^\S', line) and line.strip():
            break  # left the datasets: block
        if m := re.match(r'\s*- raw_data_dir:', line):
            if speaker:
                entries.append((speaker, prefixes))
            speaker, prefixes, in_prefixes = None, [], False
        elif m := re.match(r'\s*speaker:\s*(\S+)', line):
            speaker = m.group(1)
        elif re.match(r'\s*test_prefixes:\s*\[\]', line):
            in_prefixes = False
        elif re.match(r'\s*test_prefixes:\s*$', line):
            in_prefixes = True
        elif in_prefixes and (m := re.match(r"\s*-\s*'?([0-9a-f]{8})'?", line)):
            prefixes.append(m.group(1))
        elif in_prefixes and re.match(r'\s*\S+:', line):
            in_prefixes = False
    if speaker:
        entries.append((speaker, prefixes))
    if not entries:
        raise RuntimeError('parsed zero dataset entries from data_pansori.yaml')
    return entries


_DATASETS = _load_datasets()
HELD_OUT_PREFIXES = {p for _, ps in _DATASETS for p in ps}


class NotHeldOut(RuntimeError):
    pass


def assert_held_out(item):
    """Raise unless `item`'s song is held out of the NEW model's training set."""
    prefix = item.split('_', 1)[0]
    if prefix not in HELD_OUT_PREFIXES:
        raise NotHeldOut(
            f'{item}: song {prefix} is in the new model\'s TRAINING set '
            f'(test_prefixes in configs/pansori/data_pansori.yaml). '
            f'Held out: {sorted(HELD_OUT_PREFIXES)}'
        )


# ---------------------------------------------------------------- item lookup

_INDEX = None


def _index():
    """item name -> (speaker, wav path, csv row). Built once by scanning the corpus."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for csv_path in sorted(DATA.glob('*/transcriptions.csv')):
            speaker = csv_path.parent.name
            with open(csv_path, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    wav = csv_path.parent / 'wavs' / f'{row["name"]}.wav'
                    _INDEX[row['name']] = (speaker, wav, row)
    return _INDEX


def corpus_rows(data_dir):
    """item name -> transcriptions.csv row, for any corpus laid out like data/pansori."""
    root = Path(data_dir)
    root = root if root.is_absolute() else C.REPO / root
    rows = {}
    for csv_path in sorted(root.glob('*/transcriptions.csv')):
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows[row['name']] = row
    if not rows:
        raise FileNotFoundError(f'no transcriptions.csv under {root}')
    return rows


def lookup(item):
    try:
        return _index()[item]
    except KeyError:
        raise KeyError(f'{item} not found under {DATA}') from None


def speaker_of(item):
    return lookup(item)[0]


def wav_of(item):
    return lookup(item)[1]


def row_of(item):
    return lookup(item)[2]


# ---------------------------------------------------------------- item sets

def old_ladder_items(run=None):
    """The items the OLD model actually rendered, read off disk.

    Not hardcoded: if the ladder directory changes, the eval follows it.
    """
    d = C.OLD_LADDER[run or C.OLD_PRIMARY]
    if not d.is_dir():
        raise FileNotFoundError(f'old ladder not found: {d}')
    return sorted({p.name.split('__', 1)[0] for p in d.glob('*__1_orig.wav')})


def paired_items(run=None):
    """The old ladder items that are ALSO held out for the new model.

    Returns (fair, excluded). `fair` carries every between-system claim.
    """
    fair, excluded = [], []
    for item in old_ladder_items(run):
        try:
            assert_held_out(item)
        except NotHeldOut:
            excluded.append(item)
        else:
            fair.append(item)
    return fair, excluded


def holdout_items():
    """Every held-out utterance in the corpus (the new model has never seen these)."""
    return sorted(name for name in _index() if name.split('_', 1)[0] in HELD_OUT_PREFIXES)


def holdout_sample(n, seed=20260823):
    """n held-out items, stratified evenly across the held-out songs.

    Stratified rather than uniform because the songs differ wildly in length
    (이일주 220 utts vs 김소희 157), and an unstratified draw would let one
    singer's register dominate the absolute numbers.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    by_song = {}
    for item in holdout_items():
        by_song.setdefault(item.split('_', 1)[0], []).append(item)
    per = max(1, n // len(by_song))
    out = []
    for song in sorted(by_song):
        pool = by_song[song]
        take = min(per, len(pool))
        out.extend(rng.choice(pool, size=take, replace=False).tolist())
    return sorted(out)[:n]


def main():
    fair, excluded = paired_items()
    print(f'held-out songs (new model): {sorted(HELD_OUT_PREFIXES)}')
    print(f'old ladder items: {len(old_ladder_items())}')
    print(f'\n{len(fair)} fair pairs:')
    by_spk = {}
    for item in fair:
        by_spk.setdefault(speaker_of(item), []).append(item)
    for spk in sorted(by_spk, key=lambda s: -len(by_spk[s])):
        print(f'  {spk:6s} {len(by_spk[spk])}  {" ".join(by_spk[spk])}')
    print(f'\n{len(excluded)} excluded (training data for the new model):')
    for item in excluded:
        print(f'  {item}  {speaker_of(item)}')
    print(f'\nhold-out total: {len(holdout_items())} utterances')


if __name__ == '__main__':
    main()

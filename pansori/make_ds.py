#!/usr/bin/env python3
"""Build .ds inference inputs — from a labelled item, or from lyrics + a score.

openvpi's inference reads .ds files, which normally come out of an editor like
OpenUtau. There is no CLI that makes one, and for pansori there is no editor
project to export from either, so this fills the gap at both ends of the
pipeline:

  from-label   an item that exists in data/pansori/{speaker}/transcriptions.csv.
               Use it to isolate one stage: hand the acoustic model the real
               durations and the real f0 and what comes back is the acoustic
               ceiling; withhold them and the variance model has to invent them.

  from-score   Korean lyrics plus a melody, with no recording behind them.
               This is the path the MoonInTheRiver pipeline never had — it needs
               a grapheme-to-phoneme step, which only exists now that
               dictionaries/korean-coda.txt does.

Score files are one note per line, `syllable note duration_seconds`:

    춘  A4    0.45
    향  G4    0.40
    -   F4    0.55        # the previous syllable continues (melisma) -- common in pansori
    _   rest  0.30        # a rest -> SP

Anything after `#` is a comment. Blank lines are ignored.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/make_ds.py from-label 01ed19bb_0003 --with-dur --with-f0
  PYTHONPATH=. python pansori/make_ds.py from-score pansori/examples/scores/sample.txt
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# One frame of f0 every 5 ms is what the .ds format uses; it is resampled to the
# model's hop internally, so it does not have to match hop_size.
F0_TIMESTEP = 0.005

_COMMENT = re.compile(r'(?:^|\s)#')

MELISMA = '-'    # this note continues the previous syllable
REST = '_'       # this note is a rest


def load_dictionary(path):
    entries = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                syllable, phones = line.rstrip('\n').split('\t', 1)
                entries[syllable] = phones.split()
    return entries


def find_item(item_name, data_root):
    """(speaker, row, wav_path) for a labelled item."""
    for speaker_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        csv_path = speaker_dir / 'transcriptions.csv'
        if not csv_path.exists():
            continue
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['name'] == item_name:
                    return speaker_dir.name, row, speaker_dir / 'wavs' / f'{item_name}.wav'
    raise SystemExit(f'item not found in the labels: {item_name}')


def extract_f0(wav_path, n_frames_hint=None):
    """Ground-truth f0 with the same extractor and range the binarizer used.

    Reading it back off the wav rather than out of the binary keeps this usable
    for any wav, and guarantees the f0_min/f0_max of the config are applied --
    the default 65 Hz would cut off male tongseong (see pansori/docs/SEGMENTATION_VERDICT.md).
    """
    import librosa
    import numpy as np
    from utils.hparams import set_hparams, hparams
    set_hparams(config=str(REPO / 'configs' / 'exp' / 'ac_v1.yaml'), print_hparams=False)
    from modules.pe import initialize_pe

    waveform, _ = librosa.load(wav_path, sr=hparams['audio_sample_rate'], mono=True)
    length = round(len(waveform) / hparams['hop_size'])
    f0, uv = initialize_pe().get_pitch(
        waveform, samplerate=hparams['audio_sample_rate'], length=length,
        hop_size=hparams['hop_size'], f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
        interp_uv=True
    )
    # hop frames -> 5 ms frames
    src_t = np.arange(len(f0)) * hparams['hop_size'] / hparams['audio_sample_rate']
    dst_t = np.arange(0, src_t[-1] + F0_TIMESTEP, F0_TIMESTEP)
    return np.interp(dst_t, src_t, f0), float(uv.mean())


def parse_score(path):
    """Score file -> [(syllable|MELISMA|REST, note, duration), ...]."""
    notes = []
    for lineno, raw in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        # A '#' only starts a comment at the start of a line or after whitespace --
        # note names carry their own ('C#4') and must survive.
        line = _COMMENT.split(raw, maxsplit=1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise SystemExit(f'{path}:{lineno}: expected 3 fields, `syllable note duration` -- {raw!r}')
        syllable, note, dur = parts
        notes.append((syllable, note, float(dur)))
    if not notes:
        raise SystemExit(f'{path}: no notes')
    return notes


def derive_note_slur(ph_num, ph_dur, note_dur):
    """note_slur for a labelled item, whose labels do not carry one.

    Without ph_dur the variance model groups notes into words with note_slur
    (inference/ds_variance.py:193): a 0 starts a new word, a 1 continues the
    previous one, so exactly one note per word must be a 0. Our labels keep
    notes and words on independent axes and never recorded that grouping -- a
    real score would have it -- so recover it from the ground-truth timings by
    picking, for each word, the note that starts closest to it.

    Done as a DP rather than nearest-neighbour because the assignment has to be
    strictly increasing and every word has to receive a note; a greedy pass can
    strand a short word with none and silently produce an invalid .ds.
    """
    import numpy as np
    counts = [int(n) for n in ph_num.split()]
    durs = [float(d) for d in ph_dur.split()]
    assert sum(counts) == len(durs), 'ph_num and ph_dur lengths disagree'

    word_start, at, i = [], 0.0, 0
    for n in counts:
        word_start.append(at)
        at += sum(durs[i:i + n])
        i += n
    note_start = np.concatenate([[0.0], np.cumsum([float(d) for d in note_dur.split()])[:-1]])

    n_w, n_n = len(word_start), len(note_start)
    if n_n < n_w:
        raise SystemExit(f'fewer notes ({n_n}) than words ({n_w}); note_slur cannot be built')

    INF = float('inf')
    cost = np.abs(note_start[None, :] - np.array(word_start)[:, None])   # [T_w, T_n]
    dp = np.full((n_w, n_n), INF)
    back = np.zeros((n_w, n_n), dtype=int)
    dp[0] = cost[0]
    for w in range(1, n_w):
        best, arg = INF, -1
        for n in range(n_n):
            if n - 1 >= 0 and dp[w - 1, n - 1] < best:
                best, arg = dp[w - 1, n - 1], n - 1
            if best < INF:
                dp[w, n] = best + cost[w, n]
                back[w, n] = arg
    end = int(np.argmin(dp[n_w - 1]))
    picked = [end]
    for w in range(n_w - 1, 0, -1):
        end = back[w, end]
        picked.append(end)
    picked = set(picked)
    return ' '.join('0' if n in picked else '1' for n in range(n_n))


def score_to_ds(notes, dictionary):
    """Score -> (text, ph_seq, ph_num, note_seq, note_dur).

    A word here is one syllable, or one SP for a rest. Notes stay a separate
    axis: a syllable held over several notes just contributes its phones once
    while the extra notes sit alongside, which is how openvpi represents
    melisma and how sigimsae lands in the labels.
    """
    words, note_seq, note_dur, note_slur = [], [], [], []
    for syllable, note, dur in notes:
        note_seq.append(note)
        note_dur.append(f'{dur:.4f}')
        if syllable == MELISMA:
            if not words:
                raise SystemExit(f"'{MELISMA}' cannot be the first note")
            note_slur.append('1')
            continue
        if syllable == REST:
            if words and words[-1][1] == ['SP']:
                note_slur.append('1')         # consecutive rests collapse into one SP
                continue
            words.append(('SP', ['SP']))
        else:
            phones = dictionary.get(syllable)
            if phones is None:
                raise SystemExit(f"syllable not in the dictionary: '{syllable}'")
            words.append((syllable, phones))
        note_slur.append('0')

    return {
        'text': ' '.join(w for w, _ in words),
        'ph_seq': ' '.join(p for _, ph in words for p in ph),
        'ph_num': ' '.join(str(len(ph)) for _, ph in words),
        'note_seq': ' '.join(note_seq),
        'note_dur': ' '.join(note_dur),
        'note_slur': ' '.join(note_slur),
    }


def set_speaker(segment, speaker):
    """Write the speaker into the .ds instead of passing --spk.

    scripts/infer.py parses --spk with `[0-9A-Za-z_-]+` (utils/infer_utils.py:55),
    which rejects every myeongchang's name. The .ds fields it would have populated take
    the same dict, so set them here and leave the upstream parser alone.
    """
    segment['spk_mix'] = {speaker: 1.0}
    segment['ph_spk_mix'] = {speaker: 1.0}


def write_ds(segment, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([segment], ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
    print(f'{out_path}')
    if 'spk_mix' in segment:
        print(f'  {"spk":9s} {"|".join(segment["spk_mix"])}')
    for key in ('text', 'ph_seq', 'ph_num', 'note_seq', 'note_dur', 'note_slur'):
        if key in segment:
            print(f'  {key:9s} {str(segment[key])[:110]}')
    length = sum(float(d) for d in segment['note_dur'].split())
    extras = [k for k in ('ph_dur', 'f0_seq') if k in segment]
    print(f'  length {length:.2f}s | {len(segment["note_dur"].split())} notes'
          f' | {len(segment["ph_seq"].split())} phonemes'
          f' | supplied: {", ".join(extras) if extras else "nothing (variance fills it in)"}')


def cmd_from_label(args):
    speaker, row, wav_path = find_item(args.item, REPO / args.data)
    segment = {
        'offset': 0.0,
        'ph_seq': row['ph_seq'],
        'ph_num': row['ph_num'],
        'note_seq': row['note_seq'],
        'note_dur': row['note_dur'],
    }
    if args.with_dur:
        segment['ph_dur'] = row['ph_dur']
    else:
        segment['note_slur'] = derive_note_slur(row['ph_num'], row['ph_dur'], row['note_dur'])
    set_speaker(segment, args.spk or speaker)
    if args.with_f0:
        f0, uv_ratio = extract_f0(wav_path)
        segment['f0_seq'] = ' '.join(f'{v:.1f}' for v in f0)
        segment['f0_timestep'] = F0_TIMESTEP
        print(f'  f0: {len(f0)} frames, unvoiced {uv_ratio:.1%}')
    out = Path(args.out) if args.out else REPO / 'pansori' / 'examples' / 'ds' / f'{args.item}.ds'
    print(f'speaker {speaker} · source {wav_path.relative_to(REPO)}')
    write_ds(segment, out)


def cmd_from_score(args):
    dictionary = load_dictionary(REPO / args.dictionary)
    segment = {'offset': 0.0, **score_to_ds(parse_score(args.score), dictionary)}
    set_speaker(segment, args.spk)
    out = Path(args.out) if args.out else \
        REPO / 'pansori' / 'examples' / 'ds' / f'{Path(args.score).stem}.ds'
    write_ds(segment, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    a = sub.add_parser('from-label', help='build a .ds from a labelled utterance')
    a.add_argument('item', help='item_name (e.g. 01ed19bb_0003)')
    a.add_argument('--data', default='data/pansori')
    a.add_argument('--with-dur', action='store_true', help='inject the real ph_dur')
    a.add_argument('--with-f0', action='store_true', help='extract f0 from the source and inject it')
    a.add_argument('--spk', help='speaker (default: the source recording\'s speaker)')
    a.add_argument('--out')
    a.set_defaults(func=cmd_from_label)

    b = sub.add_parser('from-score', help='build a .ds from lyrics + a score')
    b.add_argument('score', help='score file (syllable note duration_seconds)')
    b.add_argument('--dictionary', default='dictionaries/korean-coda.txt')
    b.add_argument('--spk', required=True, help='speaker (a directory name under data/pansori)')
    b.add_argument('--out')
    b.set_defaults(func=cmd_from_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.path.insert(0, str(REPO))
    main()

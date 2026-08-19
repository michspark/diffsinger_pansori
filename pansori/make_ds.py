#!/usr/bin/env python3
"""Build .ds inference inputs — from a labelled item, or from lyrics + a score.

openvpi's inference reads .ds files, which normally come out of an editor like
OpenUtau. There is no CLI that makes one, and for pansori there is no editor
project to export from either, so this fills the gap at both ends of the
pipeline:

  from-label   an item that exists in data/pansori/{화자}/transcriptions.csv.
               Use it to isolate one stage: hand the acoustic model the real
               durations and the real f0 and what comes back is the acoustic
               ceiling; withhold them and the variance model has to invent them.

  from-score   Korean lyrics plus a melody, with no recording behind them.
               This is the path the MoonInTheRiver pipeline never had — it needs
               a grapheme-to-phoneme step, which only exists now that
               dictionaries/korean-coda.txt does.

Score files are one note per line, `음절 노트 길이초`:

    춘  A4    0.45
    향  G4    0.40
    -   F4    0.55        # 앞 음절이 이어짐 (melisma) — 판소리에서 흔하다
    _   rest  0.30        # 쉼표 → SP

`#` 뒤는 주석. Blank lines are ignored.

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
    raise SystemExit(f'라벨에 없는 item: {item_name}')


def extract_f0(wav_path, n_frames_hint=None):
    """Ground-truth f0 with the same extractor and range the binarizer used.

    Reading it back off the wav rather than out of the binary keeps this usable
    for any wav, and guarantees the f0_min/f0_max of the config are applied --
    the default 65 Hz would cut off male 통성 (see pansori/alignment/VERDICT.md).
    """
    import librosa
    import numpy as np
    from utils.hparams import set_hparams, hparams
    set_hparams(config=str(REPO / 'configs' / 'pansori_acoustic.yaml'), print_hparams=False)
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
            raise SystemExit(f'{path}:{lineno}: `음절 노트 길이` 3칸이어야 한다 — {raw!r}')
        syllable, note, dur = parts
        notes.append((syllable, note, float(dur)))
    if not notes:
        raise SystemExit(f'{path}: 노트가 없다')
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
    assert sum(counts) == len(durs), 'ph_num 과 ph_dur 길이가 안 맞는다'

    word_start, at, i = [], 0.0, 0
    for n in counts:
        word_start.append(at)
        at += sum(durs[i:i + n])
        i += n
    note_start = np.concatenate([[0.0], np.cumsum([float(d) for d in note_dur.split()])[:-1]])

    n_w, n_n = len(word_start), len(note_start)
    if n_n < n_w:
        raise SystemExit(f'노트({n_n})가 단어({n_w})보다 적어 note_slur 를 만들 수 없다')

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
    melisma and how 시김새 lands in the labels.
    """
    words, note_seq, note_dur, note_slur = [], [], [], []
    for syllable, note, dur in notes:
        note_seq.append(note)
        note_dur.append(f'{dur:.4f}')
        if syllable == MELISMA:
            if not words:
                raise SystemExit(f"'{MELISMA}' 가 첫 노트에 올 수 없다")
            note_slur.append('1')
            continue
        if syllable == REST:
            if words and words[-1][1] == ['SP']:
                note_slur.append('1')         # 연속 쉼표는 SP 하나로
                continue
            words.append(('SP', ['SP']))
        else:
            phones = dictionary.get(syllable)
            if phones is None:
                raise SystemExit(f"사전에 없는 음절: '{syllable}'")
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
    which rejects every 명창's name. The .ds fields it would have populated take
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
    print(f'  길이 {length:.2f}s | 노트 {len(segment["note_dur"].split())}개'
          f' | 음소 {len(segment["ph_seq"].split())}개'
          f' | 함께 넣은 것: {", ".join(extras) if extras else "없음 (variance 가 채운다)"}')


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
        print(f'  f0: {len(f0)}프레임, 무성 비율 {uv_ratio:.1%}')
    out = Path(args.out) if args.out else REPO / 'pansori' / 'examples' / 'ds' / f'{args.item}.ds'
    print(f'화자 {speaker} · 원본 {wav_path.relative_to(REPO)}')
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

    a = sub.add_parser('from-label', help='라벨된 발화에서 .ds 생성')
    a.add_argument('item', help='item_name (예: 01ed19bb_0003)')
    a.add_argument('--data', default='data/pansori')
    a.add_argument('--with-dur', action='store_true', help='실제 ph_dur 를 넣는다')
    a.add_argument('--with-f0', action='store_true', help='원본에서 f0 를 뽑아 넣는다')
    a.add_argument('--spk', help='화자 (기본: 원본 화자)')
    a.add_argument('--out')
    a.set_defaults(func=cmd_from_label)

    b = sub.add_parser('from-score', help='가사+악보에서 .ds 생성')
    b.add_argument('score', help='악보 파일 (음절 노트 길이초)')
    b.add_argument('--dictionary', default='dictionaries/korean-coda.txt')
    b.add_argument('--spk', required=True, help='화자 (data/pansori 의 디렉토리 이름)')
    b.add_argument('--out')
    b.set_defaults(func=cmd_from_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.path.insert(0, str(REPO))
    main()

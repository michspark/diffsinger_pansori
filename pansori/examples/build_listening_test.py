#!/usr/bin/env python3
"""Bundle the A/B renders into one self-contained listening-test page.

`pansori/docs/LYRICBOOK_VERDICT.md` §6 and §9-4 both end at the same missing step: nobody has
listened. §2 says the 창본 corrections are right, but that is agreement with a book,
not with the recording; §9 says the corrected labels make the model's voiced/unvoiced
decisions worse, but that is a number, not a judgement. Both need an ear.

So this builds the bench: for every rendered pair, the original recording, the two
candidate lyric texts, and the two models' renders of that same utterance — blind,
in an order derived from the item name so it is stable across reloads and uncorrelated
between the two questions.

Audio goes in as base64 mp3 (mono, 96 kbps). The artifact CSP blocks every external
host, so nothing can be fetched at view time; everything ships in the file.

    python pansori/examples/build_listening_test.py --n 40

Reads pansori/examples/ab_out/{items,scores}.json (see ab_ladder.py) and writes
pansori/examples/listening_test.html.
"""

import argparse
import base64
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AB_OUT = REPO / 'pansori' / 'examples' / 'ab_out'
TEMPLATE = Path(__file__).resolve().parent / 'listening_test.template.html'
OUT_HTML = Path(__file__).resolve().parent / 'listening_test.html'

BITRATE = '96k'


def encode(wav_path: Path, tmp: Path) -> str:
    """One wav -> a base64 mp3 data URI. Mono 96 kbps: ~16 KB/s of audio."""
    mp3 = tmp / (wav_path.stem + '.mp3')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav_path),
                    '-ac', '1', '-b:a', BITRATE, str(mp3)], check=True)
    return 'data:audio/mpeg;base64,' + base64.b64encode(mp3.read_bytes()).decode()


def song_title(stem: str, speaker: str) -> str:
    """'8b4d0824-06-조상현-춘향가_그때의_어사또는' -> '춘향가 그때의 어사또는'."""
    tail = stem.split(f'-{speaker}-')[-1] if f'-{speaker}-' in stem else stem
    return tail.replace('_', ' ').strip()


def summarize(scores):
    """Recompute the paired table so the page can never disagree with the data."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ab_ladder import sign_test

    changed, sc = scores['changed'], scores['scores']
    out = []
    for label, want in (('가사 바뀜', True), ('대조군', False)):
        subset = [i for i in sorted(sc['whisper'])
                  if changed.get(i) is want and i in sc['snapped']]
        group = {'label': label, 'n': len(subset), 'rows': []}
        for key, name, lower_better, digits in (
                ('mel_l1', 'mel L1', True, 3),
                ('f0_rmse_cent', 'f0 RMSE ¢', True, 0),
                ('voiced_agree', '유무성 일치', False, 3)):
            pairs = [(sc['whisper'][i][key], sc['snapped'][i][key]) for i in subset
                     if key in sc['whisper'][i] and key in sc['snapped'][i]]
            if not pairs:
                continue
            wins, losses, p = sign_test([b - a for a, b in pairs], lower_better)
            group['rows'].append({
                'metric': name, 'digits': digits,
                'whisper': round(statistics.median(a for a, _ in pairs), digits),
                'snapped': round(statistics.median(b for _, b in pairs), digits),
                'snappedWins': wins, 'pairs': wins + losses, 'p': round(p, 3)})
        out.append(group)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=40,
                        help='항목 수. 가사가 바뀐 것을 먼저 채우고 남으면 대조군')
    parser.add_argument('--out', default=str(OUT_HTML))
    args = parser.parse_args()

    scores = json.loads((AB_OUT / 'scores.json').read_text(encoding='utf-8'))
    changed_map = scores['changed']
    per_seed = scores['per_seed']

    lyrics = {}
    for line in open(REPO / 'pansori/data/full/lyrics_snapped.jsonl', encoding='utf-8'):
        row = json.loads(line)
        if row['item_name'] in changed_map:
            lyrics[row['item_name']] = row
    segments = {}
    for line in open(REPO / 'pansori/data/full/segments.jsonl', encoding='utf-8'):
        row = json.loads(line)
        if row['item_name'] in changed_map:
            segments[row['item_name']] = row
    spk_map = json.loads(
        (REPO / 'pansori/data/full/spk_map.json').read_text(encoding='utf-8'))['item2spk']

    scored = sorted(i for i in changed_map if i in scores['scores']['snapped'])
    treated = [i for i in scored if changed_map[i]]
    control = [i for i in scored if not changed_map[i]]
    picked = treated[:args.n] + control[:max(0, args.n - len(treated))]
    picked.sort()

    items, total_bytes = [], 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for index, item in enumerate(picked, start=1):
            speaker = spk_map[item]
            row = lyrics[item]
            audio = {'orig': encode(
                REPO / 'data/pansori' / speaker / 'wavs' / f'{item}.wav', tmp)}
            seeds, metrics = {}, {}
            for arm in ('whisper', 'snapped'):
                draws = per_seed[arm][item]
                # play the draw whose number is on screen: the median-mel render
                median = statistics.median(d['mel_l1'] for d in draws.values())
                seed = min(draws, key=lambda s: abs(draws[s]['mel_l1'] - median))
                seeds[arm] = int(seed)
                metrics[arm] = scores['scores'][arm][item]
                audio[arm] = encode(AB_OUT / arm / 'wav' / f'{item}__s{seed}.wav', tmp)
            total_bytes += sum(len(v) for v in audio.values())
            items.append({
                'id': item,
                'speaker': speaker,
                'song': song_title(segments[item]['song_stem'], speaker),
                'mode': segments[item].get('mode', ''),
                'duration': round(segments[item]['duration'], 2),
                'changed': bool(changed_map[item]),
                'whisperText': row['text_whisper'],
                'bookText': row['text'],
                'matchScore': row.get('match_score'),
                'seeds': seeds,
                'metrics': metrics,
                'audio': audio,
            })
            print(f'  [{index}/{len(picked)}] {item}  {speaker}', file=sys.stderr)

    payload = {'items': items, 'summary': summarize(scores)}

    html = TEMPLATE.read_text(encoding='utf-8').replace(
        '"__PAYLOAD__"', json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    out = Path(args.out)
    out.write_text(html, encoding='utf-8')
    size = out.stat().st_size
    print(f'\n{out}  {size / 1e6:.1f} MB  (오디오 {total_bytes / 1e6:.1f} MB / '
          f'{len(items)}항목 x 3)')
    if size > 16e6:
        sys.exit('⚠ 16 MB 초과 — --n 을 줄일 것')
    if not shutil.which('ffmpeg'):
        sys.exit('ffmpeg 없음')


if __name__ == '__main__':
    main()

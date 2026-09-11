#!/usr/bin/env python3
"""Measures the synthesis's **pronunciation** -- isolating which rung of the
ladder it collapses at.

All 22 metrics in `pansori/compare/` are pitch or spectrum. Even the closest one,
`voiced_agree`, only asks "should this frame make sound", so dropping a consonant
entirely or pronouncing `g` as `d` does not move it (it sits high at 0.95-0.97 on
the hold-out). Every verdict reached so far was therefore reached without being
able to see pronunciation. This script fills that hole.

## The ladder isolates the culprit

    1_orig      the original recording              -- null control (see below)
    0_vocoder   GT mel + GT f0 -> vocoder           -- the vocoder ceiling
    2_resynth   real ph_dur + real f0 -> acoustic   -- the labels and the acoustic model
    3_pitch     real ph_dur, f0 predicted
    4_full      notes + lyrics only, dur and f0 both predicted

`2_resynth` is fed the GT ph_dur directly. That GT is exactly the labels where
22% sit on MFA's floor (`pansori/eval/label_report.py`), so a collapse already at
this rung means the culprit is the labels or the acoustic model, while a collapse
only at `4_full` means duration prediction. If `0_vocoder` is already bad,
everything below it is meaningless.

## Two rulers together -- each covers the other's weakness

**whisper**: transcribe the rendered audio with whisper-large-v3 and take CER
against the input lyrics. Closest to what a person hears, but Whisper carries a
language model and **fills in sounds that are not there**.

**mfa**: re-align the rendered audio against the known lyrics with korean_mfa and
look at whether the aligner **found acoustic evidence**. Two things:

  - `floor_rate` -- the fraction of the re-alignment pinned to MFA's minimum
    duration (30 ms). When the sound does not contain that phoneme, the aligner
    parks it at the minimum and moves on. This applies to the synthesis exactly
    the same ruler `pansori/eval/label_report.py` applies to the labels.
  - `boundary_mae_ms` -- mean difference between the GT phoneme boundaries and
    the re-aligned ones.

**`spn` is not used to compare rungs.** It went in on the theory "MFA emits spn
when the rendered sound has no such consonant", but measured, **its value was
completely identical across all five rungs of the same utterance**. spn is
emitted for out-of-vocabulary words, so it is a property of the **lyrics**, not
the audio. (That `pansori/docs/LYRICBOOK_VERDICT.md` §4 saw spn rise 164->202 is
not a contradiction -- that was an experiment where the text changed.) It is
printed once as a reference figure for how much OOV the sample contains.

`phone_match` is near-constant for the same reason -- the phoneme sequence is
fixed by the lyrics we supplied. It only moves when MFA inserts or deletes a
phoneme, so it stays as a reference value.

## `1_orig` is not a ceiling -- it is a null control

The CER reference text is `data/full/lyrics.jsonl`, which is itself **Whisper's
transcription of these very recordings** (03_transcribe.py). So re-transcribing
the original compares it against itself and the CER comes out near zero. That
does **not** mean pansori is well understood; it means Whisper's decoding
reproduces. If the value is not near zero, every number after it is void.

The real ceiling for synthesis is `0_vocoder`. On the MFA side it is the other
way round: `1_orig` is a proper ceiling (the aligner receives audio and text
independently, so there is no circularity).

## Running it -- split into stages because it needs three envs

    conda activate dsp     && PYTHONPATH=. python pansori/eval/intelligibility.py --stage render
    conda activate pansori && PYTHONPATH=. python pansori/eval/intelligibility.py --stage asr
    conda activate aligner && PYTHONPATH=. python pansori/eval/intelligibility.py --stage mfa
    conda activate dsp     && PYTHONPATH=. python pansori/eval/intelligibility.py --stage score

`--stage all` runs only the stages possible inside the current env. Each stage skips
artefacts it has already produced, so it is safe to interrupt and restart.

Output: eval/out/{asr,mfa}.jsonl -> eval/out/intelligibility.json + a console table
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'pansori' / 'preprocess'))

OUT_DIR = Path(__file__).resolve().parent / 'out'
RENDER_DIR = REPO / 'pansori' / 'compare' / 'out' / 'new'
LYRICS = REPO / 'pansori' / 'data' / 'full' / 'lyrics.jsonl'
MFA_WORK = OUT_DIR / 'mfa_work'

# Rungs that use no seed. The original and the vocoder ceiling have no stochastic part.
DETERMINISTIC = {'1_orig': None, '0_vocoder': 0}
SAMPLED = ('2_resynth', '3_pitch', '4_full')
RUNGS = ('1_orig', '0_vocoder') + SAMPLED

MFA_FLOOR = 0.030


# ---------------------------------------------------------------- shared

def wav_path(item, rung, seed, items_mod):
    if rung == '1_orig':
        return items_mod.wav_of(item)
    return RENDER_DIR / f'{item}__{rung}__s{seed}.wav'


def plan(items, seeds):
    """[(item, rung, seed), ...] -- each stage checks existence for itself."""
    out = []
    for item in items:
        for rung in RUNGS:
            if rung in DETERMINISTIC:
                out.append((item, rung, DETERMINISTIC[rung]))
            else:
                out.extend((item, rung, s) for s in seeds)
    return out


def load_lyrics():
    if not LYRICS.exists():
        sys.exit(f'no lyrics file at: {LYRICS}')
    table = {}
    with open(LYRICS, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            table[row['item_name']] = row['text']
    return table


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def edit_distance(a, b) -> int:
    """Levenshtein. The sequences are short (~30 phonemes, ~30 characters), so pure Python is enough."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def hangul_only(text: str) -> str:
    """The comparison unit for CER. Whitespace, punctuation and Hanja are not realised in singing, so they are dropped."""
    return ''.join(ch for ch in text if '가' <= ch <= '힣')


# ---------------------------------------------------------------- 1. render

def stage_render(items, seeds):
    """Hand off to compare/render_new.py -- there is not a second copy of the render logic."""
    script = REPO / 'pansori' / 'compare' / 'render_new.py'
    # render_new's --seeds is a count (it slices C.SEEDS[:n]), not a seed value.
    cmd = [sys.executable, str(script), '--items', ','.join(items),
           '--seeds', str(len(seeds))]
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print(f'| render: {" ".join(cmd[:3])} … ({len(items)} items x {len(seeds)} seeds)')
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


# ---------------------------------------------------------------- 2. asr

def stage_asr(items, seeds, force=False):
    import librosa
    import numpy as np
    sys.path.insert(0, str(REPO / 'pansori' / 'preprocess'))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'transcribe', REPO / 'pansori' / 'preprocess' / '03_transcribe.py')
    transcribe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transcribe_mod)
    from pansori_lib import normalize_repeats

    from pansori.compare import items as I

    out_path = OUT_DIR / 'asr.jsonl'
    done = {} if force else {(r['item'], r['rung'], r['seed']): r for r in read_jsonl(out_path)}
    todo = [k for k in plan(items, seeds) if k not in done]
    todo = [(i, r, s) for i, r, s in todo if Path(wav_path(i, r, s, I)).exists()]
    if not todo:
        print(f'| asr: nothing new to do ({len(done)} rows)')
        return
    print(f'| asr: transcribing {len(todo)} (loading whisper-large-v3)')

    model, processor, device, dtype = transcribe_mod.load_model()
    rows = list(done.values())
    for n, (item, rung, seed) in enumerate(todo, 1):
        path = wav_path(item, rung, seed, I)
        audio, _ = librosa.load(str(path), sr=transcribe_mod.WHISPER_SAMPLE_RATE, mono=True)
        raw = transcribe_mod.transcribe(model, processor, device, dtype, np.asarray(audio))
        rows.append({'item': item, 'rung': rung, 'seed': seed,
                     'text_raw': raw, 'text': normalize_repeats(raw)})
        if n % 25 == 0 or n == len(todo):
            print(f'  {n}/{len(todo)}')
            write_jsonl(out_path, rows)
    write_jsonl(out_path, rows)
    print(f'→ {out_path}')


# ---------------------------------------------------------------- 3. mfa

def _load_align_module():
    """04_align.py starts with a digit and cannot be reached by an import statement."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'align', REPO / 'pansori' / 'preprocess' / '04_align.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_mfa(items, seeds, force=False):
    align = _load_align_module()
    from pansori.compare import items as I
    from pansori_lib import load_ipa_to_coda

    lyrics = load_lyrics()
    corpus = MFA_WORK / 'corpus'
    aligned = MFA_WORK / 'aligned'
    out_path = OUT_DIR / 'mfa.jsonl'
    done = {} if force else {(r['item'], r['rung'], r['seed']): r for r in read_jsonl(out_path)}

    todo, key_of = [], {}
    for item, rung, seed in plan(items, seeds):
        if (item, rung, seed) in done:
            continue
        src = Path(wav_path(item, rung, seed, I))
        if not src.exists() or item not in lyrics:
            continue
        todo.append((item, rung, seed, src))
    if not todo:
        print(f'| mfa: nothing new to do ({len(done)} rows)')
        return

    corpus.mkdir(parents=True, exist_ok=True)
    for old in corpus.iterdir():
        old.unlink()
    for item, rung, seed, src in todo:
        name = f'{item}__{rung}__s{seed}'
        key_of[name] = (item, rung, seed)
        link = corpus / f'{name}.wav'
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(src.resolve(), link)
        (corpus / f'{name}.lab').write_text(lyrics[item], encoding='utf-8')
    print(f'| mfa: aligning {len(todo)}')

    align.run_mfa(corpus, aligned)

    mapping = load_ipa_to_coda()
    rows = list(done.values())
    for tg in sorted(aligned.glob('*.TextGrid')):
        if tg.stem not in key_of:
            continue
        item, rung, seed = key_of[tg.stem]
        parsed = align.textgrid_to_coda(tg, mapping)
        rows.append({'item': item, 'rung': rung, 'seed': seed,
                     'phones': parsed['phones'], 'words': parsed['words'],
                     'spn': _count_spn(tg)})
    write_jsonl(out_path, rows)
    print(f'-> {out_path}  ({len(rows) - len(done)} TextGrids parsed)')


def _count_spn(tg_path) -> int:
    """How many spans MFA acoustically gave up on. textgrid_to_coda folds them into 'sil'."""
    from praatio import textgrid as tgio
    grid = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    return sum(1 for iv in grid.getTier('phones').entries
               if iv.label.strip().lower() == 'spn')


# ---------------------------------------------------------------- 4. score

def score_asr(rows, lyrics):
    """(item, rung, seed) -> CER."""
    out = {}
    for r in rows:
        ref = hangul_only(lyrics.get(r['item'], ''))
        hyp = hangul_only(r['text'])
        if not ref:
            continue
        out[(r['item'], r['rung'], r['seed'])] = edit_distance(ref, hyp) / len(ref)
    return out


def score_mfa(rows, gt_rows):
    """(item, rung, seed) -> {spn_rate, floor_rate, boundary_mae_ms, phone_match}."""
    out = {}
    for r in rows:
        gt = gt_rows.get(r['item'])
        if gt is None:
            continue
        phones = [p for p in r['phones'] if p[0] != 'sil']
        if not phones:
            continue
        durs = [end - start for _, start, end in phones]
        floor = sum(1 for d in durs if d <= MFA_FLOOR + 1e-6) / len(durs)

        # Boundary error: pair the GT phoneme boundaries with the re-aligned ones in
        # order. If the counts differ, go only as far as the shorter -- the length
        # difference itself already shows up in phone_match.
        gt_bounds, hyp_bounds = _bounds(gt['phones']), _bounds(phones)
        n = min(len(gt_bounds), len(hyp_bounds))
        mae = (sum(abs(a - b) for a, b in zip(gt_bounds[:n], hyp_bounds[:n])) / n * 1000
               if n else float('nan'))

        gt_seq = [p[0] for p in gt['phones']]
        hyp_seq = [p[0] for p in phones]
        match = 1 - edit_distance(gt_seq, hyp_seq) / max(len(gt_seq), 1)

        out[(r['item'], r['rung'], r['seed'])] = {
            'spn_rate': r['spn'] / max(len(r['words']), 1),
            'floor_rate': floor,
            'boundary_mae_ms': mae,
            'phone_match': match,
        }
    return out


def _bounds(phones):
    return [end for _, _, end in phones]


def gt_phones(items):
    """The training labels' phoneme sequence and boundaries -- what the re-alignment is measured against."""
    from pansori.compare import items as I
    out = {}
    for item in items:
        row = I.row_of(item)
        seq = row['ph_seq'].split()
        durs = [float(x) for x in row['ph_dur'].split()]
        phones, t = [], 0.0
        for phone, dur in zip(seq, durs):
            if phone not in ('SP', 'AP'):
                phones.append([phone, t, t + dur])
            t += dur
        out[item] = {'phones': phones}
    return out


def _by_rung(scored, key=None):
    """(item, rung, seed) -> value  =>  rung -> [per-utterance median over seeds]."""
    per_item = defaultdict(lambda: defaultdict(list))
    for (item, rung, _), value in scored.items():
        v = value if key is None else value[key]
        if v == v:                       # skip NaN
            per_item[rung][item].append(v)
    return {rung: [median(vs) for vs in items.values()]
            for rung, items in per_item.items()}


def stage_score(items, seeds):
    lyrics = load_lyrics()
    asr_rows = read_jsonl(OUT_DIR / 'asr.jsonl')
    mfa_rows = read_jsonl(OUT_DIR / 'mfa.jsonl')
    if not asr_rows and not mfa_rows:
        sys.exit('nothing to measure -- run --stage asr / --stage mfa first')

    report = {'n_items': len(items), 'seeds': list(seeds), 'whisper': {}, 'mfa': {}}

    if asr_rows:
        cer = _by_rung(score_asr(asr_rows, lyrics))
        null = median(cer['1_orig']) if cer.get('1_orig') else float('nan')
        print('\n── whisper CER ' + '─' * 46)
        print(f"  {'rung':<12} {'n':>4} {'CER':>8} {'vs 0_vocoder':>15}")
        base = median(cer['0_vocoder']) if cer.get('0_vocoder') else float('nan')
        for rung in RUNGS:
            vals = cer.get(rung)
            if not vals:
                continue
            m = median(vals)
            delta = '—' if rung in ('1_orig', '0_vocoder') else f'{m - base:+.3f}'
            print(f'  {rung:<12} {len(vals):>4} {m:>8.3f} {delta:>15}')
            report['whisper'][rung] = {'n': len(vals), 'cer_median': m}
        print(f'\n  null control (1_orig): {null:.3f}  <- if not near 0, the numbers below are void')
        print('  The reference text is Whisper\'s transcription of the original, so 1_orig is not a ceiling.')
        print('  The synthesis ceiling is 0_vocoder.')
        report['whisper']['_null_control'] = null

    if mfa_rows:
        gt = gt_phones(items)
        scored = score_mfa(mfa_rows, gt)
        print('\n── MFA re-alignment ' + '─' * 47)
        print(f"  {'rung':<12} {'n':>4} {'floor':>9} {'bound MAE ms':>13} "
              f"{'ph match*':>10}")
        keys = ('floor_rate', 'boundary_mae_ms', 'phone_match', 'spn_rate')
        grouped = {key: _by_rung(scored, key) for key in keys}
        for rung in RUNGS:
            cells = {'n': 0}
            for key in keys:
                vals = grouped[key].get(rung)
                cells[key] = median(vals) if vals else float('nan')
                cells['n'] = max(cells['n'], len(vals) if vals else 0)
            if not cells['n']:
                continue
            print(f"  {rung:<12} {cells['n']:>4} {cells['floor_rate'] * 100:>8.1f}% "
                  f"{cells['boundary_mae_ms']:>11.0f} {cells['phone_match'] * 100:>9.1f}%")
            report['mfa'][rung] = cells
        oov = grouped['spn_rate'].get('1_orig')
        print(f"\n  floor = the fraction of the re-alignment pinned to 30ms. It rises when the")
        print(f"  sound does not contain that phoneme (the same ruler label_report.py applies to the labels).")
        print(f"  * ph match is near-constant because the lyrics we supplied fix the phoneme sequence -- a reference value.")
        if oov:
            print(f"  OOV (spn) rate in the sample {median(oov) * 100:.0f}% -- a property of the lyrics, so "
                  f"identical across rungs. Not used to compare rungs.")
            report['mfa']['_oov_rate'] = median(oov)

    out = OUT_DIR / 'intelligibility.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n→ {out}')
    return report


# ---------------------------------------------------------------- wandb

def push_wandb(report, project, group, name):
    import wandb
    run = wandb.init(project=project, group=group, name=name,
                     job_type='intelligibility', config={'n_items': report['n_items']})
    flat = {}
    for backend in ('whisper', 'mfa'):
        for rung, value in report.get(backend, {}).items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        flat[f'{backend}/{rung}/{k}'] = v
            elif isinstance(value, (int, float)):
                flat[f'{backend}/{rung}'] = value
    run.log(flat)
    run.finish()
    print(f'→ wandb: {run.url}')


# ---------------------------------------------------------------- driver

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', required=True,
                   choices=['render', 'asr', 'mfa', 'score', 'all'])
    p.add_argument('--items', default=None,
                   help='comma-separated list. Default: a stratified hold-out sample of --n')
    p.add_argument('--n', type=int, default=40, help='sample size (default 40)')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--force', action='store_true', help='rebuild rows that already exist')
    p.add_argument('--wandb', action='store_true', help='upload the score results to wandb')
    p.add_argument('--wandb-project', default='pansori')
    p.add_argument('--wandb-group', default=None)
    p.add_argument('--wandb-name', default='intelligibility')
    args = p.parse_args()

    from pansori.compare import items as I
    if args.items:
        items = [s for s in args.items.split(',') if s]
        for item in items:
            I.assert_held_out(item)      # if training data leaks in, the numbers mean nothing
    else:
        items = I.holdout_sample(args.n)
    seeds = tuple(range(1, args.seeds + 1))
    print(f'| {len(items)} items · seeds {seeds}')

    stages = [args.stage] if args.stage != 'all' else ['render', 'asr', 'mfa', 'score']
    for stage in stages:
        try:
            if stage == 'render':
                stage_render(items, seeds)
            elif stage == 'asr':
                stage_asr(items, seeds, force=args.force)
            elif stage == 'mfa':
                stage_mfa(items, seeds, force=args.force)
            elif stage == 'score':
                report = stage_score(items, seeds)
                if args.wandb:
                    push_wandb(report, args.wandb_project, args.wandb_group, args.wandb_name)
        except ImportError as exc:
            if args.stage != 'all':
                raise
            print(f'| {stage} skipped (a dependency missing from this env): {exc}')


if __name__ == '__main__':
    main()

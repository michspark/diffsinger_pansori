#!/usr/bin/env python3
"""합성음의 **발음**을 잰다 — 사다리 각 단에서 어디가 무너지는지 가른다.

`pansori/compare/` 의 지표 22개는 전부 피치 아니면 스펙트럼이다. 가장 가까운
`voiced_agree` 도 "이 프레임에서 소리를 내야 하는가"만 재므로, 자음을 통째로 빼먹거나
`g` 를 `d` 로 발음해도 떨어지지 않는다 (홀드아웃에서 0.95~0.97 로 높게 나온다).
그래서 지금까지의 모든 판정이 발음을 보지 못한 채 내려졌다. 이 스크립트가 그 구멍을 메운다.

## 사다리로 범인을 가린다

    1_orig      원본 녹음                        — 널 대조 (아래 참고)
    0_vocoder   GT mel + GT f0 → 보코더          — 보코더 천장
    2_resynth   실제 ph_dur + 실제 f0 → acoustic — 라벨과 음향모델
    3_pitch     실제 ph_dur, f0 예측
    4_full      노트+가사만, dur/f0 둘 다 예측

`2_resynth` 는 GT ph_dur 을 그대로 먹인다. 그 GT 가 바로 22% 가 MFA 하한선에 박혀 있는
라벨이므로 (`pansori/eval/label_report.py`), 여기서 이미 무너지면 범인은 라벨이거나
음향모델이고, `4_full` 에서만 무너지면 duration 예측이다. `0_vocoder` 가 이미 나쁘면
그 아래 전부가 무의미하다.

## 두 자를 같이 쓴다 — 서로의 약점을 가려 준다

**whisper**: 렌더 오디오를 whisper-large-v3 로 전사해 입력 가사와 CER. 사람이 알아듣는
것에 가장 가깝지만, Whisper 는 언어모델을 갖고 있어 **없는 소리를 채워 넣는다**.

**mfa**: 렌더 오디오와 아는 가사를 korean_mfa 로 다시 정렬하고, 정렬기가 **음향적 근거를
찾았는지**를 본다. 두 가지다:

  - `floor_rate` — 재정렬 결과가 MFA 최소 길이(30 ms)에 박힌 비율. 소리에 그 음소가
    없으면 정렬기는 최소값에 park 하고 넘어간다. 라벨 쪽을 재는
    `pansori/eval/label_report.py` 와 정확히 같은 자를 합성음에 대는 것이다.
  - `boundary_mae_ms` — GT 음소 경계와 재정렬 경계의 평균 차이.

**`spn` 은 단 비교에 쓰지 않는다.** 처음에는 "렌더된 소리에 그 자음이 없으면 MFA 가 spn 을
낸다"고 보고 넣었는데, 실측해 보니 **같은 발화의 다섯 단에서 값이 완전히 동일했다**.
spn 은 사전에 없는 단어(OOV)에 대해 나오는 것이라 오디오가 아니라 **가사의 성질**이다.
(`pansori/docs/LYRICBOOK_VERDICT.md` §4 에서 spn 이 164→202 로 는 것도 텍스트가 바뀐 실험이었으므로
모순되지 않는다.) 표본의 OOV 정도를 알려 주는 참고값으로만 한 번 인쇄한다.

`phone_match` 도 같은 이유로 거의 상수다 — 음소열은 우리가 준 가사가 정한다. MFA 가
음소를 삽입·삭제할 때만 움직이므로 참고로만 둔다.

## `1_orig` 은 상한선이 아니다 — 널 대조다

CER 의 참조 텍스트는 `data/full/lyrics.jsonl` 인데, 그것 자체가 **Whisper 가 이 녹음에서
뽑은 전사**다 (03_transcribe.py). 그래서 원본을 다시 전사하면 자기 자신과 비교하는 셈이라
CER 이 0 에 가깝게 나온다. 이건 판소리를 잘 알아듣는다는 뜻이 **아니고**, Whisper 의 디코딩이
재현된다는 뜻이다. 값이 0 근처가 아니면 그 뒤 숫자는 전부 무효다.

합성음의 실질적 천장은 `0_vocoder` 다. MFA 쪽에서는 반대로 `1_orig` 이 제대로 된 천장이다
(정렬기는 오디오와 텍스트를 독립적으로 받으므로 순환이 없다).

## 실행 — env 가 셋이라 단계를 나눈다

    conda activate dsp     && PYTHONPATH=. python pansori/eval/intelligibility.py --stage render
    conda activate pansori && PYTHONPATH=. python pansori/eval/intelligibility.py --stage asr
    conda activate aligner && PYTHONPATH=. python pansori/eval/intelligibility.py --stage mfa
    conda activate dsp     && PYTHONPATH=. python pansori/eval/intelligibility.py --stage score

`--stage all` 은 한 env 안에서 가능한 단계만 돌린다. 각 단계는 이미 만든 산출물을 건너뛰므로
중간에 끊고 다시 시작해도 된다.

출력: eval/out/{asr,mfa}.jsonl → eval/out/intelligibility.json + 콘솔 표
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

# 시드를 안 쓰는 단. 원본과 보코더 천장은 확률적 요소가 없다.
DETERMINISTIC = {'1_orig': None, '0_vocoder': 0}
SAMPLED = ('2_resynth', '3_pitch', '4_full')
RUNGS = ('1_orig', '0_vocoder') + SAMPLED

MFA_FLOOR = 0.030


# ---------------------------------------------------------------- 공통

def wav_path(item, rung, seed, items_mod):
    if rung == '1_orig':
        return items_mod.wav_of(item)
    return RENDER_DIR / f'{item}__{rung}__s{seed}.wav'


def plan(items, seeds):
    """[(item, rung, seed), ...] — 존재 여부는 각 단계가 따로 확인한다."""
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
        sys.exit(f'가사 파일이 없다: {LYRICS}')
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
    """Levenshtein. 시퀀스가 짧아(음소 ~30, 글자 ~30) 순수 파이썬으로 충분하다."""
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
    """CER 의 비교 단위. 공백·문장부호·한자는 노래로 실현되지 않으므로 버린다."""
    return ''.join(ch for ch in text if '가' <= ch <= '힣')


# ---------------------------------------------------------------- 1. render

def stage_render(items, seeds):
    """compare/render_new.py 에 넘긴다 — 렌더 로직을 두 벌 두지 않는다."""
    script = REPO / 'pansori' / 'compare' / 'render_new.py'
    # render_new 의 --seeds 는 개수다 (C.SEEDS[:n] 를 자른다), 시드 값이 아니다.
    cmd = [sys.executable, str(script), '--items', ','.join(items),
           '--seeds', str(len(seeds))]
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print(f'| render: {" ".join(cmd[:3])} … ({len(items)}항목 x {len(seeds)}시드)')
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
        print(f'| asr: 새로 할 것 없음 ({len(done)}행)')
        return
    print(f'| asr: {len(todo)}개 전사 (whisper-large-v3 로드 중)')

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
    """04_align.py 는 숫자로 시작해 import 문으로 못 부른다."""
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
        print(f'| mfa: 새로 할 것 없음 ({len(done)}행)')
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
    print(f'| mfa: {len(todo)}개 정렬')

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
    print(f'→ {out_path}  (TextGrid {len(rows) - len(done)}개 파싱)')


def _count_spn(tg_path) -> int:
    """MFA 가 음향적으로 포기한 구간 수. textgrid_to_coda 는 'sil' 로 접어 버린다."""
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

        # 경계 오차: GT 음소 경계와 재정렬 경계를 순서대로 짝지어 비교한다. 개수가
        # 다르면 짧은 쪽까지만 — 길이 자체가 이미 phone_match 로 드러난다.
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
    """학습 라벨의 음소열과 경계 — 재정렬 결과를 여기에 대고 잰다."""
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
    """(item, rung, seed) -> 값  =>  rung -> [발화별 시드 중앙값]."""
    per_item = defaultdict(lambda: defaultdict(list))
    for (item, rung, _), value in scored.items():
        v = value if key is None else value[key]
        if v == v:                       # NaN 제외
            per_item[rung][item].append(v)
    return {rung: [median(vs) for vs in items.values()]
            for rung, items in per_item.items()}


def stage_score(items, seeds):
    lyrics = load_lyrics()
    asr_rows = read_jsonl(OUT_DIR / 'asr.jsonl')
    mfa_rows = read_jsonl(OUT_DIR / 'mfa.jsonl')
    if not asr_rows and not mfa_rows:
        sys.exit('잴 것이 없다 — --stage asr / --stage mfa 를 먼저 돌릴 것')

    report = {'n_items': len(items), 'seeds': list(seeds), 'whisper': {}, 'mfa': {}}

    if asr_rows:
        cer = _by_rung(score_asr(asr_rows, lyrics))
        null = median(cer['1_orig']) if cer.get('1_orig') else float('nan')
        print('\n── whisper CER ' + '─' * 46)
        print(f"  {'단':<12} {'n':>4} {'CER':>8} {'0_vocoder 대비':>15}")
        base = median(cer['0_vocoder']) if cer.get('0_vocoder') else float('nan')
        for rung in RUNGS:
            vals = cer.get(rung)
            if not vals:
                continue
            m = median(vals)
            delta = '—' if rung in ('1_orig', '0_vocoder') else f'{m - base:+.3f}'
            print(f'  {rung:<12} {len(vals):>4} {m:>8.3f} {delta:>15}')
            report['whisper'][rung] = {'n': len(vals), 'cer_median': m}
        print(f'\n  널 대조 (1_orig): {null:.3f}  ← 0 근처가 아니면 아래 숫자는 무효다')
        print('  참조 텍스트가 원본의 Whisper 전사라서 1_orig 은 천장이 아니다.')
        print('  합성음의 천장은 0_vocoder 다.')
        report['whisper']['_null_control'] = null

    if mfa_rows:
        gt = gt_phones(items)
        scored = score_mfa(mfa_rows, gt)
        print('\n── MFA 재정렬 ' + '─' * 47)
        print(f"  {'단':<12} {'n':>4} {'하한선율':>9} {'경계MAE ms':>11} "
              f"{'음소일치*':>10}")
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
        print(f"\n  하한선율 = 재정렬이 30ms 에 박힌 비율. 소리에 그 음소가 없으면 오른다")
        print(f"  (label_report.py 가 라벨에 대는 것과 같은 자).")
        print(f"  * 음소일치는 우리가 준 가사가 음소열을 정하므로 거의 상수다 — 참고값.")
        if oov:
            print(f"  표본의 OOV(spn) 비율 {median(oov) * 100:.0f}% — 가사의 성질이라 "
                  f"단마다 같다. 단 비교에 쓰지 않는다.")
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
                   help='쉼표 목록. 기본은 홀드아웃 층화표본 --n 개')
    p.add_argument('--n', type=int, default=40, help='표본 크기 (기본 40)')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--force', action='store_true', help='이미 만든 행도 다시 만든다')
    p.add_argument('--wandb', action='store_true', help='score 결과를 wandb 에 올린다')
    p.add_argument('--wandb-project', default='pansori')
    p.add_argument('--wandb-group', default=None)
    p.add_argument('--wandb-name', default='intelligibility')
    args = p.parse_args()

    from pansori.compare import items as I
    if args.items:
        items = [s for s in args.items.split(',') if s]
        for item in items:
            I.assert_held_out(item)      # 학습 데이터가 새면 숫자가 무의미하다
    else:
        items = I.holdout_sample(args.n)
    seeds = tuple(range(1, args.seeds + 1))
    print(f'| 항목 {len(items)}개 · 시드 {seeds}')

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
            print(f'| {stage} 건너뜀 (이 env 에 없는 의존성): {exc}')


if __name__ == '__main__':
    main()

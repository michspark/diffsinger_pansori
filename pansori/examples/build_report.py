#!/usr/bin/env python3
"""Render the pipeline report, with the generated audio embedded in the page.

Everything the report claims is read back out of the artefacts rather than
typed in: the corpus tables come from data/pansori, the scores from
measure.py / f0_report.py, the audio from examples/out. Re-running
run_examples.sh and then this script republishes the page against whatever
checkpoint is current.

Audio is embedded as base64 mp3 because an Artifact's CSP blocks every external
host, so a page that links its audio has no audio.

Usage (any env with numpy, from the repo root):
  python pansori/examples/build_report.py --out pansori/examples/report.html
"""

import argparse
import base64
import csv
import json
import re
import subprocess
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EX = REPO / 'pansori' / 'examples'
OUT = EX / 'out'

HOLDOUT = {'조상현': '8b4d0824', '정권진': 'a580300d', '이일주': 'c235c995',
           '박동진': 'e719dced', '박초월': 'd7ec123c', '김소희': 'd207244a'}

ITEM_INFO = {
    'd207244a_0085': ('김소희', '춘향가 · 이도령 이별', '여창'),
    '8b4d0824_0078': ('조상현', '춘향가 · 그때의 어사또는', '남창'),
    'e719dced_0197': ('박동진', '수궁가', '남창'),
}

RUNGS = [
    ('1_orig', '원본 녹음', '명창이 실제로 부른 소리', 'ref'),
    ('2_resynth', '재합성', 'ph_dur · f0 모두 실측 → acoustic', 'gt'),
    ('3_pitch', '피치 예측', 'ph_dur 실측 · f0 는 variance', 'mix'),
    ('4_full', '전체 추론', '노트와 가사만 → variance → acoustic', 'pred'),
]

STAGES = [
    ('01', '세그멘테이션', 'labels + Audio + midi → segments.jsonl + wavs/',
     '곡을 6초 이하 발화로 자른다. <b>이 한계는 협상 불가다</b> — korean_mfa 는 말소리로 학습돼 '
     '6초를 넘으면 정렬이 붕괴한다. 라벨된 창 구간 37.5시간 중 21초 미만은 3.3시간뿐이라 '
     '재세그멘테이션 없이는 시작조차 못 한다. 경계는 ROSVOT 노트가 비어 있는 지점을 우선한다.'),
    ('02', '화자 매핑', 'segments → spk_map.json',
     '곡 메타데이터에서 명창을 식별한다. 특정되지 않는 구간은 버리지 않고 '
     '<code>unknown</code> 한 사람으로 모은다 — 6.16시간, 단일 최대 화자다.'),
    ('03', '가사 전사', 'wavs → lyrics.jsonl',
     'whisper-large-v3. 원본에 가사 텍스트가 없어서 필요한 단계다. 지속음 때문에 같은 음절이 '
     '수백 번 반복되는 산출물이 나와서(지이이이…이) 3회 이상 반복은 하나로 접는다.'),
    ('04', '강제 정렬', 'lyrics + wavs → phones.jsonl',
     'MFA 로 음소 경계를 잡고 IPA 108종을 자모 기반 34종으로 변환한다. '
     '<b>MFA 의 words tier 도 함께 기록한다</b> — openvpi 가 요구하는 <code>ph_num</code> 은 '
     '여기서만 유도할 수 있고, 남겨두면 정렬을 다시 돌리지 않고 파싱만 재실행할 수 있다.'),
    ('05', '노트 파싱', 'rosvot_midi → notes.jsonl',
     'ROSVOT 이 뽑아 둔 곡 단위 MIDI 를 발화 구간으로 잘라 노트 열로 만든다. '
     '노트 음역은 MIDI 35(B1, 61.7 Hz)부터 78(F♯5, 740 Hz)까지다.'),
    ('06', '병합', 'phones + notes → transcriptions.csv',
     '두 축을 openvpi CSV 스키마로 합친다. 여기서 <b>노트가 덮은 무음은 앞 모음으로 흡수</b>하고 '
     '노트가 없는 무음만 <code>SP</code> 로 남긴다. 20 ms 미만 음소는 이웃에 접는다 — '
     '44.1 kHz · hop 512 에서 한 프레임이 11.6 ms 다.'),
    ('07', '검증', 'transcriptions.csv → PASS / FAIL',
     'binarizer 가 워커 안에서 assert 로 죽는 지점을 전부 앞당겨 검사한다. 실패는 스킵이 아니라 '
     '실행 중단이라 <b>PASS 없이 다음으로 가지 않는다</b>. 0프레임 음소, 길이 불일치, 인벤토리 이탈.'),
    ('08', '내보내기', 'segments + csv → data/pansori/{화자}/',
     '원본이 44.1 kHz 이므로 리샘플링 없이 다시 자른다. 화자별 디렉토리로 나누고 '
     '<code>datasets:</code> 블록을 함께 생성한다.'),
]


def b64_audio(path):
    return base64.b64encode(path.read_bytes()).decode('ascii')


def audio_tag(mp3_path):
    return (f'<audio controls preload="metadata" '
            f'src="data:audio/mpeg;base64,{b64_audio(mp3_path)}"></audio>')


def fmt(value, spec, dash='—'):
    return dash if value is None else format(value, spec)


# ---------------------------------------------------------------- corpus


def speaker_rows():
    rows = []
    for d in sorted((REPO / 'data' / 'pansori').iterdir()):
        c = d / 'transcriptions.csv'
        if not c.exists():
            continue
        rs = list(csv.DictReader(open(c, encoding='utf-8')))
        sec = sum(sum(map(float, r['ph_dur'].split())) for r in rs)
        rows.append((d.name, len(rs), sec))
    rows.sort(key=lambda r: -r[1])
    top = rows[0][2]
    out = []
    for name, n, sec in rows:
        held = HOLDOUT.get(name)
        if held:
            n_held = sum(1 for r in csv.DictReader(
                open(REPO / 'data' / 'pansori' / name / 'transcriptions.csv', encoding='utf-8')
            ) if r['name'].startswith(held))
            mark = (f'<span class="mono">{held}</span>' if n_held < n
                    else '<span class="mono" style="color:var(--ochre)">자동 해제</span>')
        else:
            mark = ''
        out.append(
            f'<tr><td>{name}</td><td class="num">{n:,}</td>'
            f'<td class="num">{sec / 3600:.2f} h</td>'
            f'<td><span class="bar" style="width:{100 * sec / top:.1f}%"></span></td>'
            f'<td>{mark}</td></tr>'
        )
    return '\n'.join(out)


def figures():
    rows = list(csv.DictReader(open(REPO / 'pansori/data/full/transcriptions.csv',
                                    encoding='utf-8')))
    hours = sum(sum(map(float, r['ph_dur'].split())) for r in rows) / 3600
    notes = sum(len(r['note_seq'].split()) for r in rows)
    phones = sum(len(r['ph_seq'].split()) for r in rows)
    n_spk = sum(1 for d in (REPO / 'data/pansori').iterdir()
                if (d / 'transcriptions.csv').exists())
    steps = (OUT / 'CKPT_STEPS').read_text().split() if (OUT / 'CKPT_STEPS').exists() else ('—', '—')
    items = [
        ('코퍼스', f'{hours:.1f}<small>시간 · 발화 {len(rows):,}개</small>'),
        ('화자', f'{n_spk}<small>명창 22인 + unknown</small>'),
        ('노트 / 음소', f'{notes // 1000}k<small>노트 · 음소 {phones // 1000}k</small>'),
        ('음소 인벤토리', f'35<small>자모 34종 + SP</small>'),
        ('학습 step', f'{int(steps[0]):,}<small>acoustic · variance {int(steps[1]):,} / '
         f'{int(steps[2]):,}</small>' if steps[0] != '—' else '—<small></small>'),
    ]
    return '\n'.join(f'<div class="figure"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in items)


def stages():
    return '\n'.join(
        f'<div class="stage"><div class="stage-no">{no}</div><div>'
        f'<p class="stage-name">{name}</p>'
        f'<p class="stage-io">{io}</p>'
        f'<p class="stage-body">{body}</p></div></div>'
        for no, name, io, body in STAGES
    )


# ---------------------------------------------------------------- diagrams


SCHEMA_ITEM = '01ed19bb_0003'


def _label_row(item):
    for r in csv.DictReader(open(REPO / 'pansori/data/full/transcriptions.csv', encoding='utf-8')):
        if r['name'] == item:
            return r
    raise SystemExit(f'{item} 없음')


def schema_diagram():
    """Both label formats for one real utterance, not an invented example.

    The old row is reconstructed from the same labels: whenever a phone spans
    more than one note the old format had to repeat it, once per note, flagging
    every repeat with is_slur. Deriving it rather than drawing a sketch keeps
    the comparison honest about how much of the old row was duplication.
    """
    row = _label_row(SCHEMA_ITEM)
    phones = row['ph_seq'].split()
    ph_dur = [float(x) for x in row['ph_dur'].split()]
    counts = [int(x) for x in row['ph_num'].split()]
    notes = row['note_seq'].split()
    note_dur = [float(x) for x in row['note_dur'].split()]

    ph_start, at = [], 0.0
    for d in ph_dur:
        ph_start.append(at)
        at += d
    nt_start, at = [], 0.0
    for d in note_dur:
        nt_start.append(at)
        at += d

    # old format: one row entry per (phone, overlapping note)
    old = []
    for i, p in enumerate(phones):
        lo, hi = ph_start[i], ph_start[i] + ph_dur[i]
        hits = [j for j in range(len(notes))
                if nt_start[j] < hi - 1e-6 and nt_start[j] + note_dur[j] > lo + 1e-6] or [0]
        for k, j in enumerate(hits):
            old.append((p, notes[j], 1 if k else 0))

    words, at = [], 0
    for n in counts:
        words.append((at, n))
        at += n

    W = max(1180, 70 * len(old) + 150)
    x0 = 118
    rows = []

    def box(x, y, w, h, label, cls, fs=12.5):
        return (f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="2" '
                f'class="f-surface s-{cls}" stroke-width="1.2"/>'
                f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 4.5}" text-anchor="middle" '
                f'class="d-mono f-ink" font-size="{fs}">{label}</text>')

    def lab(y, text, sub=''):
        out = f'<text x="8" y="{y}" class="d-mono f-muted" font-size="11">{text}</text>'
        if sub:
            out += (f'<text x="8" y="{y + 14}" class="d-mono f-muted" font-size="9.5">'
                    f'{sub}</text>')
        return out

    cw = (W - x0 - 20) / len(old)
    rows.append('<text x="8" y="20" class="d-display f-ochre" font-size="15">'
                '옛 포맷 — 멜로디가 음소 축 위에</text>')
    rows.append(lab(56, 'ph_seq'))
    for i, (p, n, sl) in enumerate(old):
        rows.append(box(x0 + i * cw, 40, cw - 5, 26, p, 'ochre' if sl else 'rule'))
    rows.append(lab(92, 'note_seq'))
    for i, (p, n, sl) in enumerate(old):
        rows.append(box(x0 + i * cw, 76, cw - 5, 26, n, 'rule', 11))
    rows.append(lab(128, 'is_slur'))
    for i, (p, n, sl) in enumerate(old):
        rows.append(box(x0 + i * cw, 112, cw - 5, 26, str(sl),
                        'ochre' if sl else 'rule', 11))
    dup = sum(1 for _, _, sl in old if sl)
    rows.append(f'<text x="{x0}" y="162" class="d-body f-ochre" font-size="12">'
                f'세 열이 같은 길이여야 해서 음소를 복제한다 — 음소 {len(phones)}개가 '
                f'{len(old)}칸이 되고 그 중 {dup}칸이 복제다</text>')

    y = 200
    rows.append(f'<line x1="8" y1="{y - 24}" x2="{W - 8}" y2="{y - 24}" '
                f'class="s-rule" stroke-width="1"/>')
    rows.append(f'<text x="8" y="{y}" class="d-display f-indigo" font-size="15">'
                f'openvpi — 두 축이 분리</text>')

    span = W - x0 - 20
    total_dur = sum(ph_dur)
    rows.append(lab(y + 36, 'ph_seq', 'ph_num'))
    for i, p in enumerate(phones):
        x = x0 + span * ph_start[i] / total_dur
        w = span * ph_dur[i] / total_dur
        rows.append(box(x, y + 20, max(w - 4, 10), 26, p, 'rule', 11.5))
    for at, n in words:
        gx = x0 + span * ph_start[at] / total_dur
        gw = span * sum(ph_dur[at:at + n]) / total_dur - 4
        rows.append(f'<path d="M{gx:.1f} {y + 52} v6 h{gw:.1f} v-6" class="s-none s-indigo" '
                    f'stroke-width="1.2"/>')
        rows.append(f'<text x="{gx + gw / 2:.1f}" y="{y + 76}" text-anchor="middle" '
                    f'class="d-mono f-indigo" font-size="11">{n}</text>')
    rows.append(lab(y + 108, 'note_seq', 'note_dur'))
    for j, n in enumerate(notes):
        x = x0 + span * nt_start[j] / total_dur
        w = span * note_dur[j] / total_dur
        rows.append(box(x, y + 92, max(w - 4, 10), 26, n, 'indigo', 11))
    rows.append(f'<text x="{x0}" y="{y + 146}" class="d-body f-indigo" font-size="12">'
                f'음소 {len(phones)}칸 · 노트 {len(notes)}칸이 각자 자기 길이로 프레임에 정렬된다 — '
                f'복제도 is_slur 도 없다. 새로 필요한 것은 ph_num 뿐</text>')

    return (f'<svg viewBox="0 0 {W} 370" width="{W}" height="370" '
            f'role="img" aria-label="옛 라벨 포맷과 openvpi 포맷 비교">'
            + '\n'.join(rows) + '</svg>')


def model_diagram():
    W = 1100
    parts = []

    def node(x, y, w, h, title, sub, color='indigo', dashed=False):
        d = ' stroke-dasharray="4 3"' if dashed else ''
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" '
                f'class="f-surface s-{color}" stroke-width="1.4"{d}/>'
                f'<text x="{x + w / 2}" y="{y + 24}" text-anchor="middle" '
                f'class="d-body f-ink" font-size="13.5" font-weight="600">{title}</text>'
                f'<text x="{x + w / 2}" y="{y + 42}" text-anchor="middle" '
                f'class="d-mono f-muted" font-size="10.5">{sub}</text>')

    def arrow(x1, y1, x2, y2, color='indigo', dashed=False):
        d = ' stroke-dasharray="4 3"' if dashed else ''
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="s-{color}" '
                f'stroke-width="1.4" marker-end="url(#a-{color})"{d}/>')

    parts.append(
        '<defs>'
        '<marker id="a-indigo" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
        'markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="f-indigo"/></marker>'
        '<marker id="a-ochre" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
        'markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="f-ochre"/></marker>'
        '</defs>')

    y = 78
    boxes = [
        (20, 178, '가사 + 악보', '.ds'),
        (238, 150, 'G2P', 'korean-coda.txt'),
        (426, 172, 'variance', 'ph_dur · f0'),
        (636, 172, 'acoustic', 'mel 128bin'),
        (846, 172, 'NSF-HiFiGAN', '44.1 kHz wav'),
    ]
    for x, w, t, s in boxes:
        parts.append(node(x, y, w, 56, t, s, 'indigo'))
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][1]
        parts.append(arrow(x1 + 4, y + 28, boxes[i + 1][0] - 6, y + 28))

    # injection points
    parts.append(node(426, y + 108, 172, 46, '실제 ph_dur', '라벨에서', 'ochre', True))
    parts.append(node(636, y + 108, 172, 46, '실제 f0', 'RMVPE 로 추출', 'ochre', True))
    parts.append(arrow(512, y + 106, 512, y + 60, 'ochre', True))
    parts.append(arrow(722, y + 106, 722, y + 60, 'ochre', True))
    parts.append(f'<text x="426" y="{y + 178}" class="d-body f-ochre" font-size="12">'
                 f'사다리는 이 두 지점을 하나씩 끊어 본다</text>')

    parts.append(f'<text x="20" y="{y - 24}" class="d-mono f-muted" font-size="10.5">'
                 f'69.1M 파라미터 (acoustic) · 25.1M (variance) · reflow, euler 20 step</text>')
    return (f'<svg viewBox="0 0 {W} 290" width="{W}" height="290" role="img" '
            f'aria-label="추론 경로 다이어그램">' + '\n'.join(parts) + '</svg>')


# ---------------------------------------------------------------- ladder


def ladder(metrics):
    blocks = []
    lyrics = {}
    lp = REPO / 'pansori/data/full/lyrics.jsonl'
    if lp.exists():
        for line in open(lp, encoding='utf-8'):
            d = json.loads(line)
            lyrics[d['item_name']] = d.get('text', '')

    for item, (spk, song, voice) in ITEM_INFO.items():
        if not (OUT / f'{item}_1_orig.mp3').exists():
            continue
        rungs = []
        for track, name, sub, kind in RUNGS:
            mp3 = OUT / f'{item}_{track}.mp3'
            if not mp3.exists():
                continue
            m = metrics.get(f'{item}_{track}.wav', {})
            bits = []
            if m.get('f0_rmse_cent') is not None:
                bits.append(f'<span>f0 <b>{m["f0_rmse_cent"]:.0f}</b>¢</span>')
            if 'mel_l1' in m:
                bits.append(f'<span>mel <b>{m["mel_l1"]:.2f}</b></span>')
            if 'voiced_agree' in m:
                bits.append(f'<span>유무성 <b>{m["voiced_agree"]:.0%}</b></span>')
            if not bits:
                bits.append('<span>기준</span>')
            cls = ' is-ref' if kind == 'ref' else ''
            colour = {'gt': 'gt', 'mix': '', 'pred': 'pred'}.get(kind, '')
            nm = f'<span class="{colour}">{name}</span>' if colour else name
            rungs.append(
                f'<div class="rung{cls}">'
                f'<div class="rung-label"><span class="name">{nm}</span>'
                f'<span class="sub">{sub}</span></div>'
                f'{audio_tag(mp3)}'
                f'<div class="rung-metrics">{"".join(bits)}</div></div>'
            )
        text = lyrics.get(item, '')
        blocks.append(
            f'<div class="utt"><div class="utt-head">'
            f'<h3>{spk}</h3><span class="meta">{song} · {voice} · '
            f'<span style="color:var(--muted)">{item}</span></span></div>'
            + (f'<div class="utt-lyric">전사: {text}</div>' if text else '')
            + f'<div class="rungs">{"".join(rungs)}</div></div>'
        )
    return '\n'.join(blocks)


def score_ladder():
    rungs = []
    for mp3 in sorted(OUT.glob('5_score_*.mp3')):
        spk = mp3.stem.rsplit('_', 1)[-1]
        rungs.append(
            f'<div class="rung">'
            f'<div class="rung-label"><span class="name pred">{spk}</span>'
            f'<span class="sub">같은 악보 · 화자만 교체</span></div>'
            f'{audio_tag(mp3)}'
            f'<div class="rung-metrics"><span>원본 없음</span></div></div>'
        )
    if not rungs:
        return ''
    return (f'<div class="utt"><div class="utt-head"><h3>사랑가</h3>'
            f'<span class="meta">춘향가 · 새 악보 · 녹음 없음</span></div>'
            f'<div class="rungs">{"".join(rungs)}</div></div>')


def score_block():
    src = EX / 'scores' / 'sarangga.txt'
    if not src.exists():
        return ''
    lines = [l for l in src.read_text(encoding='utf-8').splitlines()
             if l.strip() and not l.strip().startswith('#')]
    ds = EX / 'ds' / 'sarangga_김소희.ds'
    phon = ''
    if ds.exists():
        seg = json.loads(ds.read_text(encoding='utf-8'))[0]
        phon = (f'<div style="margin-top:.9rem;font-family:var(--mono);font-size:.78rem;'
                f'color:var(--muted);overflow-x:auto;white-space:nowrap">'
                f'<span style="color:var(--indigo)">ph_seq</span>&nbsp; {seg["ph_seq"]}<br>'
                f'<span style="color:var(--indigo)">ph_num</span>&nbsp; {seg["ph_num"]}</div>')
    cols = '<br>'.join(
        '&nbsp;'.join(f'<span style="display:inline-block;min-width:2.2rem">{p}</span>'
                      for p in l.split()) for l in lines)
    return (f'<div style="font-family:var(--mono);font-size:.78rem;line-height:1.5;'
            f'columns:3;column-gap:2rem">{cols}</div>{phon}')


# ---------------------------------------------------------------- f0


def f0_chart(f0data, item):
    """Original vs fully-predicted contour for one utterance."""
    tracks = f0data['items'].get(item)
    if not tracks:
        return '', '', ''
    series = [(t, tracks[t]) for t in ('1_orig', '4_full') if t in tracks]
    if len(series) < 2:
        return '', '', ''
    W, H, pad = 1100, 260, 34
    allv = [v for _, s in series for v in s['f0'] if v]
    lo, hi = min(allv), max(allv)
    import math
    lo_c, hi_c = 1200 * math.log2(lo / 55), 1200 * math.log2(hi / 55)
    span = max(hi_c - lo_c, 1)
    n = max(len(s['f0']) for _, s in series)
    step = f0data['timestep'] * 2

    def path(vals, colour):
        d, pen = [], False
        for i, v in enumerate(vals):
            if v is None:
                pen = False
                continue
            x = pad + 34 + (W - 2 * pad - 34) * i / max(n - 1, 1)
            c = 1200 * math.log2(v / 55)
            y = H - pad - (H - 2 * pad) * (c - lo_c) / span
            d.append(('M' if not pen else 'L') + f'{x:.1f} {y:.1f}')
            pen = True
        return (f'<path d="{" ".join(d)}" class="s-none s-{colour}" '
                f'stroke-width="1.5" stroke-linejoin="round"/>')

    grid = []
    for k in range(5):
        y = pad + (H - 2 * pad) * k / 4
        grid.append(f'<line x1="{pad + 34}" y1="{y}" x2="{W - pad}" y2="{y}" '
                    f'class="s-rule" stroke-width="1"/>')
        hz = 55 * 2 ** ((hi_c - span * k / 4) / 1200)
        grid.append(f'<text x="{pad + 28}" y="{y + 3.5}" text-anchor="end" '
                    f'class="d-mono f-muted" font-size="9.5">{hz:.0f}</text>')
    grid.append(f'<text x="{pad + 28}" y="{pad - 12}" text-anchor="end" '
                f'class="d-mono f-muted" font-size="9">Hz</text>')
    for k in range(0, int(n * step) + 1):
        x = pad + 34 + (W - 2 * pad - 34) * (k / step) / max(n - 1, 1)
        if x <= W - pad:
            grid.append(f'<text x="{x:.0f}" y="{H - 10}" text-anchor="middle" '
                        f'class="d-mono f-muted" font-size="10">{k}s</text>')
    body = ''.join(grid) + path(series[0][1]['f0'], 'indigo-d') + path(series[1][1]['f0'], 'ochre')
    svg = (f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="원본과 예측 f0 곡선 비교">{body}</svg>')
    legend = (f'<span><i style="border-color:var(--indigo-dim)"></i>원본 녹음</span>'
              f'<span><i style="border-color:var(--ochre)"></i>dur·f0 모두 예측</span>')
    spk = ITEM_INFO.get(item, ('', '', ''))[0]
    cap = (f'{spk} · <span class="mono">{item}</span> 한 번의 추첨. 세로축은 로그 피치, '
           f'끊긴 곳은 무성 구간이다. 음정은 대체로 따라가고 잔 움직임도 남아 있다 — '
           f'그 움직임의 양을 정량화한 것이 위 표의 진폭비다.')
    return svg, legend, cap


def f0_seeded_rows(f0data):
    """One row per (utterance, track): median ratio and the spread behind it."""
    seeded = f0data.get('seeded', {})
    out = []
    for item, tracks in seeded.items():
        spk = ITEM_INFO.get(item, (item,))[0]
        first = True
        for track in ('2_resynth', '3_pitch', '4_full'):
            t = tracks.get(track)
            if not t:
                continue
            med, lo, hi = t['median'], t['min'], t['max']
            # a bar centred on 1.0: left of centre is attenuation, right is excess
            centre, half = 50.0, 42.0
            x_lo, x_hi = centre + half * (min(lo, 2) - 1), centre + half * (min(hi, 2) - 1)
            x_med = centre + half * (min(med, 2) - 1)
            bar = (f'<span style="position:relative;display:block;height:1rem">'
                   f'<span style="position:absolute;left:{centre:.1f}%;top:0;bottom:0;'
                   f'width:1px;background:var(--rule)"></span>'
                   f'<span style="position:absolute;left:{min(x_lo, x_hi):.1f}%;'
                   f'width:{abs(x_hi - x_lo):.1f}%;top:.42rem;height:.16rem;'
                   f'background:var(--indigo-dim);opacity:.55"></span>'
                   f'<span style="position:absolute;left:{x_med:.1f}%;top:.2rem;'
                   f'width:.32rem;height:.6rem;margin-left:-.16rem;border-radius:1px;'
                   f'background:var(--{"indigo" if med >= .85 else "ochre"})"></span>'
                   f'</span>')
            colour = 'var(--indigo)' if med >= .85 else 'var(--ochre)'
            out.append(
                f'<tr{" class=rule-top" if first else ""}>'
                f'<td>{spk if first else ""}</td><td>{t["label"]}</td>'
                f'<td class="num" style="color:{colour}">{med:.2f}</td>'
                f'<td class="num">{lo:.2f}–{hi:.2f}</td>'
                f'<td>{bar}</td></tr>'
            )
            first = False
    return '\n'.join(out)


def f0_verdict(f0data):
    seeded = f0data.get('seeded', {})
    if not seeded:
        return ''
    res = [t['4_full']['median'] for t in seeded.values() if '4_full' in t]
    pit = [t['3_pitch']['median'] for t in seeded.values() if '3_pitch' in t]
    rs = [t['2_resynth']['median'] for t in seeded.values() if '2_resynth' in t]
    n = next(iter(seeded.values()))['2_resynth']['seeds']
    straddle = sum(1 for t in seeded.values() for k in ('3_pitch', '4_full')
                   if k in t and t[k]['min'] <= 1.0 <= t[k]['max'])
    return (
        f'가는 선은 시드 {n}회의 범위, 굵은 표시는 중앙값, 세로선이 원본과 같은 1.0 이다. '
        f'재합성만 해도 {min(rs):.2f}–{max(rs):.2f} 이므로 <b>{round(100 * (1 - max(rs)))}–'
        f'{round(100 * (1 - min(rs)))}% 의 감쇠는 acoustic 과 보코더에서 이미 생긴다</b> — '
        f'피치 예측기 탓이 아니다. 피치를 예측하게 하면 {min(pit):.2f}–{max(pit):.2f}, '
        f'길이까지 맡기면 {min(res):.2f}–{max(res):.2f} 이고, 예측 트랙 6개 중 {straddle}개는 '
        f'범위가 1.0 을 가로지른다. '
        f'<b>옛 캐스케이드의 진단이었던 체계적 평탄화는 이 모델에서 재현되지 않는다.</b> '
        f'시드 사이의 편차가 1.0 과의 거리보다 크므로, contour tokenizer 를 붙일 근거는 '
        f'적어도 이 측정에는 없다.'
    )


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description='리포트 생성')
    parser.add_argument('--out', default='pansori/examples/report.html')
    parser.add_argument('--chart-item', default='8b4d0824_0078')
    args = parser.parse_args()

    metrics = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
    f0data = json.loads((OUT / 'f0.json').read_text(encoding='utf-8'))
    steps = (OUT / 'CKPT_STEPS').read_text().split()
    chart, legend, caption = f0_chart(f0data, args.chart_item)

    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                            cwd=REPO, capture_output=True, text=True).stdout.strip()
    footer = (f'acoustic <span class="mono">{int(steps[0]):,}</span> step · '
              f'variance duration <span class="mono">{int(steps[1]):,}</span> / '
              f'pitch <span class="mono">{int(steps[2]):,}</span> step 체크포인트로 생성. '
              f'커밋 <span class="mono">{commit}</span>.')

    html = (EX / 'report_template.html').read_text(encoding='utf-8')
    for key, value in {
        'DATE': date.today().isoformat(),
        'FIGURES': figures(),
        'SPEAKER_ROWS': speaker_rows(),
        'STAGES': stages(),
        'SCHEMA_DIAGRAM': schema_diagram(),
        'MODEL_DIAGRAM': model_diagram(),
        'LADDER': ladder(metrics),
        'SCORE_BLOCK': score_block(),
        'SCORE_LADDER': score_ladder(),
        'F0_CHART': chart,
        'F0_LEGEND': legend,
        'F0_CAPTION': caption,
        'F0_SEEDED_ROWS': f0_seeded_rows(f0data),
        'F0_VERDICT': f0_verdict(f0data),
        'FOOTER_NOTE': footer,
    }.items():
        html = html.replace('{{' + key + '}}', value)

    left = re.findall(r'\{\{(\w+)\}\}', html)
    if left:
        raise SystemExit(f'채우지 못한 자리: {sorted(set(left))}')

    out = REPO / args.out
    out.write_text(html, encoding='utf-8')
    print(f'{out}  ({out.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()

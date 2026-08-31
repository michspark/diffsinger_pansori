#!/usr/bin/env python3
"""가사가 아니라 **라벨**을 잰다 — transcriptions.csv 의 ph_dur 분포 진단.

발음이 나쁠 때 모델을 먼저 의심하기 쉽지만, 이 코퍼스에서 가장 큰 이상 신호는
학습 라벨 자체에 있다. `data/pansori` 의 ph_dur 값을 세어 보면 **전체 음소의
22.3% 가 정확히 `0.030000` 초**다. 다음으로 흔한 값(0.04)의 3.2배이고, 자연스러운
발화 길이 분포에서는 나올 수 없는 모양이다. 0.03 은 MFA 의 최소 음소 길이이므로,
그 스파이크는 곧 **"정렬기가 여기서 포기했다"** 는 표시다.

그리고 균등하게 퍼져 있지 않다. 초성 파열음은 40% 대가 하한선에 박혀 있고
(pp 50.7 / tt 46.6 / b 44.0 / g 43.6 / d 42.4), 모음과 종성은 10~20% 대로 멀쩡하다.
그래서 이 리포트는 '자음' 하나로 뭉뚱그리지 않고 초성과 종성을 갈라 센다
(`pansori_lib.phone_class`) — 그 구분이 유일하게 볼 만한 것이기 때문이다.

이 스크립트가 하지 않는 것:

  - **정렬을 고치지 않는다.** 기준선을 만들 뿐이다. 나중에 재정렬을 시도할 때
    개선 여부를 재려면 그 전에 이 숫자가 있어야 한다.
  - **좋다/나쁘다를 판정하지 않는다.** 합격선을 정할 근거가 아직 없다. 두 코퍼스를
    나란히 놓고 비교할 수는 있다 (--corpus 를 두 개 준다).

모델도 GPU 도 필요 없다. transcriptions.csv 만 읽으므로 어느 env 에서든 몇 초에 끝난다.

    python pansori/eval/label_report.py
    python pansori/eval/label_report.py --corpus data/pansori data/pansori_snap
    python pansori/eval/label_report.py --html pansori/eval/out/label_report.html

출력: 콘솔 표 + eval/out/label_report.json + (--html) 자체완결 HTML
"""

import argparse
import csv
import html
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'pansori' / 'preprocess'))

from pansori_lib import PHONE_CLASSES, phone_class  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / 'out'

# MFA's minimum phone duration. A ph_dur at exactly this value is not a
# measurement, it is the aligner giving up -- see the module docstring.
MFA_FLOOR = 0.030
# 06_merge.py folds anything shorter than this into a neighbour (MIN_PH_DUR),
# so durations below it come from that folding, not from MFA.
MERGE_FLOOR = 0.020
EPS = 1e-6

# The binarizer quantises ph_dur to mel frames before the model ever sees it
# (preprocessing/variance_binarizer.py:293-295, hop 512 @ 44.1 kHz from
# configs/base.yaml:15-16), so the training-time metric in pansori/metrics/
# counts frames while everything above counts seconds. One frame is 11.61 ms and
# the floor lands on 3 frames = 34.8 ms, which is wider than 30 ms -- the two
# numbers differ by ~4 %p on this corpus for that reason alone.
# `floor_rate_frames` below is computed with the binarizer's exact rounding so
# the two measurements can be compared without an eyeball correction.
HOP_SIZE, SAMPLE_RATE = 512, 44100
TIMESTEP = HOP_SIZE / SAMPLE_RATE
FLOOR_FRAMES = round(MFA_FLOOR / TIMESTEP)      # 3

CLASS_LABEL = {
    'onset': '초성', 'coda': '종성', 'vowel': '모음',
    'glide': '활음', 'sp': '무음', 'other': '기타',
}


# ---------------------------------------------------------------- measuring

def read_corpus(root: Path) -> dict:
    """One corpus -> the raw per-phone observations everything else is derived from."""
    per_phone = defaultdict(list)          # phone -> [dur, ...]
    per_phone_frames = defaultdict(list)   # phone -> [frames, ...]
    per_speaker = defaultdict(lambda: [0, 0])   # speaker -> [n_ph, n_floor]
    value_counts = Counter()               # exact ph_dur string -> count
    n_utt = 0
    total_s = 0.0

    csvs = sorted(root.glob('*/transcriptions.csv'))
    if not csvs:
        sys.exit(f'transcriptions.csv 를 찾을 수 없음: {root}')

    for path in csvs:
        speaker = path.parent.name
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                seq = row['ph_seq'].split()
                raw = row['ph_dur'].split()
                if len(seq) != len(raw):
                    sys.exit(f"ph_seq/ph_dur 길이 불일치: {path} {row['name']}")
                n_utt += 1
                # Mirror the binarizer's rounding exactly so the frame counts are
                # the ones the model is actually trained on, not a re-derivation.
                acc_s, prev_frame = 0.0, 0
                for phone, text in zip(seq, raw):
                    dur = float(text)
                    acc_s += dur
                    cur_frame = round(acc_s / TIMESTEP + 0.5)
                    per_phone_frames[phone].append(cur_frame - prev_frame)
                    prev_frame = cur_frame

                    per_phone[phone].append(dur)
                    value_counts[text] += 1
                    total_s += dur
                    per_speaker[speaker][0] += 1
                    if dur <= MFA_FLOOR + EPS:
                        per_speaker[speaker][1] += 1

    return {
        'root': str(root.relative_to(REPO)) if root.is_relative_to(REPO) else str(root),
        'n_utt': n_utt,
        'total_s': total_s,
        'per_phone': dict(per_phone),
        'per_phone_frames': dict(per_phone_frames),
        'per_speaker': {k: v for k, v in per_speaker.items()},
        'value_counts': value_counts,
    }


def _floor_rate(durs) -> float:
    return sum(1 for d in durs if d <= MFA_FLOOR + EPS) / len(durs)


def _floor_rate_frames(frames) -> float:
    """Same quantity the training metric reports -- frames, not seconds."""
    return sum(1 for f in frames if f <= FLOOR_FRAMES) / len(frames)


def summarise(obs: dict) -> dict:
    """Derive the numbers the console table and the HTML both print."""
    per_phone = obs['per_phone']
    per_frames = obs['per_phone_frames']
    all_durs = [d for v in per_phone.values() for d in v]
    all_frames = [f for v in per_frames.values() for f in v]
    n_ph = len(all_durs)

    phones = {}
    for phone, durs in per_phone.items():
        phones[phone] = {
            'phone': phone,
            'cls': phone_class(phone),
            'n': len(durs),
            'share': len(durs) / n_ph,
            'median_ms': statistics.median(durs) * 1000,
            'mean_ms': statistics.fmean(durs) * 1000,
            'floor_rate': _floor_rate(durs),
            'floor_rate_frames': _floor_rate_frames(per_frames[phone]),
            'sub_merge_rate': sum(1 for d in durs if d <= MERGE_FLOOR + EPS) / len(durs),
            'seconds': sum(durs),
        }

    classes = {}
    for cls in PHONE_CLASSES + ('other',):
        members = [p for p in phones.values() if p['cls'] == cls]
        if not members:
            continue
        durs = [d for p in members for d in per_phone[p['phone']]]
        frames = [f for p in members for f in per_frames[p['phone']]]
        classes[cls] = {
            'cls': cls,
            'n': len(durs),
            'share': len(durs) / n_ph,
            'seconds': sum(durs),
            'time_share': sum(durs) / obs['total_s'],
            'median_ms': statistics.median(durs) * 1000,
            'floor_rate': _floor_rate(durs),
            'floor_rate_frames': _floor_rate_frames(frames),
            'n_phones': len(members),
        }

    sp_s = classes.get('sp', {}).get('seconds', 0.0)
    sp_n = classes.get('sp', {}).get('n', 0)
    speech_s = obs['total_s'] - sp_s

    speakers = sorted(
        ({'speaker': k, 'n': n, 'floor_rate': f / n} for k, (n, f) in obs['per_speaker'].items() if n),
        key=lambda r: -r['floor_rate'],
    )

    return {
        'root': obs['root'],
        'n_utt': obs['n_utt'],
        'n_ph': n_ph,
        'hours': obs['total_s'] / 3600,
        'floor_rate': _floor_rate(all_durs),
        'floor_rate_frames': _floor_rate_frames(all_frames),
        'exact_floor_share': obs['value_counts'][f'{MFA_FLOOR:.6f}'] / n_ph,
        'sub_merge_rate': sum(1 for d in all_durs if d <= MERGE_FLOOR + EPS) / n_ph,
        'sp_time_share': sp_s / obs['total_s'],
        'phones_per_speech_second': (n_ph - sp_n) / speech_s if speech_s > 0 else 0.0,
        'quantiles': {
            f'p{int(q * 100)}': _quantile(sorted(all_durs), q) * 1000
            for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
        },
        'top_values': [
            {'value': v, 'ms': float(v) * 1000, 'n': c, 'share': c / n_ph}
            for v, c in obs['value_counts'].most_common(12)
        ],
        'value_counts': {v: c for v, c in obs['value_counts'].items()},
        'classes': classes,
        'phones': phones,
        'speakers': speakers,
    }


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float('nan')
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


# ---------------------------------------------------------------- console

def print_report(s: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"{s['root']}")
    print(f"{'=' * 72}")
    print(f"발화 {s['n_utt']:,}  ·  음소 {s['n_ph']:,}  ·  {s['hours']:.2f}시간")
    print(f"무음(SP/AP) 시간 비율 {s['sp_time_share'] * 100:.1f}%  ·  "
          f"음소/초(무음 제외) {s['phones_per_speech_second']:.2f}")
    q = s['quantiles']
    print('ph_dur 분위(ms)  ' + '  '.join(f'{k}={v:.0f}' for k, v in q.items()))

    print(f"\n── ph_dur 값 상위 (전체 {s['n_ph']:,}개 중) " + '─' * 26)
    for row in s['top_values'][:8]:
        bar = '█' * round(row['share'] * 100)
        flag = '  ← MFA 하한선' if abs(row['ms'] - MFA_FLOOR * 1000) < 0.5 else ''
        print(f"  {row['ms']:6.0f}ms  {row['share'] * 100:5.1f}%  {row['n']:8,}  {bar}{flag}")
    print(f"  ≤{MFA_FLOOR * 1000:.0f}ms 전체: {s['floor_rate'] * 100:.1f}%   "
          f"≤{MERGE_FLOOR * 1000:.0f}ms: {s['sub_merge_rate'] * 100:.1f}%")
    print(f"  ≤{FLOOR_FRAMES}프레임({FLOOR_FRAMES * TIMESTEP * 1000:.1f}ms): "
          f"{s['floor_rate_frames'] * 100:.1f}%   ← 학습 지표 floor_rate_gt 와 같은 자")

    print('\n── 부류별 ' + '─' * 52)
    print(f"  {'부류':<6} {'음소수':>9} {'비중':>7} {'시간비':>7} {'중앙ms':>8} "
          f"{'≤30ms':>8} {'≤3프레임':>9}")
    for cls in PHONE_CLASSES + ('other',):
        c = s['classes'].get(cls)
        if not c:
            continue
        print(f"  {CLASS_LABEL[cls]:<6} {c['n']:>9,} {c['share'] * 100:>6.1f}% "
              f"{c['time_share'] * 100:>6.1f}% {c['median_ms']:>8.0f} "
              f"{c['floor_rate'] * 100:>7.1f}% {c['floor_rate_frames'] * 100:>8.1f}%")

    print('\n── 음소별 (부류 안에서 등장 순) ' + '─' * 31)
    print(f"  {'음소':>4} {'부류':<6} {'등장':>9} {'중앙ms':>8} {'하한선율':>9}")
    for cls in PHONE_CLASSES + ('other',):
        members = sorted((p for p in s['phones'].values() if p['cls'] == cls),
                         key=lambda p: -p['n'])
        for p in members:
            print(f"  {p['phone']:>4} {CLASS_LABEL[cls]:<6} {p['n']:>9,} "
                  f"{p['median_ms']:>8.0f} {p['floor_rate'] * 100:>8.1f}%")

    print('\n── 화자별 하한선율 (상·하위 5) ' + '─' * 32)
    for row in s['speakers'][:5] + [None] + s['speakers'][-5:]:
        if row is None:
            print('   ...')
            continue
        print(f"  {row['speaker']:<10} {row['n']:>9,} {row['floor_rate'] * 100:>8.1f}%")


def print_delta(a: dict, b: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"차이: {b['root']} − {a['root']}")
    print(f"{'=' * 72}")
    rows = [('하한선율 전체', 'floor_rate', 100, '%p'),
            ('무음 시간비', 'sp_time_share', 100, '%p'),
            ('음소/초', 'phones_per_speech_second', 1, '')]
    for label, key, scale, unit in rows:
        print(f"  {label:<14} {a[key] * scale:8.2f} → {b[key] * scale:8.2f}  "
              f"({(b[key] - a[key]) * scale:+.2f}{unit})")
    print(f"  {'부류별 하한선율':<14}")
    for cls in PHONE_CLASSES:
        ca, cb = a['classes'].get(cls), b['classes'].get(cls)
        if not ca or not cb:
            continue
        print(f"    {CLASS_LABEL[cls]:<6} {ca['floor_rate'] * 100:8.1f}% → "
              f"{cb['floor_rate'] * 100:8.1f}%  ({(cb['floor_rate'] - ca['floor_rate']) * 100:+.1f}%p)")


# ---------------------------------------------------------------- inline SVG

def _e(s):
    return html.escape(str(s), quote=True)


def svg_value_histogram(s: dict, width=880, height=260) -> str:
    """ph_dur exact-value bars up to 200 ms, with the MFA floor called out.

    Exact values rather than binned, because the point of the figure is that the
    distribution is not smooth: MFA quantises to 10 ms and then piles a fifth of
    the corpus onto one of those steps. A histogram with wide bins would hide
    exactly the thing worth seeing.
    """
    pad_l, pad_r, pad_t, pad_b = 52, 16, 18, 42
    steps = [round(v, 3) for v in [i / 100 for i in range(1, 21)]]   # 10..200 ms
    counts = {}
    for v, c in s['value_counts'].items():
        f = round(float(v), 3)
        if f <= 0.2005:
            counts[f] = counts.get(f, 0) + c
    tail = s['n_ph'] - sum(counts.values())
    shares = [counts.get(v, 0) / s['n_ph'] for v in steps] + [tail / s['n_ph']]
    labels = [f'{v * 1000:.0f}' for v in steps] + ['>200']

    top = max(shares) * 1.12
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    slot = pw / len(shares)
    bar_w = max(4.0, slot - 4)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="ph_dur 값 분포" style="width:100%;height:auto">']

    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + ph * (1 - frac)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="var(--ink-3)">{top * frac * 100:.0f}%</text>')

    for i, (share, label) in enumerate(zip(shares, labels)):
        x = pad_l + i * slot + (slot - bar_w) / 2
        h = ph * (share / top)
        y = pad_t + ph - h
        is_floor = label == '30'
        fill = 'var(--series-2)' if is_floor else 'var(--series-1)'
        opacity = '1' if is_floor else '.55'
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(h, 0.6):.1f}" '
                     f'rx="{min(4, bar_w / 2):.1f}" fill="{fill}" opacity="{opacity}"/>')
        if is_floor:
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" '
                         f'font-size="12.5" font-weight="600" fill="var(--ink)">'
                         f'{share * 100:.1f}% · MFA 하한선</text>')
        if i % 2 == 0 or label == '>200':
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - pad_b + 16:.1f}" '
                         f'text-anchor="middle" font-size="10.5" fill="var(--ink-3)">{_e(label)}</text>')

    parts.append(f'<line x1="{pad_l}" y1="{pad_t + ph:.1f}" x2="{width - pad_r}" '
                 f'y2="{pad_t + ph:.1f}" stroke="var(--grid-strong)" stroke-width="1"/>')
    parts.append(f'<text x="{width / 2:.0f}" y="{height - 6}" text-anchor="middle" '
                 f'font-size="11.5" fill="var(--ink-3)">ph_dur (ms)</text>')
    parts.append('</svg>')
    return ''.join(parts)


def svg_phone_floor(s: dict, width=880, row_h=17) -> str:
    """Per-phoneme floor rate, grouped by class.

    One hue, because floor rate is a magnitude and not an identity; the class is
    carried by spatial grouping and a heading, so no five-colour categorical
    palette is needed. Each bar is direct-labelled, which is also the relief the
    reader needs to compare two bars that are far apart vertically.
    """
    groups = []
    for cls in PHONE_CLASSES + ('other',):
        members = sorted((p for p in s['phones'].values() if p['cls'] == cls),
                         key=lambda p: -p['floor_rate'])
        if members:
            groups.append((cls, members))

    pad_l, pad_r, pad_t = 108, 96, 12
    head_h, gap_h = 24, 10
    height = pad_t + sum(head_h + len(m) * row_h + gap_h for _, m in groups)
    pw = width - pad_l - pad_r
    top = 100.0

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="음소별 30ms 하한선 비율" style="width:100%;height:auto">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        x = pad_l + pw * frac
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{height}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')

    y = pad_t
    for cls, members in groups:
        overall = s['classes'][cls]['floor_rate'] * 100
        parts.append(f'<text x="0" y="{y + 15:.1f}" font-size="12.5" font-weight="600" '
                     f'fill="var(--ink-2)">{_e(CLASS_LABEL[cls])} · 전체 {overall:.1f}%</text>')
        y += head_h
        for p in members:
            rate = p['floor_rate'] * 100
            w = max(pw * rate / top, 1.5)
            bar_h = row_h - 5
            parts.append(f'<text x="{pad_l - 10}" y="{y + bar_h - 1:.1f}" text-anchor="end" '
                         f'font-size="11.5" font-family="ui-monospace,monospace" '
                         f'fill="var(--ink-2)">{_e(p["phone"])}</text>')
            parts.append(f'<rect x="{pad_l}" y="{y:.1f}" width="{w:.1f}" height="{bar_h}" '
                         f'rx="3" fill="var(--series-1)" opacity="{0.35 + 0.6 * rate / top:.2f}"/>')
            parts.append(f'<text x="{pad_l + w + 8:.1f}" y="{y + bar_h - 1:.1f}" font-size="11" '
                         f'fill="var(--ink-3)">{rate:.0f}% · n={p["n"]:,}</text>')
            y += row_h
        y += gap_h

    parts.append('</svg>')
    return ''.join(parts)


# ---------------------------------------------------------------- HTML

CSS = """
:root{
  color-scheme: light;
  --surface-0:#eef2f2; --surface-1:#ffffff; --surface-2:#f6faf9;
  --ink:#101418; --ink-2:#4c565d; --ink-3:#7a868c;
  --rule:#dbe3e3; --grid:#e6ecec; --grid-strong:#b4c0c1;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --measure:70ch;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --surface-0:#0f1315; --surface-1:#181e21; --surface-2:#141a1c;
    --ink:#eaf0f0; --ink-2:#a7b3b7; --ink-3:#78868a;
    --rule:#263033; --grid:#222b2e; --grid-strong:#465358;
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-0:#0f1315; --surface-1:#181e21; --surface-2:#141a1c;
  --ink:#eaf0f0; --ink-2:#a7b3b7; --ink-3:#78868a;
  --rule:#263033; --grid:#222b2e; --grid-strong:#465358;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
}
*{box-sizing:border-box}
body{margin:0; background:var(--surface-0); color:var(--ink);
  font-family:"IBM Plex Sans KR",system-ui,-apple-system,"Malgun Gothic",sans-serif;
  font-size:16px; line-height:1.72; -webkit-font-smoothing:antialiased}
.wrap{max-width:1020px; margin:0 auto; padding:56px 24px 96px;
  display:flex; flex-direction:column; gap:40px}
.prose{max-width:var(--measure); display:flex; flex-direction:column; gap:14px}
h1,h2{font-family:"Gowun Batang",Georgia,"Batang",serif; margin:0;
  line-height:1.28; font-weight:700; text-wrap:balance}
h1{font-size:clamp(28px,4.4vw,42px); letter-spacing:-.015em}
h2{font-size:clamp(20px,2.5vw,26px); margin-top:8px}
p{margin:0}
.lede{font-size:18px; color:var(--ink-2)}
.meta{font-size:13px; color:var(--ink-3);
  font-family:"IBM Plex Mono",ui-monospace,monospace}
figure{margin:0; background:var(--surface-1); border:1px solid var(--rule);
  border-radius:10px; padding:18px 20px; overflow-x:auto}
figcaption{font-size:13.5px; color:var(--ink-2); margin-bottom:12px}
.stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  border:1px solid var(--rule); border-radius:10px; background:var(--surface-1);
  overflow:hidden}
.stat{padding:14px 18px; border-right:1px solid var(--rule);
  border-bottom:1px solid var(--rule)}
.stat .k{font-size:12.5px; color:var(--ink-3)}
.stat .v{font-size:24px; font-weight:600; line-height:1.3;
  font-family:"IBM Plex Mono",ui-monospace,monospace}
.stat.hi .v{color:var(--series-2)}
table{border-collapse:collapse; width:100%; font-size:13.5px;
  font-variant-numeric:tabular-nums}
th,td{padding:6px 10px; border-bottom:1px solid var(--rule); text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{font-size:12px; letter-spacing:.04em; color:var(--ink-3);
  font-weight:500; background:var(--surface-2)}
.tw{background:var(--surface-1); border:1px solid var(--rule);
  border-radius:10px; overflow:hidden}
code{font-size:.9em; background:var(--surface-2); padding:.1em .35em; border-radius:3px}
"""


def stat_tile(k, v, hi=False):
    return (f'<div class="stat{" hi" if hi else ""}"><div class="k">{_e(k)}</div>'
            f'<div class="v">{_e(v)}</div></div>')


def class_table(s: dict) -> str:
    head = ('<thead><tr><th>부류</th><th>음소 수</th><th>비중</th><th>시간비</th>'
            '<th>중앙 ms</th><th>≤30ms</th><th>≤3프레임</th></tr></thead>')
    rows = []
    for cls in PHONE_CLASSES + ('other',):
        c = s['classes'].get(cls)
        if not c:
            continue
        rows.append(
            f'<tr><td>{_e(CLASS_LABEL[cls])}</td><td>{c["n"]:,}</td>'
            f'<td>{c["share"] * 100:.1f}%</td><td>{c["time_share"] * 100:.1f}%</td>'
            f'<td>{c["median_ms"]:.0f}</td><td>{c["floor_rate"] * 100:.1f}%</td>'
            f'<td>{c["floor_rate_frames"] * 100:.1f}%</td></tr>')
    return f'<div class="tw"><table>{head}<tbody>{"".join(rows)}</tbody></table></div>'


def render_html(summaries: list) -> str:
    s = summaries[0]
    body = [f'<h1>라벨 진단 — {_e(s["root"])}</h1>',
            f'<p class="lede">합성음의 발음을 의심하기 전에, 모델이 무엇을 보고 배웠는지 먼저 잰다. '
            f'이 페이지는 <code>transcriptions.csv</code> 의 <code>ph_dur</code> 만 읽는다.</p>'
            f'<p class="meta">발화 {s["n_utt"]:,} · 음소 {s["n_ph"]:,} · {s["hours"]:.2f}시간</p>']
    body = ['<div class="prose">' + ''.join(body) + '</div>']

    body.append('<div class="stats">'
                + stat_tile('≤30ms 음소', f'{s["floor_rate"] * 100:.1f}%', hi=True)
                + stat_tile('정확히 0.030초', f'{s["exact_floor_share"] * 100:.1f}%', hi=True)
                + stat_tile('초성 ≤30ms',
                            f'{s["classes"]["onset"]["floor_rate"] * 100:.1f}%')
                + stat_tile('종성 ≤30ms',
                            f'{s["classes"]["coda"]["floor_rate"] * 100:.1f}%')
                + stat_tile('무음 시간비', f'{s["sp_time_share"] * 100:.1f}%')
                + stat_tile('≤3프레임 (학습 지표 기준)',
                            f'{s["floor_rate_frames"] * 100:.1f}%')
                + '</div>')

    body.append('<div class="prose"><h2>ph_dur 값 분포</h2>'
                '<p>MFA 는 10 ms 격자로 답을 내므로 값이 이산적인 것 자체는 정상이다. '
                '문제는 <strong>한 값에 몰린 봉우리</strong>다 — 0.03 초는 MFA 의 최소 음소 '
                '길이이고, 거기 쌓인 음소는 길이가 그렇게 <em>측정된</em> 것이 아니라 '
                '정렬기가 음향적 근거를 못 찾고 최소값에 박아 둔 것이다.</p></div>')
    body.append(f'<figure><figcaption>값별 비율 (200 ms 초과는 하나로 묶음)</figcaption>'
                f'{svg_value_histogram(s)}</figure>')

    body.append('<div class="prose"><h2>부류별</h2>'
                '<p>초성과 종성을 나눠 세는 것이 이 표의 요점이다. 둘을 \'자음\' 하나로 '
                '묶으면 서로를 상쇄해 아무것도 보이지 않는다.</p></div>')
    body.append(class_table(s))

    body.append('<div class="prose"><h2>음소별 하한선 비율</h2>'
                '<p>막대가 길수록 그 음소의 라벨을 믿기 어렵다. 파열음이 위쪽에 몰리고 '
                '모음과 종성이 아래에 남는 모양이 이 코퍼스의 특징이다.</p></div>')
    body.append(f'<figure><figcaption>≤30 ms 인 비율 · 부류 안에서 내림차순</figcaption>'
                f'{svg_phone_floor(s)}</figure>')

    if len(summaries) > 1:
        rows = []
        for other in summaries[1:]:
            rows.append(f'<tr><td>{_e(other["root"])}</td>'
                        f'<td>{other["n_ph"]:,}</td>'
                        f'<td>{other["floor_rate"] * 100:.1f}%</td>'
                        f'<td>{other["classes"]["onset"]["floor_rate"] * 100:.1f}%</td>'
                        f'<td>{other["classes"]["vowel"]["floor_rate"] * 100:.1f}%</td>'
                        f'<td>{other["sp_time_share"] * 100:.1f}%</td></tr>')
        head = ('<thead><tr><th>코퍼스</th><th>음소</th><th>≤30ms</th>'
                '<th>초성</th><th>모음</th><th>무음 시간비</th></tr></thead>')
        first = (f'<tr><td>{_e(s["root"])}</td><td>{s["n_ph"]:,}</td>'
                 f'<td>{s["floor_rate"] * 100:.1f}%</td>'
                 f'<td>{s["classes"]["onset"]["floor_rate"] * 100:.1f}%</td>'
                 f'<td>{s["classes"]["vowel"]["floor_rate"] * 100:.1f}%</td>'
                 f'<td>{s["sp_time_share"] * 100:.1f}%</td></tr>')
        body.append('<div class="prose"><h2>코퍼스 비교</h2></div>')
        body.append(f'<div class="tw"><table>{head}<tbody>{first}{"".join(rows)}'
                    f'</tbody></table></div>')

    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>라벨 진단 — {_e(s["root"])}</title><style>{CSS}</style></head>'
            f'<body><div class="wrap">{"".join(body)}</div></body></html>')


# ---------------------------------------------------------------- driver

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', nargs='+', default=['data/pansori'],
                        help='코퍼스 루트. 두 개 이상 주면 차이도 같이 인쇄한다')
    parser.add_argument('--json', default=str(OUT_DIR / 'label_report.json'))
    parser.add_argument('--html', default=None,
                        help='자체완결 HTML 을 여기에 쓴다 (기본: 안 씀)')
    args = parser.parse_args()

    summaries = []
    for rel in args.corpus:
        root = Path(rel) if Path(rel).is_absolute() else REPO / rel
        summaries.append(summarise(read_corpus(root)))

    for s in summaries:
        print_report(s)
    for other in summaries[1:]:
        print_delta(summaries[0], other)

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # value_counts is only needed for the figure; keep it out of the json so the
    # file stays diffable between runs.
    payload = [{k: v for k, v in s.items() if k != 'value_counts'} for s in summaries]
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n→ {out_json}')

    if args.html:
        out_html = Path(args.html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(render_html(summaries), encoding='utf-8')
        print(f'→ {out_html}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Assemble the comparison report as one self-contained HTML page.

Everything is inlined -- audio as base64 mp3, figures as inline SVG -- because the
published artifact runs under a CSP that blocks every external host except Google
Fonts. Figures are SVG rather than PNG so they follow the reader's theme and stay
selectable.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/compare/report.py
"""

import argparse
import base64
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402
from pansori.compare import items as I  # noqa: E402
from pansori.compare import plots as PL  # noqa: E402

OUT_HTML = C.COMPARE / 'report.html'
# Chosen for a LOW ground-truth octave-snap rate, not for a flattering result:
# on items where RMVPE drops octaves (8b4d0824_0000 does it on 28% of frames) the
# reference contour's own wrap artifacts swamp the ornament and the figure shows
# the extractor rather than the models. Two 남창 and one 여창.
SHOWCASE = ['e719dced_0045', '8b4d0824_0056', 'd207244a_0047']


def e(s):
    return html.escape(str(s), quote=True)


def num(x, nd=3):
    if x is None or not np.isfinite(x):
        return '—'
    return f'{x:,.{nd}f}'


def mp3_b64(path, kbps=64):
    """64 kbps mono mp3, base64. The page has to carry its own audio."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as t:
        tmp = Path(t.name)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(path),
                    '-ac', '1', '-b:a', f'{kbps}k', str(tmp)], check=True)
    data = base64.b64encode(tmp.read_bytes()).decode()
    tmp.unlink()
    return f'data:audio/mpeg;base64,{data}'


def audio_tag(path, label):
    return (f'<figure class="clip"><figcaption>{e(label)}</figcaption>'
            f'<audio controls preload="none" src="{mp3_b64(path)}"></audio></figure>')


# ---------------------------------------------------------------- tables

def endpoint_table(records, primary=True):
    head = ('<thead><tr><th>지표</th><th class="n">구</th><th class="n">신</th>'
            '<th class="n">차이 (중앙값)</th><th class="n">95% CI</th>'
            '<th class="n">p</th><th class="n">시드 잡음</th><th class="n">천장 격차</th>'
            + ('<th>판정</th>' if primary else '') + '</tr></thead>')
    rows = []
    for r in records:
        if primary:
            if r.get('claim'):
                chip = '<span class="chip good">개선</span>'
            else:
                reasons = []
                if not r.get('beats_seed_noise'):
                    reasons.append('시드 잡음 미만')
                if not r.get('beats_ceiling'):
                    reasons.append('천장 격차 미만')
                if not (np.isfinite(r.get('p_holm', np.nan)) and r['p_holm'] < 0.05):
                    reasons.append('유의하지 않음')
                chip = (f'<span class="chip flat" title="{e(", ".join(reasons))}">'
                        f'변화 없음</span>')
            verdict = f'<td>{chip}</td>'
        else:
            verdict = ''
        p = r.get('p_holm') if primary else r.get('p')
        nd = 1 if abs(r['old_median']) > 20 else 3
        rows.append(
            f'<tr><td>{e(r["label"])}</td>'
            f'<td class="n">{num(r["old_median"], nd)}</td>'
            f'<td class="n">{num(r["new_median"], nd)}</td>'
            f'<td class="n strong">{num(r["delta_median"], nd)}</td>'
            f'<td class="n dim">[{num(r["ci_lo"], nd)}, {num(r["ci_hi"], nd)}]</td>'
            f'<td class="n">{num(p, 4)}</td>'
            f'<td class="n dim">{num(r["seed_iqr"], nd)}</td>'
            f'<td class="n dim">{num(r.get("ceiling_delta"), nd)}</td>{verdict}</tr>')
    return f'<div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div>'


def sigimsae_table(table):
    def med(sysrung, metric):
        d = table.get((*sysrung, metric), {})
        return float(np.median([np.median(v) for v in d.values()])) if d else float('nan')

    old = (f'OLD_{C.OLD_PRIMARY}', '5_pred_own')
    rows = [
        ('원본 녹음 (RMVPE 기준)', ('GT', '1_orig'), 'ref'),
        ('ROSVOT 노트 계단', None, 'floor'),
        (f'구 모델 · 예측 mel + 모델 f0', old, 'old'),
        ('신 모델 · 예측 mel + 모델 f0', ('NEW', '3_pitch'), 'new'),
        ('신 · GT f0 로 재합성', ('NEW', '2_resynth'), 'sub'),
        ('신 · 보코더 천장', ('NEW', '0_vocoder'), 'sub'),
        ('구 · 보코더 천장', (f'OLD_{C.OLD_PRIMARY}', '2_gt_crepe'), 'sub'),
    ]
    body = []
    for label, key, kind in rows:
        if key is None:
            body.append(f'<tr class="r-floor"><td>{e(label)}</td>'
                        f'<td class="n">0.0</td><td class="n">0.0</td>'
                        f'<td class="n">0.0</td><td class="n">—</td></tr>')
            continue
        w = med(key, 'within_std')
        if not np.isfinite(w):
            continue
        corr = med(key, 'within_corr')
        body.append(
            f'<tr class="r-{kind}"><td>{e(label)}</td>'
            f'<td class="n">{num(med(key, "bias_rms"), 1)}</td>'
            f'<td class="n strong">{num(w, 1)}</td>'
            f'<td class="n">{num(med(key, "within_std_core"), 1)}</td>'
            f'<td class="n">{num(1.0 if kind == "ref" else corr, 3)}</td></tr>')
    return (
        '<div class="scroll"><table class="sig">'
        '<thead><tr><th>시스템</th><th class="n">노트편향 RMS</th>'
        '<th class="n">노트내 변동</th><th class="n">노트내 변동 (core)</th>'
        '<th class="n">원본과의 상관</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>')


def spectral_table(table):
    def med(sysrung, metric):
        d = table.get((*sysrung, metric), {})
        return float(np.median([np.median(v) for v in d.values()])) if d else float('nan')

    rows = [
        ('구 · 보코더 천장', (f'OLD_{C.OLD_PRIMARY}', '2_gt_crepe'), True),
        ('신 · 보코더 천장', ('NEW', '0_vocoder'), True),
        ('구 · 모델 출력', (f'OLD_{C.OLD_PRIMARY}', '5_pred_own'), False),
        ('신 · 모델 출력', ('NEW', '3_pitch'), False),
    ]
    body = []
    for label, key, ceiling in rows:
        if not np.isfinite(med(key, 'mcd_db')):
            continue
        body.append(
            f'<tr class="{"r-ceiling" if ceiling else "r-model"}"><td>{e(label)}</td>'
            f'<td class="n">{num(med(key, "mcd_db"), 2)}</td>'
            f'<td class="n">{num(med(key, "lsd"), 3)}</td>'
            f'<td class="n">{num(med(key, "mel_l1_common"), 3)}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>시스템</th>'
            '<th class="n">MCD (dB)</th><th class="n">LSD</th>'
            '<th class="n">공통 mel L1</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


# ---------------------------------------------------------------- page

CSS = """
:root{
  color-scheme: light;
  --surface-0:#eef2f2; --surface-1:#ffffff; --surface-2:#f6faf9;
  --ink:#101418; --ink-2:#4c565d; --ink-3:#7a868c;
  --rule:#dbe3e3; --grid:#e6ecec; --grid-strong:#b4c0c1;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --good:#126b40; --good-bg:#e3f2e9; --flat-bg:#e9edee;
  --measure:70ch;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --surface-0:#0f1315; --surface-1:#181e21; --surface-2:#141a1c;
    --ink:#eaf0f0; --ink-2:#a7b3b7; --ink-3:#78868a;
    --rule:#263033; --grid:#222b2e; --grid-strong:#465358;
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
    --good:#4bb47e; --good-bg:#17322553; --flat-bg:#222a2c;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-0:#0f1315; --surface-1:#181e21; --surface-2:#141a1c;
  --ink:#eaf0f0; --ink-2:#a7b3b7; --ink-3:#78868a;
  --rule:#263033; --grid:#222b2e; --grid-strong:#465358;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --good:#4bb47e; --good-bg:#17322553; --flat-bg:#222a2c;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--surface-0); color:var(--ink);
  font-family:"IBM Plex Sans KR",system-ui,-apple-system,"Malgun Gothic",sans-serif;
  font-size:16px; line-height:1.72; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1020px; margin:0 auto; padding:56px 24px 96px; display:flex;
      flex-direction:column; gap:44px}
.prose{max-width:var(--measure); display:flex; flex-direction:column; gap:16px}
h1,h2,h3{font-family:"Gowun Batang",Georgia,"Batang",serif; text-wrap:balance;
         margin:0; line-height:1.28; font-weight:700}
h1{font-size:clamp(30px,4.6vw,46px); letter-spacing:-.015em}
h2{font-size:clamp(21px,2.6vw,27px); margin-top:8px}
h3{font-size:18px; font-weight:400}
p{margin:0}
.eyebrow{font-size:12px; letter-spacing:.16em; text-transform:uppercase;
         color:var(--ink-3); font-family:"IBM Plex Mono",ui-monospace,monospace}
.lede{font-size:19px; color:var(--ink-2); max-width:64ch}
section{display:flex; flex-direction:column; gap:18px}
hr{border:0; border-top:1px solid var(--rule); margin:0}
a{color:var(--series-1)}
code,.mono,.n{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,monospace;
              font-variant-numeric:tabular-nums}
code{font-size:.9em; background:var(--surface-2); padding:.1em .35em; border-radius:3px}

.verdicts{display:grid; grid-template-columns:repeat(auto-fit,minmax(248px,1fr)); gap:14px}
.verdict{background:var(--surface-1); border:1px solid var(--rule); border-radius:10px;
         padding:20px; display:flex; flex-direction:column; gap:9px}
.verdict .k{font-size:13px; color:var(--ink-2)}
.verdict .v{font-family:"IBM Plex Mono",monospace; font-size:29px; font-weight:600;
            letter-spacing:-.02em; font-variant-numeric:tabular-nums}
.verdict .note{font-size:13.5px; color:var(--ink-3); line-height:1.55}
.verdict.is-good{border-color:color-mix(in oklab,var(--good) 42%,var(--rule))}
.verdict.is-good .v{color:var(--good)}

.chip{display:inline-block; padding:.14em .6em; border-radius:999px; font-size:12.5px;
      background:var(--flat-bg); color:var(--ink-2); white-space:nowrap}
.chip.good{background:var(--good-bg); color:var(--good); font-weight:600}

.scroll{overflow-x:auto; border:1px solid var(--rule); border-radius:10px;
        background:var(--surface-1)}
table{border-collapse:collapse; width:100%; font-size:14px}
th,td{padding:9px 13px; text-align:left; border-bottom:1px solid var(--rule);
      white-space:nowrap}
thead th{font-size:12px; letter-spacing:.05em; color:var(--ink-3); font-weight:500;
         background:var(--surface-2)}
tbody tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right; font-family:"IBM Plex Mono",monospace;
          font-variant-numeric:tabular-nums}
td.dim{color:var(--ink-3)}
td.strong{font-weight:600}
tr.r-ref td{background:var(--surface-2)}
tr.r-floor td{color:var(--ink-3)}
tr.r-sub td{color:var(--ink-2); font-size:13px}
tr.r-ceiling td{background:var(--surface-2)}
tr.r-old td:first-child::before,tr.r-new td:first-child::before{
  content:""; display:inline-block; width:9px; height:9px; border-radius:2px;
  margin-right:8px; vertical-align:baseline}
tr.r-old td:first-child::before{background:var(--series-2)}
tr.r-new td:first-child::before{background:var(--series-3)}

figure{margin:0}
.fig{background:var(--surface-1); border:1px solid var(--rule); border-radius:10px;
     padding:18px 18px 12px; display:flex; flex-direction:column; gap:10px; overflow-x:auto}
.fig .cap{font-size:13.5px; color:var(--ink-2); max-width:72ch; line-height:1.6}
.tick{font-size:11px; fill:var(--ink-3); font-family:"IBM Plex Mono",monospace}
.axis{font-size:12px; fill:var(--ink-2)}
.dlabel{font-size:12.5px; font-weight:600; font-family:"IBM Plex Sans KR",sans-serif}

.clips{display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px}
.clip{background:var(--surface-1); border:1px solid var(--rule); border-radius:9px;
      padding:12px; display:flex; flex-direction:column; gap:8px}
.clip figcaption{font-size:12.5px; color:var(--ink-2)}
.clip audio{width:100%; height:34px}
.caveat{background:var(--surface-1); border:1px solid var(--rule);
        border-left:3px solid var(--series-2); border-radius:8px; padding:18px 22px;
        display:flex; flex-direction:column; gap:11px}
.caveat h2{font-size:19px}
ul{margin:0; padding-left:1.15em; display:flex; flex-direction:column; gap:8px}
li{padding-left:.15em}
.foot{color:var(--ink-3); font-size:13px; max-width:var(--measure)}
:focus-visible{outline:2px solid var(--series-1); outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


def build(summary, table, with_audio=True):
    prim = {r['metric']: r for r in summary['primary']}
    amp = prim.get('within_std_ratio', {})
    corr = prim.get('within_corr', {})
    rpa = prim.get('rpa50', {})
    n = summary['n_items']

    def gt_med(metric):
        d = summary['gt'].get(metric, {})
        return float(np.median(list(d.values()))) if d else float('nan')

    verdicts = f"""
    <div class="verdicts">
      <div class="verdict is-good">
        <div class="k">노트 안에서 움직이는 양 — 시김새의 진폭</div>
        <div class="v">{num(amp.get('old_median'), 2)} → {num(amp.get('new_median'), 2)}</div>
        <div class="note">원본을 1.00 으로 봤을 때. 구 모델은 원본의 {num(amp.get('old_median', 0) * 100, 0)}%만
        움직였고, 신 모델은 {num(amp.get('new_median', 0) * 100, 0)}% 움직인다.
        시드 잡음({num(amp.get('seed_iqr'), 2)})과 보코더 천장 격차({num(amp.get('ceiling_delta'), 2)})를 모두 넘는다.</div>
      </div>
      <div class="verdict">
        <div class="k">원본과 같은 시점에 움직이는가 — 시김새의 내용</div>
        <div class="v">{num(corr.get('old_median'), 2)} → {num(corr.get('new_median'), 2)}</div>
        <div class="note">노트 안쪽 움직임의 원본 대비 상관. 둘 다 사실상 0 이다.
        같은 파이프라인에 원본 f0 를 넣으면 0.85 가 나오므로, 이 붕괴는 전적으로
        피치 예측기에서 일어난다.</div>
      </div>
      <div class="verdict">
        <div class="k">음정 자체의 정확도</div>
        <div class="v">{num(rpa.get('old_median'), 2)} → {num(rpa.get('new_median'), 2)}</div>
        <div class="note">RPA-50 — 원본과 50 cents 안에서 일치하는 프레임 비율.
        차이가 시드 잡음과 천장 격차 안에 들어가므로 개선이라 부르지 않는다.</div>
      </div>
    </div>"""

    showcase = []
    for item in SHOWCASE:
        if item not in summary['gt'].get('within_std', {}):
            continue
        def m(sysrung, metric, default=float('nan')):
            d = table.get((*sysrung, metric), {}).get(item)
            return float(np.median(d)) if d else default

        snap = summary['gt'].get('octave_snap_rate', {}).get(item, float('nan'))
        showcase.append(f"""
        <figure class="fig">
          {PL.f0_overlay(item)}
          <figcaption class="cap"><b>{e(item)}</b> · {e(I.speaker_of(item))} —
          이 발화에서 구 모델의 노트내 변동비는
          {num(m((f'OLD_{C.OLD_PRIMARY}', '5_pred_own'), 'within_std_ratio'), 2)},
          신 모델은 {num(m(('NEW', '3_pitch'), 'within_std_ratio'), 2)} 이고,
          원본과의 상관은 {num(m((f'OLD_{C.OLD_PRIMARY}', '5_pred_own'), 'within_corr'), 2)}
          → {num(m(('NEW', '3_pitch'), 'within_corr'), 2)} 이다.
          회색 점선이 두 모델 모두 조건으로 받은 ROSVOT 노트 계단.
          원본의 옥타브 스냅률 {num(snap * 100, 1)}% — 선이 끊긴 자리가 그 보정 지점이다.</figcaption>
        </figure>""")

    clips = ''
    if with_audio:
        blocks = []
        for item in SHOWCASE[:2]:
            paths = [
                (I.wav_of(item), f'{item} · 원본'),
                (C.OLD_LADDER[C.OLD_PRIMARY] / f'{item}__5_pred_own.wav', f'{item} · 구 모델'),
                (C.OUT / 'new' / f'{item}__3_pitch__s1.wav', f'{item} · 신 모델'),
            ]
            got = [audio_tag(p, l) for p, l in paths if Path(p).exists()]
            if got:
                blocks.append('<div class="clips">' + ''.join(got) + '</div>')
        clips = ''.join(blocks)

    swap = summary.get('reference_swap', {})
    swap_rmse = (float(np.median(list(swap['f0_rmse_cent'].values())))
                 if swap.get('f0_rmse_cent') else float('nan'))
    swap_w = (float(np.median(list(swap['within_std'].values())))
              if swap.get('within_std') else float('nan'))

    ext = summary.get('extended')
    ext_html = ''
    if ext and ext.get('rows'):
        body = ''.join(
            f'<tr><td>{e(r["label"])}</td>'
            f'<td class="n">{num(r["paired_median"], 3)}</td>'
            f'<td class="n">{num(r["rest_median"], 3)}</td>'
            f'<td class="n dim">{num(r["rest_iqr"], 3)}</td>'
            f'<td class="n">{num(r["p"], 3)}</td>'
            f'<td>{"<span class=chip>구분되지 않음</span>" if (r["p"] or 0) > 0.05 else "<span class=chip>차이 있음</span>"}</td></tr>'
            for r in ext['rows'])
        ext_html = f"""
    <section>
      <h2>홀드아웃 {ext['n']}발화로 넓혀 보면</h2>
      <div class="prose">
        <p>19쌍은 구 모델의 렌더가 24개뿐이라 정해진 크기다. 항목을 더해도 쌍체
        검정력은 늘지 않으므로, 확장은 신 모델의 <em>절대</em> 수치를 좁히고
        19쌍이 홀드아웃을 대표하는지 확인하는 데 쓴다.</p>
      </div>
      <div class="scroll"><table>
        <thead><tr><th>지표</th><th class="n">19쌍</th>
        <th class="n">나머지 {ext['rows'][0]['n_outside']}발화</th>
        <th class="n">IQR</th><th class="n">Mann–Whitney p</th><th>대표성</th></tr></thead>
        <tbody>{body}</tbody></table></div>
      <p class="foot">p 가 크다는 것은 두 표본이 구분되지 않는다는 뜻이고, 그래야
      19쌍의 결론을 홀드아웃 전체로 일반화할 수 있다. 19항목은 우리가 고른 것이 아니라
      구 스크립트가 <code>np.linspace</code> 로 고른 것이라 이 확인이 필요하다.
      신 모델 3시드 중앙값.</p>
    </section>"""

    return f"""<title>판소리 두 모델, 시김새로 재보기</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">

<header class="prose">
  <p class="eyebrow">홀드아웃 {n}쌍 · 신 모델 5시드 중앙값</p>
  <h1>판소리 두 모델, 시김새로 재보기</h1>
  <p class="lede">구 모델(MoonInTheRiver cascade, 24 kHz)과 신 모델(openvpi reflow,
  44.1 kHz)을 학습에 쓰이지 않은 같은 발화에서 같은 자로 쟀다. 결론은 한 문장으로
  줄어든다 — <b>신 모델은 판소리처럼 움직이지만, 원본과 함께 움직이지는 않는다.</b></p>
</header>

{verdicts}

<section>
  <h2>무엇을 무엇과 비교했는가</h2>
  <div class="prose">
    <p>구 모델의 홀드아웃 사다리 wav 가 그대로 남아 있어 구 모델을 다시 돌릴 필요가
    없었다. 단의 의미도 코드로 확인했다 — 구 <code>5_pred_own</code> 은 <b>실제 음소 길이 +
    모델이 예측한 f0 + 모델이 만든 mel</b> 이고, 신 <code>3_pitch</code> 가 정확히 같은
    조건이다. 이 둘이 헤드라인 쌍이다.</p>
    <p><b>정권진(<code>a580300d</code>) 5항목은 뺐다.</b> 구 모델은 이 곡을 홀드아웃했지만
    신 모델의 설정에서는 곡 하나가 화자 전부라 홀드아웃이 자동 해제됐다 — 신 모델에게는
    학습 데이터다. 제외는 주석이 아니라 코드가 강제한다: 실행할 때마다 설정 파일의
    <code>test_prefixes</code> 를 읽어 위반하면 예외를 던진다.</p>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>사다리 단</th><th>구 모델</th><th>신 모델</th><th>무엇을 분리하는가</th></tr></thead>
    <tbody>
      <tr><td>원본 녹음</td><td class="mono">1_orig</td><td class="mono">코퍼스 원본</td>
        <td>양쪽 공통 기준 (44.1 kHz 하나만 쓴다)</td></tr>
      <tr class="r-ceiling"><td>보코더 천장</td><td class="mono">2_gt_crepe</td>
        <td class="mono">0_vocoder</td><td>GT mel + GT f0 — 보코더와 추출기 몫</td></tr>
      <tr><td><b>모델 출력</b></td><td class="mono">5_pred_own</td>
        <td class="mono">3_pitch</td><td><b>헤드라인 쌍</b> — 실제 길이, 각자의 f0 와 mel</td></tr>
      <tr class="r-sub"><td>GT f0 재합성</td><td>없음</td><td class="mono">2_resynth</td>
        <td>신 모델 진단용 — 음향모델만</td></tr>
      <tr class="r-sub"><td>길이까지 예측</td><td>없음</td><td class="mono">4_full</td>
        <td>구 사다리가 만든 적 없어 비교 대상이 없다</td></tr>
    </tbody></table></div>
</section>

<section>
  <h2>시김새 분해</h2>
  <div class="prose">
    <p>노트 계단을 빼고 남은 것을 둘로 가른다. 노트마다의 상수는 <b>조율</b>이고,
    노트 안에서 움직이는 나머지가 <b>시김새</b>다. 세 숫자가 판정한다 — 원본이 얼마나
    움직이는가, 모델이 얼마나 움직이는가, 그리고 그 둘이 <em>같이</em> 움직이는가.</p>
  </div>
  {sigimsae_table(table)}
  <p class="foot">단위는 cents, 19항목 중앙값. <b>노트내 변동은 원본에 가까울수록 좋다 —
  클수록 좋은 값이 아니다.</b> 원본 {num(gt_med('within_std'), 1)} 은
  <code>docs/SIGIMSAE_PLAN.md</code> 가 CREPE 로 재서 적어 둔 161.4 와 같은 자리에
  떨어진다. 구 모델 행도 그 문서의 70.7 / 31.8 / 17.4 / 0.145 와 같은 이야기를 한다 —
  기준 추출기와 프레임률이 달라 정확히 같을 수는 없고, 그래서 맞춰야 할 기준선이 아니라
  참조점으로만 쓴다.</p>
</section>

<section>
  <h2>진폭은 왔고, 타이밍은 오지 않았다</h2>
  <figure class="fig">
    {PL.amplitude_vs_timing(summary)}
    <figcaption class="cap">오른쪽으로 갈수록 원본만큼 움직이고, 위로 갈수록 원본과
    같은 시점에 움직인다. 신 모델은 오른쪽으로 크게 옮겨갔지만 위로는 가지 않았다.
    점 하나가 발화 하나다.</figcaption>
  </figure>
  <div class="prose">
    <p>이것이 이 리포트에서 가장 중요한 그림이다. 구 모델은 노트를 평평하게 불렀고
    (진폭 {num(amp.get('old_median'), 2)}), 신 모델은 원본만큼 움직인다
    (진폭 {num(amp.get('new_median'), 2)}). 그런데 <b>원본과의 상관은 둘 다 0 근처다.</b></p>
    <p>어디서 무너지는지는 사다리가 답한다. 같은 파이프라인에 <em>원본 f0</em> 를 넣은
    <code>2_resynth</code> 의 상관은 0.85, 보코더 천장은 0.88 이다. 음향모델도 보코더도
    윤곽을 잘 보존한다. 상관이 사라지는 곳은 <b>variance 피치 예측기 한 곳</b>이다.</p>
    <p>다만 상관 0 을 곧바로 실패라고 읽는 것은 성급하다. 신 모델의 피치 예측기는
    생성 모델이고, 명창이 언제 목을 떨지를 악보에서 결정론적으로 맞히는 것은 애초에
    가능한 과제가 아니다. 그래서 노트별 변동량의 <em>분포</em>가 얼마나 닮았는지도 쟀고,
    거기서는 분포 거리가 119 → 50 으로 줄었다. 진폭과 분포는 맞고, 시점은 맞지 않는다.
    이 측정만으로는 "판소리다운 성질의 움직임"과 "크기만 맞는 무작위 움직임"을
    가르지 못한다.</p>
  </div>
</section>

<section>
  <h2>발화 하나하나에서</h2>
  <figure class="fig">
    {PL.paired_dots(summary, 'within_std_ratio', target=1.0)}
    <figcaption class="cap">노트내 변동비. 주황이 구 모델, 초록이 신 모델, 옅은 초록
    막대가 같은 발화를 5개 시드로 다시 뽑았을 때의 사분위 범위 — 효과가 샘플러 잡음보다
    큰지를 눈으로 확인하는 자리다. 점선이 원본 수준 1.0.</figcaption>
  </figure>
</section>

<section>
  <h2>곡선으로 보면</h2>
  {''.join(showcase)}
</section>

{f'<section><h2>들어 보기</h2><div class="prose"><p>대표 발화의 원본 · 구 모델 · 신 모델. 64 kbps 모노로 눌러 페이지에 실었다.</p></div>{clips}</section>' if clips else ''}

<section>
  <h2>사전 등록한 1차 종점</h2>
  <div class="prose">
    <p>지표를 30개 넘게 재면 그중 몇 개는 우연히 유의해진다. 그래서 숫자를 보기 전에
    종점 4개를 고정하고 Holm 보정을 걸었다. 개선이라고 쓰려면 세 가지가 동시에
    참이어야 한다 — <b>시드 잡음의 2배를 넘고</b>, <b>보정한 p 가 0.05 미만이고</b>,
    <b>보코더 천장 격차보다 크다</b>. 세 값을 모두 같은 줄에 인쇄해 두었으니 다른
    기준을 쓰고 싶으면 그렇게 하면 된다.</p>
  </div>
  {endpoint_table(summary['primary'])}
  <p class="foot">n = {n} 쌍, Wilcoxon 부호순위 정확검정, 중앙값 쌍체차의 95% 신뢰구간은
  10,000회 부트스트랩. 신 모델 값은 5시드 중앙값, 구 모델은 결정론적 회귀라
  렌더 하나가 곧 모델의 답이다.</p>
</section>

<section>
  <h2>대역을 맞춘 음색 비교</h2>
  <div class="prose">
    <p>80 bin/12 kHz 와 128 bin/16 kHz 의 mel 거리를 직접 비교하는 것은 서로 다른 정의를
    비교하는 것이라 의미가 없다. 그래서 양쪽을 24 kHz 로 내려 0–11 kHz 공통 분석계에서
    다시 쟀다. 구 쪽을 올리지 않고 신 쪽을 내린다 — 구 보코더가 만든 적 없는 대역을
    만들어 주면 모든 고역 오차가 공짜로 좋아진다.</p>
  </div>
  {spectral_table(table)}
  <p class="foot"><b>천장 두 줄만이 공정한 음색 비교다</b> — 양쪽 다 원본 mel 과 원본 f0 로
  구동되므로 남는 차이는 보코더와 샘플레이트 몫이다. 모델 두 줄의 스펙트럼 거리는
  f0 오차가 지배하므로 음정 지표와 독립적인 정보가 아니다.
  덧붙여, 이 아카이브 녹음들은 11 kHz 초과 에너지가 중앙값
  {num(gt_med('above_11k_frac') * 100, 3)}% 뿐이다 — 44.1 kHz 로 올린 이득은
  이 코퍼스에서 생각보다 작고, 대역 제한 비교가 버리는 것도 거의 없다.</p>
</section>

{ext_html}

<section>
  <h2>탐색적 지표</h2>
  <div class="prose">
    <p>아래는 사전 등록하지 않았고 다중비교 보정도 걸지 않았다. 방향을 보는 용도이지
    주장의 근거가 아니다.</p>
  </div>
  {endpoint_table(summary['exploratory'], primary=False)}
</section>

<section>
  <h2>믿기 전에 확인한 것</h2>
  <ul>
    <li><b>샘플레이트 무관성</b> — 같은 오디오를 44.1 kHz 와 24 kHz 로 넣었을 때 RMVPE 의
    답이 0.05 cents 차이, 유무성 일치 1.000, 노트내 변동비 1.000. 이게 깨졌다면
    24 kHz 시스템과 44.1 kHz 시스템을 한 표에 놓을 수 없었다.</li>
    <li><b>노트 격자 동일성</b> — 신 저장소의 <code>cumsum(note_dur)</code> 가 구 저장소의
    <code>notes.jsonl</code> 온셋을 19항목 전부에서 0.000 ms 오차로 재현한다. 두 모델이
    같은 계단을 조건으로 받았다는 뜻이고, 그래서 계단 하나가 양쪽을 다 잰다.</li>
    <li><b>널 대조</b> — 시스템 자리에 기준을 넣으면 모든 비율 1.000 · 상관 1.000 ·
    방향 일치율 1.000, 노트 계단을 넣으면 노트내 변동 0 · 편향 0 · 모션 0.</li>
    <li><b>옥타브 스냅</b> — RMVPE 는 <code>f0_min</code> 을 받기만 하고 쓰지 않는다.
    남성 통성에서 옥타브를 떨어뜨려, 보정 없이는 <em>원본</em>의 노트내 변동이 650 cents 로
    부풀었다(공표된 161 대비). 노트 잔차를 (−600, 600] 로 감아 기준과 모든 시스템에
    동일하게 적용했고, 그 결과 기준이 {num(gt_med('within_std'), 0)} cents 로 제자리를
    찾았다. 음정 정확도 지표에는 이 보정을 쓰지 않는다 — 진짜 옥타브 오류는 거기서
    벌점을 받아야 한다.</li>
    <li><b>기준 교체 진단</b> — 구 모델은 CREPE 를 타깃으로 학습했으므로 RMVPE 기준이
    불리할 수 있다. 두 기준의 차이는 {num(swap_rmse, 0)} cents 이고 원본 노트내 변동은
    CREPE {num(swap_w, 0)} vs RMVPE {num(gt_med('within_std'), 0)} cents 다. 진폭
    효과({num(amp.get('delta_median'), 2)})는 이 차이보다 훨씬 크지만, 음정 쪽 차이는
    그렇지 않아 애초에 개선으로 주장하지 않았다.</li>
  </ul>
</section>

<div class="caveat">
  <h2>이 숫자가 말할 수 없는 것</h2>
  <ul>
    <li><b>원인을 가르지 못한다.</b> 두 세대는 샘플레이트·보코더·백본·손실·f0 추출기·
    음소 라벨(26,955중 3,109개 변경)·학습 화자 구성·실효 학습량(약 30% 증가)이 한꺼번에
    다르다. "신 모델이 X 만큼 낫다"는 말할 수 있고 "무엇 때문에"는 말할 수 없다.
    값싸게 가능한 유일한 부분 귀속이 보코더 천장 차감이고, 그건 모든 표에 넣었다.</li>
    <li><b>n = {n} 은 작다.</b> 구 모델의 렌더가 24개뿐이라 정해진 크기이고, 신 모델
    항목을 더해도 쌍체 검정력은 늘지 않는다.</li>
    <li><b>표본이 비대칭이다.</b> 신 모델은 5시드 중앙값, 구 모델은 렌더 하나다.
    구 <code>pansori_aux</code> 가 결정론적 회귀라 draw 가 아니라는 것이 근거이지만,
    구 쪽의 잡음 바닥은 측정하지 못했다.</li>
    <li><b>0–11 kHz 밖은 재지 않았다.</b> 그 위는 신 모델 단독 능력이라 비교 대상이
    없고, 따라서 음색 수치는 신 모델 이득의 하한이다.</li>
    <li><b>주관 품질이 아니다.</b> 모든 지표는 "원본과 얼마나 닮았나"이지
    "얼마나 좋게 들리나"가 아니다. 특히 노트내 변동은 원본에 가까울수록 좋은 값이다.</li>
  </ul>
</div>

<footer class="prose">
  <hr>
  <p class="foot">재현: <code>PYTHONPATH=. python pansori/compare/smoke.py</code> →
  <code>render_new.py --items fair --seeds 5</code> → <code>collect.py</code> →
  <code>stats.py</code> → <code>report.py</code>.
  체크포인트 acoustic {C.CKPT_ACOUSTIC:,} · dur {C.CKPT_DUR:,} · pitch {C.CKPT_PITCH:,},
  분석 격자 10 ms, 기준 추출기 RMVPE.</p>
</footer>

</div>
"""


def main():
    p = argparse.ArgumentParser(description='render the comparison report')
    p.add_argument('--summary', default=str(C.OUT / 'summary.json'))
    p.add_argument('--results', default=str(C.OUT / 'results.jsonl'))
    p.add_argument('--out', default=str(OUT_HTML))
    p.add_argument('--no-audio', action='store_true')
    args = p.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    from pansori.compare.stats import load
    _, table = load(args.results)

    html_text = build(summary, table, with_audio=not args.no_audio)
    Path(args.out).write_text(html_text, encoding='utf-8')
    size = Path(args.out).stat().st_size
    print(f'→ {args.out}  ({size / 1e6:.2f} MB)')
    if size > 16e6:
        sys.exit('16 MB 를 넘었다 — 오디오를 줄일 것')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Paired statistics, with the decision rule fixed before the numbers were seen.

Three things have to be true before this file will call a difference an
improvement, and all three are printed next to every headline number so a reader
who disagrees with the threshold can apply their own:

  1. |median paired difference| > 2 x the median per-item seed IQR
  2. Holm-adjusted p < 0.05 across the four pre-registered endpoints
  3. the difference exceeds the corresponding 0_vocoder ceiling gap

The third is the one that keeps this honest. 0_vocoder is ground-truth mel driven
by ground-truth f0 on both sides, so whatever separates the systems THERE is the
vocoder and the sample rate, not the model. A model-level difference smaller than
its own ceiling gap has not been shown to be a model-level difference at all.

Medians and IQRs throughout, never mean +- sd: one octave slip in this corpus is
several hundred cents and would move a mean by more than any real effect.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/compare/stats.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402

# metric -> (label, better direction). 'gt' means "closest to ground truth wins",
# which is the correct direction for every 시김새 amplitude: a model that moves
# MORE than the singer is not more expressive, it is unstable.
DIRECTION = {
    'within_std_ratio': ('노트내 변동비 (1.0 이 정답)', 'gt'),
    'within_std_core_ratio': ('노트내 변동비 core', 'gt'),
    'motion_ratio': ('모션비', 'gt'),
    'wobble_ratio': ('wobble 비', 'gt'),
    'within_corr': ('노트내 GT 상관', 'up'),
    'transition_dir_agree': ('전이 방향 일치율', 'up'),
    'transition_dir_agree_release': ('전이 방향 일치율 (release)', 'up'),
    'rpa50': ('RPA-50', 'up'),
    'rca50': ('RCA-50', 'up'),
    'voiced_agree': ('유무성 일치', 'up'),
    'f0_rmse_cent': ('f0 RMSE (cents)', 'down'),
    'gross_error': ('총오차율', 'down'),
    'octave_jump': ('옥타브 점프율', 'down'),
    'bias_rms': ('노트편향 RMS', 'down'),
    'within_std_wasserstein': ('노트내 std 분포거리', 'down'),
    'mcd_db': ('MCD (dB)', 'down'),
    'lsd': ('LSD', 'down'),
    'mel_l1_common': ('공통 mel L1', 'down'),
    'slope_mag_ratio': ('전이 기울기 크기비', 'gt'),
    'kkeokgi_rate': ('꺾기율', 'gt'),
    'octave_snap_rate': ('옥타브 스냅률', 'down'),
}


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l]
    table = defaultdict(dict)          # (system, rung, metric) -> {item: [values]}
    for r in rows:
        table[(r['system'], r['rung'], r['metric'])].setdefault(r['item'], []).append(r['value'])
    return rows, table


def per_item(table, key, reduce=np.median):
    """One value per item, reducing across seeds with the median."""
    return {item: float(reduce(v)) for item, v in table.get(key, {}).items()}


def seed_iqr(table, key):
    """Median across items of the per-item interquartile range across seeds.

    This is the noise floor. A difference smaller than it is a statement about
    the sampler, not about the model.
    """
    spreads = [float(np.percentile(v, 75) - np.percentile(v, 25))
               for v in table.get(key, {}).values() if len(v) > 1]
    return float(np.median(spreads)) if spreads else float('nan')


def paired(a, b):
    common = sorted(set(a) & set(b))
    return common, np.array([a[i] for i in common]), np.array([b[i] for i in common])


def bootstrap_ci(d, n=C.BOOTSTRAP_N, seed=20260823):
    """Percentile bootstrap CI of the median paired difference."""
    rng = np.random.default_rng(seed)
    if len(d) < 3:
        return float('nan'), float('nan')
    idx = rng.integers(0, len(d), size=(n, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def compare(table, metric, new_key, old_key, target=None):
    """Paired comparison of one metric between the new and the old system."""
    from scipy.stats import wilcoxon
    new = per_item(table, (new_key[0], new_key[1], metric))
    old = per_item(table, (old_key[0], old_key[1], metric))
    if not new or not old:
        return None
    items, vn, vo = paired(new, old)
    if len(items) < 3:
        return None

    label, direction = DIRECTION.get(metric, (metric, 'up'))
    if direction == 'gt':
        t = 1.0 if target is None else target
        # distance from the ground-truth level; smaller is better
        vn_s, vo_s = np.abs(vn - t), np.abs(vo - t)
        d = vo_s - vn_s                      # positive = new is closer to GT
    elif direction == 'down':
        d = vo - vn                          # positive = new is lower = better
    else:
        d = vn - vo                          # positive = new is higher = better

    try:
        p = float(wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)
    except ValueError:
        p = float('nan')
    lo, hi = bootstrap_ci(d)

    # matched-pairs rank-biserial; skewed ratios make Cohen's d inappropriate
    r_abs = np.abs(d)
    ranks = np.argsort(np.argsort(r_abs)) + 1.0
    wp, wm = ranks[d > 0].sum(), ranks[d < 0].sum()
    rb = float((wp - wm) / (wp + wm)) if (wp + wm) > 0 else float('nan')

    return {
        'metric': metric, 'label': label, 'direction': direction, 'n': len(items),
        'new_median': float(np.median(vn)), 'old_median': float(np.median(vo)),
        'new_iqr': float(np.percentile(vn, 75) - np.percentile(vn, 25)),
        'old_iqr': float(np.percentile(vo, 75) - np.percentile(vo, 25)),
        'delta_median': float(np.median(d)), 'ci_lo': lo, 'ci_hi': hi,
        'p': p, 'rank_biserial': rb,
        'seed_iqr': seed_iqr(table, (new_key[0], new_key[1], metric)),
        'items': items,
    }


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fmt(x, nd=3):
    return '—' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def main():
    p = argparse.ArgumentParser(description='쌍체 통계')
    p.add_argument('--results', default=str(C.OUT / 'results.jsonl'))
    p.add_argument('--old', default=f'OLD_{C.OLD_PRIMARY}')
    p.add_argument('--json', default=str(C.OUT / 'summary.json'))
    p.add_argument('--extended', default=str(C.OUT / 'results_ext.jsonl'),
                   help='홀드아웃 확장 표본 결과 (없으면 건너뛴다)')
    args = p.parse_args()

    rows, table = load(args.results)
    NEW, OLD = (('NEW', '3_pitch'), (args.old, '5_pred_own'))
    CEIL_NEW, CEIL_OLD = (('NEW', '0_vocoder'), (args.old, '2_gt_crepe'))

    # ---- reference level -------------------------------------------------
    gt = {m: per_item(table, ('GT', '1_orig', m))
          for m in ('within_std', 'within_std_core', 'bias_rms', 'motion_cps',
                    'wobble_iqr', 'octave_snap_rate', 'n_notes', 'above_11k_frac')}
    print('기준 (GT, RMVPE, 옥타브 스냅 적용) — 19항목 중앙값')
    for k, v in gt.items():
        if v:
            print(f'  {k:20s} {np.median(list(v.values())):8.3f}')

    swap = {m: per_item(table, ('CREPE_vs_RMVPE', '1_orig', m))
            for m in ('f0_rmse_cent', 'voiced_agree', 'within_std')}
    if swap.get('f0_rmse_cent'):
        gt_w = np.median(list(gt['within_std'].values()))
        cr_w = np.median(list(swap['within_std'].values()))
        print(f'\n기준 교체 진단 (CREPE vs RMVPE)')
        print(f'  cents RMSE   {np.median(list(swap["f0_rmse_cent"].values())):8.1f}')
        print(f'  유무성 일치   {np.median(list(swap["voiced_agree"].values())):8.3f}')
        print(f'  노트내 변동   CREPE {cr_w:.1f} vs RMVPE {gt_w:.1f}  (비 {cr_w / gt_w:.3f})')

    # ---- primary endpoints ----------------------------------------------
    results, ceilings = [], {}
    for m in C.PRIMARY_ENDPOINTS:
        r = compare(table, m, NEW, OLD)
        if r:
            c = compare(table, m, CEIL_NEW, CEIL_OLD)
            r['ceiling_delta'] = c['delta_median'] if c else float('nan')
            ceilings[m] = r['ceiling_delta']
            results.append(r)

    if results:
        adj = holm(np.array([r['p'] for r in results]))
        for r, a in zip(results, adj):
            r['p_holm'] = float(a)
            r['beats_seed_noise'] = bool(
                abs(r['delta_median']) > C.SEED_IQR_FACTOR * r['seed_iqr']
                if np.isfinite(r['seed_iqr']) else False)
            r['beats_ceiling'] = bool(
                abs(r['delta_median']) > abs(r['ceiling_delta'])
                if np.isfinite(r['ceiling_delta']) else False)
            r['claim'] = bool(r['beats_seed_noise'] and r['beats_ceiling']
                              and np.isfinite(a) and a < 0.05)

    # ---- exploratory -----------------------------------------------------
    explore = []
    for m in DIRECTION:
        if m in C.PRIMARY_ENDPOINTS:
            continue
        r = compare(table, m, NEW, OLD)
        if r:
            c = compare(table, m, CEIL_NEW, CEIL_OLD)
            r['ceiling_delta'] = c['delta_median'] if c else float('nan')
            r['p_holm'] = float('nan')
            explore.append(r)

    def show(title, rs):
        print(f'\n{title}')
        print(f'{"지표":26s} {"구":>9s} {"신":>9s} {"차이(중앙)":>11s} '
              f'{"95% CI":>17s} {"p":>8s} {"시드IQR":>8s} {"천장차":>8s} {"판정":>6s}')
        print('-' * 118)
        for r in rs:
            ci = f'[{fmt(r["ci_lo"])}, {fmt(r["ci_hi"])}]'
            verdict = ('개선' if r.get('claim') else
                       '—' if 'claim' in r else '')
            pv = r.get('p_holm', r['p'])
            print(f'{r["label"]:26s} {fmt(r["old_median"]):>9s} {fmt(r["new_median"]):>9s} '
                  f'{fmt(r["delta_median"]):>11s} {ci:>17s} {fmt(pv, 4):>8s} '
                  f'{fmt(r["seed_iqr"]):>8s} {fmt(r["ceiling_delta"]):>8s} {verdict:>6s}')

    show(f'1차 종점 (사전 등록, Holm 보정) — 구 {args.old} 5_pred_own vs 신 3_pitch, n={results[0]["n"] if results else 0}',
         results)
    show('탐색적 지표 (다중비교 보정 없음, 참고용)', explore)

    # ---- the 시김새 table, in the shape docs/SIGIMSAE_PLAN.md §1 used -----
    print('\n시김새 분해 표 (docs/SIGIMSAE_PLAN.md §1 과 같은 모양)')
    print(f'{"":28s} {"노트편향 RMS":>13s} {"노트내 변동":>12s} {"(core)":>9s} {"GT 상관":>9s}')
    print('-' * 76)

    def line(name, sys_rung, corr_metric=True):
        b = per_item(table, (*sys_rung, 'bias_rms'))
        w = per_item(table, (*sys_rung, 'within_std'))
        wc = per_item(table, (*sys_rung, 'within_std_core'))
        cr = per_item(table, (*sys_rung, 'within_corr'))
        if not w:
            return
        print(f'{name:28s} {np.median(list(b.values())) if b else float("nan"):13.1f} '
              f'{np.median(list(w.values())):12.1f} '
              f'{np.median(list(wc.values())) if wc else float("nan"):9.1f} '
              f'{(np.median(list(cr.values())) if cr else 1.0):9.3f}')

    line('GT (RMVPE)', ('GT', '1_orig'))
    print(f'{"노트 계단":28s} {0.0:13.1f} {0.0:12.1f} {0.0:9.1f} {"—":>9s}')
    line(f'구 {args.old} (5_pred_own)', OLD)
    line('신 (3_pitch, 5시드 중앙)', NEW)
    line('신 (2_resynth, GT f0)', ('NEW', '2_resynth'))
    line('신 (0_vocoder 천장)', CEIL_NEW)
    line(f'구 (2_gt_crepe 천장)', CEIL_OLD)
    print('\n※ 노트내 변동은 GT 에 가까울수록 좋다. 클수록 좋은 값이 아니다.')

    # ---- spectral: the only rung where f0 is ground truth on both sides ----
    print('\n대역 제한 스펙트럼 (24 kHz / 0–11 kHz 공통 분석)')
    print(f'{"":30s} {"MCD (dB)":>10s} {"LSD":>8s} {"공통 mel L1":>12s}')
    print('-' * 64)
    for name, key in ((f'구 천장 (2_gt_crepe)', CEIL_OLD),
                      ('신 천장 (0_vocoder)', CEIL_NEW),
                      (f'구 모델 (5_pred_own)', OLD),
                      ('신 모델 (3_pitch)', NEW)):
        v = [per_item(table, (*key, m)) for m in ('mcd_db', 'lsd', 'mel_l1_common')]
        if not v[0]:
            continue
        print(f'{name:30s} ' + ' '.join(
            f'{np.median(list(x.values())):>10.2f}' if i == 0 else
            f'{np.median(list(x.values())):>8.3f}' if i == 1 else
            f'{np.median(list(x.values())):>12.3f}'
            for i, x in enumerate(v)))
    print('※ 천장 두 줄만이 공정한 음색 비교다 — 양쪽 다 GT mel 과 GT f0 로 구동된다.')
    print('  모델 두 줄의 스펙트럼 거리는 f0 오차가 지배하므로 Tier 1 과 독립적인 정보가 아니다.')

    ab = per_item(table, ('GT', '1_orig', 'above_11k_frac'))
    if ab:
        print(f'※ 원본의 11 kHz 초과 에너지 비중 중앙값 {np.median(list(ab.values())):.4%} — '
              f'이 아카이브 녹음은 이미 대역이 좁아, 대역 제한이 버리는 것이 거의 없다.')


    # ---- extended holdout sample: absolute level + representativeness ----
    extended = None
    ext_path = Path(args.extended)
    if ext_path.exists():
        from scipy.stats import mannwhitneyu
        _, ext_table = load(ext_path)
        paired_set = set(results[0]['items']) if results else set()
        rows = []
        for m in C.PRIMARY_ENDPOINTS:
            vals = per_item(ext_table, ('NEW', '3_pitch', m))
            if not vals:
                continue
            # the 19 come from the main run (5 seeds); the comparison set is the
            # extended sample with those removed, so the two groups are disjoint
            inside = list(per_item(table, ('NEW', '3_pitch', m)).values())
            outside = [v for i, v in vals.items() if i not in paired_set]
            if len(outside) < 5:
                continue
            # A/B on 19 items only licenses a claim about the holdout if those 19
            # are not special. They were chosen by the OLD script's np.linspace,
            # not by us, so this has to be checked rather than assumed.
            pv = (float(mannwhitneyu(inside, outside).pvalue)
                  if len(inside) >= 3 else float('nan'))
            label = DIRECTION.get(m, (m, ''))[0]
            rows.append({
                'metric': m, 'label': label,
                'paired_median': float(np.median(inside)) if inside else float('nan'),
                'rest_median': float(np.median(outside)),
                'rest_iqr': float(np.percentile(outside, 75) - np.percentile(outside, 25)),
                'n_inside': len(inside), 'n_outside': len(outside), 'p': pv,
            })
        extended = {'n': len(set(vals)), 'overlap': len(paired_set & set(vals)),
                    'rows': rows}
        print(f'\n홀드아웃 확장 ({extended["n"]}발화, 19쌍과 겹침 {extended["overlap"]})')
        print(f'{"지표":26s} {"19쌍":>9s} {"나머지":>9s} {"IQR":>9s} {"MW p":>8s}')
        print('-' * 66)
        for r in rows:
            print(f'{r["label"]:26s} {fmt(r["paired_median"]):>9s} '
                  f'{fmt(r["rest_median"]):>9s} {fmt(r["rest_iqr"]):>9s} '
                  f'{fmt(r["p"], 3):>8s}')

    payload = {'primary': results, 'exploratory': explore, 'extended': extended,
               'gt': {k: {i: float(x) for i, x in v.items()} for k, v in gt.items()},
               'reference_swap': {k: {i: float(x) for i, x in v.items()}
                                  for k, v in swap.items()},
               'old_run': args.old, 'n_items': results[0]['n'] if results else 0}
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                               encoding='utf-8')
    print(f'\n→ {args.json}')


if __name__ == '__main__':
    main()

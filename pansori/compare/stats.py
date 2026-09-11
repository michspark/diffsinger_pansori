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
# which is the correct direction for every sigimsae amplitude: a model that moves
# MORE than the singer is not more expressive, it is unstable.
DIRECTION = {
    'within_std_ratio': ('within-note variation ratio (1.0 is correct)', 'gt'),
    'within_std_core_ratio': ('within-note variation ratio, core', 'gt'),
    'motion_ratio': ('motion ratio', 'gt'),
    'wobble_ratio': ('wobble ratio', 'gt'),
    'within_corr': ('within-note correlation with GT', 'up'),
    'transition_dir_agree': ('transition direction agreement', 'up'),
    'transition_dir_agree_release': ('transition direction agreement (release)', 'up'),
    'rpa50': ('RPA-50', 'up'),
    'rca50': ('RCA-50', 'up'),
    'voiced_agree': ('voiced agreement', 'up'),
    'f0_rmse_cent': ('f0 RMSE (cents)', 'down'),
    'gross_error': ('gross error rate', 'down'),
    'octave_jump': ('octave jump rate', 'down'),
    'bias_rms': ('note-bias RMS', 'down'),
    'within_std_wasserstein': ('within-note std distribution distance', 'down'),
    'mcd_db': ('MCD (dB)', 'down'),
    'lsd': ('LSD', 'down'),
    'mel_l1_common': ('common-basis mel L1', 'down'),
    'slope_mag_ratio': ('transition slope magnitude ratio', 'gt'),
    'kkeokgi_rate': ('kkeokgi rate', 'gt'),
    'octave_snap_rate': ('octave snap rate', 'down'),
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
    p = argparse.ArgumentParser(description='paired statistics')
    p.add_argument('--results', default=str(C.OUT / 'results.jsonl'))
    p.add_argument('--old', default=f'OLD_{C.OLD_PRIMARY}')
    p.add_argument('--json', default=str(C.OUT / 'summary.json'))
    p.add_argument('--extended', default=str(C.OUT / 'results_ext.jsonl'),
                   help='the hold-out extension sample\'s results (skipped if absent)')
    args = p.parse_args()

    rows, table = load(args.results)
    NEW, OLD = (('NEW', '3_pitch'), (args.old, '5_pred_own'))
    CEIL_NEW, CEIL_OLD = (('NEW', '0_vocoder'), (args.old, '2_gt_crepe'))

    # ---- reference level -------------------------------------------------
    gt = {m: per_item(table, ('GT', '1_orig', m))
          for m in ('within_std', 'within_std_core', 'bias_rms', 'motion_cps',
                    'wobble_iqr', 'octave_snap_rate', 'n_notes', 'above_11k_frac')}
    print('reference (GT, RMVPE, octave snap applied) — median over 19 items')
    for k, v in gt.items():
        if v:
            print(f'  {k:20s} {np.median(list(v.values())):8.3f}')

    swap = {m: per_item(table, ('CREPE_vs_RMVPE', '1_orig', m))
            for m in ('f0_rmse_cent', 'voiced_agree', 'within_std')}
    if swap.get('f0_rmse_cent'):
        gt_w = np.median(list(gt['within_std'].values()))
        cr_w = np.median(list(swap['within_std'].values()))
        print(f'\nreference-swap diagnostic (CREPE vs RMVPE)')
        print(f'  cents RMSE   {np.median(list(swap["f0_rmse_cent"].values())):8.1f}')
        print(f'  voiced agreement  {np.median(list(swap["voiced_agree"].values())):8.3f}')
        print(f'  within-note var   CREPE {cr_w:.1f} vs RMVPE {gt_w:.1f}  (ratio {cr_w / gt_w:.3f})')

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
        print(f'{"metric":36s} {"old":>9s} {"new":>9s} {"median Δ":>11s} '
              f'{"95% CI":>17s} {"p":>8s} {"seed IQR":>9s} {"ceil gap":>9s} {"verdict":>8s}')
        print('-' * 118)
        for r in rs:
            ci = f'[{fmt(r["ci_lo"])}, {fmt(r["ci_hi"])}]'
            verdict = ('improved' if r.get('claim') else
                       '—' if 'claim' in r else '')
            pv = r.get('p_holm', r['p'])
            print(f'{r["label"]:26s} {fmt(r["old_median"]):>9s} {fmt(r["new_median"]):>9s} '
                  f'{fmt(r["delta_median"]):>11s} {ci:>17s} {fmt(pv, 4):>8s} '
                  f'{fmt(r["seed_iqr"]):>8s} {fmt(r["ceiling_delta"]):>8s} {verdict:>6s}')

    show(f'primary endpoints (pre-registered, Holm-corrected) — old {args.old} 5_pred_own vs new 3_pitch, n={results[0]["n"] if results else 0}',
         results)
    show('exploratory metrics (no multiple-comparison correction, for reference)', explore)

    # ---- the sigimsae table, in the shape docs/SIGIMSAE_PLAN.md §1 used ----
    print('\nsigimsae decomposition (the same shape as docs/SIGIMSAE_PLAN.md §1)')
    print(f'{"":30s} {"note-bias RMS":>14s} {"within-note":>12s} {"(core)":>9s} {"GT corr":>9s}')
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
    print(f'{"note staircase":30s} {0.0:14.1f} {0.0:12.1f} {0.0:9.1f} {"—":>9s}')
    line(f'old {args.old} (5_pred_own)', OLD)
    line('new (3_pitch, median of 5 seeds)', NEW)
    line('new (2_resynth, GT f0)', ('NEW', '2_resynth'))
    line('new (0_vocoder ceiling)', CEIL_NEW)
    line(f'old (2_gt_crepe ceiling)', CEIL_OLD)
    print('\n* within-note variation is better the closer it is to GT. Larger is not better.')

    # ---- spectral: the only rung where f0 is ground truth on both sides ----
    print('\nband-limited spectrum (24 kHz / 0-11 kHz common analysis)')
    print(f'{"":32s} {"MCD (dB)":>10s} {"LSD":>8s} {"common mel L1":>14s}')
    print('-' * 64)
    for name, key in ((f'old ceiling (2_gt_crepe)', CEIL_OLD),
                      ('new ceiling (0_vocoder)', CEIL_NEW),
                      (f'old model (5_pred_own)', OLD),
                      ('new model (3_pitch)', NEW)):
        v = [per_item(table, (*key, m)) for m in ('mcd_db', 'lsd', 'mel_l1_common')]
        if not v[0]:
            continue
        print(f'{name:30s} ' + ' '.join(
            f'{np.median(list(x.values())):>10.2f}' if i == 0 else
            f'{np.median(list(x.values())):>8.3f}' if i == 1 else
            f'{np.median(list(x.values())):>12.3f}'
            for i, x in enumerate(v)))
    print('* only the two ceiling rows are a fair timbre comparison -- both are driven by GT mel and GT f0.')
    print('  The model rows\' spectral distance is dominated by f0 error, so it is not information independent of Tier 1.')

    ab = per_item(table, ('GT', '1_orig', 'above_11k_frac'))
    if ab:
        print(f'* median share of the original\'s energy above 11 kHz: {np.median(list(ab.values())):.4%} -- '
              f'these archive recordings are already narrowband, so band-limiting discards almost nothing.')


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
        print(f'\nhold-out extension ({extended["n"]} utterances, {extended["overlap"]} overlapping the 19 pairs)')
        print(f'{"metric":36s} {"19 pairs":>9s} {"the rest":>9s} {"IQR":>9s} {"MW p":>8s}')
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

#!/usr/bin/env python3
"""0-2: is the pitch predictor worth more than the notes it was handed?

Two questions, and they are not the same question.

**Is the predictor better than nothing?**  ROSVOT already gives us a note per
phone.  Rendering those notes as a staircase costs zero parameters, so the
staircase is the floor the predictor has to clear.  If the staircase scores
better than the 243.8 cents the cascade stage-1 predictor scores, then the
predictor is actively destroying information it was conditioned on, and the fix
is to predict a *residual* on top of the staircase rather than absolute f0.

**Does the predictor sing sigimsae?**  Cents RMSE cannot answer that.  A contour
that nails every note center and holds each one dead flat scores well while
sounding like a MIDI file.  So this script splits the deviation from the
staircase into two parts:

    residual(t) = f0(t) - note(t)          in cents
                = note_bias(n)             per-note constant -- tuning / octave
                + within_note(t)           the part that moves -- sigimsae

`within_note` is where yoseong, jeonseong, chuseong and toeseong live.  Three numbers decide it:

    within-note std of CREPE     how much ornament real pansori actually has
    within-note std of the model how much the model produces
    correlation of the two       whether it is the *right* ornament, in time

A model that has learned nothing has within-note std near zero: it reproduces
the staircase it was given.  A model that produces motion of the right size but
zero correlation is generating decorative noise, which is worse than silence
because it will sound drunk rather than plain.

Everything is measured against the Stage-8 CREPE contour on the same frames the
resynthesis ladder uses, so the numbers are directly comparable to
pansori/eval/resynth_ladder.py.

Usage (diffsinger env, from the repo root, PYTHONPATH=.):

  # staircase only -- no model, no GPU, a few seconds
  python pansori/eval/midi_baseline.py \
      --config usr/configs/midi/cascade/pansori/aux.yaml --n 24

  # staircase vs the stage-1 predictor, on identical frames
  CUDA_VISIBLE_DEVICES=0 python pansori/eval/midi_baseline.py \
      --config usr/configs/midi/cascade/pansori/aux.yaml \
      --exp_name pansori_aux --target aux --n 24 --html
"""

import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f0_metrics  # noqa: E402

PANSORI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PANSORI_DIR.parent))
from pansori.note_seg import segment_notes_np, expand_to_frames_np  # noqa: E402
RAW_F0_DIR = PANSORI_DIR / 'data' / 'full' / 'f0'


# --------------------------------------------------------------------------
# note rendering
# --------------------------------------------------------------------------

def midi_to_hz(midi):
    """A4 = MIDI 69 = 440 Hz.  Not librosa's, because librosa maps 0 to 8.18 Hz
    and 0 is our rest marker (binarize.py:319 writes 0 for 'rest')."""
    return 440.0 * (2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0))


def render_staircase(pitch_midi, midi_dur, is_slur, mel2ph, n_frames):
    """ROSVOT notes -> a frame-level f0 contour with zero learned parameters.

    Note grouping comes from pansori/note_seg.py, which is also what the
    binarizer and the training step use — the three have to agree frame for
    frame or Phase 5's per-note parameters attach to the wrong notes.
    """
    midi_f = expand_to_frames_np(pitch_midi, mel2ph, 0)[:n_frames]
    slur_f = expand_to_frames_np(is_slur, mel2ph, 0)[:n_frames]
    note_f, _, _, _ = segment_notes_np(pitch_midi, midi_dur, mel2ph, n_frames)
    f0 = np.where(midi_f > 0, midi_to_hz(midi_f), 0.0)
    return f0, note_f, slur_f


# --------------------------------------------------------------------------
# sigimsae decomposition
# --------------------------------------------------------------------------

def decompose(f0, f0_note, note_id):
    """Split a contour's deviation from the notes into bias + motion.

    Returns arrays the length of f0, NaN wherever the decomposition is
    undefined (unvoiced, or a rest with no note to measure against).
    """
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = (f0 > 0) & (f0_note > 0)

    residual = np.full(len(f0), np.nan)
    residual[voiced] = 1200.0 * np.log2(f0[voiced] / f0_note[voiced])

    bias = np.full(len(f0), np.nan)
    for nid in np.unique(note_id[voiced]):
        sel = voiced & (note_id == nid)
        # A one-frame note has no motion to speak of; its bias absorbs
        # everything and its within-note term is 0, which is honest.
        bias[sel] = residual[sel].mean()

    return voiced, residual, bias, residual - bias


def _nan(fn, x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(fn(x)) if len(x) else float('nan')


def note_core_mask(note_id, voiced, trim=0.2):
    """Frames in the interior of each note, boundaries dropped.

    ROSVOT note times and MFA phone times are both estimates, so the frames at
    a note boundary often belong to the neighbouring note.  Pooled into one
    segment they masquerade as within-note motion.  Measuring the interior
    separately says how much of the ornament survives that objection.
    """
    core = np.zeros(len(note_id), dtype=bool)
    for nid in np.unique(note_id[voiced]):
        idx = np.where(voiced & (note_id == nid))[0]
        if len(idx) < 5:
            continue
        k = max(1, int(round(len(idx) * trim)))
        core[idx[k:len(idx) - k]] = True
    return core


def ornament_stats(f0, f0_note, note_id, ref_within=None):
    """Per-utterance sigimsae numbers for one contour."""
    voiced, residual, bias, within = decompose(f0, f0_note, note_id)
    core = note_core_mask(note_id, voiced)
    out = {
        'n_voiced': int(voiced.sum()),
        # |note_bias| RMS -- how far off the note centres are.
        'bias_rms': _nan(lambda a: np.sqrt((a ** 2).mean()), bias[voiced]),
        # within-note std -- how much the contour actually moves inside a note.
        'within_std': _nan(np.std, within[voiced]),
        # same, boundary frames removed.
        'within_std_core': _nan(np.std, within[voiced & core]),
        'residual_rms': _nan(lambda a: np.sqrt((a ** 2).mean()), residual[voiced]),
    }
    if ref_within is not None:
        n = min(len(within), len(ref_within))
        a, b = within[:n], ref_within[:n]
        both = np.isfinite(a) & np.isfinite(b)
        # Needs some spread on both sides or corrcoef returns nan/garbage.
        if both.sum() >= 8 and np.std(a[both]) > 1e-6 and np.std(b[both]) > 1e-6:
            out['within_corr'] = float(np.corrcoef(a[both], b[both])[0, 1])
        else:
            out['within_corr'] = float('nan')
    return out, within


def masked_cents_rmse(pred, ref, mask):
    """cents RMSE restricted to a frame subset (used for slur vs non-slur)."""
    n = min(len(pred), len(ref), len(mask))
    pred, ref, mask = pred[:n], ref[:n], mask[:n].astype(bool)
    both = (pred > 0) & (ref > 0) & mask
    if not both.any():
        return float('nan')
    err = 1200.0 * np.log2(pred[both] / ref[both])
    return float(np.sqrt((err ** 2).mean()))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return _nan(np.mean, vals)


def f0_figure_b64(item_name, crepe, stair, model, slur_f):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    show = lambda f0: np.where(np.asarray(f0, dtype=np.float64) > 0, f0, np.nan)
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.plot(show(stair), color='#888', lw=2.4, label='ROSVOT notes (staircase)')
    ax.plot(show(crepe), color='#d62728', lw=1.1, label='CREPE (ground truth)')
    if model is not None:
        ax.plot(show(model), color='#2ca02c', lw=1.1, label='model f0')
    # Shade the frames ROSVOT marked as slur -- that is where sigimsae is claimed.
    if slur_f is not None and np.any(slur_f > 0):
        ax.fill_between(np.arange(len(slur_f)), 0, 1, where=(slur_f > 0),
                        transform=ax.get_xaxis_transform(),
                        color='#1f77b4', alpha=0.10, lw=0)
    ax.set_yscale('log')
    ax.set_ylabel('Hz')
    ax.set_xlabel('frame (5.33 ms)')
    ax.set_title(item_name, fontsize=9, loc='left')
    ax.legend(fontsize=8, ncol=3, loc='upper right')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=96)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_html(rows, out_path, title, summary_html):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figs = ''.join(
        f'<figure><img src="data:image/png;base64,{r["fig"]}">'
        f'<figcaption>{r["item_name"]} &nbsp;|&nbsp; '
        f'staircase {r["midi"]["cents_rmse"]:.0f} cents'
        + (f' &nbsp;·&nbsp; model {r["model"]["cents_rmse"]:.0f} cents'
           if r.get('model') else '')
        + '</figcaption></figure>'
        for r in rows if r.get('fig'))
    out_path.write_text(f"""<!doctype html><meta charset=utf-8>
<title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#fafafa;color:#222}}
table{{border-collapse:collapse;margin:1rem 0}}
th,td{{border:1px solid #ddd;padding:.4rem .7rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#f0f0f0}}
figure{{margin:1.5rem 0}} img{{width:100%;max-width:1300px}}
figcaption{{color:#666;font-size:.85em;font-family:monospace}}
.note{{background:#fff8e1;border-left:4px solid #ffb300;padding:.8rem 1rem;margin:1rem 0}}
</style>
<h1>{title}</h1>
{summary_html}
<div class=note>The grey staircase is the ROSVOT notes, red is the CREPE ground truth,
green is the model's f0. The blue shading marks the spans ROSVOT flagged as slur -- where
sigimsae is claimed to be. <b>If the green hugs the grey staircase, the model is merely
following the notes and producing no sigimsae</b>; if it moves around the notes the way the
red does, it is producing sigimsae.</div>
{figs}
""", encoding='utf-8')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='',
                        help='leave empty to measure only the MIDI staircase (no model or GPU needed)')
    parser.add_argument('--target', default='aux', choices=['diffusion', 'aux'],
                        help='aux = cascade stage 1 (FastSpeech2MIDI)')
    parser.add_argument('--n', type=int, default=24,
                        help='number of utterances (use 24 to match the 243.8 figure)')
    parser.add_argument('--set', dest='set_name', default='test',
                        choices=['test', 'valid', 'train'])
    parser.add_argument('--html', action='store_true', help='save an f0 overlay HTML page')
    parser.add_argument('--json', default='', help='path to also save the summary as JSON')
    parser.add_argument('--dump_npz', default='',
                        help='save the raw contours as npz -- input for pansori/eval/contour_tokens.py. '
                             'The tokenizer needs a python 3.10 env and cannot run in this process')
    args, _ = parser.parse_known_args()

    # set_hparams() reads --config / --exp_name off sys.argv itself, same as
    # pansori/warm_start.py and resynth_ladder.py.
    from utils.hparams import set_hparams, hparams
    set_hparams()

    sys.path.insert(0, str(PANSORI_DIR.parent))
    from usr.diffsinger_task import OpencpopDataset

    dataset = OpencpopDataset(args.set_name)
    # Same picking rule as resynth_ladder.py: the held-out set is six whole
    # songs in order, so range(N) would sample one singer and one passage.
    picked = np.linspace(0, len(dataset) - 1,
                         min(args.n, len(dataset))).astype(int).tolist()

    model = device = None
    if args.exp_name:
        import torch
        import utils
        from pansori.model_utils import build_model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print('| ⚠ CUDA_VISIBLE_DEVICES is empty; inferring on CPU (slow).')
        print(f'| loading model ({args.target}) — {args.exp_name}')
        model = build_model(args.target).to(device)
        best = f'checkpoints/{args.exp_name}/model_ckpt_best.pt'
        utils.load_ckpt(model,
                        best if os.path.exists(best) else f'checkpoints/{args.exp_name}',
                        'model', strict=False)
        model.eval()

    print(f'| measuring {len(picked)} of {len(dataset)} in the {args.set_name} set')

    rows, dump = [], {}
    for i in picked:
        sample = dataset.collater([dataset[i]])
        item_name = sample['item_name'][0]
        n_frames = sample['mels'].shape[1]

        crepe = np.load(RAW_F0_DIR / f'{item_name}.npy').astype(np.float64)
        if len(crepe) < n_frames:
            crepe = np.pad(crepe, [[0, n_frames - len(crepe)]], mode='edge')
        crepe = crepe[:n_frames]

        stair, note_id, slur_f = render_staircase(
            sample['pitch_midi'][0].numpy(),
            sample['midi_dur'][0].numpy(),
            sample['is_slur'][0].numpy(),
            sample['mel2ph'][0].numpy(),
            n_frames)

        # The reference ornament: what CREPE does around the notes.
        ref_stats, ref_within = ornament_stats(crepe, stair, note_id)
        midi_stats, _ = ornament_stats(stair, stair, note_id, ref_within)

        row = {
            'item_name': item_name,
            'n_frames': int(n_frames),
            'crepe': dict(ref_stats),
            'midi': dict(f0_metrics.summarize(stair, crepe), **midi_stats),
            'midi_slur': masked_cents_rmse(stair, crepe, slur_f > 0),
            'midi_nonslur': masked_cents_rmse(stair, crepe, slur_f == 0),
        }

        model_f0 = None
        if model is not None:
            import torch
            to_dev = lambda k: (sample[k].to(device)
                                if sample.get(k) is not None else None)
            with torch.no_grad():
                out = model(
                    to_dev('txt_tokens'),
                    spk_embed=(to_dev('spk_ids') if hparams['use_spk_id']
                               else to_dev('spk_embed')),
                    # GT duration on purpose -- identical condition to the
                    # ladder's rung 5 and to the wandb validation audio, so the
                    # comparison isolates pitch rather than duration drift.
                    mel2ph=to_dev('mel2ph'),
                    f0=None, uv=None, energy=to_dev('energy'), ref_mels=None,
                    infer=True, pitch_midi=to_dev('pitch_midi'),
                    midi_dur=to_dev('midi_dur'), is_slur=to_dev('is_slur'))
            if out.get('f0_denorm') is None:
                raise SystemExit(
                    f"{args.exp_name} does not emit f0_denorm "
                    f"(use_pitch_embed is off in this config). "
                    f"Run it with a cascade config (aux.yaml / cascade ds.yaml).")
            model_f0 = out['f0_denorm'][0].cpu().numpy().astype(np.float64)[:n_frames]

            m_stats, _ = ornament_stats(model_f0, stair, note_id, ref_within)
            row['model'] = dict(f0_metrics.summarize(model_f0, crepe), **m_stats)
            row['model_slur'] = masked_cents_rmse(model_f0, crepe, slur_f > 0)
            row['model_nonslur'] = masked_cents_rmse(model_f0, crepe, slur_f == 0)

            # cents_error only compares frames both contours call voiced, and
            # the two disagree about voicing at different rates (the staircase
            # voices wherever a note sits, the model decides for itself).  So
            # the headline RMSEs above are computed over *different* frame
            # sets and are not strictly comparable.  Pin both to the frames
            # all three agree are voiced.
            common = (crepe > 0) & (stair > 0) & (model_f0 > 0)
            row['fair_frames'] = int(common.sum())
            row['fair_midi'] = masked_cents_rmse(stair, crepe, common)
            row['fair_model'] = masked_cents_rmse(model_f0, crepe, common)

        if args.html:
            row['fig'] = f0_figure_b64(item_name, crepe, stair, model_f0, slur_f)

        if args.dump_npz:
            dump[f'{item_name}__crepe'] = crepe
            dump[f'{item_name}__stair'] = stair
            dump[f'{item_name}__note_id'] = note_id
            dump[f'{item_name}__slur'] = slur_f
            if model_f0 is not None:
                dump[f'{item_name}__model'] = model_f0

        rows.append(row)
        msg = f'  {item_name}  staircase {row["midi"]["cents_rmse"]:7.1f} cents'
        if 'model' in row:
            msg += f'   model {row["model"]["cents_rmse"]:7.1f} cents'
        print(msg)

    # ---------------- report ----------------
    have_model = 'model' in rows[0]
    sub = lambda key: [r[key] for r in rows]

    def agg(key):
        return f0_metrics.aggregate(sub(key))

    bar = '=' * 78
    print(f'\n{bar}\n0-2 MIDI baseline — {args.set_name}, {len(rows)} utterances, against CREPE\n{bar}')
    print(f'{"source":<22}{"cents RMSE":>12}{">semitone":>11}{"V/UV":>10}{"octave jump":>13}')
    lines = []
    for key, label in [('midi', 'ROSVOT note staircase'), ('model', f'model f0 ({args.exp_name})')]:
        if key == 'model' and not have_model:
            continue
        a = agg(key)
        lines.append((label, a))
        print(f'{label:<22}{a["cents_rmse"]:>12.1f}{a["gross_error"]*100:>10.1f}%'
              f'{a["vuv_error"]*100:>9.1f}%{a["octave_jump"]*100:>12.2f}%')

    if have_model:
        print(f'\n{bar}\nre-compared on identical frames -- only where all three are voiced (cents RMSE)\n{bar}')
        print(f'{"ROSVOT note staircase":<22}{mean_of(rows, "fair_midi"):>12.1f}')
        print(f'{"model f0":<22}{mean_of(rows, "fair_model"):>12.1f}')
        print(f'{"(mean frames per utterance":<22}{mean_of(rows, "fair_frames"):>11.0f} )')

    print(f'\n{bar}\nsigimsae decomposition -- deviation from the notes, split into bias and movement (cents)\n{bar}')
    print(f'{"source":<22}{"note-bias RMS":>15}{"within-note":>13}{"(interior)":>12}'
          f'{"corr w/ CREPE":>15}')
    orn = [('crepe', 'CREPE (ground truth)'), ('midi', 'ROSVOT note staircase')]
    if have_model:
        orn.append(('model', f'model f0 ({args.exp_name})'))
    for key, label in orn:
        b = mean_of(sub(key), 'bias_rms')
        w = mean_of(sub(key), 'within_std')
        wc = mean_of(sub(key), 'within_std_core')
        c = mean_of(sub(key), 'within_corr') if key != 'crepe' else 1.0
        cs = '   —   ' if np.isnan(c) else f'{c:>12.3f}'
        print(f'{label:<22}{b:>14.1f}{w:>14.1f}{wc:>14.1f}{cs:>14}')

    print(f'\n{bar}\nslur (sigimsae) spans vs the rest -- cents RMSE\n{bar}')
    print(f'{"source":<22}{"slur spans":>12}{"non-slur spans":>16}')
    print(f'{"ROSVOT note staircase":<22}{mean_of(rows, "midi_slur"):>12.1f}'
          f'{mean_of(rows, "midi_nonslur"):>15.1f}')
    if have_model:
        print(f'{"model f0":<22}{mean_of(rows, "model_slur"):>12.1f}'
              f'{mean_of(rows, "model_nonslur"):>15.1f}')

    # ---------------- verdict ----------------
    print(f'\n{bar}\nverdict\n{bar}')
    midi_rmse = agg('midi')['cents_rmse']
    if have_model:
        # The fair (common-frame) numbers decide this, not the headline ones.
        fm, fd = mean_of(rows, 'fair_midi'), mean_of(rows, 'fair_model')
        if fm < fd:
            print(f'▸ On identical frames the staircase ({fm:.1f}) is lower than the model ({fd:.1f}). '
                  f'The predictor is worse than just using the notes it was given.')
            print(f'  -> This raises the priority of switching to predicting the residual '
                  f'above the notes instead of absolute f0.')
        else:
            print(f'▸ On identical frames the model ({fd:.1f}) is {fm - fd:.1f} cents lower '
                  f'than the staircase ({fm:.1f}). '
                  f'The predictor is adding pitch information beyond the notes.')
        # V/UV is the one axis a staircase cannot compete on: it voices
        # wherever a note sits and knows nothing about breath or consonants.
        mv, dv = agg('midi')['vuv_error'] * 100, agg('model')['vuv_error'] * 100
        print(f'▸ V/UV error is model {dv:.1f}% vs staircase {mv:.1f}% -- '
              f'voiced/unvoiced decisions are something the notes cannot imitate.')
        gt_w = mean_of(sub('crepe'), 'within_std')
        md_w = mean_of(sub('model'), 'within_std')
        corr = mean_of(sub('model'), 'within_corr')
        ratio = md_w / gt_w if gt_w else float('nan')
        print(f'▸ Within-note variation: ground truth {gt_w:.1f} cents vs model {md_w:.1f} cents '
              f'({ratio*100:.0f}%), correlation {corr:.3f}')
        if ratio < 0.35:
            print('  -> The model sings the notes nearly flat. It has not learned sigimsae.')
        elif corr < 0.2:
            print('  -> The magnitude of the movement is similar but unrelated to ground truth. '
                  'Closer to noise than to sigimsae.')
        else:
            print('  -> Both magnitude and timing are somewhat right. It is learning sigimsae.')
    else:
        print(f'▸ The ROSVOT note staircase alone gives {midi_rmse:.1f} cents.')
        print(f'  Pass --exp_name to compare directly against the model on the same frames.')
    print(f'▸ The ground-truth CREPE within-note variation of '
          f'{mean_of(sub("crepe"), "within_std"):.1f} cents — '
          f'is how far pansori actually moves above the notes.')

    if args.html:
        summary = (f'<table><tr><th>source</th><th>cents RMSE</th>'
                   f'<th>note-bias RMS</th><th>within-note</th><th>corr w/ CREPE</th></tr>')
        for key, label in orn:
            a = agg(key) if key != 'crepe' else None
            summary += (
                f'<tr><td>{label}</td>'
                f'<td>{a["cents_rmse"]:.1f}</td>' if a else
                f'<tr><td>{label}</td><td>—</td>')
            summary += (f'<td>{mean_of(sub(key), "bias_rms"):.1f}</td>'
                        f'<td>{mean_of(sub(key), "within_std"):.1f}</td>'
                        f'<td>{"1.000" if key == "crepe" else format(mean_of(sub(key), "within_corr"), ".3f")}</td></tr>')
        summary += '</table>'
        out_html = PANSORI_DIR / 'review' / f'midi_baseline_{args.exp_name or "nomodel"}.html'
        build_html(rows, out_html,
                   f'0-2 MIDI baseline — {args.exp_name or "staircase only"}', summary)
        print(f'\nfigure: {out_html}')

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                   encoding='utf-8')
        print(f'JSON: {args.json}')

    if args.dump_npz:
        # Raw contours at 187.5 Hz, for pansori/eval/contour_tokens.py.  The
        # tokenizer needs Python 3.10+ and this runs under diffsinger's 3.8, so
        # the two cannot share a process -- a file is the handoff.
        Path(args.dump_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump_npz,
                            items=np.array([r['item_name'] for r in rows]), **dump)
        print(f'NPZ:  {args.dump_npz}  ({len(rows)} utterances)')


if __name__ == '__main__':
    main()

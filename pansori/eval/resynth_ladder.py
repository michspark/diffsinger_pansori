#!/usr/bin/env python3
"""Phase 0: find out where the quality actually goes, one rung at a time.

The output we have sounds vaguely like pansori but sings the wrong notes, and
until now there was no way to tell whether that is the acoustic model, the
pitch extractor, or the vocoder. Each rung below removes one more source of
error, so the rung where the sound stops improving is the thing to fix:

  1 orig      the segment wav itself                      -- the real target
  2 gt_crepe  GT mel + CREPE f0     -> vocoder            -- VOCODER CEILING
  3 gt_pe     GT mel + PE(GT mel)   -> vocoder            -- what wandb gt_N is
  4 pred_pe   model mel + PE(model) -> vocoder            -- what we ship today
  5 pred_own  model mel + model f0  -> vocoder            -- cascade only

Rung 2 is the one that decides whether the vocoder is worth retraining: if it
already sounds bad, no acoustic model can rescue it. Rung 3 minus rung 2 is the
damage done by 0102_xiaoma_pe, which was trained on a Chinese pop singer.

Metrics compare the f0 that *drove* each rung against the CREPE reference, so
rung 3 isolates pitch-extractor error and rung 4 shows extractor plus model.

Usage (diffsinger env, from the repo root, PYTHONPATH=.):
  python pansori/eval/resynth_ladder.py \
      --config usr/configs/midi/e2e/pansori/ds.yaml --exp_name pansori_ds2 --n 20
"""

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f0_metrics  # noqa: E402

PANSORI_DIR = Path(__file__).resolve().parent.parent
RAW_F0_DIR = PANSORI_DIR / 'data' / 'full' / 'f0'
RAW_WAV_DIR = PANSORI_DIR / 'data' / 'full' / 'wavs'

RUNGS = [
    ('1_orig', '원본 wav', '진짜 기준'),
    ('2_gt_crepe', 'GT mel + CREPE f0', 'vocoder 상한'),
    ('3_gt_pe', 'GT mel + PE f0', '= wandb gt_N'),
    ('4_pred_pe', '예측 mel + PE f0', '현재 출력'),
    ('5_pred_own', '예측 mel + 모델 f0', 'cascade'),
]


def load_crepe_f0(item_name, n_frames):
    """Stage 8's contour for one item, trimmed/padded to the mel length."""
    f0 = np.load(RAW_F0_DIR / f'{item_name}.npy')
    if len(f0) < n_frames:
        f0 = np.pad(f0, [[0, n_frames - len(f0)]], mode='edge')
    return f0[:n_frames]


def wav_to_b64(wav, sample_rate):
    import soundfile as sf
    buf = io.BytesIO()
    peak = np.max(np.abs(wav)) or 1.0
    sf.write(buf, (wav / peak * 0.95).astype(np.float32), sample_rate, format='WAV')
    return base64.b64encode(buf.getvalue()).decode()


def build_html(rows, out_path, exp_name):
    cells = ''.join(
        f'<th>{label}<br><span class=sub>{note}</span></th>'
        for _, label, note in RUNGS)
    body = []
    for row in rows:
        audio = ''.join(
            f'<td>{f"<audio controls preload=none src=data:audio/wav;base64,{row[key]}></audio>" if row.get(key) else "—"}</td>'
            for key, _, _ in RUNGS)
        body.append(
            f'<tr><td class=item>{row["item_name"]}<br>'
            f'<span class=sub>{row["text"][:40]}</span></td>{audio}</tr>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"""<!doctype html><meta charset=utf-8>
<title>판소리 재합성 사다리 — {exp_name}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#fafafa}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:.5rem;text-align:left;vertical-align:top}}
th{{background:#f0f0f0}} .sub{{color:#777;font-size:.8em;font-weight:400}}
.item{{font-family:monospace;font-size:.85em;white-space:nowrap}}
audio{{width:200px}}
</style>
<h1>재합성 사다리 — {exp_name}</h1>
<p>왼쪽에서 오른쪽으로 갈수록 오차가 하나씩 더해집니다.
<b>2단이 나쁘면 vocoder가 병목</b>이고, 2단과 3단 차이가 크면 PitchExtractor가 원인입니다.</p>
<table><tr><th>발화</th>{cells}</tr>{''.join(body)}</table>
""", encoding='utf-8')


def main():
    import argparse
    from utils.hparams import set_hparams, hparams
    import utils

    # parse_known_args, because set_hparams() reads --config / --exp_name off
    # sys.argv itself (same pattern as pansori/warm_start.py).
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--n', type=int, default=12, help='발화 수')
    parser.add_argument('--target', default='diffusion', choices=['diffusion', 'aux'],
                        help='aux = cascade 1단계 (FastSpeech2MIDI)')
    parser.add_argument('--pred_dur', action='store_true',
                        help='GT duration 대신 예측 duration 사용 (기본은 GT — '
                             'wandb 검증 오디오와 같은 조건이라 음정 문제만 분리해서 볼 수 있다)')
    parser.add_argument('--save_wav', action='store_true',
                        help='HTML 외에 wav 파일로도 저장 (infer_out/ladder_{exp_name}/)')
    args, _ = parser.parse_known_args()
    exp_name, n_items, use_pred_dur = args.exp_name, args.n, args.pred_dur

    set_hparams()

    sys.path.insert(0, str(PANSORI_DIR.parent))
    from pansori.model_utils import build_model
    from usr.diffsinger_task import OpencpopDataset
    from modules.fastspeech.pe import PitchExtractor
    from utils.pitch_utils import denorm_f0
    from vocoders.hifigan import HifiGAN

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_rate = hparams['audio_sample_rate']

    print('| 보코더 로드...')
    vocoder = HifiGAN()

    pe = None
    if hparams.get('pe_ckpt'):
        print('| PitchExtractor 로드...')
        pe = PitchExtractor().to(device)
        utils.load_ckpt(pe, hparams['pe_ckpt'], 'model', strict=True)
        pe.eval()

    print(f'| 모델 로드 ({args.target})...')
    model = build_model(args.target).to(device)
    best = f'checkpoints/{exp_name}/model_ckpt_best.pt'
    utils.load_ckpt(model, best if os.path.exists(best) else f'checkpoints/{exp_name}',
                    'model', strict=False)
    model.eval()

    dataset = OpencpopDataset('test')
    # Spread across the whole test set, not the first N: the held-out set is six
    # whole songs in order, so range(N) would sample one singer and one 대목.
    picked = np.linspace(0, len(dataset) - 1, min(n_items, len(dataset))).astype(int).tolist()
    print(f'| test set {len(dataset)}개 중 {len(picked)}개 사용'
          f' ({"예측" if use_pred_dur else "GT"} duration)')

    wav_dir = Path('infer_out') / f'ladder_{exp_name}'
    rows, metrics = [], {'3_gt_pe': [], '4_pred_pe': [], '5_pred_own': []}
    for i in picked:
        sample = dataset.collater([dataset[i]])
        item_name = sample['item_name'][0]
        to_dev = lambda k: sample[k].to(device) if sample.get(k) is not None else None

        gt_mel = to_dev('mels')
        n_frames = gt_mel.shape[1]
        crepe = load_crepe_f0(item_name, n_frames)

        with torch.no_grad():
            out = model(
                to_dev('txt_tokens'),
                spk_embed=to_dev('spk_ids') if hparams['use_spk_id'] else to_dev('spk_embed'),
                mel2ph=None if use_pred_dur else to_dev('mel2ph'),
                f0=None, uv=None, energy=to_dev('energy'), ref_mels=None, infer=True,
                pitch_midi=to_dev('pitch_midi'), midi_dur=to_dev('midi_dur'),
                is_slur=to_dev('is_slur'))
            pred_mel = out['mel_out']

            pe_gt = pe(gt_mel)['f0_denorm_pred'][0].cpu().numpy() if pe else None
            pe_pred = pe(pred_mel)['f0_denorm_pred'][0].cpu().numpy() if pe else None

        own = out.get('f0_denorm')
        own = own[0].cpu().numpy() if own is not None else None

        gt_np, pred_np = gt_mel[0].cpu().numpy(), pred_mel[0].cpu().numpy()
        row = {'item_name': item_name, 'text': sample['text'][0]}

        import soundfile as sf
        wavs = {}
        orig_wav = RAW_WAV_DIR / f'{item_name}.wav'
        if orig_wav.exists():
            wavs['1_orig'] = sf.read(str(orig_wav))[0]
        wavs['2_gt_crepe'] = vocoder.spec2wav(gt_np, f0=crepe)
        if pe_gt is not None:
            wavs['3_gt_pe'] = vocoder.spec2wav(gt_np, f0=pe_gt)
            wavs['4_pred_pe'] = vocoder.spec2wav(pred_np, f0=pe_pred)
            metrics['3_gt_pe'].append(f0_metrics.summarize(pe_gt, crepe))
            metrics['4_pred_pe'].append(f0_metrics.summarize(pe_pred, crepe))
        if own is not None:
            wavs['5_pred_own'] = vocoder.spec2wav(pred_np, f0=own)
            metrics['5_pred_own'].append(f0_metrics.summarize(own, crepe))

        for key, wav in wavs.items():
            row[key] = wav_to_b64(wav, sample_rate)
            if args.save_wav:
                wav_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(wav_dir / f'{item_name}__{key}.wav'),
                         np.asarray(wav, dtype=np.float32), sample_rate)
        row['mel_l1'] = f0_metrics.mel_l1(pred_np, gt_np)
        rows.append(row)
        print(f'  {item_name}  mel_l1={row["mel_l1"]:.4f}')

    out_html = PANSORI_DIR / 'review' / f'ladder_{exp_name}.html'
    build_html(rows, out_html, exp_name)

    print(f'\n{"="*74}\n재합성 사다리 — {exp_name}   (CREPE 기준, {len(rows)}발화)\n{"="*74}')
    print(f'{"단":<14}{"cents RMSE":>12}{"반음이상오차":>14}{"V/UV오차":>11}{"octave jump":>13}')
    for key, label, note in RUNGS:
        agg = f0_metrics.aggregate(metrics.get(key, []))
        if not agg:
            continue
        print(f'{key:<14}{agg["cents_rmse"]:>12.1f}{agg["gross_error"]*100:>13.1f}%'
              f'{agg["vuv_error"]*100:>10.1f}%{agg["octave_jump"]*100:>12.2f}%')
    print(f'\nmel L1 (예측 vs GT)  {np.mean([r["mel_l1"] for r in rows]):.4f}')
    print(f'청취: {out_html}')


if __name__ == '__main__':
    main()

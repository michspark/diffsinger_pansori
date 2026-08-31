#!/usr/bin/env python3
"""Phase 0. Five gates and two null controls, before any number is claimed.

Every gate here can invalidate the comparison, and G1 can invalidate its premise.
If RMVPE is not sample-rate invariant on this material, a 24 kHz system and a
44.1 kHz system cannot be put in one pitch table at all and the design has to
change -- so it is measured rather than assumed.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/compare/smoke.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402
from pansori.compare import f0 as F  # noqa: E402
from pansori.compare import items as I  # noqa: E402
from pansori.compare import notes as N  # noqa: E402
from pansori.compare import pitch_metrics as P  # noqa: E402
from pansori.compare import sigimsae as S  # noqa: E402

FAILED = []


def gate(name, ok, detail):
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {name}: {detail}')
    if not ok:
        FAILED.append(name)
    return ok


# ---------------------------------------------------------------- G1

def g1_sr_invariance(item):
    """RMVPE on the 44.1 kHz original vs the same audio resampled to 24 kHz.

    This is the premise of the whole comparison: the old system's output only
    exists at 24 kHz, so if the extractor's answer depends on sample rate, every
    cross-system pitch number is an artifact of the resampler.
    """
    print('\nG1 SR 무관성')
    import librosa, soundfile as sf
    src = I.wav_of(item)
    f0_a, v_a = F.contour(src)

    with tempfile.TemporaryDirectory() as tmp:
        wav, sr = librosa.load(str(src), sr=None, mono=True)
        down = librosa.resample(wav, orig_sr=sr, target_sr=24000)
        p24 = Path(tmp) / f'{item}_24k.wav'
        sf.write(p24, down, 24000)
        f0_b, v_b = F.contour(p24, use_cache=False)

    m = P.evaluate(f0_b, v_b, f0_a, v_a)
    s_a = S.note_stats(f0_a, v_a, item)
    n = min(len(f0_b), len(f0_a))
    s_b = S.note_stats(f0_b[:n], v_b[:n], item, n)
    ratio = s_b['within_std'] / s_a['within_std']

    gate('cents RMSE < 5', m['f0_rmse_cent'] < 5, f'{m["f0_rmse_cent"]:.2f} cents')
    gate('유무성 일치 > 0.99', m['voiced_agree'] > 0.99, f'{m["voiced_agree"]:.4f}')
    gate('노트내 std 비 ∈ [0.97, 1.03]', 0.97 <= ratio <= 1.03,
         f'{ratio:.4f}  (44.1k {s_a["within_std"]:.1f} vs 24k {s_b["within_std"]:.1f} cents)')


# ---------------------------------------------------------------- G2

def g2_note_grid(fair):
    """cumsum(note_dur) must reproduce the old repo's absolute note timeline.

    Both models were conditioned on the same ROSVOT notes. If that is true, one
    staircase serves both sides; if it is not, the 시김새 decomposition is
    measuring two different things and calling them one.
    """
    print('\nG2 노트 격자 동일성 (구 notes.jsonl 대조)')
    path = C.OLD_REPO / 'pansori' / 'data' / 'full' / 'notes.jsonl'
    if not path.exists():
        gate('구 notes.jsonl 존재', False, str(path))
        return
    want = set(fair)
    old = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d['item_name'] in want:
                old[d['item_name']] = d['notes']

    worst_ms, mismatched, checked = 0.0, [], 0
    for item in fair:
        if item not in old:
            mismatched.append(f'{item}(구 라벨 없음)')
            continue
        onsets, midi = N.note_grid(item)
        sounding = [(int(m), float(a), float(b)) for m, a, b in old[item]]
        ours = [(int(midi[i]), float(onsets[i]), float(onsets[i + 1]))
                for i in range(len(midi)) if midi[i] > 0]
        checked += 1
        if len(ours) != len(sounding):
            mismatched.append(f'{item}({len(ours)}!={len(sounding)}노트)')
            continue
        for (m1, a1, b1), (m2, a2, b2) in zip(ours, sounding):
            if m1 != m2:
                mismatched.append(f'{item}(MIDI {m1}!={m2})')
                break
            worst_ms = max(worst_ms, abs(a1 - a2) * 1000, abs(b1 - b2) * 1000)

    gate(f'{checked}/{len(fair)}항목 MIDI·노트수 일치', not mismatched,
         '전부 일치' if not mismatched else ', '.join(mismatched[:4]))
    gate('온셋 차이 < 1 ms', worst_ms < 1.0, f'최대 {worst_ms:.3f} ms')


# ---------------------------------------------------------------- G3

def g3_rung_semantics(fair):
    """Old `5_pred_own` should be the original's length -- the observable
    signature of ground-truth durations, which is what makes it pairable with
    the new `3_pitch` rung."""
    print('\nG3 단 의미 (구 5_pred_own 이 GT 길이를 썼는가)')
    d = C.OLD_LADDER[C.OLD_PRIMARY]
    worst, worst_item = 0, None
    for item in fair:
        a = d / f'{item}__1_orig.wav'
        b = d / f'{item}__5_pred_own.wav'
        if not (a.exists() and b.exists()):
            gate('사다리 파일 존재', False, f'{item} 누락')
            return
        import soundfile as sf
        fa = sf.info(a).frames / sf.info(a).samplerate
        fb = sf.info(b).frames / sf.info(b).samplerate
        gap = abs(fa - fb) / C.TIMESTEP
        if gap > worst:
            worst, worst_item = gap, item
    gate('길이 차 ≤ 3프레임', worst <= 3, f'최대 {worst:.1f}프레임 ({worst_item})')


# ---------------------------------------------------------------- G4

def g4_exclusion(excluded):
    print('\nG4 학습 데이터 제외')
    ok = True
    for item in excluded:
        try:
            I.assert_held_out(item)
            ok = False
        except I.NotHeldOut:
            pass
    gate(f'{len(excluded)}개 전부 예외 발생', ok and len(excluded) == 5,
         f'제외 {len(excluded)}개: {", ".join(excluded)}')


# ---------------------------------------------------------------- null controls

def null_controls(item):
    """Sign and indexing bugs mostly die here."""
    print('\n널 대조')
    f0_r, v_r = F.contour(I.wav_of(item))

    m = S.evaluate(f0_r, v_r, f0_r, v_r, item)
    p = P.evaluate(f0_r, v_r, f0_r, v_r)
    gate('sys:=ref → within_std_ratio 1.000', abs(m['within_std_ratio'] - 1) < 1e-9,
         f'{m["within_std_ratio"]:.6f}')
    gate('sys:=ref → within_corr 1.000', abs(m['within_corr'] - 1) < 1e-9,
         f'{m["within_corr"]:.6f}')
    gate('sys:=ref → 전이 방향 일치 1.000', abs(m['transition_dir_agree'] - 1) < 1e-9,
         f'{m["transition_dir_agree"]:.6f} (경계 {m["transition_n"]}개)')
    gate('sys:=ref → RPA-50 == 유성비율', abs(p['rpa50'] - 1.0) < 1e-9,
         f'{p["rpa50"]:.6f}')
    gate('sys:=ref → f0 RMSE 0', p['f0_rmse_cent'] < 1e-9, f'{p["f0_rmse_cent"]:.2e}')

    note_id, stair = N.frame_notes(item, len(f0_r))
    v_s = stair > 0
    s = S.note_stats(stair, v_s, item)
    gate('sys:=계단 → 노트내 std 0', s['within_std'] < 1e-6, f'{s["within_std"]:.2e}')
    gate('sys:=계단 → 노트편향 RMS 0', s['bias_rms'] < 1e-6, f'{s["bias_rms"]:.2e}')
    gate('sys:=계단 → 모션 0', S.motion(stair, v_s) < 1e-6, f'{S.motion(stair, v_s):.2e}')

    x = P.cross_check(f0_r, v_r, f0_r, v_r)
    if x:
        gate('mir_eval 교차확인 RPA 1.000', abs(x['mir_rpa'] - 1) < 1e-6,
             f'{x["mir_rpa"]:.6f}')


# ---------------------------------------------------------------- reference

def reference_preview(item):
    """What the reference contour itself looks like -- the row the whole 시김새
    table is measured against."""
    print('\n기준 미리보기 (GT, RMVPE)')
    f0_r, v_r = F.contour(I.wav_of(item))
    s = S.note_stats(f0_r, v_r, item)
    print(f'  {item}: 노트 {s["n_notes"]}개, 유성 {s["n_voiced"]}프레임')
    print(f'  노트편향 RMS {s["bias_rms"]:7.1f} cents')
    print(f'  노트내 변동   {s["within_std"]:7.1f} cents   (core {s["within_std_core"]:.1f})')
    print(f'  모션         {S.motion(s["_f0_snapped"], s["_voiced"]):7.1f} cents/s')
    print(f'  wobble IQR   {S.wobble(s["_f0_snapped"], s["_voiced"]):7.1f} cents')
    print(f'  옥타브 스냅   {s["octave_snap_rate"]:7.1%} 프레임 (RMVPE 옥타브 오류 보정)')

    crepe = F.crepe_contour(item, len(f0_r))
    if crepe is not None:
        f0_c, v_c = crepe
        sc = S.note_stats(f0_c, v_c, item)
        m = P.evaluate(f0_c, v_c, f0_r, v_r)
        print(f'  [기준 교체] CREPE vs RMVPE: cents RMSE {m["f0_rmse_cent"]:.1f}, '
              f'유무성 일치 {m["voiced_agree"]:.3f}')
        print(f'              노트내 변동 CREPE {sc["within_std"]:.1f} vs '
              f'RMVPE {s["within_std"]:.1f} cents '
              f'(비 {sc["within_std"] / s["within_std"]:.3f})')


def main():
    fair, excluded = I.paired_items()
    item = fair[0]
    print(f'스모크 대상: {item} ({I.speaker_of(item)}), 공정 쌍 {len(fair)}개')

    g1_sr_invariance(item)
    g2_note_grid(fair)
    g3_rung_semantics(fair)
    g4_exclusion(excluded)
    null_controls(item)
    reference_preview(item)

    print()
    if FAILED:
        print(f'실패 {len(FAILED)}건: {", ".join(FAILED)}')
        sys.exit(1)
    print('게이트 전부 통과.')


if __name__ == '__main__':
    main()

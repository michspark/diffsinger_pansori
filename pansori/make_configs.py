#!/usr/bin/env python3
"""Render configs/pansori_{acoustic,variance}.yaml from the exported corpus.

The `datasets:` list is one entry per speaker and has to agree with what
08_export.py actually wrote to disk, so it is generated rather than hand-kept.
Everything else is a thin layer over configs/acoustic.yaml and configs/variance.yaml;
only the keys that pansori actually needs to differ are set here.

Usage: python pansori/make_configs.py [--data data/pansori]
"""

import argparse
import csv
from pathlib import Path

# One held-out song per 명창, carried over from the MoonInTheRiver run so the
# test set stays directly comparable with the numbers in docs/PORTING.md.
# split_train_valid_set matches with startswith and item names are
# {uuid8}_{index}, so a bare uuid holds out a whole song.
HOLDOUT = {
    '조상현': '8b4d0824',   # 춘향가 그때의 어사또는
    '정권진': 'a580300d',   # 심청가 뺑파
    '이일주': 'c235c995',   # 심청가 심청 하직
    '박동진': 'e719dced',   # 수궁가
    '박초월': 'd7ec123c',   # 수궁가 토끼 수궁들어가는 대목
    '김소희': 'd207244a',   # 춘향가 이도령 이별
}

# 이 코퍼스의 mel 포락선. openvpi 기본값 spec_max:[0] 은 판소리에 맞지 않는다 —
# 실측 상한이 +2.84 라 정규화 후 diffusion 타깃이 [-1, 1] 밖으로 나간다.
# pansori/compute_spec_stats.py 가 binarize 결과에서 생성한다. 부트스트랩 순서는
# binarize(기본값) → compute_spec_stats.py → make_configs.py → 학습 이고,
# spec_min/max 는 학습 시점에만 쓰이므로 재-binarize 는 필요 없다.
# variance 는 mel 을 만들지 않으므로 acoustic 에만 붙인다.
EXTRA_BASE = {'acoustic': '\n  - configs/pansori_spec_stats.yaml'}

HEADER = """# 판소리 fine-tune — {kind}
# 생성물: python pansori/make_configs.py  (직접 편집하지 말고 생성기를 고칠 것)
#
# 데이터 파이프라인과 근거는 pansori/docs/PORTING.md, pansori/alignment/VERDICT.md 참조.

base_config:
  - configs/{kind}.yaml{extra_base}

dictionaries:
  ko: dictionaries/korean-coda.txt
extra_phonemes: []

# openvpi 는 'SP'(무음) 와 'AP'(들숨) 를 음소셋에 항상 넣고
# (utils/phoneme_utils.py:18), binarize 는 셋의 모든 항목이 코퍼스에 실제로
# 나타나야 통과시킨다. 우리 라벨에는 들숨 구분이 없다 — 06_merge.py 는 노트가
# 덮지 않은 무음을 전부 SP 로 만든다. 그래서 둘을 한 인덱스로 합친다.
# 없는 구분을 지어내는 것보다 라벨이 말하는 대로 두는 쪽이 맞다.
# 나중에 들숨을 따로 라벨하려면 이 줄을 지우고 재-binarize 하면 된다.
merged_phoneme_groups:
  - [SP, AP]
num_lang: 1
use_lang_id: false
"""

COMMON = """
binary_data_dir: {data}/binary_{kind}
binarization_args:
  num_workers: 4

# 화자 {n_spk}명 (명창 + pooled unknown). num_spk 는 최대 spk_id 보다 커야 한다.
use_spk_id: true
num_spk: {num_spk}

# 판소리 음역. 남성 통성은 ROSVOT 노트 기준 MIDI 35 = 61.7 Hz 까지 내려가므로
# 기본값 65 Hz 면 저음이 통째로 무성 처리된다. 측정 상한은 740 Hz.
f0_min: 55
f0_max: 750
pe: rmvpe
pe_ckpt: checkpoints/rmvpe/model.pt

# ---- fine-tune ------------------------------------------------------------
# 공개 사전학습 체크포인트를 받은 뒤 아래 두 줄을 켠다. 사전학습 모델은 중국어
# 음소셋이라 phoneme embedding 이 통째로 어긋나므로 반드시 무시 목록에 넣는다.
# 실제 state_dict 키는 체크포인트마다 다르므로 받은 뒤 대조할 것.
finetune_enabled: false
finetune_ckpt_path: null
finetune_ignored_params:
{ignored}
finetune_strict_shapes: false
"""

ACOUSTIC = """
# 첫 학습은 증강 없이 간다. 판소리는 이미 음역이 넓고, key_shift/speed 임베딩이
# 붙으면 추론 시에도 그 값을 넘겨야 해서 파이프라인이 하나 더 늘어난다.
use_key_shift_embed: false
use_speed_embed: false
augmentation_args:
  random_pitch_shifting:
    enabled: false
  fixed_pitch_shifting:
    enabled: false
  random_time_stretching:
    enabled: false

# spec_min/spec_max 는 위 base_config 의 pansori_spec_stats.yaml 에서 온다.
# 코퍼스를 다시 만들면 compute_spec_stats.py 로 다시 재고 이 스크립트를 다시 돌릴 것.

max_batch_frames: 50000
max_batch_size: 32
pl_trainer_precision: '16-mixed'
"""

VARIANCE = """
predict_dur: true
predict_pitch: true
predict_energy: false
predict_breathiness: false
predict_voicing: false
predict_tension: false
use_glide_embed: false

max_batch_frames: 80000
max_batch_size: 32
pl_trainer_precision: '16-mixed'
"""

IGNORED = {
    'acoustic': ['model.fs2.encoder.embed_tokens', 'model.fs2.txt_embed', 'model.fs2.spk_embed'],
    'variance': ['model.fs2.encoder.embed_tokens', 'model.fs2.txt_embed', 'model.fs2.spk_embed'],
}


def datasets_block(data_root: Path, rel: str) -> tuple:
    speakers = []
    for path in sorted(p for p in data_root.iterdir() if p.is_dir() and (p / 'transcriptions.csv').exists()):
        with open(path / 'transcriptions.csv', encoding='utf-8') as f:
            names = [row['name'] for row in csv.DictReader(f)]
        held = sum(1 for n in names if n.startswith(HOLDOUT.get(path.name, '\0')))
        speakers.append((path.name, len(names), held))
    speakers.sort(key=lambda x: -x[1])          # spk_id 0 = 발화 최다 화자

    lines, dropped = ['', 'datasets:'], []
    for spk_id, (name, count, held) in enumerate(speakers):
        lines += [f'  - raw_data_dir: {rel}/{name}',
                  f'    speaker: {name}',
                  f'    spk_id: {spk_id}',
                  '    language: ko']
        # A holdout that swallows a speaker's entire dataset leaves that speaker
        # with no training items at all: the embedding never gets a gradient, yet
        # validation still runs through it. Better to train the speaker and give
        # up its holdout than to keep a hold-out we cannot learn from.
        if name in HOLDOUT and held < count:
            lines += ['    test_prefixes:', f"      - '{HOLDOUT[name]}'"]
        else:
            if name in HOLDOUT:
                dropped.append((name, count))
            lines += ['    test_prefixes: []']
        lines[-1] += f'      # 발화 {count:,}개'
    return '\n'.join(lines) + '\n', len(speakers), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/pansori')
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    data_root = repo / args.data
    block, n_spk, dropped = datasets_block(data_root, args.data)

    for kind, tail in (('acoustic', ACOUSTIC), ('variance', VARIANCE)):
        ignored = '\n'.join(f'  - {p}' for p in IGNORED[kind])
        text = (HEADER.format(kind=kind, extra_base=EXTRA_BASE.get(kind, '')) + block
                + COMMON.format(data=args.data, kind=kind, n_spk=n_spk,
                                num_spk=n_spk + 1, ignored=ignored)
                + tail)
        out = repo / 'configs' / f'pansori_{kind}.yaml'
        out.write_text(text, encoding='utf-8')
        print(f'{out}  (화자 {n_spk}명)')

    for name, count in dropped:
        print(f'⚠ {name}: 홀드아웃 곡이 전체 {count:,}발화를 모두 차지 → 홀드아웃 해제 '
              f'(학습 데이터 0이면 화자 임베딩이 학습되지 않음)')
    missing = [s for s in HOLDOUT if not (data_root / s).exists()]
    if missing:
        print(f'⚠ 홀드아웃 지정 화자가 데이터에 없음: {missing}')


if __name__ == '__main__':
    main()

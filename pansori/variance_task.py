#!/usr/bin/env python3
"""VarianceTask + 부류별 음소 길이 지표 — upstream 을 한 줄도 고치지 않고.

`training/variance_task.py` 는 검증마다 `rhythm_corr` 과 `ph_dur_acc` 를 낸다.
둘 다 옳은 지표지만 이 코퍼스의 문제를 보여주지는 못한다. 두 가지 때문이다:

1. **범위.** 두 지표의 갱신이 `if data_idx < hparams['num_valid_plots']` 안에 들어
   있다 (`variance_task.py:253,260`). 그 값은 기본 10 이므로 검증셋이 몇 천 발화여도
   **앞 10발화**만 채점된다. 그림을 그리려고 세운 게이트인데 지표까지 같이 걸린 것이다.
2. **해상도.** 하나의 평균이라 초성이 무너지고 모음이 멀쩡한 상황과 그 반대를 구별하지
   못한다. 이 코퍼스에서는 라벨 기준으로 초성 ≤3프레임이 40.9%, 종성이 15.1% 다.

그래서 여기서는 (1) **검증셋 전체**에 대해 duration 만 따로 추론해 재고,
(2) `pansori_lib.phone_class` 의 부류별로 나눠 기록한다. 기존 두 지표는 **그대로 둔다** —
이전 런과의 비교선을 끊지 않기 위해서다. 두 자의 범위가 다르다는 점만 기억하면 된다
(`ph_dur_acc` 10발화 · `dur_acc_*` 전체).

**왜 duration 만 따로 추론하나.** `run_model(sample, infer=True)` 는 pitch 확산
샘플러까지 돌린다. 검증셋 전체에 그걸 돌리면 검증이 학습보다 오래 걸린다. duration 은
5층 conv 하나뿐이라(`DurationPredictor`) 사실상 공짜다. 그래서 `modules/toplevel.py:251-256`
이 하는 것과 같은 `fs2` 호출을 직접 한 번 더 한다.

`word_dur=None` 을 넘기는 것이 중요하다 — `variance_encoder.py:74` 가 그때 GT ph_dur
에서 단어 길이를 만들어 준다. 즉 "단어 길이는 악보가 준다, 모델은 그 안의 배분을 푼다"
라는 추론 시 조건과 같아진다. 예측 길이를 `RhythmRegulator` 로 GT 단어 길이에 맞추는
것도 같은 이유이고, upstream `PhonemeDurationAccuracy` 가 하는 처리와 동일하다.

config 는 `pansori/make_configs.py` 가 렌더한다:

    task_cls: pansori.variance_task.PansoriVarianceTask
    num_valid_metric_items: 0      # 0 = 검증셋 전체
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'pansori' / 'preprocess'))

from modules.fastspeech.tts_modules import RhythmRegulator  # noqa: E402
from training.variance_task import VarianceTask  # noqa: E402
from utils.hparams import hparams  # noqa: E402

from pansori.metrics import (  # noqa: E402
    DurationFloorRate,
    PerPhonemeDurationStats,
    PhoneClassDurationAccuracy,
    PhoneClassDurationMAE,
    build_token_class,
    phone_dur_to_figure,
)
from pansori_lib import PHONE_CLASSES  # noqa: E402

CLASS_LABEL = {'onset': '초성', 'coda': '종성', 'vowel': '모음',
               'glide': '활음', 'sp': '무음'}

# MFA 의 최소 음소 길이. 프레임으로 환산해 쓴다 — 근거는 metrics/phone_dur.py 머리말.
MFA_FLOOR_SEC = 0.030
# upstream `ph_dur_acc` 와 같은 값. 두 수치를 나란히 읽을 수 있게 맞춘다.
TOLERANCE = 0.2


class PansoriVarianceTask(VarianceTask):
    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        super().build_losses_and_metrics()
        if not self.predict_dur:
            return

        timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        self.phone_timestep_ms = timestep * 1000
        self.phone_floor_frames = max(1, round(MFA_FLOOR_SEC / timestep))
        self.phone_rr = RhythmRegulator()
        self.phone_metric_items = hparams.get('num_valid_metric_items', 0)

        token_class = build_token_class(self.phoneme_dictionary, PHONE_CLASSES)
        self.phone_token_class = token_class
        self.phone_id_to_name = {
            t: self._token_name(t) for t in range(1, self.phoneme_dictionary.vocab_size)
        }
        self.phone_id_to_class = {
            t: (PHONE_CLASSES[c] if c >= 0 else 'other')
            for t, c in enumerate(token_class.tolist())
        }

        for cls_id, cls in enumerate(PHONE_CLASSES):
            self.register_validation_metric(f'dur_acc_{cls}', PhoneClassDurationAccuracy(
                tolerance=TOLERANCE, token_class=token_class, cls_id=cls_id))
        for cls in ('onset', 'vowel'):
            cls_id = PHONE_CLASSES.index(cls)
            self.register_validation_metric(f'dur_mae_{cls}_ms', PhoneClassDurationMAE(
                token_class=token_class, timestep_ms=self.phone_timestep_ms, cls_id=cls_id))

        self.register_validation_metric('floor_rate_pred', DurationFloorRate(
            floor_frames=self.phone_floor_frames, token_class=token_class, on='pred'))
        self.register_validation_metric('floor_rate_gt', DurationFloorRate(
            floor_frames=self.phone_floor_frames, token_class=token_class, on='target'))
        self.register_validation_metric('floor_rate_pred_onset', DurationFloorRate(
            floor_frames=self.phone_floor_frames, token_class=token_class, on='pred',
            cls_id=PHONE_CLASSES.index('onset')))

        self.register_validation_metric('dur_acc_macro', PerPhonemeDurationStats(
            tolerance=TOLERANCE, vocab_size=self.phoneme_dictionary.vocab_size))

    def _token_name(self, token: int) -> str:
        phone = self.phoneme_dictionary.decode_one(token, scalar=False)
        return phone if isinstance(phone, str) else '/'.join(phone)

    @torch.no_grad()
    def _predict_ph_dur(self, sample):
        """검증 배치 하나에 대한 duration 예측 — pitch 헤드는 건드리지 않는다."""
        model = self.model
        spk_embed = None
        if getattr(model, 'use_spk_id', False):
            spk_embed = model.spk_embed(sample['spk_ids'])[:, None, :]
        languages = sample['languages'] if self.use_lang_id else None

        _, dur_pred = model.fs2(
            sample['tokens'], midi=sample['midi'], ph2word=sample['ph2word'],
            ph_dur=sample['ph_dur'], word_dur=None,
            spk_embed=spk_embed, languages=languages, infer=True
        )
        # run_model(infer=True) 이 하는 것과 같은 반올림. clamp 는 여기서만 건다 --
        # DurationPredictor 의 exp(x) - offset 은 음수가 될 수 있고, 지표가 학습을
        # 죽이는 이유가 되어서는 안 된다.
        return dur_pred.round().long().clamp(min=0)

    def _update_phone_dur_metrics(self, sample):
        tokens = sample['tokens']
        ph_dur = sample['ph_dur']
        ph2word = sample['ph2word']
        mask = tokens != 0

        limit = self.phone_metric_items
        if limit and limit > 0:
            keep = sample['indices'] < limit
            if not keep.any():
                return
            mask = mask & keep[:, None]
            if not mask.any():
                return

        dur_pred = self._predict_ph_dur(sample)
        word_dur = ph_dur.new_zeros(ph_dur.shape[0], ph2word.max() + 1).scatter_add(
            1, ph2word, ph_dur
        )[:, 1:]
        dur_pred = self.phone_rr(dur_pred, ph2word, word_dur)

        args = dict(pred=dur_pred, target=ph_dur, tokens=tokens, mask=mask)
        for cls in PHONE_CLASSES:
            self.valid_metrics[f'dur_acc_{cls}'].update(**args)
        for cls in ('onset', 'vowel'):
            self.valid_metrics[f'dur_mae_{cls}_ms'].update(**args)
        for name in ('floor_rate_pred', 'floor_rate_gt', 'floor_rate_pred_onset'):
            self.valid_metrics[name].update(**args)
        self.valid_metrics['dur_acc_macro'].update(
            **args, floor_frames=self.phone_floor_frames)

    def _validation_step(self, sample, batch_idx):
        outputs = super()._validation_step(sample, batch_idx)
        if self.predict_dur and sample['size'] > 0:
            self._update_phone_dur_metrics(sample)
        return outputs

    def _on_validation_epoch_end(self):
        super()._on_validation_epoch_end()
        if not self.predict_dur:
            return
        stats = self.valid_metrics['dur_acc_macro']
        self.logger.all_rank_experiment.add_figure(
            'phone_dur_acc',
            phone_dur_to_figure(
                stats, self.phone_id_to_name, self.phone_id_to_class,
                CLASS_LABEL, self.phone_floor_frames,
                title=f'step {self.global_step:,}'      # 제목의 나머지는 그림 쪽이
            ),                                          # 폰트 가용성을 보고 정한다
            self.global_step,
        )

#!/usr/bin/env python3
"""VarianceTask + per-phone-class duration metrics — without editing one line of upstream.

`training/variance_task.py` emits `rhythm_corr` and `ph_dur_acc` at every
validation. Both are correct metrics, but neither shows this corpus's problem,
for two reasons:

1. **Scope.** Both updates sit inside `if data_idx < hparams['num_valid_plots']`
   (`variance_task.py:253,260`). That value defaults to 10, so however many
   thousand utterances the validation set holds, only **the first 10** are
   scored. It is a gate put up for plotting, and the metrics got caught in it.
2. **Resolution.** Being one average, it cannot tell "onsets collapsed, vowels
   fine" from the reverse. In this corpus, by the labels, 40.9% of onsets are
   <=3 frames against 15.1% of codas.

So this file (1) infers duration on its own over **the whole validation set**,
and (2) records it split by `pansori_lib.phone_class`'s classes. The existing two
metrics are **left alone** -- so as not to break the comparison line with earlier
runs. Just remember the two rulers have different scope (`ph_dur_acc` 10
utterances; `dur_acc_*` all of them).

**Why infer duration separately.** `run_model(sample, infer=True)` also runs the
pitch diffusion sampler. Run that over the whole validation set and validation
takes longer than training. Duration is one 5-layer conv (`DurationPredictor`)
and effectively free. So this makes the same `fs2` call
`modules/toplevel.py:251-256` makes, one more time, directly.

Passing `word_dur=None` matters -- `variance_encoder.py:74` then builds the word
durations from the GT ph_dur. Which makes it the same condition as at inference:
"the score gives the word durations, the model solves the division within them".
Fitting the predicted durations to the GT word durations with `RhythmRegulator`
is for the same reason, and is identical to what upstream
`PhonemeDurationAccuracy` does.

The config is rendered by `pansori/make_configs.py`:

    task_cls: pansori.variance_task.PansoriVarianceTask
    num_valid_metric_items: 0      # 0 = the whole validation set
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
               'glide': '활음', 'sp': '무음'}   # figure labels; romanised when no CJK font

# MFA's minimum phoneme duration, converted to frames -- the reasoning is in the header
# of metrics/phone_dur.py.
MFA_FLOOR_SEC = 0.030
# The same value as upstream's `ph_dur_acc`, so the two numbers can be read side by side.
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
        """Duration prediction for one validation batch -- the pitch head is untouched."""
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
        # The same rounding run_model(infer=True) does. The clamp is only applied here --
        # DurationPredictor's exp(x) - offset can go negative, and a metric must never
        # be the reason training dies.
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
                title=f'step {self.global_step:,}'      # the rest of the title is decided
            ),                                          # by the figure, from font availability
            self.global_step,
        )

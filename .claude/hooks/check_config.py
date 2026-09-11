#!/usr/bin/env python3
"""PostToolUse(Edit|Write|MultiEdit) -- load an edited config the way training will.

A broken config does not fail where you wrote it. It fails inside a binarize worker
forty minutes in, or -- worse -- it does not fail at all and you find out from the
loss curve. So this runs the real thing: upstream's own set_hparams(), resolving the
whole base_config chain, in the env that will actually train.

It costs 2-3 s because it imports lightning. That is the point: a YAML-only check
would catch a missing colon but not a dangling base_config link, a key that resolved
to the wrong type, or a KeyError raised while merging.

Then two checks that are about *this corpus* rather than about YAML:
  * an experiment config must state binary_data_dir -- which binarized corpus a run
    reads is the single most important fact about it, so it belongs in the file
    rather than inherited from somewhere out of sight;
  * f0_min/f0_max must resolve to 55/750. The openvpi default of 65 Hz cuts off male
    통성 entirely, and nothing downstream complains -- it just trains on silence
    where the low register should be.

Never blocks on things it cannot judge: no interpreter, no repo, a config outside
configs/ -- all exit 0.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import deny, edited_paths, payload, project_dir, rel_path, run  # noqa: E402

# The one env matching requirements.txt: py3.10, torch 2.6.0+cu124, lightning 2.3.3.
# PYTHONNOUSERSITE is not optional -- a stale tensorflow in ~/.local shadows the env
# and kills the tensorboard import inside protobuf.
DSP_PYTHON = Path(os.environ.get(
    'PANSORI_DSP_PYTHON', Path.home() / 'miniconda3/envs/dsp/bin/python'))

PROBE = r'''
import json, sys
from utils.hparams import set_hparams
h = set_hparams(config=sys.argv[1], global_hparams=False, print_hparams=False)
print("---HPARAMS---" + json.dumps({
    "n": len(h),
    "binary_data_dir": h.get("binary_data_dir"),
    "f0_min": h.get("f0_min"), "f0_max": h.get("f0_max"),
    "task_cls": h.get("task_cls"), "base": h.get("base_config"),
}))
'''


def probe(root, rel):
    env = {**os.environ, 'PYTHONPATH': str(root), 'PYTHONNOUSERSITE': '1'}
    return subprocess.run([str(DSP_PYTHON), '-c', PROBE, rel], cwd=root, env=env,
                          capture_output=True, text=True, timeout=180)


def main():
    root = project_dir()
    if not DSP_PYTHON.exists():
        return  # no interpreter to check with -- stay out of the way

    for raw in edited_paths(payload()):
        rel = rel_path(raw, root)
        if rel is None or not rel.startswith('configs/') or not rel.endswith(('.yaml', '.yml')):
            continue
        if not (root / rel).exists():
            continue

        try:
            proc = probe(root, rel)
        except (subprocess.TimeoutExpired, OSError):
            return

        if proc.returncode != 0:
            tail = '\n'.join(
                ln for ln in proc.stderr.strip().splitlines()
                if 'pkg_resources is deprecated' not in ln)[-2500:]
            deny(f'{rel} does not load.\n\n{tail}\n\n'
                 'Training and binarize resolve configs with this same set_hparams() call,\n'
                 'so this fails there too. Fix it before running anything.')

        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith('---HPARAMS---')), None)
        if not line:
            continue
        h = json.loads(line[len('---HPARAMS---'):])

        notes = []
        declares = re.search(r'^binary_data_dir\s*:', (root / rel).read_text(encoding='utf-8'),
                             re.MULTILINE)
        if rel.startswith('configs/exp/') and not declares:
            notes.append(
                f'  * binary_data_dir is not declared here, so it inherits '
                f'{h["binary_data_dir"]!r}.\n'
                '    It is the one key that depends on both\n'
                '    the kind and the corpus axis, so the experiment file states it:\n'
                '        binary_data_dir: data/pansori/binary_acoustic')
        if (h['f0_min'], h['f0_max']) != (55, 750):
            notes.append(
                f"  * f0 range resolves to {h['f0_min']}-{h['f0_max']}, expected 55-750.\n"
                '    ROSVOT measured male 통성 down to MIDI 35 (61.7 Hz); openvpi\'s default\n'
                '    65 Hz makes the whole low register unvoiced, silently.')
        if notes:
            deny(f'{rel} loads ({h["n"]} keys), but:\n\n' + '\n'.join(notes))

        print(f'[config] {rel} loads: {h["n"]} keys, '
              f'f0 {h["f0_min"]}-{h["f0_max"]}, data {h["binary_data_dir"]}',
              file=sys.stderr)


run(main)

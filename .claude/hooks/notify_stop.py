#!/usr/bin/env python3
"""Stop -- make the end of a turn audible, and keep finished work from sitting loose.

Two jobs, and the second one is the one with teeth.

**Notify.** Training runs here take hours and the terminal is usually not the window
in front of you, so a finished turn rings a bell and raises a desktop notification.

**Remind to commit.** CLAUDE.md asks for a commit per completed unit of work.
Pushing is a separate, asked-for step, but committing is not -- uncommitted work in
this tree is work that a later `--reset` or a branch switch can quietly cost. So if
the tree is dirty when a turn ends, this blocks *once* and says so.

Once is the whole design. A Stop hook that blocks on a condition it cannot itself
resolve is an infinite loop, so two independent brakes are used: the payload's own
`stop_hook_active` flag, and a fingerprint of `git status` recorded on disk. Come back
with the same dirty tree and it stays quiet; change something and it speaks again.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import payload, project_dir, run  # noqa: E402

SOUND = '/usr/share/sounds/freedesktop/stereo/complete.oga'


def fire_and_forget(argv):
    try:
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError:
        pass


def notify(title, body):
    if shutil.which('notify-send'):
        fire_and_forget(['notify-send', '-a', 'Claude Code', '-u', 'low', title, body])
    if shutil.which('paplay') and Path(SOUND).exists():
        fire_and_forget(['paplay', SOUND])


def git(root, *args):
    try:
        proc = subprocess.run(('git', *args), cwd=root, capture_output=True,
                              text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def main():
    data = payload()
    root = project_dir()

    status = git(root, 'status', '--porcelain')
    dirty = [ln for ln in (status or '').splitlines() if ln.strip()]
    branch = (git(root, 'branch', '--show-current') or '?').strip()

    notify('Claude Code: turn finished',
           f'{root.name} ({branch}) -- '
           + (f'{len(dirty)} uncommitted file(s)' if dirty else 'tree clean'))

    if not dirty or data.get('stop_hook_active'):
        return

    # Second brake: speak once per distinct dirty state.
    session = str(data.get('session_id') or 'nosession').replace('/', '_')[:64]
    fingerprint = hashlib.sha256(''.join(sorted(dirty)).encode()).hexdigest()[:16]
    marker = Path(tempfile.gettempdir()) / f'pansori-stop-{session}.state'
    try:
        if marker.read_text().strip() == fingerprint:
            return
    except OSError:
        pass
    try:
        marker.write_text(fingerprint)
    except OSError:
        return  # cannot record that we spoke -> do not speak, or we would loop

    listing = '\n'.join(f'  {ln}' for ln in dirty[:20])
    more = f'\n  ... and {len(dirty) - 20} more' if len(dirty) > 20 else ''
    print(f'{len(dirty)} uncommitted file(s) on {branch}:\n{listing}{more}\n\n'
          'CLAUDE.md asks for a commit per completed unit of work. Either commit this\n'
          '(capitalized imperative subject stating intent, no prefix, no PR number), or\n'
          'say why it should stay uncommitted. Do not push -- pushing is asked about '
          'separately.\n\nThis reminder fires once per distinct dirty state.',
          file=sys.stderr)
    sys.exit(2)


run(main)

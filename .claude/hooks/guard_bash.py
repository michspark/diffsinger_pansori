#!/usr/bin/env python3
"""PreToolUse(Bash) -- refuse the commands whose damage cannot be undone.

Three kinds of damage are covered, and they are different kinds:

  * **Irrecoverable within the repo** -- history rewrites, hard resets, and the
    stash stack (a bare `git stash pop` drops the entry as it applies, so a conflict
    can leave the work in neither the tree nor the stack).
  * **Irrecoverable outside it** -- pushing to `upstream`, which is MoonInTheRiver,
    not ours. Ours is `pansori` (michspark/diffsinger_pansori), our only other
    remote. CLAUDE.md asks before any push.
  * **Irreproducible** -- `01_segment.py`. It completes happily and looks fine;
    what it destroys is the correspondence between every existing phones.jsonl
    and the audio, because librosa.effects.split lands 3-11 ms differently in a
    different env. That is the worst kind of destructive command: a silent one.

Matching is deliberately coarse -- it reads the whole command string rather than
parsing shell grammar, so a pipeline or a `&&` chain cannot slip a banned command
past it. False positives are cheap here (say so and ask); false negatives are not.

One exception to the coarseness, learned the hard way on this file's own first
commit: heredoc *bodies* are stripped before matching. A commit message that
describes a banned command is not that command, and a guard that cannot tell the
difference makes it impossible to write about what it guards.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import deny, payload, run  # noqa: E402

# (pattern, explanation). Checked in order against the whole command string.
RULES = [
    (r'\bgit\s+push\b(?=.*\bupstream\b)',
     'Refused: `upstream` is MoonInTheRiver/DiffSinger -- not yours.\n'
     'Yours is `pansori` (github.com/michspark/diffsinger_pansori).\n'
     'Ask before pushing there, per CLAUDE.md.'),

    (r'\bgit\s+push\b(?=.*(--force(?!-with-lease)|\s-f\b))',
     'Refused: force-push rewrites published history.\n'
     'If a force-push is genuinely needed, say what it would overwrite and ask first.'),

    (r'\bgit\s+reset\s+--hard\b',
     'Refused: `git reset --hard` discards working-tree changes with no undo.\n'
     'To set work aside use a WIP commit (`git commit -m wip`), which is recoverable.\n'
     'To inspect a past state use `git show <rev>:<path>` or a worktree.'),

    (r'\bgit\s+clean\s+-[a-z]*[fd]',
     'Refused: `git clean` deletes untracked files, and in this repo untracked means\n'
     'binarized corpora, checkpoints and eval output -- hours of compute.\n'
     'Delete specific paths by name instead, after saying which.'),

    (r'\bgit\s+stash\b(?!\s+(list|show|push\s+.*-m))',
     'Refused: a bare stash/pop drops the entry as it applies, so a conflict can leave\n'
     'the work in neither the tree nor the stack.\n'
     'Use a WIP commit instead, or `git stash push -u -m "<unique-tag>"` and `apply` by SHA.'),

    (r'\bgit\s+(merge|remote\s+add)\b(?=.*[Mm]oon)',
     'Refused: this repo and DiffSinger_moonInRiver have no common ancestor.\n'
     'The old work was *ported*, not merged -- see pansori/docs/PORTING.md.'),

    (r'(python[0-9.]*|bash|sh|\./)\s*\S*\b01_segment\.py\b',
     'Refused: 01_segment.py does not reproduce.\n'
     'Run in a different env, librosa.effects.split moves segment boundaries by 3-11 ms,\n'
     'which silently invalidates every phones.jsonl already aligned against the old cuts.\n'
     'To restore the wav files without re-segmenting:\n'
     '    python pansori/preprocess/01b_recut_wavs.py --set full\n'
     'If you truly mean to re-segment the corpus from scratch, say so and ask first.'),

    (r'\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rR][a-zA-Z]*f|\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*f[a-zA-Z]*[rR]',
     None),  # handled below -- only refused when aimed at an expensive tree
]

# `rm -rf` is only refused when it points at something expensive to rebuild.
EXPENSIVE = re.compile(r'(^|[\s\'"=/])(checkpoints|data|binary_(acoustic|variance)'
                       r'|pansori/data|infer_out)(/|\s|$|\'|")')

RM_MESSAGE = """\
Refused: `rm -rf` aimed at {tree}.

That tree holds binarized corpora, checkpoints, or exported audio -- tens of GB and
hours of GPU time, none of it in git. checkpoints/best/ in particular holds the only
surviving copies of the best steps, which num_ckpt_keep already deleted from the run
directories.

Name the exact files and ask before deleting anything under there."""


HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1.*?^\2$", re.DOTALL | re.MULTILINE)


def strip_heredocs(cmd):
    """Drop heredoc bodies -- they are data being written, not commands being run."""
    return HEREDOC.sub('<<HEREDOC', cmd)


def main():
    cmd = strip_heredocs((payload().get('tool_input') or {}).get('command') or '')
    if not cmd.strip():
        return

    for pattern, message in RULES:
        if not re.search(pattern, cmd):
            continue
        if message is None:  # the rm rule
            hit = EXPENSIVE.search(cmd)
            if hit:
                deny(RM_MESSAGE.format(tree=hit.group(2)))
            continue
        deny(message)


run(main)

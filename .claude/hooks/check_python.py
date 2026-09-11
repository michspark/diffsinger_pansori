#!/usr/bin/env python3
"""PostToolUse(Edit|Write|MultiEdit) -- catch broken Python before it costs a run.

Scope is narrow on purpose, in two directions.

**Only files under pansori/.** Upstream files are not ours to lint, and this fork
does not modify them anyway -- guard_upstream.py refuses those edits before they
happen.

**Only rules that mean something is actually wrong.** Ruff's default selection
reports 61 findings on the existing pansori/ tree -- ambiguous names, one-line ifs,
f-strings without placeholders -- all of it deliberate, none of it a defect. A check
that fires on code its author wrote on purpose gets ignored, and then it is not a
check. So the selection below is restricted to undefined names, redefinitions,
`is` against a literal, and malformed string operations: rules that currently have
**zero** hits across the whole tree, so anything they report was introduced just now.

py_compile runs first and is the real backstop -- a syntax error is not a matter of
taste. Nothing is reformatted; this repo has no formatter config and imposing one
would be a change nobody asked for.
"""

import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import deny, edited_paths, payload, project_dir, rel_path, run  # noqa: E402

# Verified zero-hit on the existing tree, so every report is a real regression.
RUFF_RULES = 'F821,F811,F632,F702,F502,F522,F524,E9'


def compile_error(path):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            py_compile.compile(str(path), cfile=os.path.join(tmp, 'x.pyc'),
                               doraise=True, quiet=1)
        except py_compile.PyCompileError as exc:
            return str(exc)
    return None


def ruff_report(root, rels):
    ruff = shutil.which('ruff') or str(Path.home() / '.local/bin/ruff')
    if not Path(ruff).exists():
        return None
    try:
        proc = subprocess.run(
            [ruff, 'check', '--select', RUFF_RULES, '--no-cache',
             '--output-format', 'concise', *rels],
            cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode != 0 else None


def main():
    root = project_dir()
    targets = []
    for raw in edited_paths(payload()):
        rel = rel_path(raw, root)
        if rel and rel.startswith('pansori/') and rel.endswith('.py') and (root / rel).exists():
            targets.append(rel)
    if not targets:
        return

    for rel in targets:
        err = compile_error(root / rel)
        if err:
            deny(f'{rel} does not compile.\n\n{err}')

    report = ruff_report(root, targets)
    if report:
        deny(f'{report}\n\n'
             'These rules have no other hits in pansori/, so this was introduced by '
             'the edit just made.')


run(main)

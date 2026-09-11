"""Shared plumbing for this project's hooks.

Every hook here follows the same contract, and it is worth stating once:

  * stdin carries the event JSON; stdout is ignored unless we print structured
    output; **exit 2 with a message on stderr is what blocks and tells Claude why**.
  * A hook that cannot answer stays silent and exits 0. A hook is a guardrail,
    not an oracle -- a missing interpreter, a file outside the repo, or an
    unparseable payload must never turn into a spurious block.
  * Nothing here raises. `main()` bodies are wrapped by `run()` below, so a bug
    in a guard degrades to "no guard", never to "no edits possible".
"""

import json
import os
import sys
from pathlib import Path

# Artefact trees. Writes here are always refused: they are gigabytes of generated
# data and checkpoints, they are gitignored, and an agent has no business editing
# them in place. `pansori/data/` is listed explicitly because it sits *inside* the
# fork-owned tree and would otherwise be allowed.
ARTEFACT_DIRS = ('checkpoints', 'data', 'pansori/data', 'pansori/eval/out',
                 'pansori/review', 'infer_out')

# The fork's own code. Everything else in the repo belongs to openvpi upstream.
# Kept in sync with pansori/README.md, which is where this rule is stated.
FORK_OWNED = ('pansori/', 'configs/pansori/', 'configs/exp/', '.claude/')
FORK_OWNED_FILES = ('dictionaries/korean-coda.txt', 'CLAUDE.md')

# configs/pansori/ is fork-owned but *generated*; editing it by hand is a
# different mistake from editing upstream, and gets a different message.
GENERATED_DIRS = ('configs/pansori/',)


def payload():
    """The hook event, or {} if stdin held nothing we can read."""
    try:
        return json.loads(sys.stdin.read() or '{}')
    except (json.JSONDecodeError, ValueError):
        return {}


def project_dir():
    return Path(os.environ.get('CLAUDE_PROJECT_DIR') or os.getcwd()).resolve()


def rel_path(raw, root):
    """Path relative to the repo root, or None if it falls outside the repo.

    Outside-the-repo paths (the scratchpad, /tmp, another checkout) are not ours
    to police, so callers treat None as "allow".
    """
    if not raw:
        return None
    try:
        resolved = Path(raw).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
        return resolved.resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return None


def edited_paths(data):
    """Every path an Edit/Write/MultiEdit call is about to touch."""
    ti = data.get('tool_input') or {}
    out = [ti.get('file_path') or ti.get('notebook_path')]
    for edit in ti.get('edits') or ():
        if isinstance(edit, dict):
            out.append(edit.get('file_path'))
    return [p for p in out if p]


def under(rel, prefixes):
    return any(rel == p.rstrip('/') or rel.startswith(p.rstrip('/') + '/')
               for p in prefixes)


def deny(message):
    """Refuse the call and tell Claude why. Exit 2 is the blocking code."""
    print(message, file=sys.stderr)
    sys.exit(2)


def run(main):
    """Never let a broken guard block the session."""
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 -- a guard must not have teeth here
        print(f'[hook] skipped: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(0)
    sys.exit(0)

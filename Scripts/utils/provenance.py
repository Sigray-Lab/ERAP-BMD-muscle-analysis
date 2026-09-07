"""
Provenance record for every output-producing step.

Round-2 review finding R2-F01: recording only `git rev-parse HEAD` does not
identify the executed source when the working tree is dirty. This helper
records HEAD, whether Scripts/ differs from HEAD, and a content hash of every
*.py under Scripts/ so the executed implementation can always be identified.
"""

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_DIR.parent


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_dirty_scripts():
    """True if tracked files under Scripts/ differ from HEAD (staged or unstaged); None if git unavailable."""
    out = _git("status", "--porcelain", "--untracked-files=all", "--", "Scripts")
    if out == "unknown":
        return None
    return bool(out.strip())


def scripts_tree_sha256() -> str:
    """SHA-256 over (relative path, content) of every *.py under Scripts/, sorted."""
    h = hashlib.sha256()
    for p in sorted(SCRIPTS_DIR.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(SCRIPTS_DIR)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def provenance() -> dict:
    return {
        "git_commit": git_commit(),
        "git_dirty_scripts": git_dirty_scripts(),
        "scripts_tree_sha256": scripts_tree_sha256(),
        "recorded_at": datetime.now().isoformat(),
        "note": ("git_commit is HEAD at run time. If git_dirty_scripts is true the executed Scripts/ "
                 "differ from that commit; scripts_tree_sha256 identifies the executed source in either case."),
    }

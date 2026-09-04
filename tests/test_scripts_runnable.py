"""Every script must run without PYTHONPATH being set.

This gap is why a ModuleNotFoundError reached a user: local runs and CI both
happened to set PYTHONPATH, while the Kaggle notebook invokes the scripts as
`python scripts/foo.py`, which puts scripts/ on sys.path rather than the repo
root. These tests strip PYTHONPATH and run each script the way a user actually
does, from a working directory that is deliberately not the repo root.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(p.name for p in (REPO / "scripts").glob("*.py") if not p.name.startswith("_"))


def _env_without_pythonpath():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs_without_pythonpath(script, tmp_path):
    """--help exercises every module-level import without doing real work."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--help"],
        capture_output=True, text=True, env=_env_without_pythonpath(),
        cwd=tmp_path,  # not the repo root, so nothing resolves by accident
    )
    assert r.returncode == 0, f"{script} failed:\n{r.stderr}"
    assert "ModuleNotFoundError" not in r.stderr


def test_audit_runs_end_to_end_without_pythonpath(tmp_path):
    """The exact invocation the Kaggle notebook uses."""
    out = tmp_path / "audit.json"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "text_blind_audit.py"),
         "--roots", str(REPO / "tests" / "fixtures" / "flat"), "--out", str(out)],
        capture_output=True, text=True, env=_env_without_pythonpath(), cwd=tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert "discovered" in r.stdout
    assert out.exists()

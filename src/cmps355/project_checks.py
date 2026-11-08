"""Read-only setup and environment checks for a clean CMPS 355 project.

This module is independently written for the fresh project. It imports no earlier
project code and installs nothing. Missing R is a warning until Stage 2.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT_MARKER = ".cmps355_root"
REQUIRED_DIRS = (
    "protocol", "references", "data/validated", "notebooks",
    "src/cmps355", "configs", "tests", "docs",
)
REQUIRED_FILES = (
    "protocol/proposal.pdf", "protocol/project_guidelines.pdf",
    "configs/dataset.json", "data/validated/ibd_dataset_sample_aligned.zip",
    "requirements.txt",
)


def find_project_root(start: Path | str | None = None) -> Path:
    """Find the closest parent carrying our unique root marker (Windows-safe)."""
    start = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in (start, *start.parents):
        if (candidate / ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {ROOT_MARKER} above {start}. Open JupyterLab from the extracted "
        "CMPS355_Research_Fresh directory, or move this notebook back into its notebooks/ folder."
    )


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_report(root: Path, check_r: bool = True) -> dict:
    """Return a structured, inspectable report. No third-party packages needed."""
    root = find_project_root(root)
    checks = []

    def record(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    record("Project marker", "PASS", str(root))
    record("Python version", "PASS" if sys.version_info >= (3, 11) else "FAIL", sys.version.split()[0])
    for dirname in REQUIRED_DIRS:
        p = root / dirname
        record(f"Folder {dirname}", "PASS" if p.is_dir() else "FAIL", str(p))
    for filename in REQUIRED_FILES:
        p = root / filename
        record(f"Input {filename}", "PASS" if p.is_file() else "FAIL", str(p))

    for name in ("numpy", "pandas"):
        try:
            ver = importlib.metadata.version(name)
            expected = {"numpy": "2.3.5", "pandas": "2.2.3"}[name]
            record(f"Python dependency {name}", "PASS" if ver == expected else "FAIL",
                   f"installed={ver}; expected={expected}; install requirements.txt and restart kernel if needed")
        except importlib.metadata.PackageNotFoundError:
            record(f"Python dependency {name}", "FAIL", f"Install: python -m pip install -r requirements.txt")

    if check_r:
        rscript = shutil.which("Rscript")
        if rscript:
            try:
                proc = subprocess.run([rscript, "--version"], capture_output=True, text=True, timeout=8)
                record("Rscript (required at Stage 2)", "PASS" if proc.returncode == 0 else "WARN",
                       (proc.stdout or proc.stderr).strip()[:220])
            except Exception as exc:
                record("Rscript (required at Stage 2)", "WARN", f"Found but could not run: {exc}")
        else:
            record("Rscript (required at Stage 2)", "WARN",
                   "Not on PATH. Stage 1 can proceed; install R before StableMate/TENTACLES verification.")

    manifest_path = root / "configs/dataset.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset_file = root / manifest["artifact"]
            if dataset_file.is_file():
                actual = sha256_file(dataset_file)
                expected = manifest["sha256"]
                record("Canonical dataset SHA-256", "PASS" if actual == expected else "FAIL",
                       f"actual={actual}; expected={expected}")
            else:
                record("Canonical dataset SHA-256", "FAIL", f"Missing file {dataset_file}")
        except (ValueError, KeyError, OSError) as exc:
            record("Canonical dataset manifest", "FAIL", str(exc))

    # New outputs live under runs/; do not require an empty runs/ in the released ZIP.
    try:
        with tempfile.NamedTemporaryFile(prefix="write_test_", dir=root, delete=True) as f:
            f.write(b"ok")
            f.flush()
        record("Project write access", "PASS", "Temporary write succeeded")
    except OSError as exc:
        record("Project write access", "FAIL", str(exc))

    summary = {status: sum(c["status"] == status for c in checks) for status in ("PASS", "WARN", "FAIL")}
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "summary": summary,
        "checks": checks,
    }


def write_report(report: dict, root: Path) -> Path:
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "runs" / "setup" / f"setup_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return output

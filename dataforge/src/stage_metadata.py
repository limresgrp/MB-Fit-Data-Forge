"""Write reproducible, lightweight metadata for workflow stages."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


def _git_revision(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _path_summary(path: str, root: Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    summary = {
        "path": str(resolved),
        "relative_to_dataset_root": os.path.relpath(resolved, root),
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        return summary

    if resolved.is_file():
        summary["kind"] = "file"
        summary["size_bytes"] = resolved.stat().st_size
        if resolved.stat().st_size <= 50 * 1024 * 1024:
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            summary["sha256"] = digest.hexdigest()
        return summary

    files = [candidate for candidate in resolved.rglob("*") if candidate.is_file()]
    summary["kind"] = "directory"
    summary["file_count"] = len(files)
    summary["total_size_bytes"] = sum(candidate.stat().st_size for candidate in files)
    return summary


def record_stage(
    dataset_root: str,
    stage: str,
    status: str = "completed",
    inputs: Optional[Iterable[str]] = None,
    outputs: Optional[Iterable[str]] = None,
    parameters: Optional[dict] = None,
    command: Optional[str] = None,
) -> dict:
    """Write ``metadata/stages/<stage>.json`` and append to ``pipeline.jsonl``."""
    root = Path(dataset_root).expanduser().resolve()
    stage_dir = root / "metadata" / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "status": status,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python": sys.executable,
        "platform": platform.platform(),
        "git_revision": _git_revision(root),
        "parameters": parameters or {},
        "inputs": [_path_summary(path, root) for path in (inputs or [])],
        "outputs": [_path_summary(path, root) for path in (outputs or [])],
    }
    (stage_dir / f"{stage}.json").write_text(json.dumps(record, indent=2) + "\n")
    with (root / "metadata" / "pipeline.jsonl").open("a") as history:
        history.write(json.dumps(record) + "\n")
    return record

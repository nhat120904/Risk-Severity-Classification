from __future__ import annotations

from pathlib import Path
import uuid


def unique_temp_path(directory: str | Path, suffix: str = "") -> Path:
    d = Path(directory); d.mkdir(exist_ok=True)
    return d / f"tmp_{uuid.uuid4().hex}{suffix}"



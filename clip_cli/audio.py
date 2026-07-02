"""Stage 2: extract Opus mono 16kHz audio from the source video using local ffmpeg."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("audio")


def extract(source: Path, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install with: brew install ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(source),
        "-vn",
        "-c:a", "libopus", "-b:a", "32k", "-ac", "1", "-ar", "16000",
        "-f", "opus",
        str(tmp),
    ]
    log.debug(f"ffmpeg cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"ffmpeg failed (exit {result.returncode}): {result.stderr.strip()}")
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.strip()}")
    tmp.rename(dest)
    log.info(f"audio extract ok: {dest} ({dest.stat().st_size} bytes)")
    return dest

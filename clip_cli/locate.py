"""Stage 1: locate the source video. URL -> yt-dlp download. Local path -> symlink."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("locate")


def is_url(s: str) -> bool:
    return urlparse(s).scheme in ("http", "https")


def locate(source: str, dest: Path) -> Path:
    """Ensure `dest` exists and is the source video (downloaded or symlinked)."""
    if dest.exists():
        return dest
    if dest.is_symlink():
        dest.unlink()
    if is_url(source):
        _download(source, dest)
    else:
        _symlink_local(Path(source).expanduser(), dest)
    return dest


def _download(url: str, dest: Path) -> None:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not found on PATH. Install with: brew install yt-dlp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    template = str(dest.parent / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=1080]+ba/b[height<=1080]",
        "--remux-video", "mp4",
        "--no-playlist",
        "--quiet", "--no-warnings",
        "-o", template,
        url,
    ]
    log.info(f"yt-dlp downloading {url}")
    log.debug(f"yt-dlp cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"yt-dlp failed (exit {result.returncode}): {result.stderr.strip()}")
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr.strip()}")
    candidates = sorted(dest.parent.glob("source.*"))
    mp4s = [p for p in candidates if p.suffix == ".mp4"]
    src = mp4s[0] if mp4s else (candidates[0] if candidates else None)
    if src is None:
        raise RuntimeError("yt-dlp succeeded but no source.* file found")
    if src != dest:
        src.rename(dest)
    log.info(f"download ok: {dest} ({dest.stat().st_size} bytes)")


def _symlink_local(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if not src.is_file():
        raise ValueError(f"Source is not a file: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src.resolve())
    log.info(f"symlinked local source: {dest} -> {src.resolve()} ({src.stat().st_size} bytes)")

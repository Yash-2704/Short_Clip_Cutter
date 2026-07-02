"""Per-job cache for pipeline stage artifacts."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Stage(str, Enum):
    LOCATE = "locate"
    AUDIO = "audio"
    TRANSCRIBE = "transcribe"
    RANK = "rank"
    RENDER = "render"


STAGE_ORDER = [Stage.LOCATE, Stage.AUDIO, Stage.TRANSCRIBE, Stage.RANK, Stage.RENDER]


def derive_job_id(source: str) -> str:
    """Stable id from source URL or local path. Re-runs on the same source hit the same cache."""
    p = Path(source).expanduser()
    if p.is_file():
        st = p.stat()
        key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    else:
        key = source
    return hashlib.sha256(key.encode()).hexdigest()[:12]


@dataclass
class WorkDir:
    root: Path

    @classmethod
    def for_job(cls, work_root: Path, job_id: str) -> "WorkDir":
        d = work_root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return cls(root=d)

    @property
    def source(self) -> Path: return self.root / "source.mp4"
    @property
    def audio(self) -> Path: return self.root / "audio.opus"
    @property
    def segments(self) -> Path: return self.root / "segments.json"
    @property
    def clips(self) -> Path: return self.root / "clips.json"
    @property
    def render_zip(self) -> Path: return self.root / "output.zip"
    @property
    def render_meta(self) -> Path: return self.root / "render_meta.json"
    @property
    def source_id_file(self) -> Path: return self.root / "source_id.txt"

    def has(self, stage: Stage) -> bool:
        files = self._files_for(stage)
        if not all(p.exists() and p.stat().st_size > 0 for p in files):
            return False
        if stage in (Stage.TRANSCRIBE, Stage.RANK):
            return _is_json(files[0])
        return True

    def clips_match_params(self, num_clips: int, duration_range: tuple[int, int]) -> bool:
        if not self.clips.exists():
            return False
        try:
            data = json.loads(self.clips.read_text())
        except Exception:
            return False
        p = data.get("_params") or {}
        return (
            p.get("num_clips") == num_clips
            and tuple(p.get("duration_range") or ()) == tuple(duration_range)
        )

    def render_matches_params(self, style: str, expected_count: int, fit_mode: str) -> bool:
        if not self.render_meta.exists() or not self.render_zip.exists():
            return False
        try:
            meta = json.loads(self.render_meta.read_text())
        except Exception:
            return False
        return (
            meta.get("style") == style
            and meta.get("count") == expected_count
            and meta.get("fit_mode") == fit_mode
        )

    def write_clips(self, clips_response: dict, num_clips: int, duration_range: tuple[int, int]) -> None:
        data = {
            "_params": {"num_clips": num_clips, "duration_range": list(duration_range)},
            **clips_response,
        }
        self.clips.write_text(json.dumps(data, indent=2))

    def read_clips(self) -> list[dict]:
        return json.loads(self.clips.read_text())["clips"]

    def write_render_meta(self, style: str, count: int, fit_mode: str) -> None:
        self.render_meta.write_text(json.dumps({"style": style, "count": count, "fit_mode": fit_mode}))

    def invalidate_from(self, stage: Stage) -> None:
        """Delete cached artifacts for `stage` and all later stages."""
        idx = STAGE_ORDER.index(stage)
        for s in STAGE_ORDER[idx:]:
            for p in self._files_for(s):
                if p.is_symlink() or p.exists():
                    if p.is_dir() and not p.is_symlink():
                        shutil.rmtree(p)
                    else:
                        p.unlink()

    def _files_for(self, stage: Stage) -> list[Path]:
        if stage is Stage.LOCATE:
            return [self.source]
        if stage is Stage.AUDIO:
            return [self.audio]
        if stage is Stage.TRANSCRIBE:
            return [self.segments]
        if stage is Stage.RANK:
            return [self.clips]
        if stage is Stage.RENDER:
            return [self.render_zip, self.render_meta]
        return []


def _is_json(p: Path) -> bool:
    try:
        json.loads(p.read_text())
        return True
    except Exception:
        return False

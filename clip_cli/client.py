"""HTTP client for the GPU server. One method per endpoint in gpu_server_spec.md §3."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("client")


class ServerError(Exception):
    def __init__(self, status: int, code: str, detail: str):
        self.status = status
        self.code = code
        self.detail = detail
        super().__init__(f"[{status} {code}] {detail}")


class ServerBusy(ServerError): pass
class SourceExpired(ServerError): pass
class ServerUnavailable(ServerError): pass


# ML stages can take minutes; default httpx timeout (5s) would kill every long POST.
# read budget sized for the worst observed render (long-form clips with many shots) — 30 min ceiling.
_TIMEOUT = httpx.Timeout(connect=10.0, read=1800.0, write=1800.0, pool=10.0)


class ClipServer:
    def __init__(self, base_url: str, token: Optional[str] = None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=_TIMEOUT,
            headers=headers,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def health(self) -> dict:
        log.debug("GET /health")
        r = self._client.get("/health")
        log.info(f"GET /health -> {r.status_code} ({len(r.content)} bytes)")
        if r.status_code >= 400:
            log.error(f"/health body: {r.text}")
            _raise(r)
        body = r.json()
        log.debug(f"/health body: {json.dumps(body)}")
        return body

    def transcribe(self, audio_path: Path) -> dict:
        size = audio_path.stat().st_size
        log.info(f"POST /transcribe audio={audio_path.name} ({size} bytes)")
        t0 = time.monotonic()
        with audio_path.open("rb") as f:
            r = self._post_json("/transcribe", files={"audio": (audio_path.name, f, "audio/ogg")})
        body = r.json()
        log.info(f"/transcribe -> {r.status_code} in {time.monotonic()-t0:.2f}s "
                 f"({len(body.get('segments', []))} segments, "
                 f"{body.get('duration_s', 0):.1f}s audio, language={body.get('language')})")
        return body

    def rank(self, transcript: list[dict], num_clips: int, duration_range: tuple[int, int]) -> dict:
        body = {
            "transcript": transcript,
            "num_clips": num_clips,
            "duration_range": list(duration_range),
        }
        body_bytes = len(json.dumps(body))
        log.info(f"POST /rank num_clips={num_clips} duration_range={duration_range} "
                 f"transcript_segments={len(transcript)} body_size={body_bytes} bytes")
        t0 = time.monotonic()
        r = self._post_json("/rank", json=body)
        resp = r.json()
        log.info(f"/rank -> {r.status_code} in {time.monotonic()-t0:.2f}s "
                 f"({len(resp.get('clips', []))} clips returned)")
        log.debug(f"/rank response: {json.dumps(resp)}")
        return resp

    def render(
        self,
        *,
        clips: list[dict],
        style: str,
        out_zip: Path,
        video_path: Optional[Path] = None,
        source_id: Optional[str] = None,
        transcript: Optional[list[dict]] = None,
        fit_mode: str = "auto",
    ) -> str:
        """Render clips, stream the zip response to `out_zip`. Returns X-Source-Id for reuse.

        `transcript` carries word-level segments for caption burn-in. Optional on the wire
        (server skips captions if missing) but required for the demo's karaoke look.
        """
        if source_id is None and video_path is None:
            raise ValueError("render requires video_path or source_id")

        log.info(f"POST /render clips={len(clips)} style={style} fit_mode={fit_mode} "
                 f"video={'reuse:'+source_id if source_id else video_path.name} "
                 f"transcript_segments={len(transcript) if transcript else 0}")
        t0 = time.monotonic()
        try:
            r = self._render_call(clips, style, out_zip, video_path, source_id, transcript, fit_mode,
                                  use_video=source_id is None)
        except SourceExpired:
            log.warning(f"source_id {source_id} expired on server; falling back to re-upload")
            if video_path is None:
                raise
            r = self._render_call(clips, style, out_zip, video_path, None, transcript, fit_mode,
                                  use_video=True)
        sid = r.headers.get("X-Source-Id", "")
        log.info(f"/render -> {r.status_code} in {time.monotonic()-t0:.2f}s "
                 f"(zip={out_zip.stat().st_size} bytes, X-Source-Id={sid or 'none'})")
        return sid

    # ---- internals ----

    def _render_call(self, clips, style, out_zip, video_path, source_id, transcript, fit_mode, *, use_video):
        data = {"clips": json.dumps(clips), "style": style, "fit_mode": fit_mode}
        if transcript is not None:
            data["transcript"] = json.dumps(transcript)
        if not use_video and source_id:
            data["source_id"] = source_id
        files = None
        fh = None
        if use_video:
            fh = video_path.open("rb")
            files = {"video": (video_path.name, fh, "video/mp4")}
        try:
            with self._client.stream("POST", "/render", data=data, files=files) as r:
                if r.status_code != 200:
                    r.read()
                    _raise(r)
                out_zip.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_zip.with_suffix(out_zip.suffix + ".part")
                with tmp.open("wb") as out:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        out.write(chunk)
                tmp.rename(out_zip)
                return r
        finally:
            if fh is not None:
                fh.close()

    def _post_json(self, path: str, **kwargs) -> httpx.Response:
        # One retry on 500. Don't retry busy/unavailable/bad-input.
        last: httpx.Response | None = None
        for attempt in (1, 2):
            r = self._client.post(path, **kwargs)
            if r.status_code < 400:
                return r
            if r.status_code == 500 and attempt == 1:
                last = r
                continue
            _raise(r)
        assert last is not None
        _raise(last)


def _raise(r: httpx.Response) -> None:
    try:
        body = r.json()
        code = str(body.get("error", "unknown"))
        detail = str(body.get("detail", r.text))
    except Exception:
        code = "unknown"
        detail = r.text or r.reason_phrase
    log.error(f"server error {r.status_code} ({code}): {detail}")
    cls = {409: ServerBusy, 410: SourceExpired, 503: ServerUnavailable}.get(r.status_code, ServerError)
    raise cls(r.status_code, code, detail)

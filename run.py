"""Mac client CLI: long video -> ranked 9:16 short clips via the GPU server."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

import click
from dotenv import load_dotenv

from clip_cli import audio, locate, log as log_setup, ui
from clip_cli.cache import STAGE_ORDER, Stage, WorkDir, derive_job_id
from clip_cli.client import ClipServer, ServerBusy, ServerError, ServerUnavailable

load_dotenv()
TOTAL = 5
log = logging.getLogger("run")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("source", type=str)
@click.option("--num-clips", "num_clips", type=int, default=5, show_default=True,
              help="How many clips to extract.")
@click.option("--duration", "duration", type=(int, int), default=(25, 65), show_default=True,
              metavar="MIN MAX", help="Per-clip duration range in seconds.")
@click.option("--style", type=click.Choice(["viral", "hormozi", "podcast", "minimal"]),
              default="viral", show_default=True, help="Caption style preset.")
@click.option("--fit-mode", "fit_mode",
              type=click.Choice(["auto", "crop", "fit", "stylized"]),
              default="auto", show_default=True,
              help="How to fit 16:9 source into 9:16 output. "
                   "'auto' crops when a face is tracked, fits otherwise.")
@click.option("--from-stage", "from_stage",
              type=click.Choice([s.value for s in STAGE_ORDER]),
              default=None, help="Force re-run from this stage onward, ignoring cache.")
@click.option("--job-id", "job_id_override", type=str, default=None,
              help="Reuse a specific job-id (default: derived from source).")
@click.option("--output", "output_dir", type=click.Path(file_okay=False, path_type=Path),
              default=Path("output"), show_default=True, help="Where final clips land.")
@click.option("--work", "work_root", type=click.Path(file_okay=False, path_type=Path),
              default=Path("work"), show_default=True, help="Per-job cache directory.")
def main(source: str, num_clips: int, duration: tuple[int, int], style: str,
         fit_mode: str, from_stage: str | None, job_id_override: str | None,
         output_dir: Path, work_root: Path) -> None:
    """Turn a long video (URL or local file) into ranked 9:16 short clips with captions."""
    base_url = os.environ.get("CLIP_SERVER_URL")
    if not base_url:
        ui.console.print("[red]CLIP_SERVER_URL not set.[/] Copy .env.example to .env and fill it in.")
        sys.exit(2)
    token = os.environ.get("CLIP_SERVER_TOKEN")

    job_id = job_id_override or derive_job_id(source)
    work = WorkDir.for_job(work_root, job_id)
    run_id = log_setup.make_run_id(job_id)
    run_output_dir = output_dir / run_id

    log_path = log_setup.setup(run_id)
    log.info("=== run start ===")
    log.info(f"argv: {sys.argv}")
    log.info(f"source={source!r}")
    log.info(f"run_id={run_id}")
    log.info(f"job_id={job_id} work={work.root} server={base_url} token={'set' if token else 'none'}")
    log.info(f"params: num_clips={num_clips} duration={duration} style={style} fit_mode={fit_mode} "
             f"from_stage={from_stage} output_root={output_dir} run_output={run_output_dir}")

    if from_stage:
        log.info(f"invalidating cache from stage={from_stage}")
        work.invalidate_from(Stage(from_stage))

    ui.console.print(f"[dim]job-id:[/] {job_id}   [dim]work:[/] {work.root}")
    ui.console.print(f"[dim]log:[/] {log_path}")

    t0 = time.monotonic()
    exit_code = 0
    try:
        with ClipServer(base_url, token=token) as server:
            _preflight(server)
            _stage_locate(1, source, work)
            _stage_audio(2, work)
            segments = _stage_transcribe(3, server, work)
            clips = _stage_rank(4, server, work, segments, num_clips, duration)
            ui.clips_table(clips)
            _stage_render(5, server, work, clips, segments, style, fit_mode)
            _unpack(work.render_zip, run_output_dir)
            _write_transcripts(segments, clips, run_output_dir)
            _update_latest(output_dir, run_output_dir)
            ui.console.print(f"\n[bold green]Done.[/] {len(clips)} clips in [cyan]{run_output_dir}/[/]")
            ui.console.print(f"[dim]also: {output_dir}/latest -> {run_id}/[/]")
            ui.console.print(f"[dim]log:[/] {log_path}")
    except ServerBusy as e:
        log.error(f"server busy: {e.detail}")
        ui.console.print(f"[red]Server is busy:[/] {e.detail}")
        exit_code = 3
    except ServerUnavailable as e:
        log.error(f"server unavailable: {e.detail}")
        ui.console.print(f"[red]Server unavailable:[/] {e.detail}")
        exit_code = 4
    except ServerError as e:
        log.error(f"server error: status={e.status} code={e.code} detail={e.detail}")
        ui.console.print(f"[red]Server error {e.status} ({e.code}):[/] {e.detail}")
        exit_code = 5
    except Exception:
        log.error(f"unhandled exception:\n{traceback.format_exc()}")
        raise
    finally:
        log.info(f"=== run end (exit={exit_code}, elapsed={time.monotonic()-t0:.1f}s) ===")
    if exit_code:
        sys.exit(exit_code)


def _preflight(server: ClipServer) -> None:
    try:
        h = server.health()
    except ServerUnavailable as e:
        log.error(f"preflight: server degraded: {e.detail}")
        ui.console.print(f"[red]GPU server is degraded:[/] {e.detail}")
        sys.exit(4)
    except Exception as e:
        log.error(f"preflight: cannot reach server: {e}")
        ui.console.print(f"[red]Cannot reach GPU server:[/] {e}")
        sys.exit(4)
    if h.get("status") != "ok":
        log.error(f"preflight: non-ok health: {h}")
        ui.console.print(f"[red]Health check returned non-ok:[/] {h}")
        sys.exit(4)
    log.info(f"preflight ok: vram_free={h.get('gpu', {}).get('vram_free_mb')} MB "
             f"ollama_model={h.get('ollama', {}).get('model')} "
             f"whisper={h.get('whisper', {}).get('model')}")


def _stage_locate(n: int, source: str, work: WorkDir) -> None:
    if work.has(Stage.LOCATE):
        log.info(f"stage 1 locate: cache hit ({work.source})")
        ui.cached(n, TOTAL, "Locating source")
        return
    log.info("stage 1 locate: starting")
    with ui.stage(n, TOTAL, "Locating source") as st:
        path = locate.locate(source, work.source)
        st["detail"] = f"{path.stat().st_size / 1024**2:.1f} MB"


def _stage_audio(n: int, work: WorkDir) -> None:
    if work.has(Stage.AUDIO):
        log.info(f"stage 2 audio: cache hit ({work.audio})")
        ui.cached(n, TOTAL, "Extracting audio")
        return
    log.info("stage 2 audio: starting")
    with ui.stage(n, TOTAL, "Extracting audio (Opus mono 16kHz)") as st:
        path = audio.extract(work.source, work.audio)
        st["detail"] = f"{path.stat().st_size / 1024:.0f} KB"


def _stage_transcribe(n: int, server: ClipServer, work: WorkDir) -> list[dict]:
    if work.has(Stage.TRANSCRIBE):
        log.info(f"stage 3 transcribe: cache hit ({work.segments})")
        ui.cached(n, TOTAL, "Transcribing (WhisperX)")
        return json.loads(work.segments.read_text())["segments"]
    log.info("stage 3 transcribe: starting")
    with ui.stage(n, TOTAL, "Transcribing via GPU server (WhisperX large-v3)") as st:
        resp = server.transcribe(work.audio)
        work.segments.write_text(json.dumps(resp))
        st["detail"] = f"{len(resp.get('segments', []))} segments, {resp.get('duration_s', 0):.0f}s audio"
    return resp["segments"]


def _stage_rank(n: int, server: ClipServer, work: WorkDir,
                segments: list[dict], num_clips: int, duration: tuple[int, int]) -> list[dict]:
    if work.has(Stage.RANK) and work.clips_match_params(num_clips, duration):
        log.info(f"stage 4 rank: cache hit ({work.clips})")
        ui.cached(n, TOTAL, f"Ranking top {num_clips} moments")
        return work.read_clips()
    log.info(f"stage 4 rank: starting (num_clips={num_clips}, duration={duration})")
    with ui.stage(n, TOTAL, f"Ranking top {num_clips} moments (Qwen 2.5 14B)"):
        resp = server.rank(segments, num_clips, duration)
        work.write_clips(resp, num_clips, duration)
    for c in resp["clips"]:
        flags = []
        if c.get("end_extended_by"):
            flags.append(f"end_extended_by={c['end_extended_by']:.1f}s")
        if c.get("start_extended_by"):
            flags.append(f"start_extended_by={c['start_extended_by']:.1f}s")
        if c.get("clip_split_at_boundary"):
            flags.append(f"split_at_boundary={c['clip_split_at_boundary']:.1f}s")
        flags_str = (" " + " ".join(flags)) if flags else ""
        log.info(f"clip rank={c.get('rank')} score={c.get('score')} "
                 f"range={c.get('start'):.1f}-{c.get('end'):.1f} "
                 f"duration={c.get('duration'):.1f}{flags_str} hook={c.get('hook')!r}")
    return resp["clips"]


def _stage_render(n: int, server: ClipServer, work: WorkDir,
                  clips: list[dict], segments: list[dict], style: str, fit_mode: str) -> None:
    _validate_clips(clips)
    if work.has(Stage.RENDER) and work.render_matches_params(style, len(clips), fit_mode):
        log.info(f"stage 5 render: cache hit ({work.render_zip})")
        ui.cached(n, TOTAL, f"Rendering {len(clips)} vertical clips")
        return
    cached_sid = work.source_id_file.read_text().strip() if work.source_id_file.exists() else None
    log.info(f"stage 5 render: starting (clips={len(clips)}, style={style}, fit_mode={fit_mode}, "
             f"source_id={cached_sid or 'none'})")
    with ui.stage(n, TOTAL, f"Rendering {len(clips)} vertical clips ({style}/{fit_mode}, NVENC)"):
        sid = server.render(
            clips=clips, style=style, out_zip=work.render_zip,
            video_path=work.source if cached_sid is None else None,
            source_id=cached_sid,
            transcript=segments,
            fit_mode=fit_mode,
        )
        if sid:
            work.source_id_file.write_text(sid)
        work.write_render_meta(style, len(clips), fit_mode)


def _write_transcripts(segments: list[dict], clips: list[dict], run_output_dir: Path) -> None:
    """Per-clip .txt (human-readable, timestamps relative to clip start) + .json
    (full metadata + filtered segments) alongside the rendered mp4s, plus one
    combined transcripts.json index for the whole run."""
    run_output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for c in clips:
        rank = c.get("rank", 0)
        start, end = c["start"], c["end"]
        clip_segs = [s for s in segments if s["end"] > start and s["start"] < end]

        lines = [f"# Clip {rank:02d}  —  {c.get('hook', '')}",
                 f"# range: {start:.2f}s – {end:.2f}s  ({c.get('duration', 0):.1f}s)",
                 f"# score: {c.get('score', '?')}",
                 ""]
        for s in clip_segs:
            rel = max(0.0, s["start"] - start)
            mm, ss = divmod(rel, 60)
            lines.append(f"[{int(mm):02d}:{ss:05.2f}] {s['text'].strip()}")

        txt_path = run_output_dir / f"clip_{rank:02d}.txt"
        json_path = run_output_dir / f"clip_{rank:02d}.json"
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        meta = {
            "rank": rank,
            "start": start,
            "end": end,
            "duration": c.get("duration"),
            "hook": c.get("hook"),
            "score": c.get("score"),
            "reason": c.get("reason"),
            "end_extended_by": c.get("end_extended_by"),
            "segments": clip_segs,
        }
        json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        combined.append(meta)
        log.info(f"transcripts: wrote {txt_path.name} + {json_path.name} ({len(clip_segs)} segments)")

    index_path = run_output_dir / "transcripts.json"
    index_path.write_text(
        json.dumps({"clips": combined}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"transcripts: wrote {index_path.name} ({len(combined)} clips)")


def _update_latest(output_root: Path, run_output_dir: Path) -> None:
    """Maintain `<output_root>/latest` as a symlink to the most recent run."""
    latest = output_root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_output_dir.name)
        log.info(f"latest -> {run_output_dir.name}")
    except OSError as e:
        log.warning(f"could not update latest symlink: {e}")


def _validate_clips(clips: list[dict]) -> None:
    for c in clips:
        s, e = c.get("start"), c.get("end")
        if s is None or e is None or e <= s:
            raise click.ClickException(f"Invalid clip from rank: {c}")


def _unpack(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        log.info(f"unpack: zip contents = {names}")
        if "manifest.json" in names:
            try:
                manifest = json.loads(z.read("manifest.json"))
                log.info(f"server manifest: {json.dumps(manifest)}")
                for entry in manifest:
                    miss = entry.get("miss_ratio")
                    miss_str = f"{miss:.2f}" if isinstance(miss, (int, float)) else str(miss)
                    traj = entry.get("trajectory") or {}
                    log.info(f"render result: {entry.get('filename')} "
                             f"mode={entry.get('mode')} miss_ratio={miss_str} "
                             f"elapsed_s={entry.get('elapsed_s')} "
                             f"trajectory={traj}")
                    if entry.get("fallback"):
                        log.warning(f"render fallback (face miss): "
                                    f"{entry.get('filename')} miss_ratio={miss_str}")
                    if not entry.get("ok", True):
                        log.error(f"render failure for {entry.get('filename')}: {entry}")
            except Exception as e:
                log.warning(f"could not parse manifest.json: {e}")
        for name in names:
            if not name.endswith(".mp4"):
                continue
            target = out_dir / Path(name).name
            with z.open(name) as src, target.open("wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            log.info(f"unpacked {name} -> {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

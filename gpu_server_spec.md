# GPU Server — Implementation Spec

Companion document to `clipping_demo_project.md`. This file is the source of truth for what the GPU-side FastAPI service must do, how it must behave, and the contract it owes the Mac client. The Mac client is being developed against this spec in parallel; treat anything here as binding unless explicitly renegotiated.

---

## 1. Context — what you are building

A single-host FastAPI service that runs on a **Windows 10/11 box** with an NVIDIA 12 GB GPU and serves three real endpoints plus a health probe. Native Windows install — no WSL. The service is the entire worker tier of a two-machine video-clipping pipeline:

```
Mac CLI (orchestrator)  ──HTTP over Tailscale──▶  YOUR SERVICE (this spec)
                                                     │
                                                     ├── WhisperX + faster-whisper (transcribe)
                                                     ├── Qwen 2.5 14B via Ollama (rank)
                                                     └── MediaPipe + ffmpeg/NVENC (render)
```

You own everything inside the box. The Mac client owns: source download (yt-dlp / local file), local audio extract, calling your endpoints in sequence, caching responses, unzipping `/render` output. Do not assume any state lives on the Mac between calls — every request must carry the data it needs (or reference a `source_id` you've cached server-side; see §5.4).

You do not own: any UI, auth UX, MCP wrapper, Instagram posting, scheduling. Those are out of scope. Build the service and nothing more.

---

## 2. Hard constraints

| Constraint | Value | Why it matters |
|---|---|---|
| VRAM ceiling | 12 GB | Whisper-large (~5 GB) + Qwen 14B Q4_K_M (~8.5 GB) do **not** fit simultaneously. You **must** load/free sequentially. See §4. |
| Source format | mp4/mkv/mov, up to 1080p, ≤ 30 min | Demo target is 5–10 min sources. Design for 30 min as a soft ceiling. |
| Audio format from client | Opus, mono, 16 kHz, 32 kbps | WhisperX-compatible. Don't re-decode unless required. |
| Single concurrent job | One pipeline at a time | Multi-tenant is not in scope. See §7. |
| Stateless across pipeline runs | Each endpoint call is independent | Except optional `source_id` cache for render (§5.4). |
| No outbound internet at runtime | Only during model download | Pipeline must work offline. |
| Platform | Windows 10/11 native | All install commands, paths, service manager are Windows-specific. See "Platform runbook" below. |

---

## Platform runbook — Windows native

Everything below assumes a recent NVIDIA driver is installed and `nvidia-smi.exe` works from a fresh PowerShell. CUDA toolkit is **not** required — PyTorch ships its own CUDA runtime via pip wheels.

### Install order

1. **Python 3.11** — installer from python.org. Tick "Add to PATH". Verify: `python --version`.
2. **ffmpeg with NVENC** — download a "full" build from https://www.gyan.dev/ffmpeg/builds/ (the `full` build includes NVENC). Extract to `C:\ffmpeg\`, add `C:\ffmpeg\bin` to System PATH. Verify: `ffmpeg -hide_banner -encoders | findstr nvenc` shows `h264_nvenc` and `hevc_nvenc`.
3. **Ollama for Windows** — installer from https://ollama.com/download. Installs as a Windows service (`OllamaService`) listening on `http://localhost:11434`. Verify: `ollama list` works in a new shell.
4. **Pull Qwen** — `ollama pull qwen2.5:14b-instruct-q4_K_M` (~8.5 GB download).
5. **Tailscale for Windows** — installer from tailscale.com. After install, run `tailscale ip -4` to get the address the Mac client will use.
6. **NSSM** — the service wrapper. Either `scoop install nssm` or download from https://nssm.cc/. Used in §service below.

### Project layout

Place everything under `C:\services\clip-server\` (or any path without spaces — paths with spaces complicate NSSM args).

```
C:\services\clip-server\
├── .venv\                          # python3.11 -m venv .venv
├── server.py
├── app\
│   ├── transcribe.py
│   ├── rank.py
│   ├── reframe.py
│   ├── captions.py
│   └── render.py
├── work\
│   ├── jobs\<job_id>\
│   └── sources\
└── logs\
    ├── service.out.log
    └── service.err.log
```

### Python environment

```powershell
cd C:\services\clip-server
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Core ML — pin CUDA 12.1 wheels (match your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper whisperx

# Service
pip install fastapi "uvicorn[standard]" python-multipart httpx pydantic

# CV / face track
pip install mediapipe opencv-python-headless numpy

# Pre-download Whisper weights (~3 GB)
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16')"
```

If PowerShell blocks the `Activate.ps1` script, run once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

### Service via NSSM

NSSM wraps `uvicorn.exe` as a real Windows service so it survives reboots and restarts on failure. Run an **elevated** PowerShell:

```powershell
nssm install ClipServer "C:\services\clip-server\.venv\Scripts\uvicorn.exe" "server:app --host 0.0.0.0 --port 8000"
nssm set ClipServer AppDirectory "C:\services\clip-server"
nssm set ClipServer AppStdout "C:\services\clip-server\logs\service.out.log"
nssm set ClipServer AppStderr "C:\services\clip-server\logs\service.err.log"
nssm set ClipServer AppRotateFiles 1
nssm set ClipServer AppRotateBytes 10485760

# Critical: keep the Qwen 14B model from sitting in VRAM across stages (see §4)
nssm set ClipServer AppEnvironmentExtra OLLAMA_KEEP_ALIVE=0

nssm set ClipServer Start SERVICE_AUTO_START
nssm set ClipServer DependOnService OllamaService
nssm start ClipServer

# Smoke
curl http://localhost:8000/health
```

To update after a code change:

```powershell
nssm restart ClipServer
```

To debug interactively (foreground, see tracebacks):

```powershell
nssm stop ClipServer
.venv\Scripts\Activate.ps1
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Windows-specific gotchas

- **`OLLAMA_KEEP_ALIVE=0` must be set in the NSSM env block**, not just a user shell env var — services don't inherit interactive shell env. This is the #1 way VRAM sequencing breaks (see §4).
- **File handles block deletion.** Unlike Linux, Windows will not let you `os.unlink()` a file with an open handle. The "stream to `.part`, rename on success" pattern is mandatory for `output.zip` and any cached source — not optional. Close `ffmpeg` subprocess pipes explicitly before renaming or deleting.
- **Path length cap.** Default Windows path limit is 260 chars. Keep job IDs short (12 char hash) and `work\sources\` shallow. If you ever hit it, enable long paths via group policy or registry, but design around it instead.
- **No `~` expansion** in subprocess args you pass to ffmpeg/ollama — Python's `os.path.expanduser` is fine, but don't paste `~/...` strings into ffmpeg's command line.
- **Antivirus / Defender.** Real-time scanning on `work\` can add seconds to every large file write. Exclude `C:\services\clip-server\work\` from Defender scanning (Settings → Virus & threat protection → Exclusions → add folder).
- **Firewall.** On first launch, Windows will prompt to allow uvicorn through the firewall. Approve "Private network." Tailscale handles its own tunneling but the local listener still needs the rule.
- **Newline handling in subprocess.** Always pass `text=True` to `subprocess.run` and don't assume `\n` separators when parsing tool stdout.
- **Sweeper for old jobs.** No cron. Either an in-process background task (asyncio scheduled cleanup) or a Task Scheduler entry pointing at a small Python script. Prefer in-process so the cleanup lives with the service.

### Verifying the install

Before writing any service code, prove the stack works:

```powershell
# GPU visible to PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Whisper loads on GPU
python -c "from faster_whisper import WhisperModel; m = WhisperModel('large-v3', device='cuda', compute_type='float16'); print('ok')"

# Ollama responds
curl http://localhost:11434/api/tags

# Qwen responds in JSON mode
curl -X POST http://localhost:11434/api/generate -d '{\"model\":\"qwen2.5:14b-instruct-q4_K_M\",\"prompt\":\"Return JSON {\\\"ok\\\":true}\",\"format\":\"json\",\"stream\":false}'

# ffmpeg + NVENC encode test
ffmpeg -f lavfi -i testsrc=duration=2:size=1280x720 -c:v h264_nvenc -y test.mp4
```

All five must work before any further server development.

---

## 3. HTTP contract

Base URL: `http://0.0.0.0:8000`. The Mac client reaches you via Tailscale IP. CORS not required.

All JSON bodies use UTF-8. All timestamps are seconds as float. All durations in seconds. All file uploads use `multipart/form-data`. Errors return JSON with `{"error": "<code>", "detail": "<human readable>"}` and an appropriate HTTP status (see §6).

### 3.1 `GET /health`

Required behavior — not just "service up":

```json
{
  "status": "ok",
  "ollama": {"reachable": true, "model_pulled": true, "model": "qwen2.5:14b-instruct-q4_K_M"},
  "whisper": {"weights_cached": true, "model": "large-v3"},
  "gpu": {"vram_total_mb": 12288, "vram_free_mb": 11240},
  "current_job": null
}
```

If any subsystem is degraded (Ollama unreachable, model not pulled, GPU not visible), return HTTP **503** with the same shape but `"status": "degraded"` and the failing component populated. The Mac client uses this as pre-flight and will refuse to start a job otherwise.

`current_job` is `null` when idle, or `{"id": "...", "stage": "transcribe|rank|render", "started_at": "<iso8601>"}` when busy.

### 3.2 `POST /transcribe`

- **Request:** `multipart/form-data`
  - file field `audio` — any format ffmpeg accepts; Mac client sends Opus mono 16 kHz.
- **Response:** `application/json`
  ```json
  {
    "language": "en",
    "duration_s": 612.4,
    "segments": [
      {
        "start": 0.0, "end": 4.2, "text": "So I was talking to...",
        "words": [
          {"word": "So", "start": 0.00, "end": 0.12, "score": 0.99},
          {"word": "I",  "start": 0.13, "end": 0.18, "score": 0.99}
        ]
      }
    ]
  }
  ```

Word-level timestamps are **required**, not optional. The caption render in `/render` depends on per-word timing. If WhisperX alignment fails on a segment, still return the segment but with an empty `words` array — don't drop the segment.

### 3.3 `POST /rank`

- **Request:** `application/json`
  ```json
  {
    "transcript": [<segments array from /transcribe>],
    "num_clips": 5,
    "duration_range": [25, 65]
  }
  ```
- **Response:** `application/json`
  ```json
  {
    "clips": [
      {
        "rank": 1,
        "start": 187.2, "end": 232.8, "duration": 45.6,
        "hook": "controversial take on AI training data",
        "score": 92,
        "reason": "strong opinion + clean intro + natural endpoint"
      }
    ]
  }
  ```

**`duration_range` must actually be substituted into the Qwen prompt** — don't hardcode 25–65 in the prompt template. The Mac client exposes this as a flag and will be confused if it's silently ignored.

Sort by `score` descending. Enforce no-overlap server-side; if Qwen returns overlapping clips, drop the lower-scored one and continue. Each clip must fall within `duration_range`. If Qwen returns fewer than `num_clips` valid clips after filtering, return what you have — do not pad.

Use Ollama's `format: "json"` mode and the prompt from §7 of `clipping_demo_project.md`. One retry on invalid JSON with a stricter "RETURN JSON ONLY, NO PROSE" preamble; then fail with 502.

### 3.4 `POST /render`

- **Request:** `multipart/form-data`
  - file field `video` — full source mp4 (large; see §5.4 for caching)
  - form field `clips` — JSON string of the clips array (same shape as `/rank` response's `clips`)
  - form field `style` — string preset name, one of: `viral` (default), `hormozi`, `podcast`, `minimal`
  - optional form field `source_id` — if present, server reuses a cached source upload and `video` may be omitted (see §5.4)
- **Response:** `application/zip`
  - Filenames inside zip: `clip_01.mp4` … `clip_NN.mp4`, **zero-padded to width 2 minimum**, widen to match N if N ≥ 100 (i.e. always wide enough that lexicographic sort = rank order).
  - `Content-Disposition: attachment; filename="clips_<job_id>.zip"` so the Mac client can stream-write it.

**Critical:** if render fails mid-pipeline, do **not** stream a partial zip and then close the socket. Buffer the zip server-side (to disk in `work/<job_id>/`), then either send the complete zip on success or return JSON 500 on failure. Streaming-then-erroring is unrecoverable on the client.

---

## 4. VRAM lifecycle — the most important section

The 12 GB ceiling is the single biggest risk in this design. Sequential model loading must be implemented explicitly. Do not rely on Python GC or PyTorch lazy eviction.

### 4.1 The three states

| State | Resident in VRAM | When |
|---|---|---|
| Idle | nothing | between jobs |
| Transcribing | faster-whisper-large-v3 (~3 GB) + WhisperX alignment model (~1.5 GB) | inside `/transcribe` only |
| Ranking | Qwen 2.5 14B Q4_K_M loaded by Ollama (~8.5 GB) | inside `/rank` only |
| Rendering | nothing meaningful (NVENC encoder block + MediaPipe on CPU) | inside `/render` only |

Transitions:
- After `/transcribe` returns response: `del whisper_model; del align_model; gc.collect(); torch.cuda.empty_cache()`. Verify with `nvidia-smi` or `torch.cuda.memory_allocated()` that VRAM dropped below ~500 MB before returning the HTTP response. If it didn't, log a warning — you have a leak.
- Before `/rank` calls Ollama: it's safe to assume Whisper is gone (you freed it). Ollama runs in its own process; you do not manage its memory directly.
- Ollama keep-alive: **set `OLLAMA_KEEP_ALIVE=0` in the NSSM service env block** (see Platform runbook above), and/or pass `"keep_alive": 0` in every Ollama API call. Default 5-min keep-alive means the 14B model stays resident across the next `/transcribe` call → OOM. This is the most likely way this design breaks in practice. A user-shell env var is **not enough** — Windows services don't inherit it.
- After `/rank` returns: explicitly tell Ollama to unload via a `keep_alive: 0` request, or rely on it having unloaded already if you set the env var.
- Before `/render`: verify VRAM is mostly free. NVENC uses ~500 MB of VRAM through ffmpeg; you do not load any ML model in this stage.

### 4.2 VRAM check between stages

Add an internal helper:

```python
def assert_vram_free(min_free_mb: int):
    free = torch.cuda.mem_get_info()[0] / 1024**2
    if free < min_free_mb:
        raise RuntimeError(f"VRAM not freed: only {free:.0f} MB free, need {min_free_mb}")
```

Call this at the top of each stage to fail fast with a meaningful error instead of dying inside a model load with a confusing CUDA OOM trace.

### 4.3 Why this is risky and how you'll know it's broken

If VRAM sequencing isn't implemented correctly, the symptom is: first job in a freshly-started service works; second job OOMs at the rank-or-render boundary. Always test the **second** consecutive job through the pipeline before declaring it works.

---

## 5. Per-endpoint implementation notes

### 5.1 `/transcribe`

- WhisperX wraps faster-whisper. Use `compute_type="float16"` on GPU.
- VAD filtering (`vad_filter=True`) is recommended — improves segmentation on long-form audio.
- The Mac sends Opus mono 16 kHz. faster-whisper resamples internally if needed; you do not need to pre-decode.
- For audio > 30 min, chunked transcription is slower than single-pass but uses less peak VRAM. Don't optimize for it now; the soft ceiling is 30 min.
- WhisperX alignment uses a wav2vec2 model. It needs to be downloaded on first run; pre-cache during setup.
- Return the response **before** freeing VRAM if it's simpler (the Mac doesn't care about timing of the free), but make sure freeing happens before the next request can land. Easiest: free inside a `finally:` block inside the lock-guarded handler (§7).

### 5.2 `/rank`

- Call Ollama via its HTTP API (`POST http://localhost:11434/api/generate`) with `format: "json"` and `stream: false`.
- The transcript can be long. A 10-min transcript is ~1500 words ≈ ~2000 tokens. Qwen 14B's context window handles this fine. For >30 min, you may need to truncate or chunk-summarize — out of scope for now.
- The prompt from §7 of `clipping_demo_project.md` is the starting point. Substitute `num_clips` and `duration_range` into it dynamically.
- Validate Qwen's output before returning:
  - JSON parseable
  - All required fields present and well-typed
  - All clips' start/end within `[0, transcript_duration]`
  - `end > start`
  - Duration within `duration_range`
  - No clip overlaps another (drop lower-scored)
- On validation failure: one retry with a stricter prompt, then 502.

### 5.3 `/render`

This is the biggest endpoint. Sub-stages:

1. **Receive video upload** (and/or look up `source_id` from cache; see §5.4).
2. **Per clip, in parallel where safe:**
   - **Face track:** Sample frames at ~5 fps (not native), run MediaPipe face detector, build a list of `(t, x, y)` face centers across the clip window.
   - **Smooth:** Apply an EMA (alpha ~0.2, ~500 ms window) to the face-center trajectory. Crop trajectory should never jump more than ~10% of frame width per second.
   - **Fallback:** If no face detected in > 30% of frames, fall back to fixed center crop. Log the fallback.
   - **Generate ASS subtitle:** From word timestamps in the transcript, build an `.ass` file with one event per word, highlighted-on-spoken style ("karaoke"). Use a single style preset per the `style` form field.
   - **ffmpeg one-pass:** Seek to `start`, encode `duration` seconds, apply a `crop+scale` to 1080x1920 (9:16) using the per-frame trajectory (use the `crop` filter with `sendcmd` for time-varying crop, or pre-compute and use `crop=w:h:x'(t):y'(t)`), burn in the ASS subtitle, encode with `h264_nvenc -preset p5 -tune hq -cq 23 -b:v 0` (or comparable quality target — defaults will look bad on Reels).

3. **Zip the resulting `clip_NN.mp4` files** to a temp file, then stream-respond.

ffmpeg specifics:
- Audio: copy if compatible (`-c:a copy`), else AAC `-c:a aac -b:a 128k`. Reels prefers AAC.
- Use `+faststart` (`-movflags +faststart`) so the moov atom is at the head — required for web playback.
- `-pix_fmt yuv420p` for compatibility.
- `nvenc` preset `p5` is the quality sweet spot at the time of writing. Defaults (`-preset default`) produce noticeably worse output than libx264 — the demo's visual impression depends on tuning this.

MediaPipe specifics:
- Run on CPU (`mediapipe.solutions.face_detection`). GPU detection is finicky and offers no win at the frame rates we use.
- Detection model selection: `1` (full range) for general; `0` (short range) for talking-head close-ups. Start with `1`.

Failure handling:
- If any single clip fails to render, do not abort the whole job. Log the failure, omit that clip from the zip, return the rest. Include a `manifest.json` inside the zip listing what's present and which clips failed.

### 5.4 `source_id` caching (strong recommendation)

**Without this:** every `/render` call re-uploads a 100–500 MB video over Tailscale (~30–90 s). If the user iterates on style, every iteration eats that cost.

**With this:**
- `/render` accepts `source_id` (string) as an optional form field.
- On first call with a video upload, compute `source_id = sha256(video_bytes)[:16]`, save under `work/sources/<source_id>.mp4`, return it in a custom response header `X-Source-Id`.
- On subsequent calls, the Mac client can send `source_id` and omit the `video` field. Server looks up the cached file. If missing, respond 410 Gone with `{"error": "source_expired"}` so the client knows to re-upload.

Cache eviction policy: LRU by file mtime, evict when `work/sources/` exceeds (say) 20 GB. Cap at 50 cached sources max.

Without `source_id` support, the Mac client will work but the iteration loop will be painful.

---

## 6. Error responses and status codes

| Status | When | Body |
|---|---|---|
| 200 | success | normal response |
| 400 | bad input (malformed JSON, missing required field, audio file unreadable) | `{"error": "bad_request", "detail": "..."}` |
| 409 | another job is in progress | `{"error": "busy", "detail": "current_job: <id>"}` — Mac client surfaces and exits; does NOT retry-loop |
| 410 | `source_id` referenced but evicted from cache | `{"error": "source_expired"}` — Mac client knows to re-upload |
| 422 | input shape valid but content rejected (e.g. audio is silent, transcript empty) | `{"error": "unprocessable", "detail": "..."}` |
| 500 | unexpected server error | `{"error": "internal", "detail": "..."}` — Mac retries once |
| 502 | downstream model returned garbage even after retry (Qwen invalid JSON) | `{"error": "model_failure", "detail": "..."}` |
| 503 | model not loaded, Ollama down, GPU not visible | `{"error": "unavailable", "detail": "..."}` — also used by `/health` when degraded |

Never return 200 with an error body. Never return a non-JSON body on error (especially not on `/render` — see §3.4).

---

## 7. Concurrency model

Single-tenant. Wrap each pipeline endpoint (`/transcribe`, `/rank`, `/render`) in a shared `asyncio.Lock`. If a request arrives while the lock is held by another job, respond **immediately with 409** — do not block, do not queue.

```python
job_lock = asyncio.Lock()

@app.post("/transcribe")
async def transcribe(...):
    if job_lock.locked():
        raise HTTPException(409, {"error": "busy", ...})
    async with job_lock:
        ...
```

`/health` must not take the lock — it must work concurrently with a running job to report `current_job` accurately.

---

## 8. Performance targets

For a 10-min 1080p source on a 12 GB GPU (4070-class):

| Stage | Target wall time | Notes |
|---|---|---|
| `/transcribe` | ≤ 30 s | faster-whisper-large-v3 fp16, VAD on |
| `/rank` | ≤ 10 s | Qwen 14B Q4_K_M, ~2k input tokens |
| `/render` (5 clips) | ≤ 60 s | NVENC, parallel where ffmpeg allows |
| Total server time | ≤ 100 s | excludes Mac→server upload |

If you're not hitting these, look in this order: NVENC preset, MediaPipe sample rate, WhisperX VAD config, Ollama keep_alive thrashing.

---

## 9. Filesystem layout

```
C:\services\clip-server\
├── server.py
├── app\
│   ├── transcribe.py
│   ├── rank.py
│   ├── reframe.py
│   ├── captions.py
│   └── render.py
├── work\
│   ├── jobs\<job_id>\          # per-job scratch, deleted on success
│   │   ├── input_audio.opus
│   │   ├── input_video.mp4
│   │   ├── clips.json
│   │   └── output.zip
│   └── sources\                # source_id cache (§5.4); LRU evicted
│       └── <source_id>.mp4
└── logs\
    ├── service.out.log
    └── service.err.log
```

`work\jobs\<job_id>\` is cleaned on job completion (success or failure, after response is sent). Keep failed jobs for 24 h for debugging; run an in-process asyncio sweeper (preferred) or a Task Scheduler entry — Windows has no cron.

`work\sources\` is the cross-job cache, evicted by LRU. Never deleted on job completion.

Use the system temp dir (`tempfile.gettempdir()` → `C:\Users\<user>\AppData\Local\Temp`) for ffmpeg scratch but never for cache. Don't hand-write `C:\Temp` or similar paths.

**Reminder from the Platform runbook:** exclude `C:\services\clip-server\work\` from Windows Defender real-time scanning, and always write large files via `.part` + rename — Windows refuses to delete or overwrite a file with an open handle.

---

## 10. Security posture

Tailnet-only is the access control. The service binds `0.0.0.0:8000` but Tailscale ACLs (or the lack of public routing) keep it private. This is acceptable for the demo. If the service ever needs to be exposed beyond Tailnet:

- Add bearer-token auth via an `Authorization: Bearer <token>` header.
- Token sourced from an env var `CLIP_SERVER_TOKEN`. Reject all requests without a matching token (including `/health` — or expose a separate unauthenticated `/ping` for liveness).
- The Mac client config has room for this; coordinate with the Mac side before adding.

Do not log audio or video content. Log job IDs, durations, error codes, and VRAM stats. That's it.

---

## 11. Observability

Minimum logging per job, structured (JSON lines is fine):

```json
{"ts": "...", "job_id": "...", "stage": "transcribe", "event": "start", "input_bytes": 2400000}
{"ts": "...", "job_id": "...", "stage": "transcribe", "event": "vram_pre", "free_mb": 11200}
{"ts": "...", "job_id": "...", "stage": "transcribe", "event": "end", "duration_s": 23.4, "vram_post_free_mb": 11180}
```

Key things to record:
- VRAM free before and after each stage. If `vram_post_free_mb` drifts down across jobs, you have a leak.
- Stage wall time. Compare to §8 targets.
- Ollama response time and validation pass/fail count.
- For `/render`: per-clip time, face-detection fallback count.

---

## 12. Open decisions you need to make (and confirm with Mac side)

| # | Decision | Why it matters | Default if unsure |
|---|---|---|---|
| 1 | Implement `source_id` caching for `/render`? | Removes re-upload cost on style iteration | **Yes, implement.** |
| 2 | Progress streaming for `/transcribe` and `/render`? (SSE or chunked) | Mac CLI shows a useless spinner otherwise | No initially. Mac can use a smarter spinner. Revisit if demo feels dead. |
| 3 | Filename padding width inside `/render` zip | Sort order | `clip_01.mp4` always, widen to 3 if N ≥ 100. |
| 4 | Auth token? | Future-proofing | No for demo. Plumb the env var in but don't enforce. |
| 5 | Diarization in `/transcribe`? | Multi-speaker support | No for demo. Single speaker only. |
| 6 | Max upload size? | Reverse-proxy timeouts | If running uvicorn directly, no limit. If nginx fronts it, raise `client_max_body_size 2g;` and `proxy_read_timeout 600s;`. |

Anything else you change in the contract (response shapes, status codes, field names) breaks the Mac client — coordinate before merging.

---

## 13. Testing without the Mac client

You can validate the whole service before the Mac client exists. Windows ships `curl.exe` (in `System32`) so the same commands work in PowerShell — but PowerShell's `curl` alias points at `Invoke-WebRequest`, which has different syntax. Always invoke `curl.exe` explicitly. `jq` is not bundled; either `scoop install jq` or use the Python one-liner alternative shown below.

```powershell
# 1. Health
curl.exe -s http://localhost:8000/health

# 2. Extract audio locally for testing
ffmpeg -i sample.mp4 -c:a libopus -b:a 32k -ac 1 -ar 16000 sample.opus

# 3. Transcribe
curl.exe -s -F "audio=@sample.opus" http://localhost:8000/transcribe -o segments.json

# 4. Rank (no jq required — build the request body in Python)
python -c "import json; s=json.load(open('segments.json')); json.dump({'transcript':s['segments'],'num_clips':3,'duration_range':[25,65]}, open('rank_req.json','w'))"
curl.exe -s -X POST -H "Content-Type: application/json" --data-binary "@rank_req.json" http://localhost:8000/rank -o clips.json

# 5. Render
python -c "import json; json.dump(json.load(open('clips.json'))['clips'], open('clips_arr.json','w'))"
curl.exe -s -X POST -F "video=@sample.mp4" -F "clips=<clips_arr.json" -F "style=viral" http://localhost:8000/render -o clips.zip

# 6. Inspect
python -c "import zipfile; [print(n) for n in zipfile.ZipFile('clips.zip').namelist()]"
```

After each stage, eyeball the output:
- `segments.json` — has `words` arrays populated
- `clips.json` — clips are sorted by score, non-overlapping, within duration range
- `clips.zip` — N mp4s, each 9:16, each shows the face roughly centered, each has burned captions

Then run the same job twice in a row. If the second one OOMs, you have a VRAM leak; see §4.

---

## 14. Out of scope (do not build)

- yt-dlp integration (Mac side)
- Local audio extraction (Mac side)
- Any UI
- MCP wrapper
- Instagram posting or social scheduling
- Multi-tenant queueing
- Cloud / non-NVIDIA fallback
- Authentication UX (token plumbing is fine; user-facing login flow is not)
- Cross-source clip mashups
- B-roll insertion
- Multi-speaker diarization

---

## 15. Quick checklist before declaring done

- [ ] `/health` returns rich status including Ollama and Whisper readiness
- [ ] All four endpoints implemented per §3
- [ ] Sequential VRAM load/free verified by running two jobs back-to-back without restart
- [ ] `OLLAMA_KEEP_ALIVE=0` set in NSSM service env (not just user shell)
- [ ] Concurrent job request returns 409, not blocked queue
- [ ] `/render` zip contains zero-padded filenames sorted by rank
- [ ] `/render` failure returns JSON 500, not partial zip
- [ ] `source_id` caching working end-to-end
- [ ] Per-stage VRAM logging in place
- [ ] NVENC preset tuned (do not ship defaults)
- [ ] NSSM service installed + Tailscale running + `tailscale ip -4` documented in README
- [ ] `work\` excluded from Windows Defender real-time scanning
- [ ] Verified second consecutive pipeline run does not OOM (proves VRAM sequencing + Ollama keep-alive)
- [ ] Smoke test in §13 passes from a clean boot

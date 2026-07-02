# MCP Server Roadmap

**Purpose:** Design roadmap for wrapping the clipping pipeline as an MCP server that Claude can drive. This is not a week-by-week schedule — it's a description of *how* to build it, the critical requirements, and the mistakes to avoid.

**Audience:** whoever builds the MCP layer (likely you) and the GPU-server dev who owns the box it will run on.

---

## 0. The one fact that reshapes everything

The Mac was the development machine. **The Mac is not part of the deployment.** Today all orchestration — the 5-stage sequencing, caching, retries, timeouts, per-run outputs, transcript writing — lives on the Mac in `run.py` + `clip_cli/`. The GPU server only exposes stateless `/transcribe`, `/rank`, `/render`.

So the MCP server **cannot** "wrap the Mac CLI." The orchestration brain has to be re-homed onto a machine that is actually in the deployment. Moving the files (scp them to the GPU box — no GitHub needed for a one-shot transfer to a single machine) is only how the code *travels* to that machine — it is not a runtime. Something has to *execute* the Python when a job runs, and that something is no longer the Mac.

**Baseline decision for this roadmap: the orchestration + MCP server run ON the GPU box.**
- The GPU box is already always-on and already runs the FastAPI server the orchestration calls.
- The Tailscale HTTP hops (`http://100.105.228.103:8000`) collapse into `http://localhost:8000` calls — faster, no upload over the wire, fewer failure modes.
- One machine to keep alive instead of two.

The alternative (a separate always-on host running the orchestration and talking to the GPU box over Tailscale, exactly like the Mac did) stays valid if you ever want to decouple from the GPU box. Everything below is written for the baseline; the deltas for the decoupled variant are called out where they matter.

---

## 1. Target architecture

```
                    ┌─────────────────────── GPU BOX (Windows, always-on) ───────────────────────┐
                    │                                                                              │
  [Claude client]   │   ┌──────────────┐      calls        ┌───────────────────────────────────┐  │
  (Desktop / Code / │   │  MCP server   │ ───────────────►  │  Orchestrator  (ex-clip_cli)      │  │
   web) ──MCP────────►  │  (tool layer) │                   │  locate → audio → transcribe →    │  │
                    │   └──────────────┘                    │  rank → render, caching, jobs     │  │
                    │          │                            └───────────────┬───────────────────┘  │
                    │          │ reads job state / artifacts                │ localhost HTTP        │
                    │          ▼                                            ▼                        │
                    │   ┌──────────────┐                    ┌───────────────────────────────────┐  │
                    │   │ job store +   │                    │  FastAPI server  :8000            │  │
                    │   │ artifact dir  │                    │  /health /transcribe /rank /render│  │
                    │   └──────────────┘                    │  + Ollama (Qwen) + ffmpeg + NVENC │  │
                    │                                        └───────────────────────────────────┘  │
                    └──────────────────────────────────────────────────────────────────────────────┘
```

Three logical components, all on one box:
1. **MCP server** — the thin tool-call surface Claude talks to. Owns nothing but request validation, job dispatch, and formatting results back to Claude.
2. **Orchestrator** — the re-homed `clip_cli` logic. Owns the 5-stage sequence, caching, retries, timeouts, job state, artifact writing.
3. **FastAPI worker** — unchanged. The MCP layer must not touch it directly; it goes through the orchestrator so all the hard-won retry/cache/timeout logic is reused, not reimplemented.

**Critical boundary rule:** the MCP server must NOT re-implement pipeline logic. It calls the orchestrator. If the MCP layer starts making raw `/rank` calls itself, you will fork the retry/cache/count-guarantee behavior and it will rot. One brain.

---

## 2. What to reuse vs. rewrite

The `clip_cli` code is mostly portable — it was deliberately kept as pure Python with no ML deps. Reuse it; don't rewrite the pipeline.

| Module | Verdict | Notes |
|---|---|---|
| `clip_cli/client.py` | **Reuse as-is** | `base_url` becomes `http://localhost:8000`. Bump `read`/`write` timeout to cover worst-case renders (news clip hit 21m46s; the 30-min ceiling is close — raise to 60 min). |
| `clip_cli/cache.py` | **Reuse** | The per-job artifact cache is exactly what makes re-runs cheap and recovers orphaned renders. Keep it. |
| `clip_cli/locate.py` | **Reuse, re-test** | yt-dlp on Windows works but check the download path handling. |
| `clip_cli/audio.py` | **Reuse, re-test** | ffmpeg `-f opus` extract. Verify the `.part` temp-file logic and ffmpeg path resolution on Windows. |
| `clip_cli/log.py` | **Reuse** | Per-run logs remain the primary debug interface. |
| `run.py` | **Refactor, don't call as CLI** | Its stage-orchestration *functions* (`_stage_locate/audio/transcribe/rank/render`, `_write_transcripts`, `_validate_clips`, `_unpack`) are the reusable core. Extract them into an importable `orchestrator.run_job(...)` that returns a result object. Drop the Click/argv layer — the MCP server calls the function directly. |
| `clip_cli/ui.py` | **Drop / replace** | `rich` console output is for a human at a terminal. The MCP layer emits structured data + progress, not ANSI. Replace with structured status updates. |

**The refactor that makes this work:** turn `run.py`'s procedural flow into one callable — `run_job(source, num_clips, duration_range, style, fit_mode, from_stage, job_id) -> JobResult` — that (a) is import-safe (no side effects at import, no `sys.exit`), (b) reports progress via a callback rather than printing, and (c) never raises past the boundary without a typed, serializable error. That single function is what both a future CLI *and* the MCP server call. Do this refactor first; everything else builds on it.

---

## 3. The MCP tool surface

Keep it small. Start with these:

| Tool | Shape | Purpose |
|---|---|---|
| `clip_video` | `(source, num_clips=5, duration=[30,600], style="viral", fit_mode="auto") -> {job_id, status}` | Kick off a job. Returns immediately with a `job_id` (see async section). Does NOT block. |
| `get_job` | `(job_id) -> {status, stage, progress, clips[], artifacts[], error?}` | Poll a job. Claude calls this to check progress and retrieve results. |
| `list_jobs` | `() -> [{job_id, source, status, created}]` | See recent/running jobs. |
| `get_health` | `() -> {status, current_job, gpu:{vram_free_mb, processes}}` | Pre-flight. Let Claude check the box is idle and uncontended before starting a long render. |

Later, cheap additions once you've used it: `cancel_job`, `render_alternative_style(job_id, style)`, `get_transcript(job_id)`.

**Tool design rules:**
- **Descriptions are the UX.** Claude picks tools and fills arguments from the descriptions. Spell out defaults, units (seconds), and the `duration` guidance (avoid tiny lower bounds — Qwen picks 3-second fragments; `[30,600]` is the good default, per the ops learnings). A vague description makes Claude call the tool wrong.
- **Validate and normalize at the tool boundary.** Reject a `duration` with min > max, clamp `num_clips` to a sane range, default `fit_mode` correctly. Return a clear error string, not a stack trace — Claude will read it and can self-correct.
- **Keep arguments flat and primitive.** Strings, numbers, small arrays. Don't make Claude construct nested objects.

---

## 4. The async problem (the single biggest design decision)

Renders take **minutes to tens of minutes** — the 46-min news source rendered in **21m46s**. A synchronous tool that blocks for 20 minutes is a broken UX: Claude sits frozen, the client may time out, and you can't check progress.

**Do not build `clip_video` as a blocking call.** Build it async from day one:

1. `clip_video` starts the job in a background worker, writes an initial job record, and returns `{job_id, status: "running"}` **immediately**.
2. The orchestrator runs the stages in the background, updating the job record's `stage`/`progress` as it goes (`locate → audio → transcribe → rank → render`).
3. Claude polls `get_job(job_id)` to watch progress and collect results when `status == "done"`.

This mirrors how the pipeline already thinks in cached stages — each stage completion is a natural progress checkpoint.

**Concurrency reality — the pipeline is single-tenant.** The GPU server holds a VRAM lock and returns **409 busy** if a second job arrives mid-flight. So:
- The orchestrator must run **one job at a time**. Queue additional `clip_video` calls, or reject them with a clear "busy, job X is running" message. Do NOT fire concurrent jobs at the FastAPI worker — you'll just collect 409s.
- `get_health.current_job` already tells you if the box is busy. Surface it.

**Fallback / heartbeat:** if the MCP runtime supports it, a long job should have a keepalive so the client doesn't consider the tool dead. If not, the async+poll design sidesteps it entirely — the kickoff call is fast and only `get_job` is ever in flight.

---

## 5. Getting the clips back to the user (the second non-obvious problem)

MCP is a **structured-data / text channel, not a file-download pipe.** The pipeline's output is binary: a zip of `clip_01.mp4 … clip_NN.mp4`. You cannot "return the mp4" through a tool result the way a website hands over a download.

Decide the delivery model explicitly:

- **Local MCP (stdio, runs where Claude runs):** the orchestrator writes clips to a known output dir and `get_job` returns **absolute file paths**. The user opens them from disk. Simple. But note: in the baseline the MCP server runs on the **GPU box**, so "local paths" are paths on the GPU box, not on the user's machine — only useful if the user is working on that box.
- **Remote MCP (the realistic case here):** the clips live on the GPU box; the user is elsewhere. You need a retrieval path:
  - Serve artifacts over HTTP from the box (a static `/artifacts/<job_id>/clip_01.mp4` route, Tailscale-only) and return **URLs** from `get_job`. Cleanest.
  - Or MCP **resources** — expose each clip as a resource the client can fetch. Good fit for MCP semantics if your client supports resource retrieval well.
  - Returning base64 blobs inline is possible for tiny files but **not** for multi-MB/-hundred-MB videos — don't.

**Recommendation:** add a read-only, Tailscale-scoped static file route on the GPU box and return artifact URLs. It reuses the network you already trust and keeps the tool results small (URLs + metadata, not bytes). Include per-clip metadata in `get_job` (rank, hook, score, start, end, duration, the transcript path) so Claude can talk about the clips without downloading them.

---

## 6. State and persistence

- **Reuse the existing `work/<job-id>/` cache** — it already survives restarts and already recovers orphaned renders via the render cache (Fix B). This is a feature, not scaffolding. Don't throw it away for an in-memory job dict.
- **Job records must be durable.** If the MCP process restarts mid-job, `get_job` should still find the record (status, stage, artifact paths). A tiny SQLite file or one JSON per job in the job dir is enough. In-memory-only state means a restart orphans every running job's visibility.
- **Job id vs. cache job-id:** the pipeline's `derive_job_id` (hash of source path+size+mtime) is what makes re-runs cache-hit. Keep using it as the cache key. The MCP-facing `job_id` can be the same value — that way asking to clip the same source twice naturally reuses cached stages.

---

## 7. Transport, deployment, auth

**stdio vs. remote — decide up front, it changes everything below.**
- **stdio** = MCP server launched by the client, talks over stdin/stdout, lives on the same machine as Claude. Simplest, zero network exposure, but then the *client machine* runs the orchestration — which contradicts "GPU box runs it" unless the user is literally on the GPU box.
- **remote (HTTP/SSE)** = MCP server runs as a long-lived service on the GPU box, client connects over the network. This is what matches "Mac gone, GPU box serves." **This is the recommended mode.**

**Deployment on the GPU box (Windows):**
- Run the MCP server as a managed service alongside FastAPI. The FastAPI/Ollama stack already uses **NSSM** (not systemd — this is Windows). Wrap the MCP server the same way so it survives reboots and logs consistently.
- Same environment discipline as the worker: `OLLAMA_KEEP_ALIVE=0` still critical, the `RANK_*` and `RENDER_*` tunables still apply — the MCP layer doesn't change any of that, it just triggers the stages.

**Auth:** today the contract is open-on-Tailnet. If the MCP endpoint is Tailscale-only, that may be acceptable to start. But an MCP server that can spend 20 minutes of GPU time per call is a real resource — put at least a bearer token on it before it's reachable by anything but you. The `client.py` already supports a bearer token; mirror that on the inbound side.

---

## 8. Windows portability — re-test these specific spots

The orchestration was written and run on macOS. Before it runs on the GPU box, verify (don't assume):
- **`ffmpeg` / `ffprobe` / `yt-dlp` resolution** — full paths or on `PATH`? The `subprocess` calls in `audio.py`/`locate.py` need the Windows executables found.
- **Audio extract** — the `-f opus` fix and the `.part` temp-file rename. Windows file-rename-over-open-handle semantics differ; verify the temp→final rename works.
- **Path handling** — any `/`-joined or POSIX-assuming paths. Use `pathlib` throughout (the code already leans on it — confirm no raw string paths slipped in).
- **The `uv`/venv setup** — the Tahoe `pyexpat` problem was macOS-specific; Windows has its own Python-env quirks. Pin the environment.
- **Long-render timeout** — with everything local, the upload disappears but renders still take 20+ min. Keep the client read timeout at 60 min.

---

## 9. Critical failure modes to carry over (do NOT rediscover these)

These are field-tested from the iteration cycle. The MCP layer must surface them cleanly, because Claude (and the user) can only react to what the tool result says.

| Failure | What the MCP layer must do |
|---|---|
| **GPU compute contention** (Enscape/Revit/other CUDA apps starve Qwen even with free VRAM → `/rank` times out) | `get_health` must expose `gpu.processes`. Encourage a pre-flight health check before long jobs. Surface "GPU contended by <app>" rather than a bare timeout. |
| **Ollama wedges** (health looks fine, `/rank` hangs with empty `ReadTimeout`) | Detect the wedge (rank timeout with VRAM math not adding up) and return an actionable error. Remember: Ollama is **user-mode, not a Windows service** — recovery is kill `llama-server.exe` + `ollama.exe`, restart `ollama serve`. `Restart-Service` does nothing. |
| **409 busy** (VRAM lock held) | Never retry-loop. Queue or reject with "job X running." Single-tenant is a hard constraint. |
| **410 source_expired** (source_id cache evicted) | Already auto-handled in `client.py` by re-upload fallback. Keep that path — don't strip it in the refactor. |
| **Orphan render on timeout** | The render cache (Fix B) already recovers this — an identical retry is a ~0.2s cache hit. Make sure the orchestrator's retry passes *identical* inputs so the cache key matches. |
| **Cold-load timeout** (Qwen 14B takes 3–4 min to load on first call) | Don't set the rank timeout below the load time. 600s server-side is calibrated. |
| **Empty error bodies** | The lesson holds: any new error path must surface the error *class* and *repr*, not just `str(e)`. Propagate the server's `err_class`/`err_repr`/traceback into the tool result. |
| **Cache keyed on params, not server behavior** | If the GPU dev ships a new rank prompt or render algorithm, the cache will hit stale results. Expose a way to force `from_stage` invalidation through a tool arg (e.g., `refresh_from="rank"`) so you can bust it without RDP'ing the box. |

---

## 10. Known-open items that intersect the MCP work

These are pre-existing pipeline TODOs; note them so the MCP layer accounts for them rather than papering over:
- **Caption burn-in bug** — captions are sent but not visible in output. Unresolved server-side. The MCP layer can't fix it; just don't claim captions work until it's verified.
- **`server_version` in `/health`** — if the GPU dev adds it, the orchestrator can auto-invalidate stale caches on server upgrade instead of manual `from_stage`. Wire `get_health` to read it.
- **Open-ended `duration_range: [N, null]`** — until the server supports it, keep sending `[N, 600]` as the "no upper bound" workaround, and document that in the `clip_video` tool description.
- **Per-clip transcripts should use absolute source timestamps** — if `_write_transcripts` gets touched during the refactor, apply this (it's the deferred feedback item). Since you're refactoring `run.py` anyway, this is the natural moment.

---

## 11. Build order (logical, not calendar)

1. **Refactor `run.py` into an importable `run_job(...)`** with a progress callback and typed results. No MCP yet. Prove it still works by calling it from a tiny script. *This is the foundation — do it first.*
2. **Move + green the orchestration on the GPU box.** scp the files over, fix the Windows portability spots (§8), point `client.py` at `localhost:8000`, run one full job end-to-end locally on the box. *Prove the Mac is truly unnecessary.*
3. **Add durable job state + background execution** (async worker + job store). Single-tenant queue. Prove `get_job`-style polling against the job store before any MCP is involved.
4. **Wrap it in the MCP server** — the 4 core tools, validation, health passthrough. Test with a real Claude client over stdio locally on the box.
5. **Solve artifact delivery** (§5) — static Tailscale route + artifact URLs in `get_job`.
6. **Promote to a remote service** — NSSM-managed, bearer token, reachable over Tailscale. Retest from a Claude client that is NOT on the box.
7. **Harden the failure surfaces** (§9) — make each known failure return an actionable tool result, not a raw timeout.

Each step is independently testable and leaves you with something that works. Don't build the MCP tools before step 2 proves the orchestration runs without the Mac — that's the whole point of the project.

---

## 12. Mistakes to avoid (summary)

- **Treating the file transfer as deployment.** scp moves the code; it doesn't run it. Code executes on the box, not by virtue of landing there.
- **Wrapping the Mac CLI.** The Mac is gone; wrap the *orchestrator function*, not `argv`.
- **Letting the MCP layer call `/rank` etc. directly.** One brain — go through the orchestrator so retry/cache/count-guarantee are reused, not forked.
- **A blocking `clip_video`.** 20-minute renders demand async + poll from day one.
- **Returning video bytes through MCP.** Return URLs/paths/resources; MCP is not a download pipe.
- **In-memory-only job state.** A restart must not blind you to running jobs.
- **Firing concurrent jobs.** Single-tenant VRAM lock → 409. Queue.
- **Assuming macOS code runs on Windows.** Re-test ffmpeg/yt-dlp/paths/audio-extract explicitly.
- **Swallowing errors into bare timeouts.** Surface `gpu.processes`, error class/repr, and the Ollama-wedge signature so failures are actionable.
- **Forgetting the cache is param-keyed.** Give yourself a `from_stage` bust so server-behavior changes don't serve stale clips.

---

Related memory: [[architecture]], [[api-contract]], [[mac-cli]], [[caching-resume]], [[operational-learnings]].

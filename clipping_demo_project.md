# AI Video Clipping Pipeline — Demo Project

A working recreation of the **"Best of Claude · Vol 20"** Instagram reel workflow: turn any long-form video — YouTube URL **or local file** — into ranked 9:16 short clips with burned-in captions, suitable for posting to Reels / Shorts / TikTok.

Demo target: produce 3-5 publishable clips from a 5-10 minute source video, end-to-end, fully offline after model download, with $0 in software costs.

---

## 1. Setup constraints

| Resource | Available | Implication |
|---|---|---|
| Dev machine | Mac, 8 GB RAM | Orchestrates the pipeline, plays back clips, runs the CLI |
| Compute server | NVIDIA GPU, 12 GB VRAM | Runs all ML inference + ffmpeg with NVENC |
| Transport between them | Tailscale (or LAN) | Same pattern as the Trijya 4090 broadcast pipeline |
| Internet | Required only for one-time model downloads | Pipeline runs offline after setup |
| Cost | $0 | All open source, no APIs, no SaaS |
| Instagram Business account | Not assumed | Auto-posting deferred — manual upload for demo |

**Net effect:** The Mac is the control plane. The GPU box does the actual work. The two communicate over a small FastAPI service — same architecture pattern used for the Wan diffusion broadcast video service. No API quotas, no upload caps, no rate limits.

### Hard limits to design around

- **VRAM budget = 12 GB.** Whisper-large + a 13B LLM at FP16 won't fit simultaneously. The pipeline loads/unloads sequentially: transcribe → free → rank → free → render.
- **Mac disk space.** Source videos and intermediate files can chew through 20-30 GB. Source files are uploaded to the GPU box for processing, clips are pulled back. Mac only holds inputs + final outputs.
- **GPU box availability.** If the Tailscale link is down, the pipeline can't run. Mitigation: pipeline is stateless per stage; resume from any cached JSON.
- **No autopost to Instagram.** Meta Graph API + Business account + FB app review is its own multi-day side quest. Out of scope — produce MP4s, post manually.

---

## 2. Architecture

```
┌─────────────────────────────┐                ┌──────────────────────────────────────┐
│       LOCAL (Mac)           │                │      GPU SERVER (12GB VRAM)          │
│                             │                │                                      │
│  CLI: run.py <url|path>     │                │  FastAPI service @ :8000             │
│                             │  Tailscale     │                                      │
│  ┌───────────────────────┐  │  HTTPS/HTTP    │  ┌────────────────────────────────┐ │
│  │ 1. download / locate  │──┼────────────────┼─▶│ POST /transcribe (audio file)  │ │
│  └───────────────────────┘  │                │  │   → faster-whisper-large-v3    │ │
│           │                 │                │  │   → WhisperX word alignment    │ │
│           │  audio.opus     │                │  │   → returns timestamped JSON   │ │
│           ▼                 │                │  └────────────────────────────────┘ │
│  ┌───────────────────────┐  │                │           │                          │
│  │ 2. POST /transcribe   │──┼────────────────┼──────────▶│                          │
│  └───────────────────────┘  │                │           ▼                          │
│           │                 │                │  ┌────────────────────────────────┐ │
│           │  segments.json  │                │  │ POST /rank (transcript)        │ │
│           ▼                 │                │  │   → Qwen 2.5 14B via Ollama    │ │
│  ┌───────────────────────┐  │                │  │   → JSON mode, virality prompt │ │
│  │ 3. POST /rank         │──┼────────────────┼─▶│   → returns top-N clips JSON   │ │
│  └───────────────────────┘  │                │  └────────────────────────────────┘ │
│           │                 │                │           │                          │
│           │  clips.json     │                │           ▼                          │
│           ▼                 │                │  ┌────────────────────────────────┐ │
│  ┌───────────────────────┐  │                │  │ POST /render (clips JSON +     │ │
│  │ 4. POST /render       │──┼────────────────┼─▶│        video file)             │ │
│  └───────────────────────┘  │                │  │   → MediaPipe face track       │ │
│           │                 │                │  │   → ffmpeg crop 9:16 + NVENC   │ │
│           │  zip of mp4s    │                │  │   → ASS karaoke captions       │ │
│           ▼                 │                │  │   → returns clip_NN.mp4 files  │ │
│  output/clip_01.mp4 ...     │                │  └────────────────────────────────┘ │
│                             │                │                                      │
└─────────────────────────────┘                └──────────────────────────────────────┘
```

Mac CLI is a thin orchestrator. Heavy work happens on the GPU box.

### Why this split

- **Reuses the Trijya pattern** — FastAPI on the GPU box behind Tailscale is already proven for video work.
- **The Mac stays responsive.** No GPU contention, no thermal issues, no 8GB RAM ceiling.
- **Stages are independently cacheable.** Re-render with different caption styles without re-transcribing. Re-rank with a different prompt without re-running Whisper.
- **Path to MCP wrapper is short.** The FastAPI service becomes a Claude-callable tool with a thin MCP shim later.

---

## 3. Tech stack

| Layer | Tool | Why | VRAM |
|---|---|---|---|
| Download | **yt-dlp** | Already installed | — |
| Audio extract | **ffmpeg** | `-c:a libopus -b:a 32k -ac 1 -ar 16000` | — |
| Transcription | **WhisperX** wrapping **faster-whisper-large-v3** | Word-level timestamps, optional diarization | ~5 GB |
| Moment ranking | **Qwen 2.5 14B Instruct** at Q4_K_M, served by **Ollama** | Best small model for structured JSON output | ~8 GB |
| Face tracking | **MediaPipe** (Python) | Fast, accurate, CPU-friendly | <1 GB |
| Reframe encode | **ffmpeg with h264_nvenc** | Hardware encode = 5-10× faster than libx264 | <1 GB |
| Captions | **ffmpeg** with `libass` | Word-level karaoke highlight |  — |
| GPU service | **FastAPI + Uvicorn** | Pattern user already runs in production |  — |
| Mac CLI | **Python + httpx + click** | Lean, async-friendly |  — |
| Transport | **Tailscale** | Already in user's stack |  — |

Stages run sequentially — `transcribe` releases VRAM before `rank` loads. 12 GB is enough headroom.

---

## 4. Setup — GPU server

### 4.1 System prep (one-time)

```bash
# Assumes Ubuntu 22.04+ with NVIDIA driver + CUDA 12.x already installed
nvidia-smi    # verify GPU visible
ffmpeg -version | grep nvenc   # verify NVENC support compiled in

sudo apt install -y python3.11 python3.11-venv ffmpeg
```

### 4.2 Ollama + Qwen model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama

# Pull the ranking model (~8.5 GB download)
ollama pull qwen2.5:14b-instruct-q4_K_M

# Test
ollama run qwen2.5:14b-instruct-q4_K_M "Return JSON: {\"ok\": true}"
```

### 4.3 Transcription + service environment

```bash
mkdir -p ~/services/clip-server && cd ~/services/clip-server
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Core ML
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper whisperx

# Service
pip install fastapi uvicorn[standard] python-multipart httpx
pip install mediapipe opencv-python-headless numpy

# Pre-download Whisper large-v3 weights (~3 GB)
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16')"
```

### 4.4 Service layout

```
~/services/clip-server/
├── .venv/
├── server.py               # FastAPI app, mounts the four endpoints
├── app/
│   ├── __init__.py
│   ├── transcribe.py       # WhisperX wrapper, manages VRAM
│   ├── rank.py             # Ollama client, ranking prompt
│   ├── reframe.py          # MediaPipe face track → crop trajectory
│   ├── captions.py         # ASS subtitle generator from word timestamps
│   └── render.py           # ffmpeg orchestration (crop + caption + NVENC)
├── work/                   # per-job scratch dir, auto-cleaned
└── systemd/clip-server.service
```

### 4.5 Run as a service

```ini
# /etc/systemd/system/clip-server.service
[Unit]
Description=Clip server (FastAPI)
After=network.target ollama.service

[Service]
Type=simple
User=yash
WorkingDirectory=/home/yash/services/clip-server
ExecStart=/home/yash/services/clip-server/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now clip-server
curl http://localhost:8000/health   # smoke test
```

### 4.6 Expose via Tailscale

```bash
sudo tailscale up
tailscale ip -4        # note the Tailscale IP — this is what the Mac uses
```

Service is now reachable at `http://<tailscale-ip>:8000` from the Mac.

---

## 5. Setup — Mac client

### 5.1 Project scaffold

```bash
brew install ffmpeg yt-dlp python@3.11

mkdir -p ~/projects/clip-demo && cd ~/projects/clip-demo
python3 -m venv .venv && source .venv/bin/activate
pip install httpx click rich
```

### 5.2 Configuration

```bash
cat > .env << 'EOF'
CLIP_SERVER_URL=http://100.x.x.x:8000   # Tailscale IP of the GPU box
EOF
```

### 5.3 CLI usage

```bash
python run.py "https://www.youtube.com/watch?v=<id>"
python run.py ~/Movies/my_unpublished_interview.mp4
python run.py ~/Downloads/podcast.mkv --num-clips 3 --style hormozi
```

Single command, accepts URLs or local paths transparently.

---

## 6. Pipeline contract (the JSON glue)

### `POST /transcribe`
Request: `multipart/form-data` with audio file (any format ffmpeg supports)
Response:
```json
{
  "language": "en",
  "duration_s": 612.4,
  "segments": [
    {
      "start": 0.0, "end": 4.2, "text": "So I was talking to...",
      "words": [
        { "word": "So", "start": 0.00, "end": 0.12, "score": 0.99 },
        { "word": "I",  "start": 0.13, "end": 0.18, "score": 0.99 }
      ]
    }
  ]
}
```

### `POST /rank`
Request: `{ "transcript": <segments>, "num_clips": 5, "duration_range": [25, 65] }`
Response:
```json
{
  "clips": [
    {
      "rank": 1, "start": 187.2, "end": 232.8, "duration": 45.6,
      "hook": "controversial take on AI training data",
      "score": 92,
      "reason": "strong opinion + clean intro + natural endpoint"
    }
  ]
}
```

### `POST /render`
Request: `multipart/form-data` with video file + JSON `{clips, style}`
Response: ZIP archive containing `clip_01.mp4` ... `clip_NN.mp4`

Stages are independently runnable. Cache each response to disk between runs while iterating.

---

## 7. The Qwen ranking prompt

```
You are a short-form video editor. You receive a timestamped transcript of a
long-form video. Return the TOP N moments that work as standalone 30-60 second
clips for TikTok / Reels / Shorts.

Criteria, priority order:
1. HOOK — does the clip start with something that stops a scroll?
2. SELF-CONTAINED — can a viewer understand it without watching the rest?
3. EMOTIONAL PEAK — strong opinion, surprise, conflict, revelation, laugh
4. CLEAN BOUNDARIES — starts/ends at a sentence break, not mid-word

Return STRICT JSON only. No prose. Schema:
{ "clips": [ { "rank":int, "start":float, "end":float,
               "hook":str, "score":int, "reason":str } ] }

Constraints:
- Each clip 25-65 seconds long
- No overlapping clips
- Sort by score descending
```

Call Ollama with `format: "json"` to lock structured output. Qwen 2.5 14B is reliable at this — far better than 8B at producing valid JSON without retries.

---

## 8. Demo script (what to show)

**Source:** Pick a 5-10 min interview / talk / podcast segment. Either a YouTube URL or a local MP4.

**Live flow (~2 minutes wall time, end to end):**

```bash
python run.py ~/Movies/sample_interview.mp4
```

Console output (designed for presentation):

```
[1/5] Locating source...                                    ✓ 8m 24s, 1080p
[2/5] Extracting audio (Opus mono 16kHz)...                 ✓ 3.8 MB
[3/5] Transcribing via GPU server (WhisperX large-v3)...    ✓ 24s
[4/5] Ranking top 5 moments (Qwen 2.5 14B)...               ✓ 6s

   #1  score 94   t=187-228   "controversial take on AI training data"
   #2  score 88   t=312-358   "story about a 3am Slack message"
   #3  score 81   t=421-465   "definition of 'taste' for engineers"
   #4  score 78   t=502-548   "the one habit she changed everything"
   #5  score 73   t=87-141    "why she quit her last job"

[5/5] Rendering 5 vertical clips with captions (NVENC)...   ✓ 52s

Done. 5 clips in output/ (total wall time: 1m 28s)
```

Open `output/clip_01.mp4` in QuickTime. Show the 9:16 frame, the face-tracked crop, the karaoke-style word-by-word captions.

### Demo numbers vs. previous (cloud-API) design

| Metric | Cloud API (Groq) | GPU server (12GB) | Win |
|---|---|---|---|
| End-to-end for 10-min video | ~5 min | **~1.5 min** | 3× faster |
| Upload size cap | 25 MB | none | ∞ |
| Ongoing cost | $0 (free tier) | $0 | tie |
| Works offline | no | **yes** | local |
| Concurrent jobs | rate-limited | as many as VRAM allows | local wins |
| Privacy of source video | uploaded to Groq | **never leaves Tailnet** | local wins |

The privacy win matters for the user's actual use case — unpublished video files don't get shipped to third-party APIs.

---

## 9. What's out of scope for the demo

| Out | Why | When to add |
|---|---|---|
| Auto-post to Instagram | Meta Graph API + Business account + FB app review | Phase 2 |
| Scheduling across platforms | Needs Postiz/Mixpost integration | Phase 2 |
| MCP wrapper around the FastAPI service | Demo doesn't need Claude-driven UX | Phase 2 |
| B-roll insertion | Complexity > demo value | Phase 3 |
| Multi-speaker diarization | WhisperX supports it but adds latency + VRAM | If targeting multi-host podcasts |
| Custom caption fonts / animation presets | One karaoke style is enough for demo | Polish phase |

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tailscale link down during demo | Low | Have last-run JSON cached; show pre-rendered clips as fallback |
| Qwen returns invalid JSON | Low | `format: "json"` + 1 retry with stricter prompt |
| Face tracker loses subject during pan | Medium | EMA smoothing over 0.5s window; fixed center crop fallback |
| Whisper mis-times short overlapping speech | Low | WhisperX word alignment handles this; pad clip ends by 200ms |
| GPU OOM if a second job kicks off | Low | Service uses an asyncio lock; reject concurrent /transcribe |
| YouTube blocks yt-dlp on the demo URL | Medium | Use a local file as the demo source — better UX anyway |

---

## 11. Effort estimate

| Task | Hours |
|---|---|
| GPU box: install stack, Ollama pull, smoke test | 2 |
| `server.py` + `/transcribe` endpoint | 2 |
| `/rank` endpoint + prompt tuning | 3 |
| `/render` endpoint (MediaPipe + ffmpeg + captions) | 5 |
| Mac CLI (`run.py`) + auth + retries | 2 |
| systemd service + Tailscale verify | 1 |
| End-to-end debug + polish + sample data | 3 |
| **Total** | **~18 hours / 2-3 evenings** |

Same total as the cloud-API design — the GPU server adds complexity but removes the chunking + rate-limit logic, so it washes out.

---

## 12. Stretch goals (post-demo)

1. **MCP server wrapper** — expose the four FastAPI endpoints as MCP tools so Claude can drive the pipeline conversationally, matching the reel UX. Roughly half a day.
2. **Postiz integration** — pipe rendered clips into Postiz's MCP server for cross-platform scheduling.
3. **Multi-style caption presets** — `--style hormozi|viral|podcast|minimal` flag with different ASS templates.
4. **Speaker-aware reframe** — Pyannote diarization + per-speaker crop centers for multi-host content.
5. **Web UI** — small Next.js frontend for the FastAPI service so non-technical users can drop a video and get clips back.
6. **Batch mode** — directory watcher: drop files into `inbox/`, get clips in `outbox/`.

---

## 13. References

- WhisperX (word-level timestamps + diarization): https://github.com/m-bain/whisperX
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Ollama: https://ollama.com/
- Qwen 2.5 14B model card: https://ollama.com/library/qwen2.5
- MediaPipe face detection: https://developers.google.com/mediapipe/solutions/vision/face_detector
- NVENC in ffmpeg: https://trac.ffmpeg.org/wiki/HWAccelIntro#NVENC
- ffmpeg ASS subtitle filter: https://ffmpeg.org/ffmpeg-filters.html#subtitles-1
- ClipsAI library (reference patterns): https://github.com/ClipsAI/clipsai
- The reel that started it: Instagram @Claude, "Best of Claude · Vol 20"

# Short Clip Cutter

Turns a long-form video (YouTube URL or local file) into ranked 9:16 vertical
short clips with burned-in captions, suitable for Reels / Shorts / TikTok.

This repository is the **client / control plane**. It runs a thin Python CLI that
orchestrates the pipeline but performs no ML itself. All heavy work (transcription,
ranking, rendering) runs on a separate **GPU server** exposed over HTTP.

## Architecture

```
CLI (this repo)  --HTTP-->  GPU server (:8000)
  locate                     /transcribe   WhisperX large-v3
  extract audio              /rank         Qwen 2.5 14B (Ollama)
  orchestrate                /render       ffmpeg + NVENC
  cache + output             /health
```

The pipeline runs in five sequential stages: `locate -> audio -> transcribe ->
rank -> render`. Each stage's output is cached per job so re-runs resume instead
of recomputing.

## Prerequisites

- Python 3.11+
- `ffmpeg` (local audio extraction)
- `yt-dlp` (only if sourcing from a URL)
- A reachable GPU server implementing the HTTP contract (see `gpu_server_spec.md`)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set the server URL:

```
CLIP_SERVER_URL=http://<server-host>:8000
# CLIP_SERVER_TOKEN=<token>   # only if the server requires bearer auth
```

## Usage

```bash
python run.py <url-or-path> [options]
```

Examples:

```bash
python run.py https://www.youtube.com/watch?v=VIDEO_ID
python run.py ./input/talk.mp4 --num-clips 3 --duration 30 60 --style hormozi
```

### Options

| Option | Default | Description |
|---|---|---|
| `--num-clips N` | `5` | Number of clips to extract. |
| `--duration MIN MAX` | `25 65` | Per-clip duration range, in seconds. |
| `--style NAME` | `viral` | Caption preset: `viral`, `hormozi`, `podcast`, `minimal`. |
| `--fit-mode MODE` | `auto` | 16:9 to 9:16 fit: `auto`, `crop`, `fit`, `stylized`. |
| `--from-stage STAGE` | — | Force re-run from `locate`/`audio`/`transcribe`/`rank`/`render` onward, ignoring cache. |
| `--job-id ID` | derived | Reuse a specific job id (default: hashed from the source). |
| `--output DIR` | `output` | Where final clips are written. |
| `--work DIR` | `work` | Per-job cache directory. |

## Output

Each run writes to `output/<run-id>/`:

- `clip_NN.mp4` — rendered vertical clips
- `clip_NN.txt` — human-readable transcript for the clip
- `clip_NN.json` — clip metadata (range, hook, score, segments)
- `transcripts.json` — combined index for the run

`output/latest` is a symlink to the most recent run.

## Caching and resume

Stage artifacts are cached in `work/<job-id>/` and survive across runs. A re-run
with the same source and parameters skips completed stages. The cache is keyed on
request parameters, not server behavior: if the GPU server changes how a stage
computes its result, invalidate that stage explicitly with
`--from-stage <stage>`.

## Related documents

- `gpu_server_spec.md` — HTTP contract the GPU server must implement.
- `mcp_server_roadmap.md` — plan for exposing the pipeline as an MCP server.

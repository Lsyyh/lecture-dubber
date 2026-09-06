# lecture-dubber

Local-first cross-language dubbing for lectures and educational videos. Input can be a local video file or a URL that `yt-dlp` can handle (including many YouTube/Bilibili pages). The pipeline downloads/imports media, runs WhisperX alignment, merges ASR fragments into translation units, translates through a local OpenAI-compatible Qwen server, synthesizes Mandarin with CosyVoice, duration-fits segments, writes subtitles, and renders a dubbed MP4.

> Use only media you are authorized to download/process. Site support changes frequently; URL import is a convenience layer, not a guaranteed crawler.

## Architecture

```text
local file / URL
  -> yt-dlp (URL only; also attempts subtitles)
  -> ffmpeg audio extraction
  -> WhisperX ASR + forced alignment
  -> 5-15 s semantic segment merger
  -> Qwen via OpenAI-compatible HTTP
  -> CosyVoice TTS
  -> duration-aware rewrite + light ffmpeg atempo
  -> timeline composition + Chinese SRT
  -> ffmpeg render
```

The repository intentionally avoids agent frameworks, databases, queues, and lip-sync. Every expensive stage is cached to disk and `dubber run` resumes from existing artifacts.

## Recommended deployment on RTX 3090

Use Ubuntu/Linux. Keep the orchestrator, ASR, LLM, and TTS dependencies separated when possible; CUDA/torch dependency conflicts are common in speech repos.

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git git-lfs sox libsox-dev
```

### 2. Main environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp config.example.yaml config.yaml
```

For a quick all-in-one experiment, install WhisperX in this environment:

```bash
pip install -e '.[asr]'
```

For a durable setup, run ASR in its own environment and later replace the in-process adapter with a subprocess/service if your torch stacks conflict.

### 3. Local Qwen server

The translator talks to an OpenAI-compatible `/v1/chat/completions` endpoint. One convenient option is vLLM:

```bash
pip install vllm
vllm serve Qwen/Qwen3-8B-Instruct --host 127.0.0.1 --port 8000
```

Adjust `llm_model` and `llm_base_url` in `config.yaml` to match your server. A 7B/8B-class instruct model is the intended starting point for a 24 GB 3090; use quantization or a smaller model if your local stack needs more headroom.

### 4. CosyVoice

Clone the official CosyVoice repo recursively and follow its current installation instructions in a dedicated environment if necessary:

```bash
mkdir -p third_party
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
```

Download a compatible model and set these paths in `config.yaml`:

```yaml
cosyvoice_root: third_party/CosyVoice
cosyvoice_model_dir: pretrained_models/CosyVoice-300M-SFT
cosyvoice_speaker: 中文女
```

`lecture-dubber` uses CosyVoice's current `AutoModel(...).inference_sft(...)` adapter. If the upstream API changes, only `src/lecture_dubber/tts.py` should need editing.

## Usage

Check the environment:

```bash
dubber doctor
```

Local video:

```bash
dubber run ./Stanford_CS229_Lecture01.mp4
```

URL import:

```bash
dubber run 'https://www.youtube.com/watch?v=...'
dubber run 'https://www.bilibili.com/video/BV...'
```

Download/import only:

```bash
dubber fetch 'https://www.youtube.com/watch?v=...' -o outputs/test-fetch
```

Use a fixed job directory:

```bash
dubber run VIDEO_OR_URL -o outputs/cs229-l01
```

Re-running the same job resumes from `segments.json`, `units.json`, and completed TTS files. To force recomputation:

```bash
dubber run VIDEO_OR_URL -o outputs/cs229-l01 --no-resume
```

### Real-time interpretation (Web UI)

```bash
dubber serve            # open http://127.0.0.1:8010
```

The landing page offers two modes. **Real-time** lists the processes that are
currently playing audio; pick one (e.g. an English livestream running in the
background) and the system captures its loopback, lowers its volume to a
whisper, and speaks a live Mandarin interpretation: streaming ASR -> clause
segmentation -> incremental translation -> Piper TTS. No video download or
processing involved. Session traces land in `outputs/realtime/<ts>/`.
Requires the `realtime` extras (`pip install -e '.[realtime]'`) and the local
Qwen server for translation.

**Offline** is the full dubbing pipeline described below.

## Job directory

```text
outputs/cs229-l01/
  state.json
  source/                 # URL downloads/subtitles, if applicable
  audio.wav
  segments.json           # WhisperX-aligned transcript
  units.json              # merged text + translations + duration metadata
  tts/
  tts_fit/
  zh.srt
  dub.wav
  final.zh.mp4
```

`units.json` is the most useful checkpoint for manual inspection. You can edit translations there and rerun after removing only the corresponding `tts_fit/*.wav` file.

## Translation policy

The prompt is optimized for lecture dubbing rather than literal subtitles. It uses previous/next context, a per-segment time budget, and `glossary.yaml`. If generated speech is >22% too long, the translation is compressed and synthesized again; smaller mismatch is corrected with light `ffmpeg atempo`.

Edit `glossary.yaml` before processing a course. This has a disproportionately large effect on technical lecture quality.

## URL import behavior

URL support comes from `yt-dlp`. The importer:

- tries one video rather than playlists;
- caps video height at `source_max_height`;
- requests English human/automatic subtitles when exposed by the site;
- keeps downloaded subtitle files for possible inspection;
- falls back cleanly to ASR for the actual pipeline.

There is deliberately no custom anti-bot bypass, DRM circumvention, account automation, or site-specific scraper logic. If `yt-dlp` cannot access a source, download it manually and pass the local file path.

## Known limitations in v0.1

- single-speaker lecture workflow; no diarization;
- no lip-sync;
- the downloaded source subtitle is not yet used as an ASR replacement because subtitle quality/format varies greatly across sites;
- CosyVoice is invoked in-process, so a conflicting torch/CUDA environment may require a small subprocess adapter;
- `ffmpeg subtitles` requires a build with libass;
- URL extractor support is inherently brittle because sites change.

## Why this design

The hard part of lecture dubbing is semantic segmentation and duration control, not orchestration. The code therefore keeps the model boundaries thin and makes the timeline/cached artifacts explicit. This is easier to debug and extend than an agent framework.

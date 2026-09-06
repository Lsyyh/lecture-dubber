# RealtimeDub — Coding Agent Implementation Plan

## 0. Project Goal

Build a local, GPU-accelerated real-time speech interpretation system for desktop media.

Primary use case:

> Capture audio from a browser/player/application, suppress or route away the original audio, perform streaming ASR → incremental English-to-Chinese translation → streaming TTS, and play the Chinese interpretation with stable low latency.

Target environment:

- OS: Windows 10/11 first
- GPU: NVIDIA RTX 3090 24 GB
- Python: 3.10/3.11
- CUDA-capable PyTorch environment available
- Primary language pair: English → Simplified Chinese
- Primary content: Stanford/MIT lectures, YouTube/Bilibili technical videos, online talks
- Deployment: single-machine local application
- Internet connection must not be required after models are downloaded

This is a **stream-processing system**, not an Agent framework project.

Do not introduce LangChain, CrewAI, Dify, message brokers, databases, Kubernetes, or distributed infrastructure unless later requirements prove they are necessary.

---

# 1. Product Definition

The finished V1 should support three modes:

```bash
# Capture an audio device / virtual cable
realdub live

# Capture a selected application/process where supported
realdub live --process chrome.exe

# Existing offline mode may remain available
realdub file lecture.mp4
realdub url "https://..."
```

The primary feature is `realdub live`.

Expected user experience:

1. User plays an English lecture/video.
2. Original application audio is muted or routed to a virtual device.
3. RealtimeDub captures the source audio.
4. Chinese interpretation begins approximately 1.5–3 seconds later.
5. Interpretation continues indefinitely without latency accumulating.
6. Technical terminology remains consistent.
7. If processing falls behind, the system automatically compresses translations and/or slightly increases TTS speed.

The system must prioritize:

1. bounded latency;
2. intelligibility;
3. semantic correctness;
4. terminology consistency;
5. natural TTS;
6. literal translation fidelity.

Literal sentence-by-sentence translation is explicitly NOT the goal.

---

# 2. Success Metrics

Target V1 metrics for an RTX 3090:

| Metric | Target |
|---|---:|
| Median end-to-end latency | < 2.5 s |
| P95 end-to-end latency | < 4.0 s |
| Processing real-time factor | < 0.7 |
| Long-session lag | bounded; must not continuously increase |
| Continuous run | >= 60 min without crash |
| Audio underrun / overlap | rare / none in normal lecture speech |
| ASR partial revision | minimized |
| Translation terminology consistency | stable for session |
| TTS output | understandable, no severe discontinuity |

Latency means:

```text
source speech timestamp
    →
first corresponding audible Chinese speech
```

Track latency explicitly in logs.

---

# 3. Non-Goals for V1

Do NOT spend time on:

- lip synchronization;
- video manipulation;
- voice cloning of the original speaker;
- multi-speaker diarization;
- multilingual auto-detection beyond basic source language config;
- web UI;
- cloud deployment;
- mobile deployment;
- browser extension;
- subtitle rendering;
- training/fine-tuning models;
- complex site-specific YouTube/Bilibili scraping;
- perfect simultaneous translation research implementation.

The V1 must first prove reliable real-time audio interpretation.

---

# 4. High-Level Architecture

```text
┌──────────────────────┐
│ Browser / VLC / App  │
└──────────┬───────────┘
           │ application/device audio
           ▼
┌──────────────────────┐
│ Audio Capture Layer  │
│ WASAPI / virtual dev │
└──────────┬───────────┘
           │ 20–100 ms PCM frames
           ▼
┌──────────────────────┐
│ Ring Buffer + VAD    │
└──────────┬───────────┘
           │ voiced chunks
           ▼
┌──────────────────────┐
│ Streaming ASR        │
└──────────┬───────────┘
           │ partial hypotheses
           ▼
┌──────────────────────┐
│ Stable Prefix Buffer │
└──────────┬───────────┘
           │ committed source text
           ▼
┌──────────────────────┐
│ Clause Segmenter     │
└──────────┬───────────┘
           │ semantic clauses
           ▼
┌──────────────────────┐
│ Incremental MT       │
└──────────┬───────────┘
           │ committed Chinese text
           ▼
┌──────────────────────┐
│ Latency Controller   │
└──────────┬───────────┘
           │ text + speed policy
           ▼
┌──────────────────────┐
│ Streaming / Chunk TTS│
└──────────┬───────────┘
           │ PCM chunks
           ▼
┌──────────────────────┐
│ Jitter Buffer        │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Audio Playback       │
└──────────────────────┘
```

The processing path must use queues and explicit timestamps.

Every intermediate event should carry:

```python
timestamp_start
timestamp_end
created_at
sequence_id
```

Do not pass plain strings between workers when timing metadata is available.

---

# 5. Recommended Repository Structure

```text
realdub/
├── pyproject.toml
├── README.md
├── PLAN.md
├── config.example.yaml
├── src/
│   └── realdub/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── logging.py
│       │
│       ├── audio/
│       │   ├── capture.py
│       │   ├── playback.py
│       │   ├── devices.py
│       │   ├── ring_buffer.py
│       │   └── resample.py
│       │
│       ├── vad/
│       │   ├── base.py
│       │   └── funasr_vad.py
│       │
│       ├── asr/
│       │   ├── base.py
│       │   ├── paraformer_streaming.py
│       │   └── whisper_streaming.py
│       │
│       ├── text/
│       │   ├── stable_prefix.py
│       │   ├── clause_segmenter.py
│       │   └── normalization.py
│       │
│       ├── translation/
│       │   ├── base.py
│       │   ├── qwen.py
│       │   ├── nllb.py
│       │   ├── glossary.py
│       │   └── prompts.py
│       │
│       ├── tts/
│       │   ├── base.py
│       │   └── cosyvoice.py
│       │
│       ├── runtime/
│       │   ├── events.py
│       │   ├── queues.py
│       │   ├── pipeline.py
│       │   ├── latency.py
│       │   └── metrics.py
│       │
│       └── offline/
│           └── ...
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── scripts/
│   ├── bootstrap_windows.ps1
│   ├── download_models.ps1
│   └── benchmark.py
│
└── examples/
    ├── config_low_latency.yaml
    └── glossary_cs.yaml
```

Keep model adapters isolated from orchestration logic.

---

# 6. Core Data Contracts

Define typed dataclasses or Pydantic models.

## 6.1 AudioFrame

```python
@dataclass
class AudioFrame:
    seq: int
    pcm: np.ndarray
    sample_rate: int
    start_ts: float
    end_ts: float
```

## 6.2 ASRHypothesis

```python
@dataclass
class ASRHypothesis:
    seq: int
    text: str
    start_ts: float
    end_ts: float
    is_final: bool
    produced_at: float
```

## 6.3 CommittedText

```python
@dataclass
class CommittedText:
    seq: int
    text: str
    source_start_ts: float
    source_end_ts: float
    committed_at: float
```

## 6.4 TranslationUnit

```python
@dataclass
class TranslationUnit:
    seq: int
    source_text: str
    translated_text: str
    source_start_ts: float
    source_end_ts: float
    translation_mode: str
    produced_at: float
```

## 6.5 SpeechChunk

```python
@dataclass
class SpeechChunk:
    seq: int
    pcm: np.ndarray
    sample_rate: int
    source_start_ts: float
    source_end_ts: float
    generated_at: float
```

---

# 7. Audio Capture

## 7.1 V0 Implementation

First implement audio capture using:

- `sounddevice` or equivalent PortAudio-based backend;
- selectable input device;
- support for virtual audio cable workflow.

User workflow:

```text
Chrome/VLC output
    →
Virtual Audio Cable Input
    →
RealtimeDub captures corresponding virtual output
```

TTS playback must use a separate physical output device.

This is the simplest reliable V0.

## 7.2 V1 Windows Process Capture

After device-based capture is stable, add optional Windows-specific process loopback capture.

Desired CLI:

```bash
realdub devices
realdub processes
realdub live --process chrome.exe
```

Do not block the project on per-process capture.

If implementation is difficult or brittle, retain virtual-device capture as the supported production path.

## 7.3 Audio Format

Normalize internally to:

```text
mono
16 kHz
float32 or int16
```

Capture frames should be approximately:

```text
20–100 ms
```

Do not submit each tiny frame directly to ASR.

Use ring buffers/chunks.

---

# 8. VAD

Use streaming VAD before ASR where beneficial.

Preferred initial implementation:

```text
FunASR FSMN-VAD
```

VAD responsibilities:

- detect speech onset;
- detect likely speech offset;
- avoid sending long silence;
- provide soft utterance boundaries;
- never introduce excessive delay.

Important:

Do not wait for full silence to perform ASR.

Streaming ASR must continue while speech is active.

---

# 9. Streaming ASR

## 9.1 Primary Backend

Implement:

```text
FunASR Streaming Paraformer
```

as the first backend.

Reasons:

- designed for streaming;
- relatively lightweight;
- stable chunk-based decoding;
- good latency characteristics;
- suitable for a 3090.

Do not optimize model selection before benchmark data exists.

## 9.2 Secondary Backend

Add an optional second adapter:

```text
Whisper / SimulStreaming-style backend
```

Only after the primary pipeline works.

The ASR interface must make backend replacement trivial.

## 9.3 Required ASR Behavior

The streaming ASR adapter must emit partial hypotheses repeatedly.

Example:

```text
t0:
"The gradient"

t1:
"The gradient of"

t2:
"The gradient of this loss"

t3:
"The gradient of this loss function"
```

Do not wait for a complete sentence.

---

# 10. Stable Prefix / Commitment Logic

This is a critical component.

Implement a `StablePrefixBuffer`.

Purpose:

- compare consecutive ASR hypotheses;
- determine text that is sufficiently stable;
- commit stable text;
- never revise already committed text;
- retain unstable suffix for future ASR revisions.

Minimal strategy:

1. normalize punctuation/case;
2. tokenize;
3. compare current and previous N hypotheses;
4. compute longest common prefix;
5. only commit tokens unchanged across a configurable number of hypotheses;
6. optionally require minimum token age.

Example:

```text
H1: the gradient of this loss
H2: the gradient of this loss function
H3: the gradient of this loss function is

Stable:
the gradient of this loss

Commit:
the gradient of this loss
```

Configuration:

```yaml
stable_prefix:
  agreement_window: 2
  minimum_tokens: 2
  max_uncommitted_age_ms: 1200
```

Add unit tests covering revision edge cases.

---

# 11. Clause Segmentation

Do not translate individual ASR tokens.

Do not wait for sentence punctuation.

Implement an incremental clause segmenter.

Emit a clause when one or more conditions are met:

- punctuation indicates semantic boundary;
- pause detected;
- source text exceeds target token/word count;
- conjunction boundary is acceptable;
- maximum waiting time exceeded.

Target source clause duration:

```text
~1–4 seconds
```

Typical result:

```text
"The reason gradient descent diverges here"
"is because the learning rate"
"is simply too large"
```

Avoid extremely short fragments unless latency is already high.

---

# 12. Translation

## 12.1 Translation Objective

Translation must be optimized for speech interpretation:

- semantically faithful;
- concise;
- spoken Mandarin;
- preserve technical meaning;
- drop unnecessary English filler;
- obey glossary;
- do not explain or expand;
- prefer short constructions;
- minimize future latency.

Example:

```text
"So what we're going to do here is take the derivative..."

Bad:
“所以我们接下来在这里要做的是求这个表达式的导数……”

Good:
“接下来求这个表达式的导数……”
```

## 12.2 Backend

Primary implementation:

```text
small Qwen instruct model
```

Run locally using one of:

- Transformers;
- vLLM;
- llama.cpp-compatible backend if practical.

Avoid requiring HTTP if direct inference is simpler.

Keep a clean adapter so backend can later change.

A dedicated MT model such as NLLB may be implemented as a benchmark backend.

## 12.3 Context

Translator receives:

```text
previous committed source clause
current source clause
recent translated context
session glossary
latency mode
```

Do not send the entire lecture transcript on every call.

Use a bounded context window.

Example:

```python
TranslatorInput(
    previous_source="...",
    current_source="...",
    recent_target="...",
    glossary={...},
    mode="normal",
)
```

## 12.4 Output Constraint

Return JSON or structured output:

```json
{
  "translation": "接下来求这个表达式的导数。",
  "terms": []
}
```

Robustly recover from malformed model output.

---

# 13. Session Glossary

Implement session-level terminology support.

Config file example:

```yaml
glossary:
  gradient descent: 梯度下降
  learning rate: 学习率
  objective function: 目标函数
  loss function: 损失函数
  likelihood: 似然
  log likelihood: 对数似然
  embedding: embedding
  token: token
  transformer: Transformer
```

Optional future feature:

- infer candidate terms from the first few minutes;
- allow manual acceptance.

Do not make glossary extraction part of V1 critical path.

---

# 14. Latency Controller

This is a first-class component.

Track:

```python
lag = latest_source_audio_ts - latest_played_source_ts
```

Do not estimate lag only from queue length.

Define operating modes.

Example:

```yaml
latency:
  target_seconds: 2.5
  warning_seconds: 4.0
  critical_seconds: 6.0
```

Policy:

```text
lag < 2.5 s:
    mode = NORMAL
    natural translation
    tts_speed = 1.00

2.5–4.0 s:
    mode = CONCISE
    shorter translation
    tts_speed ≈ 1.05–1.10

4.0–6.0 s:
    mode = AGGRESSIVE
    remove fillers
    strongly compress wording
    tts_speed ≈ 1.10–1.15

lag > 6.0 s:
    mode = RECOVERY
    translate core meaning only
    combine/drop low-information fragments
    tts_speed <= 1.20
```

Translation mode must be passed to the translator.

The system must never solve lag by indefinitely buffering.

---

# 15. TTS

Primary backend:

```text
CosyVoice
```

Requirements:

- fixed Chinese lecturer voice by default;
- no voice cloning in V1;
- support incremental/chunk synthesis if stable;
- otherwise synthesize clause-level chunks;
- allow speed adjustment;
- avoid obvious prosody discontinuity.

Preferred synthesis unit:

```text
roughly 4–20 Chinese characters
or one short semantic clause
```

Do not synthesize token-by-token.

The TTS adapter should expose:

```python
async def synthesize(
    text: str,
    speed: float = 1.0,
) -> AsyncIterator[AudioChunk]:
    ...
```

Even if the first implementation internally returns one full chunk.

---

# 16. Playback / Jitter Buffer

Implement a bounded playback queue.

Responsibilities:

- prevent tiny gaps between generated chunks;
- prevent overlapping chunks;
- preserve sequence order;
- expose played source timestamps for lag computation.

Suggested target buffer:

```text
100–300 ms
```

Do not let the playback buffer become an uncontrolled latency sink.

If a chunk arrives too late:

- play immediately;
- do not add artificial spacing unless required for intelligibility.

---

# 17. Concurrency Model

Use `asyncio` for orchestration.

Heavy model inference may use:

- dedicated worker tasks;
- model-specific execution thread;
- separate processes if required by CUDA/runtime conflicts.

Initial pipeline:

```text
capture_task
    ↓ asyncio.Queue
vad/asr_task
    ↓
stable_prefix_task
    ↓
translation_task
    ↓
tts_task
    ↓
playback_task
```

All queues must have bounded `maxsize`.

Backpressure behavior must be explicit.

Avoid unbounded queues.

Example:

```python
audio_q = asyncio.Queue(maxsize=64)
text_q = asyncio.Queue(maxsize=16)
translation_q = asyncio.Queue(maxsize=8)
tts_q = asyncio.Queue(maxsize=8)
```

If a queue is saturated, log it and invoke recovery policy rather than silently accumulating infinite latency.

---

# 18. GPU Strategy

V1 goal:

Keep all primary models resident simultaneously if possible.

Target memory budget for RTX 3090:

```text
ASR          small
Translator   3B–4B quantized or compact model
TTS          ~0.5B class
Runtime      remaining VRAM
```

Do not unload/reload models between streaming stages.

At startup print:

```text
GPU
VRAM total
model names
allocated VRAM
```

Optional later optimization:

- separate CUDA streams;
- quantized translator;
- TensorRT;
- ONNX;
- torch.compile.

Do not optimize before profiling.

---

# 19. Configuration

Example `config.yaml`:

```yaml
source_language: en
target_language: zh

audio:
  input_device: null
  output_device: null
  sample_rate: 16000
  frame_ms: 40

vad:
  backend: funasr
  enabled: true

asr:
  backend: paraformer
  model: paraformer-streaming
  chunk_size: [0, 10, 5]

stable_prefix:
  agreement_window: 2
  max_uncommitted_age_ms: 1200

translation:
  backend: qwen
  model: Qwen-small
  max_context_clauses: 4

tts:
  backend: cosyvoice
  voice: default_zh_lecturer
  base_speed: 1.0

latency:
  target_seconds: 2.5
  warning_seconds: 4.0
  critical_seconds: 6.0

logging:
  level: INFO
  save_session_trace: true
```

Do not hard-code model paths throughout the codebase.

---

# 20. CLI

Use `typer`.

Required commands:

```bash
realdub doctor
realdub devices
realdub live
realdub benchmark
```

Optional:

```bash
realdub processes
realdub file
realdub url
```

Examples:

```bash
realdub live \
  --input-device "CABLE Output" \
  --output-device "Headphones"
```

and later:

```bash
realdub live --process chrome.exe
```

`doctor` should verify:

- Python version;
- ffmpeg if offline mode enabled;
- CUDA;
- torch CUDA availability;
- GPU name;
- input/output audio devices;
- required model paths;
- optional dependency availability.

---

# 21. Logging and Session Trace

Each live run should optionally save a compact trace:

```text
runs/2026-xx-xx_xxxxxx/
├── config.yaml
├── events.jsonl
├── transcript.jsonl
├── translations.jsonl
└── metrics.json
```

Do NOT save raw captured audio by default.

Event example:

```json
{
  "type": "translation",
  "seq": 42,
  "source_start_ts": 101.3,
  "source_end_ts": 104.1,
  "source": "the learning rate is simply too large",
  "target": "学习率实在太大了",
  "mode": "normal",
  "latency_ms": 2110
}
```

This trace is essential for profiling and debugging.

---

# 22. Benchmark Harness

Create a repeatable benchmark script.

Input:

- 5–10 minute lecture audio file;
- optional reference transcript.

Simulate streaming by feeding audio frames in real-time or controlled accelerated mode.

Measure:

```text
ASR first partial latency
ASR stable commit latency
translation latency
TTS first-audio latency
end-to-end latency
queue sizes over time
real-time factor
translation length ratio
lag curve
```

Output:

```text
benchmark_results.json
benchmark_summary.md
```

The most important graph is:

```text
lag_seconds vs wall_clock_time
```

A successful system has bounded lag.

A failed system has monotonically increasing lag.

---

# 23. Testing Strategy

## 23.1 Unit Tests

Required:

- ring buffer;
- stable-prefix logic;
- clause segmentation;
- latency policy transitions;
- translation JSON parser;
- queue overflow behavior;
- timestamp propagation.

## 23.2 Integration Tests

Use mocked ASR/translation/TTS adapters.

Test:

```text
synthetic audio frames
→ ASR mock
→ stable prefix
→ translation mock
→ TTS mock
→ playback mock
```

Verify:

- event order;
- timestamps;
- bounded queues;
- graceful shutdown;
- no deadlock.

## 23.3 Hardware Test

Manual acceptance test on RTX 3090:

- 10 min Stanford lecture;
- 60 min continuous lecture;
- fast speaker segment;
- silence;
- music-only section;
- application pause/resume.

---

# 24. Error Handling

Pipeline must survive transient component failures.

Examples:

### ASR error

- log;
- reset ASR state if needed;
- continue capture.

### Translation timeout

- retry once;
- fall back to shorter context;
- if still failing, skip clause rather than block the entire stream.

### TTS failure

- retry once;
- skip failed chunk if necessary;
- continue.

### Audio device disconnect

- terminate live pipeline gracefully with clear error.

Never allow one failed clause to crash a 1-hour session.

---

# 25. Graceful Shutdown

Ctrl+C must:

1. stop accepting new audio;
2. stop capture device;
3. optionally flush already committed translation/TTS;
4. stop playback;
5. save metrics;
6. release models/audio devices;
7. exit without hanging.

Implement shutdown behavior early.

---

# 26. Implementation Phases

## Phase 0 — Repository and Contracts

Goal:

Establish architecture before using real models.

Tasks:

- create project structure;
- configure `pyproject.toml`;
- add CLI;
- add config loader;
- add typed event/data models;
- add bounded queue abstraction;
- add logging;
- add test suite.

Acceptance:

```bash
pytest
realdub doctor
```

run successfully.

---

## Phase 1 — Audio Loopback Prototype

Goal:

Capture audio and play it through the pipeline without AI.

Implement:

```text
input device
→ ring buffer
→ output device
```

Add configurable delay.

Acceptance:

- route browser audio to virtual cable;
- RealtimeDub captures it;
- audio can be replayed through headphones;
- stable for >= 20 minutes;
- no uncontrolled memory growth.

Do not proceed until this works reliably.

---

## Phase 2 — Streaming ASR

Goal:

Real-time English transcript.

Implement:

- VAD;
- Paraformer streaming adapter;
- partial hypothesis events;
- CLI transcript output.

Acceptance:

- transcript appears while speaker is talking;
- no need to wait for sentence completion;
- measure ASR latency;
- run 10 min lecture without crash.

---

## Phase 3 — Stable Prefix + Clause Commit

Goal:

Produce irreversible source clauses suitable for translation.

Implement:

- hypothesis agreement logic;
- committed/uncommitted buffers;
- clause segmentation;
- unit tests.

Acceptance:

Terminal output should resemble:

```text
PARTIAL: the gradient of this ...
COMMIT:  the gradient of this loss
CLAUSE:  the gradient of this loss function
```

Committed text must not later change.

---

## Phase 4 — Incremental Translation

Goal:

Output concise Chinese clauses.

Implement:

- Qwen adapter;
- structured prompts;
- glossary;
- bounded recent context;
- translation modes.

Acceptance:

For a technical lecture:

- technical meaning preserved;
- fillers commonly removed;
- terminology stable;
- per-clause inference fast enough for streaming.

Record latency.

---

## Phase 5 — TTS + Playback

Goal:

Hear continuous Chinese interpretation.

Implement:

- CosyVoice adapter;
- chunk/clause synthesis;
- playback queue;
- jitter buffer;
- output-device selection.

Acceptance:

```text
English source
→ Chinese voice audible
```

with no source video modification.

Initial latency < 5 seconds is acceptable at this phase.

---

## Phase 6 — Latency Controller

Goal:

Prevent lag accumulation.

Implement:

- lag tracking;
- NORMAL / CONCISE / AGGRESSIVE / RECOVERY modes;
- dynamic translation prompt;
- dynamic TTS speed;
- queue monitoring.

Acceptance:

Run a 30 min lecture.

Lag must not monotonically increase.

Target:

```text
median < 2.5–3.0 s
P95 < 4–5 s
```

---

## Phase 7 — Production Hardening

Implement:

- session traces;
- metrics;
- graceful recovery;
- config presets;
- Windows bootstrap script;
- better `doctor`;
- 60 min endurance test.

Acceptance:

1-hour technical lecture completes without manual intervention.

---

## Phase 8 — Optional Process Audio Capture

Only after V1 works.

Implement Windows process-level loopback capture if feasible.

Desired:

```bash
realdub live --process chrome.exe
```

Also optionally mute source application session automatically.

If this becomes disproportionately complex, document virtual-audio-cable routing as the supported production method and leave process capture experimental.

---

# 27. Recommended Coding Order

Coding agent should follow this order strictly:

```text
1. project skeleton
2. core dataclasses/events
3. audio device enumeration
4. capture/playback loop
5. bounded queues
6. mocked streaming pipeline
7. real streaming ASR
8. stable prefix
9. clause segmenter
10. translation adapter
11. glossary/context
12. TTS adapter
13. jitter buffer
14. latency controller
15. metrics/trace
16. benchmark harness
17. endurance testing
18. process capture
```

Do not begin by integrating all three real AI models simultaneously.

---

# 28. Engineering Rules for the Coding Agent

1. Keep each model behind an interface.
2. Keep orchestration independent from model implementation.
3. Every queue must be bounded.
4. Every event must preserve source timestamps.
5. Never introduce sleeps as a synchronization mechanism unless explicitly implementing playback timing.
6. Do not swallow exceptions silently.
7. Do not use global mutable state for session pipeline state.
8. Keep configuration external.
9. Prefer simple Python over framework abstractions.
10. Write tests before changing commitment/latency logic.
11. Preserve offline mode only if it does not complicate the live core.
12. Never block the audio capture callback with model inference.
13. Never perform heavy work directly inside the audio callback.
14. Avoid premature GPU optimization.
15. Profile before optimizing.
16. Favor reliability over clever abstractions.
17. All model-specific dependencies should be optional extras when practical.

---

# 29. Dependency Strategy

Preferred core dependencies:

```text
typer
pydantic
pyyaml
numpy
sounddevice
rich
pytest
```

Model dependencies should be isolated.

Example extras:

```toml
[project.optional-dependencies]
asr = [...]
translate = [...]
tts = [...]
dev = [...]
```

If FunASR/CosyVoice dependency conflicts become severe, use separate subprocess workers with a simple local IPC protocol instead of forcing one Python environment.

IPC can initially be:

```text
stdin/stdout JSONL
```

or local sockets.

Do not introduce Redis/RabbitMQ.

---

# 30. Model Isolation Fallback

If one environment cannot cleanly host:

```text
FunASR
Qwen runtime
CosyVoice
```

split into:

```text
core process
├── ASR worker process
├── Translation worker process
└── TTS worker process
```

Communication contract:

```json
{
  "request_id": "...",
  "type": "translate",
  "payload": {...}
}
```

This is preferable to dependency pinning hacks.

---

# 31. Translation Prompt Template

Initial system instruction:

```text
You are a simultaneous interpreter translating an English technical lecture
into concise natural spoken Mandarin Chinese.

Rules:
- Preserve technical meaning.
- Do not explain, elaborate, or add information.
- Prefer concise spoken Chinese.
- Remove filler expressions when safe.
- Preserve model names, code identifiers, equations, variables, and selected
  technical terms according to the glossary.
- Maintain consistency with prior context.
- The output will immediately be synthesized as speech.
- Shorter wording is preferred when meaning is preserved.
```

Mode augmentation:

```text
NORMAL:
Natural concise interpretation.

CONCISE:
Shorten aggressively while preserving all important technical information.

AGGRESSIVE:
Remove discourse fillers and redundant phrasing. Prefer the shortest natural
Chinese expression that preserves the core technical meaning.

RECOVERY:
Transmit only the essential semantic content needed to follow the lecture.
Do not repeat context already conveyed.
```

---

# 32. Benchmark Dataset

For initial engineering tests use short excerpts from:

- Stanford CS229;
- MIT machine learning / AI lectures;
- clear single-speaker English technical talks.

Create a local benchmark folder:

```text
benchmarks/
├── lecture_01.wav
├── lecture_02.wav
└── metadata.yaml
```

Do not make web downloading part of benchmark reproducibility.

---

# 33. Definition of Done for V1

V1 is done only if all conditions hold:

- [ ] Windows audio device capture works.
- [ ] User can route a browser/player into RealtimeDub.
- [ ] Source audio does not need to be audible to the user.
- [ ] Streaming ASR produces incremental English text.
- [ ] Stable-prefix commitment works.
- [ ] Translation occurs clause-by-clause before sentence end where possible.
- [ ] Chinese TTS plays continuously.
- [ ] User can select input/output devices.
- [ ] Lag is measured continuously.
- [ ] Adaptive latency policy prevents unbounded lag.
- [ ] Session trace is saved.
- [ ] 60-minute session completes without crash.
- [ ] Median latency is approximately <= 2.5–3.0 seconds on target hardware.
- [ ] CLI and setup instructions are sufficient for a fresh Windows machine.
- [ ] Core unit/integration tests pass.

---

# 34. First Deliverable Expected From Coding Agent

Before attempting a full implementation, produce and commit:

```text
1. repository skeleton
2. pyproject.toml
3. config schema
4. event dataclasses
5. audio input/output device enumeration
6. capture → delayed playback prototype
7. bounded queue pipeline
8. mocked ASR → translator → TTS pipeline
9. unit tests
10. README with exact Windows setup instructions
```

The first milestone should contain no large model downloads and must be runnable immediately.

After that milestone passes, integrate real ASR.

---

# 35. Final Design Principle

The system should behave like a human simultaneous interpreter:

```text
listen
→ wait only as long as needed for meaning
→ commit stable information
→ compress unnecessary phrasing
→ speak immediately
→ adapt when falling behind
```

The main engineering problem is not model accuracy alone.

The main problem is maintaining:

```text
semantic quality
+
stable incremental commitment
+
bounded latency
+
continuous audio playback
```

for long-running sessions.

Optimize the system around those four properties.

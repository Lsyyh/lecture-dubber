from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float | None = None
    end: float | None = None
    word: str
    score: float | None = None


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class TranslationUnit(BaseModel):
    id: int
    start: float
    end: float
    source: str
    translation: str | None = None
    tts_path: str | None = None
    tts_duration: float | None = None
    speed: float = 1.0
    attempts: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class SourceInfo(BaseModel):
    kind: Literal["local", "url"]
    input: str
    video_path: str
    title: str | None = None
    webpage_url: str | None = None
    subtitle_paths: list[str] = Field(default_factory=list)


class JobState(BaseModel):
    source: SourceInfo
    audio_path: str | None = None
    language: str | None = None
    segments_path: str | None = None
    units_path: str | None = None
    subtitle_path: str | None = None
    final_path: str | None = None


class Config(BaseModel):
    work_dir: Path = Path("outputs")

    source_max_height: int = 1080
    prefer_subtitles: bool = False
    subtitle_languages: list[str] = Field(default_factory=lambda: ["en", "en-US", "en-GB"])

    asr_model: str = "large-v3"
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    asr_batch_size: int = 8
    asr_language: str | None = "en"
    # Optional local directory of a wav2vec2 alignment model; None uses the
    # WhisperX per-language default (downloaded from the HF hub at runtime).
    asr_align_model: str | None = None

    merge_min_duration: float = 4.0
    merge_target_duration: float = 9.0
    merge_max_duration: float = 15.0
    merge_max_gap: float = 1.2

    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen3-8B-Instruct"
    llm_temperature: float = 0.2
    target_language: str = "zh-CN"
    glossary_path: Path = Path("glossary.yaml")

    cosyvoice_root: Path = Path("third_party/CosyVoice")
    cosyvoice_model_dir: Path = Path("pretrained_models/CosyVoice-300M-SFT")
    cosyvoice_speaker: str = "中文女"
    cosyvoice_fp16: bool = True
    # CosyVoice2 backend: reference voice for zero-shot cloning.
    cosyvoice_prompt_wav: Path | None = None
    cosyvoice_prompt_text: str = ""

    duration_soft_min: float = 0.88
    duration_soft_max: float = 1.12
    duration_rewrite_threshold: float = 1.22
    max_rewrite_attempts: int = 2

    original_audio_gain_db: float = -24.0
    dub_audio_gain_db: float = 0.0
    # x264 speed preset for the final render; faster presets trade some size for time.
    render_preset: str = "medium"

    # Parallel translation requests. Keep at 1: llama.cpp concurrent slots can
    # mix up responses with the Qwen3-VL template, and the 2B model is fast
    # enough serially (~0.5 s per unit).
    translate_workers: int = 1
    # "auto" lets the VLM detect burned-in source subtitles; "top"/"bottom" force it.
    subtitle_alignment: str = "auto"
    # VLM quality control: hardsub pre-check and rendered-subtitle post-check.
    qc_enabled: bool = True
    qc_frames: int = 6
    # Delete tts/, source/*.m4s and dub.wav after a successful render.
    cleanup_intermediates: bool = True

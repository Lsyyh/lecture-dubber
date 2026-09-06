"""Typed events flowing through the realtime interpretation pipeline.

Every event carries source-audio timestamps so the latency controller can
measure lag from the moment speech was spoken, not from queue lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import numpy as np


class SeqGen:
    """Monotonic sequence numbers, one generator per stream."""

    def __init__(self) -> None:
        self._it = count()

    def next(self) -> int:
        return next(self._it)


@dataclass
class AudioSpan:
    """A captured PCM chunk already resampled to 16 kHz mono."""

    seq: int
    pcm: np.ndarray  # float32 mono 16 kHz
    start_ts: float  # wall-clock seconds of the first sample
    end_ts: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)


@dataclass
class WordTok:
    """One recognized word with source-audio timestamps."""

    word: str
    start_ts: float
    end_ts: float


@dataclass
class Clause:
    """A committed source clause ready for translation."""

    seq: int
    text: str
    start_ts: float
    end_ts: float


@dataclass
class TranslationResult:
    seq: int
    clause_seq: int
    source_text: str
    zh_text: str
    source_end_ts: float
    mode: str
    produced_at: float


@dataclass
class DubChunk:
    """Synthesized Chinese speech ready for playback."""

    seq: int
    pcm: np.ndarray  # float32 mono
    sample_rate: int
    clause_seq: int
    source_end_ts: float  # source speech timestamp this dub corresponds to
    generated_at: float
    speed: float = 1.0


@dataclass
class SessionStatus:
    """Snapshot of a live session for the web UI."""

    running: bool = False
    phase: str = "idle"  # idle | loading | live | stopping | error
    target_name: str = ""
    target_pid: int = 0
    lag_s: float = 0.0
    mode: str = "NORMAL"
    tts_speed: float = 1.0
    partial_text: str = ""
    clauses: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    error: str = ""

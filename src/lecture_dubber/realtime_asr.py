"""Streaming ASR for the realtime mode.

Chunked faster-whisper: the growing window since the last commit point is
re-transcribed roughly once per second, and StablePrefixBuffer keeps only the
text that survives re-transcription. This is the "whisper_streaming" pattern -
much simpler than a true streaming decoder and easily fast enough on a 3090.
"""

from __future__ import annotations

import threading

import numpy as np

from .realtime_events import WordTok

WINDOW_SR = 16000


class StreamingASR:
    def __init__(
        self,
        model: str,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str | None = "en",
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language or "en"

    def transcribe(self, pcm: np.ndarray, window_start_ts: float) -> list[WordTok]:
        """Transcribe one 16 kHz mono window. Timestamps are mapped onto
        wall-clock source time via window_start_ts."""
        if len(pcm) < 800:  # < 50 ms
            return []
        segments, _info = self._model.transcribe(
            pcm,
            language=self._language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        out: list[WordTok] = []
        for seg in segments:
            for word in seg.words or []:
                text = word.word.strip()
                if not text:
                    continue
                # Collapse whisper hallucination loops ("z, z, z, z") - real
                # speech rarely repeats a word three times in a row.
                if (
                    len(out) >= 2
                    and text.lower() == out[-1].word.lower()
                    and text.lower() == out[-2].word.lower()
                ):
                    continue
                out.append(
                    WordTok(
                        word=text,
                        start_ts=window_start_ts + float(word.start),
                        end_ts=window_start_ts + float(word.end),
                    )
                )
        return out


class AudioRing:
    """Thread-safe rolling buffer of 16 kHz mono PCM with wall-clock bounds."""

    def __init__(self, max_seconds: float = 30.0) -> None:
        self.sr = WINDOW_SR
        self._chunks: list[np.ndarray] = []
        self._ts: list[float] = []  # end_ts per chunk
        self._start_ts: list[float] = []
        self._max_samples = int(max_seconds * WINDOW_SR)
        self._lock = threading.Lock()
        self.last_ts = 0.0

    def append(self, pcm: np.ndarray, end_ts: float) -> None:
        with self._lock:
            self._chunks.append(pcm)
            dur = len(pcm) / self.sr
            self._start_ts.append(end_ts - dur)
            self._ts.append(end_ts)
            self.last_ts = end_ts
            total = sum(len(c) for c in self._chunks)
            while total > self._max_samples and len(self._chunks) > 1:
                total -= len(self._chunks[0])
                self._chunks.pop(0)
                self._ts.pop(0)
                self._start_ts.pop(0)

    def window(self, since_ts: float) -> tuple[np.ndarray, float]:
        """Concatenated PCM covering audio after since_ts, plus the exact
        wall-clock time of the first sample in the returned window."""
        with self._lock:
            picked = [
                (c, s)
                for c, s in zip(self._chunks, self._start_ts)
                if s + len(c) / self.sr > since_ts
            ]
            if not picked:
                return np.zeros(0, dtype=np.float32), 0.0
            first_start = picked[0][1]
            cut = max(0, int((since_ts - first_start) * WINDOW_SR))
            pcm = np.concatenate([c for c, _ in picked])[cut:]
        return pcm.astype(np.float32), first_start + cut / WINDOW_SR

    def trim_before(self, ts: float) -> None:
        with self._lock:
            while self._chunks and self._ts[0] < ts:
                self._chunks.pop(0)
                self._ts.pop(0)
                self._start_ts.pop(0)

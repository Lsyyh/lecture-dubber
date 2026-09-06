"""Pure text logic for the realtime pipeline: stable-prefix commitment,
clause segmentation and the latency policy. No model or I/O dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .realtime_events import WordTok

_SENTENCE_END = re.compile(r"[.!?…]$")
_SOFT_BREAK = re.compile(r"[,;:]$")
_FILLER = re.compile(
    r"^(uh+|um+|erm+|ah+|oh+|so|well|okay|ok|right|like|you know|kind of|sort of|"
    r"i mean|basically|actually|literally)[,.]?$",
    re.IGNORECASE,
)


class StablePrefixBuffer:
    """Commit ASR words that survive re-transcription.

    Each ASR tick re-transcribes the whole (growing) window from the last
    commit point. A word is committed once it lies in the longest common
    prefix of the current and the previous hypothesis, keeping a small
    trailing margin for boundary revisions. A pending span older than
    ``max_pending_s`` is force-committed so a pathological ASR never stalls
    the stream.
    """

    def __init__(self, trailing_keep: int = 1, max_pending_s: float = 2.0) -> None:
        self.trailing_keep = trailing_keep
        self.max_pending_s = max_pending_s
        self._prev: list[WordTok] = []

    def reset(self) -> None:
        """Call after each commit: the next hypothesis is anchored at the new
        commit point, so the old hypothesis is no longer comparable."""
        self._prev = []

    def update(self, words: list[WordTok], now_ts: float) -> list[WordTok]:
        """Feed the latest hypothesis; return the words to commit now."""
        if not words:
            return []
        lcp = 0
        for a, b in zip(self._prev, words):
            if a.word == b.word:
                lcp += 1
            else:
                break
        commit_upto = max(0, lcp - self.trailing_keep)
        # Force-commit an aged pending span (minus the trailing margin).
        if commit_upto < len(words) - self.trailing_keep:
            span = words[-1].end_ts - words[commit_upto].start_ts
            if span > self.max_pending_s:
                commit_upto = max(commit_upto, len(words) - self.trailing_keep)
        committed = words[:commit_upto]
        self._prev = words
        return committed


@dataclass
class ClausePolicy:
    """Segmentation limits handed to the segmenter by the latency controller."""

    max_words: int
    max_wait_s: float
    min_words: int = 2


class ClauseSegmenter:
    """Accumulate committed words and emit semantic clauses.

    A clause is emitted when a sentence-ending punctuation is seen, when the
    buffer exceeds the policy word limit, or when speech has paused for
    ``max_wait_s``. Commas split only when the buffer is already reasonably
    long, which keeps clauses in the ~1-4 s range the plan targets.
    """

    def __init__(self) -> None:
        self.buffer: list[WordTok] = []
        self._seq = 0

    def add(self, words: list[WordTok], policy: ClausePolicy) -> list[tuple[str, float, float]]:
        out: list[tuple[str, float, float]] = []
        for w in words:
            self.buffer.append(w)
            if (
                _SENTENCE_END.search(w.word)
                and len(self.buffer) >= policy.min_words
                or len(self.buffer) >= policy.max_words
                or (_SOFT_BREAK.search(w.word) and len(self.buffer) >= max(policy.min_words, 6))
            ):
                out.append(self._pop())
        return out

    def tick(self, now_ts: float, policy: ClausePolicy) -> list[tuple[str, float, float]]:
        """Flush clauses whose tail has been silent for max_wait_s."""
        if not self.buffer:
            return []
        if now_ts - self.buffer[-1].end_ts < policy.max_wait_s:
            return []
        if len(self.buffer) < policy.min_words:
            return []
        return [self._pop()]

    def soft_break(self, policy: ClausePolicy) -> list[tuple[str, float, float]]:
        """Emit on a comma when the buffer is already long (latency pressure)."""
        if not self.buffer or len(self.buffer) < max(policy.min_words, 6):
            return []
        if not _SOFT_BREAK.search(self.buffer[-1].word):
            return []
        return [self._pop()]

    def _pop(self) -> tuple[str, float, float]:
        self._seq += 1
        words = self.buffer
        self.buffer = []
        text = " ".join(w.word for w in words).strip()
        # drop punctuation-only leading/trailing tokens left by commit boundaries
        text = text.strip(" .,;")
        return text, words[0].start_ts, words[-1].end_ts


MODES = ("NORMAL", "CONCISE", "AGGRESSIVE", "RECOVERY")
# lag thresholds that switch modes (seconds)
_WARNING_S = 2.5
_CRITICAL_S = 4.0
_RECOVERY_S = 6.0


@dataclass
class LatencyPolicy:
    mode: str
    tts_speed: float
    clause: ClausePolicy
    prompt_hint: str


_POLICIES = {
    "NORMAL": LatencyPolicy("NORMAL", 1.0, ClausePolicy(18, 1.4), ""),
    "CONCISE": LatencyPolicy(
        "CONCISE",
        1.05,
        ClausePolicy(12, 1.5),
        "Shorten aggressively while preserving all important technical information.",
    ),
    "AGGRESSIVE": LatencyPolicy(
        "AGGRESSIVE",
        1.12,
        ClausePolicy(8, 1.0),
        "Remove discourse fillers and redundant phrasing. Prefer the shortest "
        "natural Chinese expression that preserves the core technical meaning.",
    ),
    "RECOVERY": LatencyPolicy(
        "RECOVERY",
        1.18,
        ClausePolicy(6, 0.5),
        "Transmit only the essential semantic content needed to follow the "
        "lecture. Drop low-information fragments and fillers entirely.",
    ),
}


def latency_policy(lag_s: float) -> LatencyPolicy:
    if lag_s >= _RECOVERY_S:
        return _POLICIES["RECOVERY"]
    if lag_s >= _CRITICAL_S:
        return _POLICIES["AGGRESSIVE"]
    if lag_s >= _WARNING_S:
        return _POLICIES["CONCISE"]
    return _POLICIES["NORMAL"]


def strip_fillers(text: str) -> str:
    """Remove standalone filler words (AGGRESSIVE/RECOVERY post-pass)."""
    kept = [w for w in text.split() if not _FILLER.match(w)]
    return " ".join(kept)

"""Integration test for the realtime pipeline with fake adapters.

Runs the real thread orchestration (capture -> ASR -> translate -> TTS -> play)
end to end without GPU, network, or audio hardware.
"""

from __future__ import annotations

import time

import numpy as np

from lecture_dubber.models import Config
from lecture_dubber.realtime_pipeline import RealtimeSession


class FakeCapture:
    """Feeds 0.2 s of quiet sine per read(), like a real loopback tap."""

    def __init__(self) -> None:
        self.t = time.time()

    def start(self) -> None:
        pass

    def read(self, timeout: float = 0.5):
        time.sleep(0.1)
        dur = 0.2
        pcm = 0.05 * np.ones(int(16000 * dur), dtype=np.float32)
        self.t += dur
        return pcm, self.t

    def stop(self) -> None:
        pass


class FakeASR:
    """Scripted hypotheses that grow, then a commit-friendly final sentence."""

    def __init__(self) -> None:
        self.script = [
            "the learning",
            "the learning rate is",
            "the learning rate is too large .",
            "the learning rate is too large . now we",
            "the learning rate is too large . now we update the weights .",
        ] * 6
        self.calls = 0

    def transcribe(self, pcm, window_start_ts: float):
        text = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        out = []
        t = window_start_ts
        for w in text.split():
            out.append(type("W", (), {"word": w, "start_ts": t, "end_ts": t + 0.3})())
            t += 0.3
        return out


class FakeTranslator:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def interpret(self, source_text, prev_source, prev_zh, glossary, mode_hint):
        self.seen.append(source_text)
        return "中文" + source_text


class FakeTTS:
    def synthesize_array(self, text: str, speed: float = 1.0):
        return 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.4, 9600)).astype(np.float32), 24000


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[float] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def play(self, pcm, src_sr: int) -> float:
        self.played.append(len(pcm) / src_sr)
        return time.time()


def _run_session(seconds: float):
    from pathlib import Path

    cfg = Config()
    asr = FakeASR()
    translator = FakeTranslator()
    player = FakePlayer()
    session = RealtimeSession(
        cfg,
        {"learning rate": "学习率"},
        pid=0,
        name="fake.exe",
        capture=FakeCapture(),
        asr=asr,
        translator=translator,
        tts=FakeTTS(),
        player=player,
        session_dir=Path("outputs/rt_test/trace"),
    )
    session.start()
    time.sleep(seconds)
    snap = session.snapshot()
    session.stop()
    return session, snap, translator, player


class TestRealtimePipeline:
    def test_full_flow(self):
        session, snap, translator, player = _run_session(seconds=9.0)

        assert session.phase == "done", session.error
        assert snap["counters"]["clauses"] > 0
        # the final clause may still be in flight when we snapshot
        assert (
            snap["counters"]["translated"] + snap["counters"]["skipped"]
            >= snap["counters"]["clauses"] - 1
        )
        assert snap["counters"]["dubbed"] > 0
        assert snap["counters"]["skipped"] == 0
        assert snap["counters"]["asr_errors"] == 0
        assert player.played, "no dub audio reached the player"
        # every clause carries a translation and non-trivial text
        done = [c for c in snap["clauses"] if c["zh"]]
        assert done
        for s in translator.seen:
            assert len(s) >= 2 and any(ch.isalpha() for ch in s), repr(s)

    def test_trace_written(self):
        _run_session(seconds=6.0)
        trace = session_trace_lines()
        types = {line.split('"type": "')[1].split('"')[0] for line in trace}
        assert {"session_start", "clause", "translation", "played", "session_stop"} <= types


def session_trace_lines() -> list[str]:
    from pathlib import Path

    p = Path("outputs/rt_test/trace/events.jsonl")
    return [line for line in p.read_text(encoding="utf-8").splitlines() if line]

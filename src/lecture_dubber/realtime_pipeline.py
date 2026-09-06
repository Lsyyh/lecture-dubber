"""Realtime interpretation pipeline: capture -> ASR -> translate -> TTS -> play.

Thread-based orchestration (all model calls are blocking; threads are simpler
than asyncio here). Every queue is bounded - backpressure is resolved by the
latency controller (shorter clauses, faster TTS), never by unbounded buffering.
Each stage tolerates transient failures: a bad clause is skipped, never allowed
to kill the session.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

from .realtime_capture import ProcessCapture, duck_session, restore_session_volume
from .realtime_events import DubChunk, SeqGen
from .realtime_text import (
    ClauseSegmenter,
    LatencyPolicy,
    StablePrefixBuffer,
    latency_policy,
    strip_fillers,
)

ASR_TICK_S = 1.0
WINDOW_PREROLL_S = 0.25
MAX_WINDOW_S = 28.0
TTS_MERGE_MAX = 4
# Dub more than this far behind the live speech is skipped by the play loop.
STALE_SKIP_S = 6.0


class RealtimeSession:
    def __init__(
        self,
        cfg,
        glossary: dict[str, str],
        pid: int,
        name: str,
        *,
        duck_original: bool = True,
        capture=None,
        asr=None,
        translator=None,
        tts=None,
        player=None,
        session_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.pid = pid
        self.name = name
        self.duck_original = duck_original

        self._capture = capture
        self._asr = asr
        self._translator = translator
        self._tts = tts
        self._player = player

        self.stop_event = threading.Event()
        self.phase = "loading"  # loading | live | stopping | error | done
        self.error = ""
        self.started_at = time.time()

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = session_dir or (Path(cfg.work_dir) / "realtime" / stamp)
        self.dir.mkdir(parents=True, exist_ok=True)

        self._ring = None
        self._spb = StablePrefixBuffer(max_pending_s=2.0)
        self._segmenter = ClauseSegmenter()
        self._seq = SeqGen()
        self._trace_lock = threading.Lock()
        self._trace_file = (self.dir / "events.jsonl").open("a", encoding="utf-8")

        self._q_translate: queue.Queue = queue.Queue(maxsize=8)
        self._q_tts: queue.Queue = queue.Queue(maxsize=8)
        self._q_play: queue.Queue = queue.Queue(maxsize=4)

        self._state_lock = threading.Lock()
        # None until the first dub starts playing; the session is not
        # "behind" before that point, it simply has not spoken yet.
        self._played_source_ts: float | None = None
        self._policy: LatencyPolicy = latency_policy(0.0)
        self._partial = ""
        self._clauses: deque[dict] = deque(maxlen=40)
        self._counters = {"clauses": 0, "translated": 0, "dubbed": 0, "skipped": 0, "asr_errors": 0}
        self._commit_ts = time.time()
        self._threads: list[threading.Thread] = []
        self._orig_volume: float | None = None

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._capture is None:
            duck = self.cfg.realtime_duck_volume
            self._orig_volume = duck_session(self.pid, duck) if self.duck_original else None
            gain = 1.0 / duck if self._orig_volume is not None else 1.0
            self._capture = ProcessCapture(self.pid, gain=gain)
        if self._asr is None:
            self._asr = _get_asr(self.cfg)
        if self._translator is None:
            self._translator = _get_translator(self.cfg, self.glossary)
        if self._tts is None:
            self._tts = _get_tts(self.cfg)
        if self._player is None:
            from .realtime_capture import DubPlayer

            self._player = DubPlayer()
        self._log("session_start", pid=self.pid, name=self.name, duck=self.duck_original)
        # Warm TTS kernels once so the first real clause doesn't eat a
        # multi-second cold start; the rendered audio is discarded.
        try:
            self._tts.synthesize_array("准备完毕", speed=1.0)
        except Exception:
            pass
        self._capture.start()
        self.phase = "live"
        for target in (
            self._capture_loop,
            self._asr_loop,
            self._translate_loop,
            self._tts_loop,
            self._play_loop,
        ):
            t = threading.Thread(target=target, daemon=True)
            self._threads.append(t)
            t.start()

    def stop(self) -> None:
        self.phase = "stopping"
        self.stop_event.set()
        for t in self._threads:
            t.join(timeout=6.0)
        try:
            self._capture.stop()
        except Exception:
            pass
        if self._orig_volume is not None:
            restore_session_volume(self.pid, self._orig_volume)
        self._log("session_stop", counters=dict(self._counters))
        self._trace_file.close()
        self.phase = "done"

    # ---------- stage: capture ----------

    def _capture_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                got = self._capture.read(timeout=0.5)
                if got is None:
                    continue
                pcm, end_ts = got
                if self._ring is None:
                    from .realtime_asr import AudioRing

                    self._ring = AudioRing(max_seconds=MAX_WINDOW_S)
                self._ring.append(pcm, end_ts)
        except Exception as e:  # device disconnect etc.
            if not self.stop_event.is_set():
                self.phase = "error"
                self.error = f"capture failed: {e}"
                self.stop_event.set()

    # ---------- stage: ASR + commit + segment ----------

    def _asr_loop(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(ASR_TICK_S)
            try:
                self._asr_tick()
            except Exception as e:
                self._counters["asr_errors"] += 1
                self._log("asr_error", error=str(e))
                time.sleep(2.0)

    def _asr_tick(self) -> None:
        if self._ring is None:
            return
        now = time.time()
        since = self._commit_ts - WINDOW_PREROLL_S
        pcm, window_start = self._ring.window(since)
        if len(pcm) < 1600:  # < 100 ms
            return
        words = self._asr.transcribe(pcm, window_start)
        # The window includes a pre-roll covering already-committed audio;
        # drop words anchored there so committed text is never re-proposed.
        words = [w for w in words if w.start_ts > self._commit_ts - 0.02]
        committed = self._spb.update(words, now)
        with self._state_lock:
            self._partial = " ".join(w.word for w in words[-24:])
            lag = 0.0
            if self._played_source_ts is not None:
                lag = (
                    max(0.0, self._ring.last_ts - self._played_source_ts)
                    if self._played_source_ts is not None
                    else 0.0
                )
            self._policy = latency_policy(lag)
        if committed:
            self._commit_ts = committed[-1].end_ts
            self._ring.trim_before(self._commit_ts - 1.0)
            self._spb.reset()
            clauses = self._segmenter.add(committed, self._policy.clause)
            for text, start_ts, end_ts in clauses:
                self._emit_clause(text, start_ts, end_ts)
        # flush clauses that ended with a pause
        for text, start_ts, end_ts in self._segmenter.tick(now, self._policy.clause):
            self._emit_clause(text, start_ts, end_ts)
        # comma breaks when latency pressure is on
        if self._policy.mode != "NORMAL":
            for text, start_ts, end_ts in self._segmenter.soft_break(self._policy.clause):
                self._emit_clause(text, start_ts, end_ts)

    def _emit_clause(self, text: str, start_ts: float, end_ts: float) -> None:
        text = text.strip()
        if len(text) < 2:
            return
        self._counters["clauses"] += 1
        entry = {
            "seq": self._seq.next(),
            "source": text,
            "zh": "",
            "start_ts": start_ts,
            "end_ts": end_ts,
            "mode": self._policy.mode,
        }
        with self._state_lock:
            self._clauses.append(entry)
        self._log("clause", seq=entry["seq"], source=text, start_ts=start_ts, end_ts=end_ts)
        try:
            self._q_translate.put(entry, timeout=2.0)
        except queue.Full:
            self._counters["skipped"] += 1
            self._log("clause_dropped", seq=entry["seq"], reason="translate queue full")

    # ---------- stage: translation ----------

    def _translate_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                entry = self._q_translate.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.time()
            prev_source = self._clauses[-2]["source"] if len(self._clauses) >= 2 else ""
            prev_zh = next(
                (c["zh"] for c in reversed(self._clauses) if c["zh"] and c is not entry), ""
            )
            hint = self._policy.prompt_hint
            try:
                zh = self._translator.interpret(
                    entry["source"], prev_source, prev_zh, self.glossary, hint
                )
            except Exception as e:
                self._counters["skipped"] += 1
                self._log("translate_error", seq=entry["seq"], error=str(e))
                continue
            if self._policy.mode in ("AGGRESSIVE", "RECOVERY"):
                zh = strip_fillers(zh) or zh
            latency_ms = int(1000 * (time.time() - t0))
            entry["zh"] = zh
            self._counters["translated"] += 1
            self._log(
                "translation",
                seq=entry["seq"],
                zh=zh,
                mode=self._policy.mode,
                latency_ms=latency_ms,
            )
            payload = {
                "entry": entry,
                "zh": zh,
                "speed": self._policy.tts_speed,
                "source_end_ts": entry["end_ts"],
            }
            try:
                self._q_tts.put(payload, timeout=2.0)
            except queue.Full:
                self._counters["skipped"] += 1
                self._log("dub_dropped", seq=entry["seq"], reason="tts queue full")

    # ---------- stage: TTS ----------

    def _tts_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                payload = self._q_tts.get(timeout=0.5)
            except queue.Empty:
                continue
            # Merge any already-translated backlog into one synthesis call:
            # CosyVoice per-call overhead dominates short clauses, and merged
            # text also synthesizes with better prosody.
            batch = [payload]
            while len(batch) < TTS_MERGE_MAX:
                try:
                    batch.append(self._q_tts.get_nowait())
                except queue.Empty:
                    break
            text = "，".join(p["zh"].strip("。，") for p in batch)
            speed = max(p["speed"] for p in batch)
            t0 = time.time()
            try:
                pcm, sr = self._tts.synthesize_array(text, speed=speed)
            except Exception as e:
                self._counters["skipped"] += len(batch)
                self._log("tts_error", seq=payload["entry"]["seq"], error=str(e))
                continue
            self._log(
                "tts",
                seq=payload["entry"]["seq"],
                merged=[p["entry"]["seq"] for p in batch],
                synth_ms=int(1000 * (time.time() - t0)),
                duration_s=len(pcm) / sr,
            )
            last = batch[-1]
            chunk = DubChunk(
                seq=self._seq.next(),
                pcm=pcm,
                sample_rate=sr,
                clause_seq=payload["entry"]["seq"],
                source_end_ts=last["source_end_ts"],
                generated_at=time.time(),
                speed=speed,
            )
            try:
                self._q_play.put(chunk, timeout=5.0)
            except queue.Full:
                self._counters["skipped"] += len(batch)
                self._log("dub_dropped", seq=chunk.clause_seq, reason="play queue full")

    # ---------- stage: playback ----------

    def _play_loop(self) -> None:
        player_ctx = self._player
        entered = False
        try:
            if hasattr(player_ctx, "__enter__"):
                player_ctx.__enter__()
                entered = True
            while not self.stop_event.is_set():
                try:
                    chunk = self._q_play.get(timeout=0.5)
                except queue.Empty:
                    continue
                # Catch-up: dub spoken more than STALE_SKIP_S in the past can
                # never be heard in sync; skip it and move on, or the queue
                # grows unboundedly when the dub rate trails the source rate.
                last_ts = self._ring.last_ts if self._ring is not None else 0.0
                if last_ts - chunk.source_end_ts > STALE_SKIP_S:
                    self._counters["stale_skipped"] = self._counters.get("stale_skipped", 0) + 1
                    self._log(
                        "stale_skipped",
                        seq=chunk.clause_seq,
                        behind_s=round(last_ts - chunk.source_end_ts, 1),
                    )
                    continue
                started = player_ctx.play(chunk.pcm, chunk.sample_rate)
                self._played_source_ts = chunk.source_end_ts
                self._counters["dubbed"] += 1
                self._log(
                    "played",
                    seq=chunk.clause_seq,
                    started=started,
                    duration_s=len(chunk.pcm) / chunk.sample_rate,
                )
        except Exception as e:
            if not self.stop_event.is_set():
                self.phase = "error"
                self.error = f"playback failed: {e}"
                self.stop_event.set()
        finally:
            if entered:
                try:
                    player_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    # ---------- status + trace ----------

    def snapshot(self) -> dict:
        with self._state_lock:
            lag = 0.0
            if self._ring is not None:
                lag = (
                    max(0.0, self._ring.last_ts - self._played_source_ts)
                    if self._played_source_ts is not None
                    else 0.0
                )
            return {
                "running": self.phase in ("loading", "live"),
                "phase": self.phase,
                "target_name": self.name,
                "target_pid": self.pid,
                "lag_s": round(lag, 2),
                "mode": self._policy.mode,
                "tts_speed": self._policy.tts_speed,
                "partial_text": self._partial,
                "clauses": list(self._clauses),
                "counters": dict(self._counters),
                "error": self.error,
            }

    def _log(self, event_type: str, **fields) -> None:
        record = {"type": event_type, "ts": time.time(), **fields}
        with self._trace_lock:
            self._trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._trace_file.flush()


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict = {}


def _get_asr(cfg):
    from .realtime_asr import StreamingASR

    with _MODEL_LOCK:
        if "asr" not in _MODEL_CACHE:
            model = cfg.realtime_asr_model or cfg.asr_model
            # int8 keeps the ASR footprint small so TTS keeps RTF < 1 on the
            # shared GPU; the offline pipeline keeps its own float16 model.
            _MODEL_CACHE["asr"] = StreamingASR(
                str(model),
                device=cfg.asr_device,
                compute_type="int8_float16",
                language=cfg.asr_language or "en",
            )
        return _MODEL_CACHE["asr"]


def _get_translator(cfg, glossary):
    from .realtime_translate import RealtimeTranslator

    with _MODEL_LOCK:
        key = ("translator", cfg.llm_base_url, cfg.llm_model)
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = RealtimeTranslator(cfg, glossary)
        return _MODEL_CACHE[key]


def _get_tts(cfg):
    from .realtime_tts import build_realtime_tts

    with _MODEL_LOCK:
        key = ("tts", cfg.realtime_tts_backend)
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = build_realtime_tts(cfg)
        return _MODEL_CACHE[key]


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict = {}

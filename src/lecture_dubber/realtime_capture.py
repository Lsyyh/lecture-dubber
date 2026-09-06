"""Windows audio I/O for the realtime mode.

Capture: per-process WASAPI loopback via proc-tap (no admin, Windows 10 20H1+),
resampled to 16 kHz mono float32. Muting the target session would also silence
the loopback tap (it samples post-session-volume), so "ducking" lowers the
session volume to a whisper and the capture compensates numerically.

Playback: default output device through soundcard, one persistent stream.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

TARGET_SR = 16000
DUCK_VOLUME = 0.08  # original audio is barely audible at this level


def list_audio_processes() -> list[dict]:
    """Processes that currently have an audio session (pid, name, volume)."""
    import comtypes
    import psutil
    from pycaw.pycaw import AudioUtilities

    comtypes.CoInitialize()
    try:
        out = []
        seen: set[int] = set()
        for session in AudioUtilities.GetAllSessions():
            pid = session.ProcessId
            if not pid or pid in seen:
                continue
            seen.add(pid)
            try:
                name = psutil.Process(pid).name()
            except Exception:
                name = session.Process and session.Process.name() or f"pid {pid}"
            volume, muted = 1.0, False
            try:
                volume = session.SimpleAudioVolume.GetMasterVolume()
                muted = bool(session.SimpleAudioVolume.GetMute())
            except Exception:
                pass
            out.append({"pid": pid, "name": name, "volume": volume, "muted": muted})
        return out
    finally:
        comtypes.CoUninitialize()


def _with_session_volume(pid: int, fn):
    """Run fn(ISimpleAudioVolume) inside a COM scope on the calling thread -
    the interface must be fully used before CoUninitialize, so nothing escapes."""
    import comtypes
    from pycaw.pycaw import AudioUtilities

    comtypes.CoInitialize()
    try:
        for session in AudioUtilities.GetAllSessions():
            if session.ProcessId == pid:
                return fn(session.SimpleAudioVolume)
        return None
    finally:
        comtypes.CoUninitialize()


def get_session_volume(pid: int) -> float | None:
    def action(vol):
        try:
            return float(vol.GetMasterVolume())
        except Exception:
            return None

    return _with_session_volume(pid, action)


def duck_session(pid: int, volume: float = DUCK_VOLUME) -> float | None:
    """Lower the target's session volume; return the original volume to restore."""

    def action(vol):
        try:
            orig = float(vol.GetMasterVolume())
            vol.SetMute(False, None)
            vol.SetMasterVolume(volume, None)
            return orig
        except Exception:
            return None

    return _with_session_volume(pid, action)


def restore_session_volume(pid: int, original: float) -> None:
    def action(vol):
        try:
            vol.SetMasterVolume(original, None)
        except Exception:
            pass

    _with_session_volume(pid, action)


def _resample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    # 3:1 decimation with a 3-tap average pre-filter; speech content is
    # band-limited well below the 8 kHz Nyquist, so this is sufficient.
    n = (len(x) // 3) * 3
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    x = x[:n]
    smoothed = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    return smoothed[::3].astype(np.float32)


class ProcessCapture:
    """Read PCM from one process's loopback tap, converted to 16 kHz mono."""

    def __init__(self, pid: int, gain: float = 1.0) -> None:
        from proctap import ProcessAudioCapture  # lazy: native extension

        self._tap = ProcessAudioCapture(pid)
        self.gain = gain

    def start(self) -> None:
        self._tap.start()

    def read(self, timeout: float = 0.5) -> tuple[np.ndarray, float] | None:
        raw = self._tap.read(timeout=timeout)
        if not raw:
            return None
        pcm = np.frombuffer(raw, dtype=np.float32)
        if pcm.ndim != 0 and len(pcm) % 2 == 0:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
        mono = _resample_48k_to_16k(pcm.astype(np.float32))
        if self.gain != 1.0:
            mono = np.clip(mono * self.gain, -1.0, 1.0)
        return mono, time.time()

    def stop(self) -> None:
        try:
            self._tap.stop()
        except Exception:
            pass
        try:
            self._tap.close()
        except Exception:
            pass


class DubPlayer:
    """Continuous-stream dub playback with a jitter buffer.

    Synthesized chunks are resampled to the output rate and appended to one
    continuous buffer; a PortAudio callback pulls frames at all times and
    inserts silence when the buffer runs dry, so playback never glitches
    between clauses. While the dub trails the live speech the consumption
    rate is gently raised (up to 1.3x, i.e. faster speech instead of drops);
    chunks more than ``stale_drop_s`` behind are discarded outright.
    """

    CLAUSE_GAP_S = 0.08  # tiny pause between clauses so words don't collide

    def __init__(self, sample_rate: int = 48000, stale_drop_s: float = 10.0) -> None:
        import sounddevice as sd

        self.sr = sample_rate
        self.stale_drop_s = stale_drop_s
        self._lock = threading.Lock()
        self._buf = np.zeros(0, dtype=np.float32)  # buffered output frames
        self._meta: deque[tuple[float, float]] = deque()  # (cum_end_f, source_end_ts)
        self._consumed = 0  # input frames consumed so far
        self._speed = 1.0
        self.played_source_ts: float | None = None
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=0,
            callback=self._cb,
        )
        self._stream.start()

    # ---- callback thread ----

    def _cb(self, outdata, frames, _t, _status) -> None:
        try:
            with self._lock:
                take = min(int(frames * self._speed), len(self._buf))
                if take <= 0:
                    outdata.fill(0.0)
                    return
                idx = np.linspace(0.0, take - 1, num=frames)
                out = np.interp(idx, np.arange(take), self._buf[:take])
                outdata[:] = out.reshape(-1, 1)
                self._buf = self._buf[take:]
                self._consumed += take
                while self._meta and self._consumed >= self._meta[0][0]:
                    self.played_source_ts = self._meta.popleft()[1]
        except Exception:
            outdata.fill(0.0)

    # ---- producer side ----

    @staticmethod
    def resample_to(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return pcm.astype(np.float32)
        n_out = int(len(pcm) * dst_sr / src_sr)
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.linspace(0.0, len(pcm) - 1, num=n_out)
        return np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)

    def enqueue(self, pcm: np.ndarray, src_sr: int, source_end_ts: float) -> None:
        out = self.resample_to(pcm, src_sr, self.sr)
        with self._lock:
            if len(self._buf) > 0:
                gap = np.zeros(int(self.CLAUSE_GAP_S * self.sr), dtype=np.float32)
                self._buf = np.concatenate([self._buf, gap])
            self._buf = np.concatenate([self._buf, out])
            self._meta.append((self._consumed + len(self._buf), source_end_ts))

    def tick(self, last_ts: float) -> dict:
        """Called periodically: adapt playback speed to lag, drop stale chunks.

        Returns {"dropped": n, "lag_s": lag}."""
        lag = 0.0 if self.played_source_ts is None else max(0.0, last_ts - self.played_source_ts)
        # gentle catch-up: above 3 s behind, consume up to 30% faster
        self._speed = min(1.3, 1.0 + max(0.0, lag - 3.0) / 10.0)
        dropped = 0
        with self._lock:
            while self._meta and last_ts - self._meta[0][1] > self.stale_drop_s:
                cum_end = self._meta[0][0]
                cut = min(cum_end, self._consumed + len(self._buf)) - self._consumed
                if cut > 0:
                    self._buf = self._buf[cut:]
                    self._consumed += cut
                self._meta.popleft()
                dropped += 1
        if dropped:
            self.played_source_ts = None  # jumped ahead; recompute on next fresh chunk
        return {"dropped": dropped, "lag_s": round(lag, 2)}

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

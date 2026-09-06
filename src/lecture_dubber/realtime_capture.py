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
from typing import Self

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
    """Serialize dub chunks onto the default output device."""

    def __init__(self, sample_rate: int = 48000) -> None:
        import soundcard as sc

        self.sr = sample_rate
        self._spk = sc.default_speaker()
        self._player = self._spk.player(samplerate=sample_rate, channels=1)
        self._lock = threading.Lock()

    def __enter__(self) -> Self:
        self._player.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._player.__exit__(*exc)

    @staticmethod
    def resample_to(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return pcm.astype(np.float32)
        n_out = int(len(pcm) * dst_sr / src_sr)
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.linspace(0.0, len(pcm) - 1, num=n_out)
        return np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)

    def play(self, pcm: np.ndarray, src_sr: int) -> float:
        """Play one chunk, block until finished. Returns wall-clock seconds when
        playback actually started (for lag tracking)."""
        out = self.resample_to(pcm, src_sr, self.sr)
        with self._lock:
            started = time.time()
            self._player.play(out)
            time.sleep(len(out) / self.sr + 0.02)
            return started

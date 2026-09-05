"""Compatibility shim for pyannote.audio 3.x on torchaudio >= 2.9.

pyannote.audio 3.x still uses torchaudio.info / torchaudio.load / torchaudio.AudioMetaData,
which were removed from torchaudio 2.9 in favor of torchcodec (not available on this
deployment). Backfill the removed APIs with soundfile-based equivalents so the bundled
WhisperX VAD keeps working. Call ensure_torchaudio_compat() before importing whisperx.
"""
from __future__ import annotations

from collections import namedtuple


def ensure_torchaudio_compat() -> None:
    try:
        import torchaudio
    except ImportError:
        return
    if hasattr(torchaudio, "AudioMetaData") and hasattr(torchaudio, "load"):
        return

    AudioMetaData = namedtuple(
        "AudioMetaData",
        ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
    )

    import soundfile as sf

    def _info(source: object, *args: object, **kwargs: object) -> AudioMetaData:
        info = sf.info(str(source))
        return AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=16,
            encoding=info.subtype or "PCM_16",
        )

    def _load(filepath: object, *args: object, **kwargs: object):
        import torch

        data, sample_rate = sf.read(str(filepath), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sample_rate

    def _list_audio_backends() -> list[str]:
        return ["soundfile"]

    torchaudio.AudioMetaData = AudioMetaData
    torchaudio.info = _info
    torchaudio.load = _load
    torchaudio.list_audio_backends = _list_audio_backends


def ensure_lightning_load_trusted() -> None:
    """Let lightning load trusted local checkpoints (whisperx bundled VAD) with
    weights_only=False; torch >= 2.6 flipped the default and the checkpoint stores
    omegaconf hyper-parameters that the safe loader rejects."""
    try:
        from lightning_fabric.utilities import cloud_io
    except ImportError:
        return
    original = cloud_io._load
    if getattr(original, "_lecture_dubber_trusted_load", False):
        return

    def _load(*args: object, **kwargs: object):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    _load._lecture_dubber_trusted_load = True
    cloud_io._load = _load

"""TTS adapter for the realtime mode.

Piper runs on CPU with RTF ~0.05, which keeps dub production ahead of the
source speech while ASR and TTS share one GPU. CosyVoice sounds better but
sustains RTF ~1 on this machine - offered as the quality option via
``realtime_tts_backend: cosyvoice``.
"""

from __future__ import annotations

import numpy as np


class PiperTTS:
    def __init__(self, model_path: str) -> None:
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(model_path))

    def synthesize_array(self, text: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        from piper import SynthesisConfig

        config = SynthesisConfig(length_scale=max(0.4, 1.0 / max(0.5, speed)))
        chunks = list(self._voice.synthesize(text, syn_config=config))
        if not chunks:
            raise RuntimeError("Piper returned no audio")
        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, int(chunks[0].sample_rate)


def build_realtime_tts(cfg):
    """Dispatch on config: 'piper' (default, fast) or 'cosyvoice' (quality)."""
    if cfg.realtime_tts_backend == "cosyvoice":
        from .tts import CosyVoiceTTS

        return CosyVoiceTTS(cfg)
    from .realtime_tts import PiperTTS

    return PiperTTS(cfg.realtime_tts_model)

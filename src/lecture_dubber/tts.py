from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

from ._torchaudio_compat import ensure_torchaudio_compat
from .models import Config
from .textnorm import spoken_form


def _patch_cosyvoice_load_wav() -> None:
    """CosyVoice2 re-reads already-loaded prompt tensors through load_wav, which
    only accepts file paths; teach it to pass tensors through (assumed 16 kHz)."""
    import torch
    import torchaudio.transforms
    from cosyvoice.cli import frontend

    if getattr(frontend.load_wav, "_lecture_dubber_tensor_passthrough", False):
        return
    original = frontend.load_wav

    def load_wav(wav, target_sr, min_sr=16000):
        if isinstance(wav, torch.Tensor):
            speech = wav if wav.ndim == 2 else wav.reshape(1, -1)
            if target_sr != 16000:
                speech = torchaudio.transforms.Resample(orig_freq=16000, new_freq=target_sr)(speech)
            return speech
        return original(wav, target_sr, min_sr)

    load_wav._lecture_dubber_tensor_passthrough = True
    frontend.load_wav = load_wav


class CosyVoiceTTS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        ensure_torchaudio_compat()
        root = cfg.cosyvoice_root.resolve()
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "third_party" / "Matcha-TTS"))
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as e:
            raise RuntimeError(
                f"Could not import CosyVoice from {root}. Clone the official repo recursively and install its environment."
            ) from e
        self.model = AutoModel(model_dir=str(cfg.cosyvoice_model_dir), fp16=cfg.cosyvoice_fp16)
        # With a reference voice configured we clone it via inference_zero_shot
        # (works for CosyVoice v1 and v2 models, including speaker-less 25Hz dirs);
        # otherwise we use the built-in SFT speakers of the loaded model.
        self.use_zero_shot = cfg.cosyvoice_prompt_wav is not None
        self.prompt_speech = None
        if self.use_zero_shot:
            _patch_cosyvoice_load_wav()
            self.prompt_speech = self._load_prompt_16k(cfg)

    def _load_prompt_16k(self, cfg: Config):
        if cfg.cosyvoice_prompt_wav is None or not Path(cfg.cosyvoice_prompt_wav).exists():
            raise RuntimeError(
                "Zero-shot TTS requires a reference voice: set cosyvoice_prompt_wav and cosyvoice_prompt_text in config.yaml"
            )
        import torch
        import torchaudio.functional

        data, sr = sf.read(str(cfg.cosyvoice_prompt_wav), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        speech = torch.from_numpy(data).reshape(1, -1)
        return torchaudio.functional.resample(speech, sr, 16000)

    def synthesize(self, text: str, out_wav: Path, speed: float = 1.0) -> Path:
        text = spoken_form(text)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if self.use_zero_shot:
            chunks = list(self.model.inference_zero_shot(
                text, self.cfg.cosyvoice_prompt_text, self.prompt_speech, stream=False, speed=speed,
            ))
        else:
            chunks = list(self.model.inference_sft(
                text,
                self.cfg.cosyvoice_speaker,
                stream=False,
                speed=speed,
            ))
        if not chunks:
            raise RuntimeError("CosyVoice returned no audio")
        import torch
        speech = torch.cat([x["tts_speech"] for x in chunks], dim=1)
        # soundfile instead of torchaudio.save: torchaudio >= 2.9 needs torchcodec for wav writes,
        # which is unavailable in this deployment.
        sf.write(str(out_wav), speech.reshape(-1).to(torch.float32).cpu().numpy(), self.model.sample_rate)
        return out_wav

from __future__ import annotations

import sys
from pathlib import Path
import soundfile as sf
from .models import Config
from .textnorm import spoken_form


class CosyVoiceTTS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
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

    def synthesize(self, text: str, out_wav: Path, speed: float = 1.0) -> Path:
        text = spoken_form(text)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
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

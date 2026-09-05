from __future__ import annotations

import gc
from pathlib import Path

from .models import Config, Segment, Word
from ._torchaudio_compat import ensure_lightning_load_trusted, ensure_torchaudio_compat


def transcribe_whisperx(audio_path: Path, cfg: Config) -> tuple[str, list[Segment]]:
    ensure_torchaudio_compat()
    ensure_lightning_load_trusted()
    try:
        import torch
        import whisperx
    except ImportError as e:
        raise RuntimeError("WhisperX not installed. Install the ASR environment with: pip install 'lecture-dubber[asr]'") from e

    model = whisperx.load_model(
        cfg.asr_model,
        cfg.asr_device,
        compute_type=cfg.asr_compute_type,
        language=cfg.asr_language,
    )
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=cfg.asr_batch_size)
    language = result.get("language") or cfg.asr_language or "en"
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=cfg.asr_device, model_name=cfg.asr_align_model
    )
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, cfg.asr_device,
        return_char_alignments=False,
    )
    del align_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    segments: list[Segment] = []
    for i, seg in enumerate(aligned.get("segments", [])):
        words = [Word.model_validate(w) for w in seg.get("words", []) if w.get("word")]
        segments.append(Segment(
            id=i,
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            text=str(seg.get("text", "")).strip(),
            words=words,
        ))
    return language, segments

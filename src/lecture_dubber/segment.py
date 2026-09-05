from __future__ import annotations

from .models import Config, Segment, TranslationUnit


def _sentence_like(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", ":", ";"))


def merge_segments(segments: list[Segment], cfg: Config) -> list[TranslationUnit]:
    if not segments:
        return []
    units: list[TranslationUnit] = []
    current: list[Segment] = []

    def flush() -> None:
        if not current:
            return
        units.append(TranslationUnit(
            id=len(units),
            start=current[0].start,
            end=current[-1].end,
            source=" ".join(s.text.strip() for s in current if s.text.strip()).strip(),
        ))
        current.clear()

    for seg in segments:
        if not seg.text.strip():
            continue
        if not current:
            current.append(seg)
            continue
        proposed_duration = seg.end - current[0].start
        gap = max(0.0, seg.start - current[-1].end)
        current_duration = current[-1].end - current[0].start
        should_flush = (
            gap > cfg.merge_max_gap
            or proposed_duration > cfg.merge_max_duration
            or (current_duration >= cfg.merge_target_duration and _sentence_like(current[-1].text))
        )
        if should_flush:
            flush()
        current.append(seg)
        if (seg.end - current[0].start) >= cfg.merge_max_duration:
            flush()
    flush()
    return units

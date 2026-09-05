from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .asr import transcribe_whisperx
from .glossary import load_glossary
from .media import compose_timeline, extract_audio, fit_audio, probe_duration, render_video
from .models import Config, JobState, Segment, TranslationUnit
from .segment import merge_segments
from .source import bilibili_cache_title, import_source
from .subtitles import write_srt
from .translate import OpenAICompatibleTranslator
from .tts import CosyVoiceTTS
from .utils import read_json, slugify, write_json

console = Console()


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def job_dir_for(self, value: str) -> Path:
        if value.startswith(("http://", "https://")):
            stem = "url-job"
        else:
            stem = bilibili_cache_title(value) or Path(value).stem
        return self.cfg.work_dir / slugify(stem)

    def run(self, value: str, job_dir: Path | None = None, resume: bool = True) -> Path:
        job_dir = job_dir or self.job_dir_for(value)
        job_dir.mkdir(parents=True, exist_ok=True)
        state_path = job_dir / "state.json"

        if resume and state_path.exists():
            state = JobState.model_validate(read_json(state_path))
            console.print(f"[cyan]Resuming[/cyan] {job_dir}")
        else:
            console.print("[bold]1/6 Import source[/bold]")
            source = import_source(value, job_dir, self.cfg)
            state = JobState(source=source)
            write_json(state_path, state.model_dump(mode="json"))

        video = Path(state.source.video_path)
        audio = job_dir / "audio.wav"
        if not (resume and state.audio_path and Path(state.audio_path).exists()):
            console.print("[bold]2/6 Extract audio[/bold]")
            extract_audio(video, audio)
            state.audio_path = str(audio)
            write_json(state_path, state.model_dump(mode="json"))
        else:
            audio = Path(state.audio_path)

        segments_path = job_dir / "segments.json"
        if resume and segments_path.exists():
            segments = [Segment.model_validate(x) for x in read_json(segments_path)]
        else:
            console.print("[bold]3/6 Transcribe + align[/bold]")
            language, segments = transcribe_whisperx(audio, self.cfg)
            state.language = language
            write_json(segments_path, [s.model_dump() for s in segments])
            state.segments_path = str(segments_path)
            write_json(state_path, state.model_dump(mode="json"))

        units_path = job_dir / "units.json"
        if resume and units_path.exists():
            units = [TranslationUnit.model_validate(x) for x in read_json(units_path)]
        else:
            units = merge_segments(segments, self.cfg)
            write_json(units_path, [u.model_dump() for u in units])
            state.units_path = str(units_path)

        console.print("[bold]4/6 Translate[/bold]")
        translator = OpenAICompatibleTranslator(self.cfg, load_glossary(self.cfg.glossary_path))
        for i, unit in enumerate(units):
            if resume and unit.translation:
                continue
            prev = units[i - 1].source if i > 0 else ""
            nxt = units[i + 1].source if i + 1 < len(units) else ""
            unit.translation = translator.translate(unit, prev, nxt)
            write_json(units_path, [u.model_dump() for u in units])

        console.print("[bold]5/6 Synthesize + duration fit[/bold]")
        tts = CosyVoiceTTS(self.cfg)
        seg_dir = job_dir / "tts"
        fit_dir = job_dir / "tts_fit"
        for unit in units:
            raw = seg_dir / f"{unit.id:05d}.wav"
            fitted = fit_dir / f"{unit.id:05d}.wav"
            if resume and fitted.exists():
                unit.tts_path = str(fitted)
                unit.tts_duration = probe_duration(fitted)
                continue
            while True:
                tts.synthesize(unit.translation or unit.source, raw)
                actual = probe_duration(raw)
                ratio = actual / max(unit.duration, 0.01)
                if ratio > self.cfg.duration_rewrite_threshold and unit.attempts < self.cfg.max_rewrite_attempts:
                    unit.translation = translator.compress(unit, actual)
                    unit.attempts += 1
                    write_json(units_path, [u.model_dump() for u in units])
                    continue
                unit.speed = fit_audio(raw, fitted, unit.duration, self.cfg.duration_soft_min, self.cfg.duration_soft_max)
                unit.tts_path = str(fitted)
                unit.tts_duration = probe_duration(fitted)
                break
            write_json(units_path, [u.model_dump() for u in units])

        console.print("[bold]6/6 Compose + render[/bold]")
        subtitle = write_srt(units, job_dir / "zh.srt")
        total = probe_duration(video)
        timeline = compose_timeline([(u.start, Path(u.tts_path)) for u in units if u.tts_path], total, job_dir / "dub.wav")
        final = render_video(
            video, timeline, subtitle, job_dir / "final.zh.mp4",
            self.cfg.original_audio_gain_db, self.cfg.dub_audio_gain_db,
        )
        state.subtitle_path = str(subtitle)
        state.final_path = str(final)
        write_json(state_path, state.model_dump(mode="json"))
        return final

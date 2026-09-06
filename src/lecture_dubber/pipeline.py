from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console

from .asr import transcribe_whisperx
from .glossary import load_glossary
from .media import compose_timeline, extract_audio, fit_audio, probe_duration, render_video
from .models import Config, JobState, Segment, TranslationUnit
from .qc import VisionQC
from .segment import merge_segments
from .source import bilibili_cache_title, import_source
from .subtitles import write_srt
from .translate import OpenAICompatibleTranslator
from .tts import CosyVoiceTTS
from .utils import cleanup_intermediates, read_json, slugify, write_json

console = Console()

# ASS alignment numpad codes for the subtitles force_style.
_ALIGNMENT_CODES = {"bottom": 2, "top": 8}


class PipelineCancelled(RuntimeError):
    """Raised when a run is cancelled through its stop event."""


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def job_dir_for(self, value: str) -> Path:
        if value.startswith(("http://", "https://")):
            stem = "url-job"
        else:
            stem = bilibili_cache_title(value) or Path(value).stem
        return self.cfg.work_dir / slugify(stem)

    def run(
        self,
        value: str,
        job_dir: Path | None = None,
        resume: bool = True,
        stop_event: threading.Event | None = None,
    ) -> Path:
        job_dir = job_dir or self.job_dir_for(value)
        job_dir.mkdir(parents=True, exist_ok=True)
        state_path = job_dir / "state.json"

        def check_stop() -> None:
            if stop_event is not None and stop_event.is_set():
                raise PipelineCancelled(str(job_dir))

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

        # Hardsub detection runs in the background while we translate/synthesize;
        # its result is only needed when rendering.
        qc_thread: threading.Thread | None = None
        qc_result: dict[str, str] = {}
        if self.cfg.qc_enabled and self.cfg.subtitle_alignment == "auto":
            qc_thread = threading.Thread(
                target=self._detect_hardsubs, args=(video, qc_result), daemon=True
            )
            qc_thread.start()

        console.print("[bold]4-5/6 Translate + synthesize (overlapped)[/bold]")
        translator = OpenAICompatibleTranslator(self.cfg, load_glossary(self.cfg.glossary_path))
        tts = CosyVoiceTTS(self.cfg)
        seg_dir = job_dir / "tts"
        fit_dir = job_dir / "tts_fit"

        def context(i: int) -> tuple[str, str]:
            prev = units[i - 1].source if i > 0 else ""
            nxt = units[i + 1].source if i + 1 < len(units) else ""
            return prev, nxt

        def synthesize_unit(unit: TranslationUnit) -> None:
            raw = seg_dir / f"{unit.id:05d}.wav"
            fitted = fit_dir / f"{unit.id:05d}.wav"
            if resume and fitted.exists():
                unit.tts_path = str(fitted)
                unit.tts_duration = probe_duration(fitted)
                return
            check_stop()
            while True:
                tts.synthesize(unit.translation or unit.source, raw)
                actual = probe_duration(raw)
                ratio = actual / max(unit.duration, 0.01)
                if (
                    ratio > self.cfg.duration_rewrite_threshold
                    and unit.attempts < self.cfg.max_rewrite_attempts
                ):
                    unit.translation = translator.compress(unit, actual)
                    unit.attempts += 1
                    write_json(units_path, [u.model_dump() for u in units])
                    continue
                unit.speed = fit_audio(
                    raw,
                    fitted,
                    unit.duration,
                    self.cfg.duration_soft_min,
                    self.cfg.duration_soft_max,
                )
                unit.tts_path = str(fitted)
                unit.tts_duration = probe_duration(fitted)
                break
            write_json(units_path, [u.model_dump() for u in units])

        pending = [i for i, unit in enumerate(units) if not (resume and unit.translation)]
        workers = max(1, min(self.cfg.translate_workers, len(pending) or 1))
        if workers > 1 and pending:
            console.print(f"[dim]{workers} parallel translation workers[/dim]")
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(translator.translate, units[i], *context(i)): i for i in pending}
            # Synthesize units whose translations already exist (fresh jobs: none;
            # resumed jobs: all previously translated ones) while translations run.
            for unit in units:
                if not (resume and unit.translation):
                    continue
                synthesize_unit(unit)
            for fut in as_completed(futures):
                check_stop()
                i = futures[fut]
                units[i].translation = fut.result()
                write_json(units_path, [u.model_dump() for u in units])
                synthesize_unit(units[i])
            pool.shutdown(wait=True)
        except PipelineCancelled:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

        console.print("[bold]6/6 Compose + render[/bold]")
        check_stop()
        subtitle = write_srt(units, job_dir / "zh.srt")
        total = probe_duration(video)
        timeline = compose_timeline(
            [(u.start, Path(u.tts_path)) for u in units if u.tts_path], total, job_dir / "dub.wav"
        )
        alignment = self.cfg.subtitle_alignment
        if qc_thread is not None:
            qc_thread.join(timeout=180)
            if "error" in qc_result:
                console.print(f"[yellow]Hardsub detection failed: {qc_result['error']}[/yellow]")
            alignment = qc_result.get("alignment", "bottom")
        alignment_code = _ALIGNMENT_CODES.get(alignment, 2)
        final = render_video(
            video,
            timeline,
            subtitle,
            job_dir / "final.zh.mp4",
            self.cfg.original_audio_gain_db,
            self.cfg.dub_audio_gain_db,
            subtitle_force_style=f"Alignment={alignment_code}",
            render_preset=self.cfg.render_preset,
        )
        state.subtitle_path = str(subtitle)
        state.final_path = str(final)
        write_json(state_path, state.model_dump(mode="json"))

        if self.cfg.qc_enabled:
            self._check_render(final, job_dir)

        if self.cfg.cleanup_intermediates:
            console.print("[dim]Cleaning intermediates (tts/, dub.wav, source/*.m4s)[/dim]")
            cleanup_intermediates(job_dir)
        return final

    def _detect_hardsubs(self, video: Path, qc_result: dict[str, str]) -> None:
        try:
            qc_result["alignment"] = VisionQC(self.cfg).detect_subtitles(video)
            console.print(f"[dim]Hardsub detection: {qc_result['alignment']}[/dim]")
        except Exception as e:
            qc_result["error"] = str(e)

    def _check_render(self, final: Path, job_dir: Path) -> None:
        try:
            report = VisionQC(self.cfg).check_render(final)
            (job_dir / "qc_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if report.get("status") == "ok":
                if report.get("subtitles_present") and not report.get("issues"):
                    console.print("[green]Render QC passed[/green]")
                else:
                    console.print(f"[yellow]Render QC issues: {report.get('issues')}[/yellow]")
            else:
                console.print(f"[yellow]Render QC skipped/failed: {report.get('issues')}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Render QC failed: {e}[/yellow]")

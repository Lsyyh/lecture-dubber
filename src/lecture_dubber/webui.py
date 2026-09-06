"""Local web UI for the dubbing pipeline.

Serves a single-page frontend plus a small JSON API. Job progress is derived
from the on-disk resume artifacts (units.json, state.json, final.zh.mp4), so it
stays consistent with CLI runs and needs no extra bookkeeping. Only one job
runs at a time (single GPU); further requests are rejected while busy.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from .models import Config

try:  # fastapi is optional; only `dubber serve` needs it
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, Response
except ImportError:
    FastAPI = HTTPException = Request = FileResponse = Response = None
from .pipeline import Pipeline, PipelineCancelled
from .utils import read_json

_NAME_RE = re.compile(r"[\w\-.]+")
_UNIT_RE = re.compile(r"\d+")


class StartBody(BaseModel):
    source: str


@dataclass
class RunningJob:
    name: str
    thread: threading.Thread
    stop_event: threading.Event
    error: str | None = None


class JobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: RunningJob | None = None

    def start(self, cfg: Config, value: str, job_dir: Path) -> RunningJob:
        with self._lock:
            if self._running is not None and self._running.thread.is_alive():
                raise RuntimeError("another job is already running")
            stop_event = threading.Event()
            job: RunningJob | None = None

            def target() -> None:
                try:
                    Pipeline(cfg).run(value, job_dir=job_dir, resume=True, stop_event=stop_event)
                except PipelineCancelled:
                    if job is not None:
                        job.error = "cancelled"
                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    (cfg.work_dir / "webui_error.log").write_text(tb, encoding="utf-8")
                    if job is not None:
                        job.error = str(e)

            thread = threading.Thread(target=target, daemon=True)
            job = RunningJob(name=job_dir.name, thread=thread, stop_event=stop_event)
            self._running = job
            thread.start()
            return job

    def current(self) -> RunningJob | None:
        with self._lock:
            return self._running


def _units_summary(units_path: Path, fit_dir: Path) -> list[dict]:
    if not units_path.exists():
        return []
    units = read_json(units_path)
    return [
        {
            "id": u["id"],
            "start": u["start"],
            "end": u["end"],
            "source": u["source"],
            "translation": u["translation"],
            "audio_ready": (fit_dir / f"{u['id']:05d}.wav").exists(),
        }
        for u in units
    ]


def job_status(cfg: Config, name: str, registry: JobRegistry | None = None) -> dict:
    """Derive the display state of one job from its on-disk artifacts."""
    job_dir = cfg.work_dir / name
    if not job_dir.is_dir():
        raise FileNotFoundError(name)
    fit_dir = job_dir / "tts_fit"
    units = _units_summary(job_dir / "units.json", fit_dir)
    translated = sum(1 for u in units if u["translation"])
    synthesized = sum(1 for u in units if u["audio_ready"])
    final = job_dir / "final.zh.mp4"

    if not (job_dir / "state.json").exists():
        stage = "starting"
    elif not (job_dir / "audio.wav").exists():
        stage = "importing"
    elif not (job_dir / "segments.json").exists():
        stage = "transcribing"
    elif not units:
        stage = "merging"
    elif final.exists():
        stage = "done"
    elif translated < len(units):
        stage = "translating"
    else:
        stage = "synthesizing"

    running = bool(registry and registry.current() and registry.current().name == name
                   and registry.current().thread.is_alive())
    error = None
    if registry and registry.current() and registry.current().name == name:
        error = registry.current().error

    qc = None
    qc_path = job_dir / "qc_report.json"
    if qc_path.exists():
        qc = read_json(qc_path)

    return {
        "name": name,
        "stage": stage,
        "running": running,
        "error": error,
        "units_total": len(units),
        "units_translated": translated,
        "units_synthesized": synthesized,
        "final_ready": final.exists(),
        "qc": qc,
    }


def create_app(cfg: Config) -> object:
    from .source import bilibili_cache_title

    static_dir = Path(__file__).parent / "static"
    registry = JobRegistry()

    app = FastAPI(title="lecture-dubber")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        import httpx

        try:
            r = httpx.get(cfg.llm_base_url.rstrip("/") + "/health", timeout=3)
            return {"llm_ok": r.status_code == 200 and r.json().get("status") == "ok"}
        except Exception:
            return {"llm_ok": False}

    @app.post("/api/jobs/{name}/cancel")
    def cancel(name: str) -> dict:
        job = registry.current()
        if job is None or job.name != name:
            raise HTTPException(404, "job is not running")
        job.stop_event.set()
        return {"cancelling": True}

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        out = []
        if cfg.work_dir.is_dir():
            for d in sorted(cfg.work_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if d.is_dir() and (d / "state.json").exists() and _NAME_RE.fullmatch(d.name):
                    out.append(job_status(cfg, d.name, registry))
        return out

    @app.post("/api/jobs")
    def start_job(body: StartBody) -> dict:
        value = body.source.strip().strip('"')
        if not value:
            raise HTTPException(400, "empty source")
        job_dir = Pipeline(cfg).job_dir_for(value)
        try:
            registry.start(cfg, value, job_dir)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"name": job_dir.name, "title": bilibili_cache_title(value) or Path(value).stem}

    @app.get("/api/jobs/{name}/status")
    def status(name: str) -> dict:
        if not _NAME_RE.fullmatch(name):
            raise HTTPException(400, "bad job name")
        try:
            return job_status(cfg, name, registry)
        except FileNotFoundError as e:
            raise HTTPException(404, name) from e

    @app.get("/api/jobs/{name}/units")
    def units(name: str) -> list[dict]:
        if not _NAME_RE.fullmatch(name):
            raise HTTPException(400, "bad job name")
        return _units_summary(cfg.work_dir / name / "units.json", cfg.work_dir / name / "tts_fit")

    @app.get("/api/jobs/{name}/audio/{unit_id}")
    def audio(name: str, unit_id: str):
        if not _NAME_RE.fullmatch(name) or not _UNIT_RE.fullmatch(unit_id):
            raise HTTPException(400, "bad path")
        path = cfg.work_dir / name / "tts_fit" / f"{int(unit_id):05d}.wav"
        if not path.exists():
            raise HTTPException(404, "audio not ready")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/jobs/{name}/srt")
    def srt(name: str):
        if not _NAME_RE.fullmatch(name):
            raise HTTPException(400, "bad job name")
        path = cfg.work_dir / name / "zh.srt"
        if not path.exists():
            raise HTTPException(404, "srt not ready")
        return FileResponse(path, media_type="application/x-subrip", filename="zh.srt")

    def _range_response(path: Path, request: Request, media_type: str):
        range_header = request.headers.get("range")
        file_size = path.stat().st_size
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m and (m.group(1) or m.group(2)):
                start = int(m.group(1) or 0)
                end = int(m.group(2)) if m.group(2) else min(start + 4 * 1024 * 1024, file_size - 1)
                end = min(end, file_size - 1)
                with path.open("rb") as f:
                    f.seek(start)
                    data = f.read(end - start + 1)
                return Response(
                    content=data, media_type=media_type, status_code=206,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                    },
                )
        return FileResponse(path, media_type=media_type)

    @app.get("/api/jobs/{name}/video")
    def video(name: str, request: Request):
        if not _NAME_RE.fullmatch(name):
            raise HTTPException(400, "bad job name")
        path = cfg.work_dir / name / "final.zh.mp4"
        if not path.exists():
            raise HTTPException(404, "video not ready")
        return _range_response(path, request, "video/mp4")

    @app.get("/api/jobs/{name}/source_video")
    def source_video(name: str, request: Request):
        if not _NAME_RE.fullmatch(name):
            raise HTTPException(400, "bad job name")
        state_path = cfg.work_dir / name / "state.json"
        if not state_path.exists():
            raise HTTPException(404, "job not found")
        video_path = Path(read_json(state_path)["source"]["video_path"])
        if not video_path.exists():
            raise HTTPException(404, "source video missing")
        return _range_response(video_path, request, "video/mp4")

    return app

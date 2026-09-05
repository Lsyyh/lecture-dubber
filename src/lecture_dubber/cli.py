from __future__ import annotations

from pathlib import Path
import typer
from rich.console import Console

from .config import load_config
from .pipeline import Pipeline
from .source import import_source

app = typer.Typer(no_args_is_help=True, help="Local-first lecture dubbing pipeline")
console = Console()


@app.command()
def run(
    source: str = typer.Argument(..., help="Local video path or YouTube/Bilibili/other yt-dlp URL"),
    config: Path | None = typer.Option(Path("config.yaml"), "--config", "-c"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Job directory"),
    no_resume: bool = typer.Option(False, help="Ignore cached stage outputs"),
):
    cfg = load_config(config if config and config.exists() else None)
    final = Pipeline(cfg).run(source, job_dir=output, resume=not no_resume)
    console.print(f"[green]Done:[/green] {final}")


@app.command()
def fetch(
    source: str,
    output: Path = typer.Option(Path("outputs/fetch"), "--output", "-o"),
    config: Path | None = typer.Option(Path("config.yaml"), "--config", "-c"),
):
    cfg = load_config(config if config and config.exists() else None)
    info = import_source(source, output, cfg)
    console.print_json(data=info.model_dump(mode="json"))


@app.command()
def doctor(config: Path | None = typer.Option(Path("config.yaml"), "--config", "-c")):
    import shutil
    cfg = load_config(config if config and config.exists() else None)
    checks = {
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "yt-dlp": True,
        "glossary": cfg.glossary_path.exists(),
        "cosyvoice_root": cfg.cosyvoice_root.exists(),
        "cosyvoice_model": cfg.cosyvoice_model_dir.exists(),
    }
    try:
        import whisperx  # noqa: F401
        checks["whisperx"] = True
    except Exception:
        checks["whisperx"] = False
    for k, ok in checks.items():
        console.print(f"{'[green]OK[/green]' if ok else '[red]MISS[/red]'} {k}")

from lecture_dubber.models import Config
from lecture_dubber.webui import job_status


def _make_job(tmp_path):
    cfg = Config(work_dir=tmp_path)
    job = tmp_path / "my-job"
    job.mkdir()
    (job / "state.json").write_text("{}", encoding="utf-8")
    (job / "audio.wav").write_bytes(b"x")
    (job / "segments.json").write_text("[]", encoding="utf-8")
    (job / "units.json").write_text(
        '[{"id":0,"start":0,"end":2,"source":"a","translation":"甲"},'
        '{"id":1,"start":2,"end":4,"source":"b","translation":null}]',
        encoding="utf-8",
    )
    return cfg, job


def test_status_stages_and_counts(tmp_path):
    cfg, _job = _make_job(tmp_path)
    st = job_status(cfg, "my-job")
    assert st["stage"] == "translating"
    assert st["units_total"] == 2
    assert st["units_translated"] == 1
    assert st["units_synthesized"] == 0
    assert not st["final_ready"]


def test_status_done_and_missing_job(tmp_path):
    cfg, job = _make_job(tmp_path)
    (job / "final.zh.mp4").write_bytes(b"x")
    (job / "tts_fit").mkdir()
    (job / "tts_fit" / "00000.wav").write_bytes(b"x")
    st = job_status(cfg, "my-job")
    assert st["stage"] == "done" and st["final_ready"]
    import pytest
    with pytest.raises(FileNotFoundError):
        job_status(cfg, "nope")

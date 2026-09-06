from lecture_dubber.models import TranslationUnit
from lecture_dubber.subtitles import write_srt


def test_srt(tmp_path):
    p = write_srt(
        [TranslationUnit(id=0, start=1.2, end=3.4, source="x", translation="你好")],
        tmp_path / "x.srt",
    )
    txt = p.read_text(encoding="utf-8")
    assert "00:00:01,200 --> 00:00:03,400" in txt
    assert "你好" in txt

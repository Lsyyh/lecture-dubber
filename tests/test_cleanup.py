from lecture_dubber.utils import cleanup_intermediates


def test_cleanup_intermediates_removes_only_regenrable_artifacts(tmp_path):
    tts = tmp_path / "tts"
    fit = tmp_path / "tts_fit"
    src = tmp_path / "source"
    tts.mkdir()
    fit.mkdir()
    src.mkdir()
    (tts / "00000.wav").write_bytes(b"x")
    (fit / "00000.wav").write_bytes(b"x")
    (src / "page1_video.m4s").write_bytes(b"x")
    (src / "lecture.mp4").write_bytes(b"x")
    (tmp_path / "dub.wav").write_bytes(b"x")
    (tmp_path / "units.json").write_bytes(b"x")

    cleanup_intermediates(tmp_path)

    assert not tts.exists()
    assert not (tmp_path / "dub.wav").exists()
    assert not (src / "page1_video.m4s").exists()
    assert (fit / "00000.wav").exists()
    assert (src / "lecture.mp4").exists()
    assert (tmp_path / "units.json").exists()

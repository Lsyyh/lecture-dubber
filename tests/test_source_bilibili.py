from lecture_dubber.source import _cache_pages, _copy_stripped, bilibili_cache_title


def _make_cache(folder, title="My Lecture"):
    (folder / "123-1-30032.m4s").write_bytes(b"000000000ftypvideo")
    (folder / "123-1-30280.m4s").write_bytes(b"000000000ftypaudio")
    (folder / "123-2-30032.m4s").write_bytes(b"000000000ftypvideo2")
    (folder / "videoInfo.json").write_text(
        f'{{"title": "{title}", "bvid": "BV1xx"}}', encoding="utf-8"
    )


def test_cache_pages_groups_by_page_and_kind(tmp_path):
    _make_cache(tmp_path)
    pages = _cache_pages(tmp_path)
    assert set(pages) == {1, 2}
    assert pages[1]["video"][0][0] == 30032
    assert pages[1]["audio"][0][0] == 30280
    assert pages[2]["audio"] == []


def test_cache_title(tmp_path):
    _make_cache(tmp_path)
    assert bilibili_cache_title(str(tmp_path)) == "My Lecture"
    assert bilibili_cache_title(str(tmp_path / "nonexistent")) is None


def test_copy_stripped_removes_nine_byte_header(tmp_path):
    src = tmp_path / "in.m4s"
    src.write_bytes(b"000000000REALDATA")
    dst = _copy_stripped(src, tmp_path / "out.m4s")
    assert dst.read_bytes() == b"REALDATA"


def test_copy_stripped_keeps_unknown_header(tmp_path):
    src = tmp_path / "in.m4s"
    src.write_bytes(b"NOTJUNK")
    dst = _copy_stripped(src, tmp_path / "out.m4s")
    assert dst.read_bytes() == b"NOTJUNK"

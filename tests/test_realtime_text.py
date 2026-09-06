from __future__ import annotations

from lecture_dubber.realtime_text import (
    ClauseSegmenter,
    StablePrefixBuffer,
    latency_policy,
    strip_fillers,
)


def words(text: str, t0: float = 0.0, wps: float = 3.0) -> list:
    out = []
    t = t0
    for w in text.split():
        out.append(type("W", (), {"word": w, "start_ts": t, "end_ts": t + 0.3})())
        t += 1 / wps
    return out


class TestStablePrefixBuffer:
    def test_commits_growing_prefix(self):
        buf = StablePrefixBuffer(trailing_keep=1)
        h1 = words("the gradient of this loss", t0=10.0)
        assert buf.update(h1, now_ts=11.0) == []
        h2 = words("the gradient of this loss function", t0=10.0)
        got = buf.update(h2, now_ts=11.5)
        # LCP=5 words, minus trailing margin of 1 -> commit 4
        assert [w.word for w in got] == ["the", "gradient", "of", "this"]
        assert got[0].start_ts == 10.0
        assert abs(got[-1].end_ts - 11.0) < 1.0

    def test_revision_shrinks_commit(self):
        buf = StablePrefixBuffer(trailing_keep=1)
        buf.update(words("the gradient of this loss", t0=10.0), now_ts=11.0)
        # ASR revises "gradient" -> "gradients": LCP is just "the"
        got = buf.update(words("the gradients of this loss", t0=10.0), now_ts=11.0)
        assert got == []

    def test_force_commit_on_age(self):
        buf = StablePrefixBuffer(trailing_keep=1, max_pending_s=2.5)
        buf.update(words("the gradient", t0=10.0), now_ts=11.0)
        # same hypothesis again, but the pending span is now old
        got = buf.update(words("the gradient", t0=10.0), now_ts=14.0)
        # LCP=2, trailing_keep=1 -> commit 1 word; age forces nothing new here
        assert [w.word for w in got] == ["the"]

    def test_force_commit_long_pending(self):
        buf = StablePrefixBuffer(trailing_keep=1, max_pending_s=2.5)
        buf.update(words("aaa bbb ccc", t0=10.0), now_ts=11.0)
        got = buf.update(
            words("aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll", t0=10.0), now_ts=11.0
        )
        # LCP=3, pending span words[3]..end is 9/3=3s > 2.5 -> force commit all but last
        assert [w.word for w in got] == [
            "aaa",
            "bbb",
            "ccc",
            "ddd",
            "eee",
            "fff",
            "ggg",
            "hhh",
            "iii",
            "jjj",
            "kkk",
        ]

    def test_empty_hypothesis_keeps_state(self):
        buf = StablePrefixBuffer()
        buf.update(words("the gradient", t0=10.0), now_ts=11.0)
        assert buf.update([], now_ts=12.0) == []


class TestClauseSegmenter:
    def test_sentence_punctuation_emits(self):
        seg = ClauseSegmenter()
        got = seg.add(
            words("the learning rate is too large .", t0=10.0), latency_policy(0.0).clause
        )
        assert len(got) == 1
        text, start, _end = got[0]
        assert text == "the learning rate is too large"
        assert start == 10.0

    def test_word_limit_emits(self):
        seg = ClauseSegmenter()
        policy = latency_policy(0.0).clause
        got = seg.add(words(" ".join(f"w{i}" for i in range(20)), t0=10.0), policy)
        assert len(got) == 1
        assert got[0][0].split()[0] == "w0"

    def test_tick_flushes_after_silence(self):
        seg = ClauseSegmenter()
        policy = latency_policy(0.0).clause
        assert seg.add(words("the learning rate is too large", t0=10.0), policy) == []
        assert seg.tick(now_ts=11.0, policy=policy) == []
        tail_end = seg.buffer[-1].end_ts
        got = seg.tick(now_ts=tail_end + policy.max_wait_s + 0.1, policy=policy)
        assert len(got) == 1

    def test_soft_break_on_comma(self):
        seg = ClauseSegmenter()
        policy = latency_policy(0.0).clause
        got = seg.add(words("first we compute the gradient , then we update", t0=10.0), policy)
        # comma lands after 6 words -> soft break fires inside add()
        assert len(got) == 1
        assert got[0][0].startswith("first we compute")

    def test_short_buffer_not_flushed_by_tick(self):
        seg = ClauseSegmenter()
        policy = latency_policy(0.0).clause
        seg.add(words("okay", t0=10.0), policy)
        assert seg.tick(now_ts=20.0, policy=policy) == []


class TestLatencyPolicy:
    def test_mode_ladder(self):
        assert latency_policy(1.0).mode == "NORMAL"
        assert latency_policy(3.0).mode == "CONCISE"
        assert latency_policy(5.0).mode == "AGGRESSIVE"
        assert latency_policy(7.0).mode == "RECOVERY"

    def test_speed_rises_and_clauses_shrink(self):
        normal = latency_policy(1.0)
        recovery = latency_policy(7.0)
        assert recovery.tts_speed > normal.tts_speed
        assert recovery.clause.max_words < normal.clause.max_words
        assert recovery.clause.max_wait_s < normal.clause.max_wait_s

    def test_strip_fillers(self):
        assert strip_fillers("um so the gradient is basically large") == "the gradient is large"
        assert strip_fillers("loss function") == "loss function"

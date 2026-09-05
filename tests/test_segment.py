from lecture_dubber.models import Config, Segment
from lecture_dubber.segment import merge_segments


def test_merge_segments_groups_short_phrases():
    cfg = Config(merge_target_duration=6, merge_max_duration=10, merge_max_gap=1.0)
    segs = [
        Segment(id=0, start=0, end=2, text="So today"),
        Segment(id=1, start=2.1, end=4.2, text="we discuss gradients."),
        Segment(id=2, start=4.4, end=7.0, text="First, consider this objective."),
    ]
    units = merge_segments(segs, cfg)
    assert units
    assert units[0].source.startswith("So today")
    assert units[-1].end == 7.0

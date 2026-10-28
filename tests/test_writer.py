from __future__ import annotations

import json

import pytest

from coco_detect import SCHEMA
from coco_detect.writer import RawDetectionsWriter


def _header():
    return {"schema": SCHEMA, "start_frame": 0}


def test_header_is_first_line_and_frames_follow(tmp_path):
    path = tmp_path / "out.jsonl"
    with RawDetectionsWriter(path, _header()) as w:
        w.write_frame(0, 0.0, [])
        w.write_frame(1, 1 / 30, [{"bbox": [1, 2, 3, 4], "score": 0.9, "label": "person"}])

    lines = [json.loads(x) for x in path.read_text().strip().splitlines()]
    assert lines[0]["schema"] == SCHEMA
    assert [r["frame_index"] for r in lines[1:]] == [0, 1]
    assert lines[2]["detections"][0]["label"] == "person"


def test_gap_raises(tmp_path):
    with RawDetectionsWriter(tmp_path / "out.jsonl", _header()) as w:
        w.write_frame(10, 0.0, [])
        with pytest.raises(ValueError, match="non-contiguous"):
            w.write_frame(12, 0.0, [])


def test_counters(tmp_path):
    with RawDetectionsWriter(tmp_path / "out.jsonl", _header()) as w:
        w.write_frame(0, 0.0, [{"bbox": [0, 0, 1, 1], "score": 0.5, "label": "person"}] * 3)
        w.write_frame(1, 0.0, [])
        assert w.frames_written == 2
        assert w.detections_written == 3


def test_frame_step_allows_stride_and_rejects_other_gaps(tmp_path):
    with RawDetectionsWriter(tmp_path / "out.jsonl", _header(), frame_step=10) as w:
        w.write_frame(5, 0.0, [])
        w.write_frame(15, 0.0, [])
        with pytest.raises(ValueError, match="non-contiguous"):
            w.write_frame(16, 0.0, [])

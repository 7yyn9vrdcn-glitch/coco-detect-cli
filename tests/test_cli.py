from __future__ import annotations

import json

import pytest

from coco_detect import SCHEMA, __version__
from coco_detect.cli import build_parser, cmd_run


class _FakeDetector:
    """Returns one fixed detection per frame."""

    def detect(self, image):
        return [{"bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9, "label": "person"}]

    def detect_batch(self, images):
        return [self.detect(i) for i in images]


def _make_video(path, frames=12, fps=30.0, size=(64, 48)):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size,
    )
    assert writer.isOpened()
    for i in range(frames):
        img = np.full((size[1], size[0], 3), i * 10 % 255, dtype=np.uint8)
        writer.write(img)
    writer.release()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_run_end_to_end_with_fake_detector(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=12, fps=30.0)
    out = tmp_path / "raw.jsonl"

    args = build_parser().parse_args([
        "run", "--input", str(video), "--output", str(out),
        "--model", "fake.pt", "--confidence", "0.25",
        "--classes", "person,sports ball", "--batch-size", "4",
    ])
    rc = cmd_run(args, detector=_FakeDetector())
    assert rc == 0

    lines = [json.loads(x) for x in out.read_text().strip().splitlines()]
    header, frames = lines[0], lines[1:]
    assert header["schema"] == SCHEMA
    assert header["start_frame"] == 0
    assert header["confidence"] == 0.25
    assert header["classes"] == ["person", "sports ball"]
    assert header["runner_version"] == __version__
    # one record per frame, contiguous from 0
    assert [r["frame_index"] for r in frames] == list(range(len(frames)))
    assert len(frames) == 12
    assert all(len(r["detections"]) == 1 for r in frames)


def test_run_window_uses_absolute_frame_numbers(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=30, fps=30.0)
    out = tmp_path / "raw.jsonl"

    args = build_parser().parse_args([
        "run", "--input", str(video), "--output", str(out),
        "--model", "fake.pt",
        "--start-seconds", "0.5", "--window-seconds", "0.2",
    ])
    rc = cmd_run(args, detector=_FakeDetector())
    assert rc == 0

    lines = [json.loads(x) for x in out.read_text().strip().splitlines()]
    header, frames = lines[0], lines[1:]
    # start 0.5s @30fps -> frame 15; window 0.2s -> 6 frames
    assert header["start_frame"] == 15
    assert [r["frame_index"] for r in frames] == list(range(15, 21))
    assert frames[0]["timestamp"] == pytest.approx(0.5)


def test_run_start_past_end_errors(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=10, fps=30.0)
    args = build_parser().parse_args([
        "run", "--input", str(video), "--output", str(tmp_path / "o.jsonl"),
        "--model", "fake.pt", "--start-seconds", "999",
    ])
    assert cmd_run(args, detector=_FakeDetector()) == 2

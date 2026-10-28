from __future__ import annotations

import json

import pytest

from coco_detect import SCHEMA, __version__
from coco_detect.cli import build_parser, cmd_run


class _FakeDetector:
    """Returns one fixed detection per frame."""

    def detect(self, image):
        return [{"bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9, "label": "person"}]

    def detect_batch(self, images, imgsz=None):
        return [self.detect(i) for i in images]


class _MeanDetector:
    """Reports the frame's mean brightness as the score, to check frame order."""

    def detect(self, image):
        return [{"bbox": [0.0, 0.0, 1.0, 1.0], "score": float(image.mean()), "label": "x"}]

    def detect_batch(self, images, imgsz=None):
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


def _run(tmp_path, video, *extra, detector=None):
    out = tmp_path / "raw.jsonl"
    args = build_parser().parse_args([
        "run", "--input", str(video), "--output", str(out), "--model", "fake.pt", *extra,
    ])
    rc = cmd_run(args, detector=detector or _FakeDetector())
    if rc != 0:
        return rc, None, None
    lines = [json.loads(x) for x in out.read_text().strip().splitlines()]
    return rc, lines[0], lines[1:]


def test_frame_step_keeps_source_frame_numbers(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=30, fps=30.0)
    rc, header, frames = _run(tmp_path, video, "--frame-step", "10", detector=_MeanDetector())
    assert rc == 0
    assert header["frame_step"] == 10
    assert [r["frame_index"] for r in frames] == [0, 10, 20]
    # frame i is filled with i*10: the decoded image must be that frame, not a neighbour
    assert [round(r["detections"][0]["score"] / 10) for r in frames] == [0, 10, 20]


def test_frame_step_with_window(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=30, fps=30.0)
    rc, header, frames = _run(
        tmp_path, video, "--frame-step", "5", "--start-seconds", "0.5",
        "--window-seconds", "0.5", detector=_MeanDetector(),
    )
    assert rc == 0
    assert header["start_frame"] == 15
    assert [r["frame_index"] for r in frames] == [15, 20, 25]
    assert [round(r["detections"][0]["score"] / 10) for r in frames] == [15, 20, 25]


def test_vf_is_applied(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=3, fps=30.0)
    rc, header, frames = _run(tmp_path, video, "--vf", "negate", detector=_MeanDetector())
    assert rc == 0
    assert header["vf"] == "negate"
    assert [r["frame_index"] for r in frames] == [0, 1, 2]
    assert frames[0]["detections"][0]["score"] > 240  # black frame negated


def test_bad_vf_fails_loudly(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=3, fps=30.0)
    with pytest.raises(RuntimeError, match="ffmpeg exited"):
        _run(tmp_path, video, "--vf", "no_such_filter")


def test_regions_map_to_source_and_header_keeps_source_size(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=2, fps=30.0, size=(64, 48))
    rc, header, frames = _run(tmp_path, video, "--region", "32,8,32,32@32")
    assert rc == 0
    assert (header["input"]["width"], header["input"]["height"]) == (64, 48)
    assert header["regions"] == ["32,8,32,32@32"]
    assert frames[0]["detections"][0]["bbox"] == [42.0, 18.0, 52.0, 28.0]


def test_region_outside_frame_errors(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=2, fps=30.0, size=(64, 48))
    rc, _, _ = _run(tmp_path, video, "--region", "40,0,32,32@32")
    assert rc == 2


def test_region_with_input_scale_errors(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, frames=2, fps=30.0)
    rc, _, _ = _run(tmp_path, video, "--region", "full@32", "--input-scale", "0.5")
    assert rc == 2

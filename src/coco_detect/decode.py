"""Video decoding helpers.

Frame numbering contract (kept deliberately simple and deterministic so that
downstream consumers can decode the same video independently and line up):

- ``frame_index`` is the ABSOLUTE frame number in the source video. Seeking
  to ``start_frame`` does not reset numbering.
- ``timestamp = frame_index / fps`` where fps comes from the container probe.
- No frame sampling/striding: every frame in the window is decoded.
- ``input_scale`` resizes frames before inference; emitted bounding boxes are
  in the scaled coordinate space (recorded in the output header).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass
class Frame:
    frame_index: int
    timestamp: float
    image: Any  # BGR ndarray


def probe_video(video_path: Path, fallback_fps: float = 30.0) -> VideoInfo:
    """Return frame count / fps / dimensions by probing the container."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or fallback_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    finally:
        cap.release()
    return VideoInfo(frame_count=frame_count, fps=float(fps), width=width, height=height)


def iter_frames(
    video_path: Path,
    fps: float,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
    input_scale: float = 1.0,
) -> Iterator[Frame]:
    """Yield BGR frames with absolute frame numbering.

    ``start_frame`` seeks before reading; ``max_frames`` caps the number of
    yielded frames (None = until end of stream).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    frame_index = start_frame
    yielded = 0
    try:
        while True:
            if max_frames is not None and yielded >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if input_scale != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * input_scale), int(h * input_scale)))
            yield Frame(
                frame_index=frame_index,
                timestamp=frame_index / fps,
                image=frame,
            )
            frame_index += 1
            yielded += 1
    finally:
        cap.release()

    LOGGER.info("decoded %d frames from %s (start=%d)", yielded, video_path, start_frame)

"""Video decoding helpers.

Frame numbering contract (kept deliberately simple and deterministic so that
downstream consumers can decode the same video independently and line up):

- ``frame_index`` is the ABSOLUTE frame number in the source video. Seeking
  to ``start_frame`` does not reset numbering.
- ``timestamp = frame_index / fps`` where fps comes from the container probe.
- Every frame in the window is decoded. With ``frame_step=N`` only every Nth
  frame (counted from the window start) is yielded, so consecutive indices
  differ by exactly N.
- ``input_scale`` resizes frames before inference; emitted bounding boxes are
  in the scaled coordinate space (recorded in the output header).
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
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


def probe_video(video_path: str | Path, fallback_fps: float = 30.0) -> VideoInfo:
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
    video_path: str | Path,
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


def iter_frames_ffmpeg(
    source: str | Path,
    fps: float,
    width: int,
    height: int,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
    frame_step: int = 1,
    vf: str | None = None,
    input_scale: float = 1.0,
) -> Iterator[Frame]:
    """Decode through an ffmpeg pipe (needed for ``vf`` and cheap striding).

    ffmpeg decodes every frame; the ``select`` filter drops the ones between
    steps before ``vf`` runs, so an expensive ``vf`` (e.g. tonemapping) only
    touches yielded frames. ``vf`` must not change the frame size.
    """
    filters = []
    if frame_step > 1:
        filters.append(f"select='not(mod(n\\,{frame_step}))'")
    if vf:
        filters.append(vf)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start_frame > 0:
        cmd += ["-ss", f"{start_frame / fps:.6f}"]
    cmd += ["-i", str(source)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-fps_mode", "passthrough"]
    if max_frames is not None:
        cmd += ["-frames:v", str(math.ceil(max_frames / frame_step))]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    LOGGER.info("ffmpeg decode: %s", " ".join(cmd))

    import cv2
    import numpy as np

    frame_bytes = width * height * 3
    yielded = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    if buf:
                        LOGGER.warning("ffmpeg: dropped a partial frame (%d bytes)", len(buf))
                    break
                image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
                if input_scale != 1.0:
                    image = cv2.resize(
                        image, (int(width * input_scale), int(height * input_scale)),
                    )
                frame_index = start_frame + yielded * frame_step
                yield Frame(frame_index=frame_index, timestamp=frame_index / fps, image=image)
                yielded += 1
        except BaseException:  # consumer stopped early or failed: do not leave ffmpeg running
            proc.kill()
            proc.wait()
            raise
        finally:
            proc.stdout.close()
        returncode = proc.wait()
        if returncode != 0:
            err.seek(0)
            tail = err.read()[-2000:].decode("utf-8", "replace")
            raise RuntimeError(f"ffmpeg exited with {returncode}: {tail}")

    LOGGER.info(
        "decoded %d frames from %s via ffmpeg (start=%d step=%d)",
        yielded, source, start_frame, frame_step,
    )

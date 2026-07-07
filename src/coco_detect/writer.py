"""JSONL output writer for the ``raw_detections/v1`` schema.

File layout:

- Line 1: header record — ``schema``, ``input`` (path/fps/width/height/
  frame_count), ``model``, ``confidence``, ``classes``, ``start_frame``,
  ``input_scale``, ``runner_version``.
  ``input.width/height`` describe the COORDINATE SPACE of the emitted
  bounding boxes (i.e. after ``input_scale`` is applied).
- Following lines: exactly one record per decoded frame, in ascending
  ``frame_index`` order with NO GAPS — a frame with zero detections still
  gets a line. Consumers rely on this to detect misalignment early.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO


class RawDetectionsWriter:
    def __init__(self, path: Path, header: dict, *, flush_every: int = 100) -> None:
        self._path = Path(path)
        self._flush_every = max(1, int(flush_every))
        self._pending = 0
        self._last_frame_index: int | None = None
        self.frames_written = 0
        self.detections_written = 0
        self._fh: IO[str] = self._path.open("w", encoding="utf-8")
        self._fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_frame(self, frame_index: int, timestamp: float, detections: list[dict]) -> None:
        if self._last_frame_index is not None and frame_index != self._last_frame_index + 1:
            raise ValueError(
                f"non-contiguous frame_index: {frame_index} after {self._last_frame_index} "
                "(the schema requires one record per frame with no gaps)"
            )
        self._last_frame_index = frame_index
        rec = {
            "frame_index": int(frame_index),
            "timestamp": float(timestamp),
            "detections": detections,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.frames_written += 1
        self.detections_written += len(detections)
        self._pending += 1
        if self._pending >= self._flush_every:
            self._fh.flush()
            self._pending = 0

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "RawDetectionsWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

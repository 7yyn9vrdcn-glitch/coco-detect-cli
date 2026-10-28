"""Ultralytics YOLO wrapper.

Runs batched inference and converts results to plain dicts with the model's
own class names (COCO names for stock models, e.g. ``person`` /
``sports ball``). No post-processing beyond a confidence threshold and an
optional class-name filter — consumers get the raw detections.

FP16 is intentionally disabled (``half=False``): FP32 keeps results
deterministic across export formats, and half precision has shown recall
regressions on small objects with some engines.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        classes: set[str] | None = None,
        *,
        model: Any = None,
    ) -> None:
        """``model`` is injectable for tests; normally loaded from ``model_path``."""
        self.model_path = model_path
        self.confidence = float(confidence)
        self.classes = set(classes) if classes else None
        self._half = False

        if model is not None:
            self._model = model
        else:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics is not installed; coco-detect requires it "
                    "(pip install ultralytics)"
                ) from exc
            self._model = YOLO(model_path)
            LOGGER.info("loaded YOLO model: %s", model_path)

    def _result_to_dicts(self, result: Any) -> list[dict]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", None) or getattr(self._model, "names", {}) or {}
        out: list[dict] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            label = str(names.get(cls_id, cls_id))
            if self.classes is not None and label not in self.classes:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            out.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(boxes.conf[i].item()),
                    "label": label,
                }
            )
        return out

    def _predict(self, source: Any, imgsz: int | None) -> Any:
        kwargs = {"conf": self.confidence, "half": self._half, "verbose": False}
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        return self._model(source, **kwargs)

    def detect(self, image: Any, imgsz: int | None = None) -> list[dict]:
        results = self._predict(image, imgsz)
        detections: list[dict] = []
        for result in results:
            detections.extend(self._result_to_dicts(result))
        return detections

    def detect_batch(self, images: list[Any], imgsz: int | None = None) -> list[list[dict]]:
        """One forward pass over N frames; returns per-frame lists in input order."""
        if not images:
            return []
        results = self._predict(images, imgsz)
        out = [self._result_to_dicts(r) for r in results]
        if len(out) != len(images):
            # Defensive: some export formats have been observed to collapse
            # batches; fall back to per-frame inference rather than misalign.
            LOGGER.warning(
                "batch returned %d results for %d frames; falling back to per-frame",
                len(out), len(images),
            )
            return [self.detect(img, imgsz) for img in images]
        return out

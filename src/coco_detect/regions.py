"""Region-based inference.

A region is either a rectangle of the source frame (``x,y,w,h@imgsz``) that is
cropped and run at ``imgsz``, or the whole frame shrunk to fit ``imgsz``
(``full@imgsz``, optionally ``full@imgsz:ymin=Y`` to keep only detections
whose box centre is at or below source-frame row ``Y``).

Bounding boxes are always mapped back to source-frame coordinates. Detections
from different regions are concatenated as-is (no cross-region NMS).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Region:
    spec: str
    imgsz: int
    rect: tuple[int, int, int, int] | None  # None = full frame
    ymin: float | None = None


def parse_region(raw: str) -> Region:
    body, sep, tail = raw.partition("@")
    if not sep:
        raise argparse.ArgumentTypeError(f"region needs '@imgsz': {raw!r}")
    imgsz_str, _, option = tail.partition(":")
    try:
        imgsz = int(imgsz_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad imgsz in region: {raw!r}") from None
    if imgsz <= 0:
        raise argparse.ArgumentTypeError(f"imgsz must be positive: {raw!r}")

    if body == "full":
        ymin = None
        if option:
            key, _, value = option.partition("=")
            if key != "ymin":
                raise argparse.ArgumentTypeError(f"unknown full-frame option: {raw!r}")
            try:
                ymin = float(value)
            except ValueError:
                raise argparse.ArgumentTypeError(f"bad ymin in region: {raw!r}") from None
        return Region(raw, imgsz, None, ymin)

    if option:
        raise argparse.ArgumentTypeError(f"options are only valid for 'full': {raw!r}")
    try:
        x, y, w, h = (int(v) for v in body.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"region must be x,y,w,h@imgsz: {raw!r}") from None
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"region out of range: {raw!r}")
    return Region(raw, imgsz, (x, y, w, h))


def region_fits(region: Region, width: int, height: int) -> bool:
    if region.rect is None:
        return True
    x, y, w, h = region.rect
    return x + w <= width and y + h <= height


def _crop(image: Any, region: Region) -> tuple[Any, float, float, float]:
    """Return (image, offset_x, offset_y, scale) mapping crop coords -> source."""
    if region.rect is not None:
        x, y, w, h = region.rect
        return image[y:y + h, x:x + w], float(x), float(y), 1.0
    import cv2

    h, w = image.shape[:2]
    scale = max(w, h) / region.imgsz
    if scale <= 1.0:
        return image, 0.0, 0.0, 1.0
    small = cv2.resize(
        image, (round(w / scale), round(h / scale)), interpolation=cv2.INTER_AREA,
    )
    return small, 0.0, 0.0, scale


def detect_regions(detector, images: list[Any], regions: list[Region]) -> list[list[dict]]:
    """Run every region over every image; one forward pass per distinct imgsz."""
    out: list[list[dict]] = [[] for _ in images]
    by_imgsz: dict[int, list[Region]] = {}
    for region in regions:
        by_imgsz.setdefault(region.imgsz, []).append(region)

    for imgsz, group in by_imgsz.items():
        jobs = []  # (image index, region, ox, oy, scale)
        crops = []
        for i, image in enumerate(images):
            for region in group:
                crop, ox, oy, scale = _crop(image, region)
                crops.append(crop)
                jobs.append((i, region, ox, oy, scale))
        results = detector.detect_batch(crops, imgsz=imgsz)
        for (i, region, ox, oy, scale), detections in zip(jobs, results):
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                bbox = [x1 * scale + ox, y1 * scale + oy, x2 * scale + ox, y2 * scale + oy]
                if region.ymin is not None and (bbox[1] + bbox[3]) / 2 < region.ymin:
                    continue
                out[i].append({**det, "bbox": bbox})
    return out

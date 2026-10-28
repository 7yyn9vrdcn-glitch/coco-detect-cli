"""coco-detect: video object detection pre-pass CLI.

Decodes a video (or a time window of it), runs an Ultralytics YOLO model on
every frame (or every Nth frame, optionally per region), and writes the raw
detections to a JSONL file — one header line followed by one line per
processed frame, with no gaps.

Licensed under AGPL-3.0 (this package links against Ultralytics, which is
AGPL-3.0 licensed).
"""

__version__ = "0.2.0"

SCHEMA = "raw_detections/v1"

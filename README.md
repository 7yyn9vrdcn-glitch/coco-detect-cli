# coco-detect

Video object detection pre-pass CLI. Decodes a video (or a time window of
it), runs an [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
model on every frame, and writes the raw detections to a JSONL file.

It is a generic video→detections tool and contains no application logic:
downstream systems consume the JSONL output through a documented file
interface.

## License

AGPL-3.0 (see `LICENSE`). This package depends on Ultralytics, which is
licensed under AGPL-3.0; this repository is published so that users of
services built on it can obtain the source of the AGPL-covered component.

## Install

```bash
pip install .
# GPU inference requires a working PyTorch + CUDA setup (installed via ultralytics).
```

## Usage

```bash
# Detect persons and balls over the first 10 minutes of a video
coco-detect run \
  --input input.mp4 \
  --output raw_detections.jsonl \
  --model yolo26l.pt \
  --classes "person,sports ball" \
  --confidence 0.25 \
  --batch-size 16 \
  --start-seconds 0 --window-seconds 600

# Export a TensorRT engine (FP32, dynamic batch) for faster inference
coco-detect export-engine --model yolo26l.pt --imgsz 640 --batch 16
```

## Output format (`raw_detections/v1`)

Line 1 is a header record; every following line is one decoded frame, in
ascending `frame_index` order **with no gaps** (frames with zero detections
still get a line):

```jsonl
{"schema":"raw_detections/v1","input":{"path":"input.mp4","fps":30.0,"width":1920,"height":1080,"frame_count":108000},"model":"yolo26l.pt","confidence":0.25,"classes":["person","sports ball"],"start_frame":0,"input_scale":1.0,"runner_version":"0.1.0"}
{"frame_index":0,"timestamp":0.0,"detections":[{"bbox":[100.5,200.1,140.2,300.0],"score":0.87,"label":"person"}]}
{"frame_index":1,"timestamp":0.0333,"detections":[]}
```

Contract notes:

- `frame_index` is the absolute frame number in the source video (window
  runs keep absolute numbering; `timestamp = frame_index / fps`).
- Every frame in the window is processed — no sampling or striding.
- Bounding boxes are `[x1, y1, x2, y2]` in the coordinate space described by
  `input.width` / `input.height` (i.e. after `--input-scale` is applied).
- Labels are the model's own class names (COCO names for stock models,
  e.g. `person`, `sports ball`).
- FP16 is disabled (FP32 only) for deterministic results across export
  formats.

## Development

```bash
pip install -e ".[dev]"
pytest -q   # tests use fake backends; no GPU, model weights, or ultralytics needed
```

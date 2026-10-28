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

# Small objects in a 4K video: every 10th frame, tonemapped, three 1280-wide
# tiles across one band plus a shrunk full frame for the lower part
coco-detect run \
  --input "https://example.com/match.mp4" \
  --output balls.jsonl \
  --model yolo26l.pt --classes "sports ball" \
  --frame-step 10 \
  --vf "zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" \
  --region "0,600,1280,640@1280" --region "1280,600,1280,640@1280" \
  --region "2560,600,1280,640@1280" --region "full@1280:ymin=1240"

# Export a TensorRT engine (FP32, dynamic batch) for faster inference
coco-detect export-engine --model yolo26l.pt --imgsz 640 --batch 16
```

Options of `run` beyond the window:

- `--frame-step N` — infer on every Nth frame of the window. `frame_index`
  stays the source frame number, so consecutive records differ by N.
- `--vf <filter>` — an ffmpeg filter applied after decoding (e.g. HDR→SDR
  tonemapping). It must not change the frame size. Decoding goes through
  `ffmpeg` (must be on `PATH`) whenever `--vf` or `--frame-step` > 1 is used.
- `--region x,y,w,h@imgsz` — crop that rectangle and infer at `imgsz`.
  `--region full@imgsz[:ymin=Y]` shrinks the whole frame to fit `imgsz` and,
  with `ymin`, keeps only boxes whose centre is at or below row `Y`.
  Repeatable; boxes are mapped back to source coordinates and concatenated
  without cross-region NMS. Cannot be combined with `--input-scale`.
  `--batch-size` caps the crops per forward pass (a TensorRT engine reserves
  memory for its largest batch, so a batch-1 engine with `--batch-size 1`
  keeps the footprint small).

## Output format (`raw_detections/v1`)

Line 1 is a header record; every following line is one processed frame, in
ascending `frame_index` order **with no gaps** other than the header's
`frame_step` (frames with zero detections still get a line):

```jsonl
{"schema":"raw_detections/v1","input":{"path":"input.mp4","fps":30.0,"width":1920,"height":1080,"frame_count":108000},"model":"yolo26l.pt","confidence":0.25,"classes":["person","sports ball"],"start_frame":0,"input_scale":1.0,"frame_step":1,"vf":null,"regions":null,"runner_version":"0.3.0"}
{"frame_index":0,"timestamp":0.0,"detections":[{"bbox":[100.5,200.1,140.2,300.0],"score":0.87,"label":"person"}]}
{"frame_index":1,"timestamp":0.0333,"detections":[]}
```

Contract notes:

- `frame_index` is the absolute frame number in the source video (window
  runs keep absolute numbering; `timestamp = frame_index / fps`).
- Every frame in the window is processed unless `frame_step` > 1.
- Bounding boxes are `[x1, y1, x2, y2]` in the coordinate space described by
  `input.width` / `input.height` (i.e. after `--input-scale` is applied;
  with `--region` this is always the source frame).
- Labels are the model's own class names (COCO names for stock models,
  e.g. `person`, `sports ball`).
- FP16 is disabled (FP32 only) for deterministic results across export
  formats.

## Development

```bash
pip install -e ".[dev]"
pytest -q   # tests use fake backends; no GPU, model weights, or ultralytics needed
```

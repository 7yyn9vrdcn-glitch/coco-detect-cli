"""Command line interface.

Subcommands:

- ``run``           — decode a video window, run YOLO, write raw detections JSONL
- ``export-engine`` — export a ``.pt`` model to a TensorRT ``.engine`` file
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from coco_detect import SCHEMA, __version__
from coco_detect.decode import iter_frames, iter_frames_ffmpeg, probe_video
from coco_detect.regions import detect_regions, parse_region, region_fits
from coco_detect.writer import RawDetectionsWriter

LOGGER = logging.getLogger("coco_detect")


def _parse_classes(raw: str) -> set[str]:
    classes = {c.strip() for c in raw.split(",") if c.strip()}
    if not classes:
        raise argparse.ArgumentTypeError("at least one class name is required")
    return classes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coco-detect",
        description="Video object detection pre-pass: YOLO over a video window -> JSONL",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="decode + detect + write raw detections JSONL")
    run.add_argument("--input", required=True, help="input video path or URL")
    run.add_argument("--output", required=True, type=Path, help="output JSONL path")
    run.add_argument("--model", required=True, help="YOLO weights (.pt or .engine)")
    run.add_argument("--classes", type=_parse_classes, default=None,
                     help="comma-separated class names to keep (default: all)")
    run.add_argument("--confidence", type=float, default=0.25)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--start-seconds", type=float, default=None,
                     help="window start (absolute frame numbering is preserved)")
    run.add_argument("--window-seconds", type=float, default=None,
                     help="window length (default: to end of video)")
    run.add_argument("--input-scale", type=float, default=1.0,
                     help="resize factor applied before inference; bboxes are in scaled space")
    run.add_argument("--fallback-fps", type=float, default=30.0,
                     help="fps to assume when the container reports none")
    run.add_argument("--flush-every", type=int, default=100)
    run.add_argument("--frame-step", type=int, default=1,
                     help="run inference on every Nth frame of the window "
                          "(frame_index stays the source frame number)")
    run.add_argument("--vf", default=None,
                     help="ffmpeg filter applied after decoding (must keep the frame size)")
    run.add_argument("--region", dest="regions", action="append", type=parse_region,
                     default=None, metavar="x,y,w,h@imgsz | full@imgsz[:ymin=Y]",
                     help="crop this rectangle (or shrink the full frame) and infer at imgsz; "
                          "repeatable. bboxes are mapped back to source coordinates")

    exp = sub.add_parser("export-engine", help="export .pt weights to a TensorRT .engine")
    exp.add_argument("--model", required=True, help="YOLO .pt weights")
    exp.add_argument("--imgsz", type=int, default=640)
    exp.add_argument("--batch", type=int, default=16)
    exp.add_argument("--half", action="store_true", help="FP16 engine (default FP32)")
    exp.add_argument("--device", default="0")
    return parser


def cmd_run(args: argparse.Namespace, *, detector=None) -> int:
    """``detector`` is injectable for tests."""
    info = probe_video(args.input, fallback_fps=args.fallback_fps)
    fps = info.fps
    start_frame = 0
    if args.start_seconds is not None and args.start_seconds > 0:
        start_frame = max(0, int(round(float(args.start_seconds) * fps)))
    max_frames = None
    if args.window_seconds is not None and args.window_seconds > 0:
        max_frames = max(1, int(round(float(args.window_seconds) * fps)))
    if start_frame >= info.frame_count > 0:
        LOGGER.error(
            "--start-seconds=%s is past the end of the video (frame_count=%d)",
            args.start_seconds, info.frame_count,
        )
        return 2

    frame_step = int(args.frame_step)
    if frame_step < 1:
        LOGGER.error("--frame-step must be >= 1 (got %d)", frame_step)
        return 2
    regions = args.regions or []
    if regions and args.input_scale != 1.0:
        LOGGER.error("--region and --input-scale cannot be combined")
        return 2
    for region in regions:
        if not region_fits(region, info.width, info.height):
            LOGGER.error(
                "--region %s does not fit the %dx%d frame", region.spec, info.width, info.height,
            )
            return 2

    if detector is None:
        from coco_detect.detector import YoloDetector

        detector = YoloDetector(args.model, args.confidence, args.classes)

    scale = float(args.input_scale)
    bbox_w = int(info.width * scale)
    bbox_h = int(info.height * scale)
    header = {
        "schema": SCHEMA,
        "input": {
            "path": str(args.input),
            "fps": fps,
            "width": bbox_w,
            "height": bbox_h,
            "frame_count": info.frame_count,
        },
        "model": str(args.model),
        "confidence": float(args.confidence),
        "classes": sorted(args.classes) if args.classes else None,
        "start_frame": start_frame,
        "input_scale": scale,
        "frame_step": frame_step,
        "vf": args.vf,
        "regions": [r.spec for r in regions] or None,
        "runner_version": __version__,
    }
    use_ffmpeg = frame_step > 1 or bool(args.vf)
    LOGGER.info(
        "run: input=%s model=%s conf=%.3f classes=%s window=[%s,%s)s "
        "start_frame=%d max_frames=%s batch=%d scale=%.2f step=%d decoder=%s "
        "vf=%s regions=%s -> %s",
        args.input, args.model, args.confidence,
        ",".join(sorted(args.classes)) if args.classes else "<all>",
        args.start_seconds or 0, args.window_seconds or "end",
        start_frame, max_frames, args.batch_size, scale, frame_step,
        "ffmpeg" if use_ffmpeg else "opencv", args.vf or "-",
        ",".join(r.spec for r in regions) or "-", args.output,
    )

    started = time.monotonic()
    batch_size = max(1, int(args.batch_size))
    if use_ffmpeg:
        frames = iter_frames_ffmpeg(
            args.input, fps, info.width, info.height,
            start_frame=start_frame, max_frames=max_frames,
            frame_step=frame_step, vf=args.vf, input_scale=scale,
        )
    else:
        frames = iter_frames(
            args.input, fps,
            start_frame=start_frame, max_frames=max_frames, input_scale=scale,
        )
    with RawDetectionsWriter(
        args.output, header, flush_every=args.flush_every, frame_step=frame_step,
    ) as writer:
        chunk: list = []
        for frame in frames:
            chunk.append(frame)
            if len(chunk) >= batch_size:
                _write_chunk(writer, detector, chunk, regions)
                chunk = []
        if chunk:
            _write_chunk(writer, detector, chunk, regions)
        frames_written = writer.frames_written
        detections_written = writer.detections_written

    elapsed = time.monotonic() - started
    LOGGER.info(
        "done: frames=%d detections=%d elapsed=%.1fs (%.1f fps)",
        frames_written, detections_written, elapsed,
        frames_written / elapsed if elapsed > 0 else 0.0,
    )
    if frames_written == 0:
        LOGGER.error("no frames decoded — output contains only the header")
        return 3
    return 0


def _write_chunk(writer: RawDetectionsWriter, detector, chunk: list, regions: list) -> None:
    if regions:
        results = detect_regions(detector, [f.image for f in chunk], regions)
    elif len(chunk) == 1:
        results = [detector.detect(chunk[0].image)]
    else:
        results = detector.detect_batch([f.image for f in chunk])
    for frame, detections in zip(chunk, results):
        writer.write_frame(frame.frame_index, frame.timestamp, detections)


def cmd_export_engine(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        LOGGER.error("ultralytics is not installed (pip install ultralytics)")
        return 2
    model = YOLO(args.model)
    LOGGER.info(
        "exporting %s -> TensorRT engine (imgsz=%d batch=%d half=%s device=%s)",
        args.model, args.imgsz, args.batch, args.half, args.device,
    )
    out = model.export(
        format="engine",
        imgsz=args.imgsz,
        batch=args.batch,
        half=args.half,
        dynamic=True,
        device=args.device,
    )
    LOGGER.info("engine written: %s", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "export-engine":
        return cmd_export_engine(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())

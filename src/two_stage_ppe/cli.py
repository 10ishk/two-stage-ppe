"""Command-line interface kept intentionally thin around PPEPipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from .pipeline import PPEPipeline


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--person-model", required=True, help="Ultralytics person checkpoint")
    parser.add_argument("--ppe-model", required=True, help="Ultralytics PPE checkpoint")
    parser.add_argument("--person-conf", type=float, default=0.30)
    parser.add_argument("--ppe-conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--crop-padding", type=float, default=0.05)
    parser.add_argument("--device")
    parser.add_argument("--person-class", default="person", help="Person class name or numeric class ID")
    parser.add_argument(
        "--ppe-batch-size",
        type=positive_int,
        help="Maximum PPE crops per backend call; defaults to all crops in each frame",
    )


def create_pipeline(args: argparse.Namespace) -> PPEPipeline:
    return PPEPipeline(
        args.person_model,
        args.ppe_model,
        person_conf=args.person_conf,
        ppe_conf=args.ppe_conf,
        iou=args.iou,
        crop_padding=args.crop_padding,
        device=args.device,
        person_class=args.person_class,
        ppe_batch_size=args.ppe_batch_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="two-stage-ppe", description="Two-stage person-to-PPE detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    detect = subparsers.add_parser("detect", help="Run detection on an image or directory")
    detect.add_argument("--input", required=True, help="Input image or non-recursive image directory")
    detect.add_argument("--output", required=True, help="Output directory")
    add_model_options(detect)
    detect.add_argument("--save-json", action="store_true", help="Write one JSON file per image")
    detect.add_argument("--no-images", action="store_true", help="Do not write annotated images")

    video = subparsers.add_parser("video", help="Stream detection over a video")
    video.add_argument("--input", required=True, help="Input video")
    video.add_argument("--output", help="Output video; selected frames are annotated by default")
    add_model_options(video)
    video.add_argument("--frame-stride", type=positive_int, default=1)
    video.add_argument("--start-frame", type=non_negative_int, default=0)
    video.add_argument("--max-frames", type=positive_int)
    video.add_argument(
        "--jsonl",
        nargs="?",
        const="",
        metavar="PATH",
        help="Write one JSON object per processed frame; omit PATH for an automatic name",
    )
    video.add_argument("--no-render", action="store_true", help="Write frames without annotations")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "detect":
        if args.no_images and not args.save_json:
            raise SystemExit("Nothing to save: combine --no-images with --save-json")
        results = create_pipeline(args).process(
            args.input, args.output, save_images=not args.no_images, save_json=args.save_json
        )
        logging.info("Processed %d image(s)", len(results))
        return 0

    if args.output is None and args.jsonl is None:
        raise SystemExit("Nothing to save: provide --output, --jsonl, or both")
    summary = create_pipeline(args).predict_video(
        args.input,
        args.output,
        save_json=args.jsonl is not None,
        jsonl_path=args.jsonl or None,
        frame_stride=args.frame_stride,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        render=not args.no_render,
    )
    logging.info(
        "Processed %d video frames in %.2fs (%.2f processed frames/s)",
        summary.processed_frames,
        summary.elapsed_seconds,
        summary.average_processing_fps,
    )
    return 0

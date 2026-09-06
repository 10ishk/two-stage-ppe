"""Command-line interface kept intentionally thin around PPEPipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from .pipeline import PPEPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="two-stage-ppe", description="Two-stage person-to-PPE detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    detect = subparsers.add_parser("detect", help="Run detection on an image or directory")
    detect.add_argument("--input", required=True, help="Input image or non-recursive image directory")
    detect.add_argument("--output", required=True, help="Output directory")
    detect.add_argument("--person-model", required=True, help="Ultralytics person checkpoint")
    detect.add_argument("--ppe-model", required=True, help="Ultralytics PPE checkpoint")
    detect.add_argument("--person-conf", type=float, default=0.30)
    detect.add_argument("--ppe-conf", type=float, default=0.30)
    detect.add_argument("--iou", type=float, default=0.45)
    detect.add_argument("--crop-padding", type=float, default=0.05)
    detect.add_argument("--device")
    detect.add_argument("--person-class", default="person", help="Person class name or numeric class ID")
    detect.add_argument("--save-json", action="store_true", help="Write one JSON file per image")
    detect.add_argument("--no-images", action="store_true", help="Do not write annotated images")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.no_images and not args.save_json:
        raise SystemExit("Nothing to save: combine --no-images with --save-json")
    pipeline = PPEPipeline(
        args.person_model,
        args.ppe_model,
        person_conf=args.person_conf,
        ppe_conf=args.ppe_conf,
        iou=args.iou,
        crop_padding=args.crop_padding,
        device=args.device,
        person_class=args.person_class,
    )
    results = pipeline.process(args.input, args.output, save_images=not args.no_images, save_json=args.save_json)
    logging.info("Processed %d image(s)", len(results))
    return 0


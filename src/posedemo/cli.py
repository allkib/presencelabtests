"""Command-line entry point: ``posedemo`` / ``python -m posedemo``."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from posedemo.config import Settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time YOLO pose recognition web demo")
    p.add_argument(
        "--source", help="camera index (default 0) or path/URL of a video, image or stream"
    )
    p.add_argument("--model", help="YOLO pose weights, e.g. yolo11s-pose.pt")
    p.add_argument("--device", help="auto | mps | cuda | cpu")
    p.add_argument("--imgsz", type=int, help="inference image size")
    p.add_argument("--conf", type=float, help="person detection confidence threshold")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--no-mirror", dest="mirror", action="store_false", default=None)
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "log_level"}
    settings = Settings(**overrides)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    from posedemo.web.app import create_app

    print(f"posedemo → http://{settings.host}:{settings.port}")
    uvicorn.run(
        create_app(settings), host=settings.host, port=settings.port, log_level=args.log_level
    )


if __name__ == "__main__":
    main()

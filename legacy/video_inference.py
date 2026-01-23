from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from emotion_ai.predictor import EmotionPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate a video file with emotion predictions")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--out", default="output_annotated.avi", help="Output video path")
    p.add_argument("--threshold", type=float, default=None, help="Confidence threshold override")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    predictor = EmotionPredictor(config_path=args.config)
    threshold = args.threshold

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(args.out, fourcc, float(fps), (width, height))

    print(f"Processing: {video_path} -> {args.out}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = predictor.predict(frame)
            if threshold is not None:
                dets = [d for d in dets if d["confidence"] >= threshold]

            annotated = predictor.annotate_image(frame, dets)
            out.write(annotated)
    finally:
        cap.release()
        out.release()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

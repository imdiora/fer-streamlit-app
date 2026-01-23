from __future__ import annotations

import argparse
import cv2

from emotion_ai.predictor import EmotionPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenCV webcam demo (local window)")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--cam", type=int, default=0, help="Camera index")
    p.add_argument("--threshold", type=float, default=None, help="Confidence threshold override")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    predictor = EmotionPredictor(config_path=args.config)
    threshold = args.threshold

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.cam}")

    print("Starting webcam demo. Press q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = predictor.predict(frame)
            if threshold is not None:
                dets = [d for d in dets if d["confidence"] >= threshold]

            annotated = predictor.annotate_image(frame, dets)
            cv2.imshow("Emotion AI (webcam)", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

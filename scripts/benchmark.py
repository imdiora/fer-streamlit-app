from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from emotion_ai.predictor import EmotionPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple latency benchmark")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    predictor = EmotionPredictor(config_path=args.config)

    size = int(predictor.config.model.input_size)
    x = torch.randn(1, 3, size, size, device=predictor.device)

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = predictor.model(x)

        # Timed
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = predictor.model(x)
        t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000.0
    avg_ms = total_ms / args.iters

    print("Device:", predictor.device)
    print("Iters:", args.iters)
    print(f"Avg forward latency: {avg_ms:.3f} ms")
    if avg_ms > 0:
        print(f"Approx FPS: {1000.0/avg_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

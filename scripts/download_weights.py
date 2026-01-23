from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download model weights referenced in config/config.yaml")
    p.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    p.add_argument("--out", default=None, help="Override output path (defaults to model.weights_path)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    auto = cfg.get("model", {}).get("auto_download", {}) or {}
    if not auto.get("repo_id") or not auto.get("filename"):
        raise ValueError("config.model.auto_download.repo_id/filename are required")

    out_path = Path(args.out or cfg.get("model", {}).get("weights_path", "models/model.pth"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    print(f"Downloading: {auto['repo_id']} / {auto['filename']}")
    downloaded = hf_hub_download(repo_id=auto["repo_id"], filename=auto["filename"])
    src = Path(downloaded)
    out_path.write_bytes(src.read_bytes())
    print(f"Saved weights -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

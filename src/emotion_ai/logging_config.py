# src/emotion_ai/logging_config.py
from __future__ import annotations

import logging
import logging.config
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def _make_run_id() -> str:
    # One run id per process (so API + predictor share the same file)
    rid = os.environ.get("EMOTION_AI_RUN_ID")
    if rid:
        return rid
    rid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    os.environ["EMOTION_AI_RUN_ID"] = rid
    return rid


def setup_logging(
    service_name: str = "app",
    level: str | int = "INFO",
    log_dir: str | Path | None = None,
    also_console: bool = True,
) -> Path:
    """
    Configure logging for the whole process (root logger).
    - Writes to logs/<service_name>_<run_id>.log
    - Optionally also prints to console
    - Captures uvicorn loggers too (uvicorn.error, uvicorn.access)

    Returns: Path to the log file.
    """
    run_id = _make_run_id()

    # Where to store logs:
    # - EMOTION_AI_LOG_DIR if set
    # - else `logs/` under current working directory
    env_dir = os.environ.get("EMOTION_AI_LOG_DIR")
    base_dir = Path(env_dir) if env_dir else Path(log_dir or "logs")
    base_dir.mkdir(parents=True, exist_ok=True)

    log_path = base_dir / f"{service_name}_{run_id}.log"

    # Normalize level
    if isinstance(level, str):
        level = level.upper()

    handlers: dict = {
        "file": {
            "class": "logging.FileHandler",
            "level": level,
            "filename": str(log_path),
            "mode": "a",
            "encoding": "utf-8",
            "formatter": "standard",
        }
    }

    root_handlers = ["file"]

    if also_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": level,
            "stream": "ext://sys.stdout",
            "formatter": "standard",
        }
        root_handlers.append("console")

    config = {
        "version": 1,
        "disable_existing_loggers": False,  # important: don't silence libraries
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": handlers,
        "root": {"level": level, "handlers": root_handlers},
        "loggers": {
            # Ensure uvicorn logs flow into our handlers too
            "uvicorn": {"level": level, "propagate": True},
            "uvicorn.error": {"level": level, "propagate": True},
            "uvicorn.access": {"level": level, "propagate": True},
            # Your package namespace
            "emotion_ai": {"level": level, "propagate": True},
        },
    }

    logging.config.dictConfig(config)

    # Log basic run header once
    logging.getLogger("emotion_ai.run").info(
        "Run started | service=%s | run_id=%s | python=%s | platform=%s | cwd=%s",
        service_name,
        run_id,
        sys.version.replace("\n", " "),
        platform.platform(),
        os.getcwd(),
    )
    logging.getLogger("emotion_ai.run").info("Log file: %s", log_path)

    return log_path

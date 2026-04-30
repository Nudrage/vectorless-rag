#!/usr/bin/env python3
"""
Structured logging configuration for the chunker package.

Configures logging with optional file rotation, console output, and JSON format.
Environment variables CHUNKER_LOG_LEVEL and CHUNKER_LOG_FILE are respected
when not explicitly passed.
"""

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Default format: timestamp, level, module, function, line, message
DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
    rotation_max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configure application logging with file and console handlers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Overridden by
            CHUNKER_LOG_LEVEL env var if set.
        log_file: Optional path for log file. If set, a RotatingFileHandler
            is added. CHUNKER_LOG_FILE env var can provide a path if not set.
        json_format: If True, use JSON formatter for file/console (e.g. for production).
        rotation_max_bytes: Max bytes per log file before rotation (default 10 MB).
        backup_count: Number of backup log files to keep.

    Returns:
        None. Configures the root logger and common chunker loggers.
    """
    env_level = os.environ.get("CHUNKER_LOG_LEVEL", "").strip().upper()
    if env_level:
        level = env_level

    env_log_file = os.environ.get("CHUNKER_LOG_FILE", "").strip()
    if env_log_file and log_file is None:
        log_file = Path(env_log_file)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicate handlers when called multiple times (e.g. in tests)
    if root.handlers:
        for h in root.handlers[:]:
            root.removeHandler(h)

    if json_format:
        formatter: logging.Formatter = JsonFormatter()
        formatter.datefmt = DEFAULT_DATE_FORMAT
    else:
        formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                str(log_path),
                maxBytes=rotation_max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as e:
            # Fallback: log to stderr that file logging failed
            sys.stderr.write(f"Warning: could not open log file {log_path}: {e}\n")

"""
Structured logging configuration for the agent framework.

- JSON-formatted logs for production
- Correlation IDs for request tracing
- Configurable log level and format
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

# Default log directory and file
LOG_DIR = "logs"
LOG_FILE = "agent.log"

# Context variable for correlation ID (works with async and threads)
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Return the current request correlation ID."""
    return _correlation_id.get()


def set_correlation_id(value: Optional[str]) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(value)


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current context."""
    try:
        _correlation_id.set(None)
    except LookupError:
        pass


def new_correlation_id() -> str:
    """Generate and set a new correlation ID. Returns the new ID."""
    cid = str(uuid.uuid4())
    set_correlation_id(cid)
    return cid


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs JSON for production or human-readable for development.
    """

    def __init__(
        self,
        use_json: bool = False,
        include_correlation_id: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_json = use_json
        self.include_correlation_id = include_correlation_id

    def format(self, record: logging.LogRecord) -> str:
        if self.use_json:
            return self._format_json(record)
        return self._format_standard(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if self.include_correlation_id:
            cid = get_correlation_id()
            if cid:
                log_obj["correlation_id"] = cid
        if record.exc_info:
            log_obj["exception"] = self.formatterException(record.exc_info)
        # Include extra fields set by StructuredLogger
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_obj.update(record.extra_fields)
        return json.dumps(log_obj, default=str)

    def _format_standard(self, record: logging.LogRecord) -> str:
        parts = [super().format(record)]
        if self.include_correlation_id:
            cid = get_correlation_id()
            if cid:
                parts.append(f" [cid={cid[:8]}]")
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            parts.append(" ")
            parts.append(str(record.extra_fields))
        return "".join(parts)


class StructuredLogger(logging.LoggerAdapter):
    """
    Logger adapter that adds correlation ID and optional extra fields to every log.
    """

    def process(self, msg: str, kwargs: Any) -> tuple:
        extra = kwargs.get("extra", {})
        if "extra_fields" not in extra:
            extra["extra_fields"] = {}
        cid = get_correlation_id()
        if cid:
            extra["extra_fields"]["correlation_id"] = cid
        extra["extra_fields"].update(self.extra or {})
        kwargs["extra"] = extra
        return msg, kwargs

    def with_fields(self, **fields: Any) -> "StructuredLogger":
        """Return a new adapter with additional permanent extra fields."""
        new_extra = dict(self.extra or {})
        new_extra.update(fields)
        return StructuredLogger(self.logger, new_extra)


def get_logger(name: str) -> StructuredLogger:
    """Get a StructuredLogger for the given name."""
    return StructuredLogger(logging.getLogger(name), {})


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_correlation_id: bool = True,
    log_to_file: bool = True,
    log_file_path: str = None,
    file_json_format: bool = False,
) -> None:
    """
    Configure root logging for the agent package.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, use JSON formatter for console
        include_correlation_id: Include correlation_id in log output
        log_to_file: If True, also write logs to a file
        log_file_path: Custom log file path (default: logs/agent.log)
        file_json_format: If True, use JSON formatter for log file
    """
    # Ensure log directory exists
    if log_to_file:
        os.makedirs(LOG_DIR, exist_ok=True)
        if log_file_path is None:
            log_file_path = os.path.join(LOG_DIR, LOG_FILE)

    # Also set agent root if we have a parent
    for log_name in ("utils", "agent"):
        log = logging.getLogger(log_name)
        log.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not log.handlers:
            # Console handler (stderr)
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(
                StructuredFormatter(
                    use_json=json_format,
                    include_correlation_id=include_correlation_id,
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
            log.addHandler(console_handler)

            # File handler
            if log_to_file:
                file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
                file_handler.setFormatter(
                    StructuredFormatter(
                        use_json=file_json_format,
                        include_correlation_id=include_correlation_id,
                        datefmt="%Y-%m-%dT%H:%M:%S",
                    )
                )
                log.addHandler(file_handler)


def timed(logger: Optional[StructuredLogger] = None, name: Optional[str] = None):
    """
    Decorator that logs elapsed time for a function.

    Usage:
        @timed(get_logger(__name__), "planner_node")
        def planner_node(state): ...
    """
    _log = logger or get_logger(__name__)

    def decorator(func: Any) -> Any:
        _name = name or func.__name__

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                _log.info(
                    "%s completed in %.3fs",
                    _name,
                    elapsed,
                    extra={"extra_fields": {"operation": _name, "elapsed_seconds": round(elapsed, 3)}},
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                _log.warning(
                    "%s failed after %.3fs: %s",
                    _name,
                    elapsed,
                    e,
                    extra={"extra_fields": {"operation": _name, "elapsed_seconds": round(elapsed, 3)}},
                )
                raise

        return wrapper

    return decorator

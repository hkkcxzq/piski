"""Structured logging with secret redaction."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

_REDACT_MARKERS = ("secret", "api_key", "apikey", "password", "token", "signature", "private_key")
REDACTED = "***"


def redact_secrets(_logger: Any, _method: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """structlog processor: mask values whose key looks like a credential (recursively)."""

    def _clean(value: Any) -> Any:
        if isinstance(value, MutableMapping):
            return {
                k: (REDACTED if any(m in str(k).lower() for m in _REDACT_MARKERS) else _clean(v))
                for k, v in value.items()
            }
        return value

    for key in list(event_dict.keys()):
        if any(marker in key.lower() for marker in _REDACT_MARKERS):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _clean(event_dict[key])
    return event_dict


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    numeric = logging.getLevelNamesMapping().get(level.upper())
    if numeric is None:
        raise ValueError(f"unknown log level: {level}")
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_secrets,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        # resolve sys.stderr at call time (it may be swapped by test runners or daemonizers)
        logger_factory=lambda *_args: structlog.PrintLogger(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger

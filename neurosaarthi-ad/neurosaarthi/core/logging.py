"""Structured local logging with conservative identifier redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

_REDACTION_PATTERNS = (
    re.compile(r"(?i)\b((?:patient|participant|subject|record|mrn)[-_ ]?id\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)\b((?:patientname|patientbirthdate|accessionnumber)\s*[=:]\s*)[^\s,;]+"),
)


def redact_text(value: str) -> str:
    """Redact identifier-like key/value fragments from a message."""

    result = value
    for pattern in _REDACTION_PATTERNS:
        result = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", result)
    return result


class JsonFormatter(logging.Formatter):
    """Emit one local JSON record per event without arbitrary record fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
        }
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str = "INFO") -> None:
    """Configure the process root logger for local, redacted JSON output."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

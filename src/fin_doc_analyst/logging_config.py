"""JSON logs with the same PII redaction used on model outputs."""

from __future__ import annotations

import logging

import structlog

from fin_doc_analyst.guardrails.pii import redact


def _redact_processor(
    _logger: object, _method: str, event_dict: dict[str, object]
) -> dict[str, object]:
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = redact(value)
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "fin_doc_analyst") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

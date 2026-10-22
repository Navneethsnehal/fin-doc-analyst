"""Parse JSON objects from model text, including fenced blocks."""

from __future__ import annotations

import json
import re

from fin_doc_analyst.llm.base import LLMError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def parse_json_object(text: str) -> dict[str, object]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = _FENCE.sub("", raw).strip()
    start = raw.find("{")
    if start == -1:
        raise LLMError("model response did not contain a JSON object")
    try:
        payload, _end = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        raise LLMError("model response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise LLMError("model response JSON was not an object")
    return payload


def coerce_bool(value: object, default: bool = False) -> bool:
    """Read a JSON boolean. The string "false" is false; bool("false") is not."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "yes", "1"}:
            return True
        if cleaned in {"false", "no", "0", ""}:
            return False
    return default

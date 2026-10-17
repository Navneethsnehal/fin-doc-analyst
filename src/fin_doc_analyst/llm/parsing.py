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
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError("model response did not contain a JSON object")
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError("model response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise LLMError("model response JSON was not an object")
    return payload

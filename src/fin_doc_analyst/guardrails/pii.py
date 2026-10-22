"""Redact SSNs and account numbers from answers and logs."""

from __future__ import annotations

import re

SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Labeled account, routing, or SSN. The digits may be contiguous or split
# by spaces or hyphens ("4821-9377-01", "SSN 999999999").
LABELED_NUMBER_RE = re.compile(
    r"(?i)\b(account(?:\s+number)?|acct|routing(?:\s+number)?|social\s+security(?:\s+number)?|ssn)"
    r"(\s*[:#-]?\s*)(\d(?:[ \t-]?\d){5,16})"
)
LONG_NUMBER_RE = re.compile(r"\b\d{10,17}\b")


def _mask(digits: str) -> str:
    return f"****{digits[-4:]}"


def _is_ssn_label(label: str) -> bool:
    lowered = label.lower()
    return lowered == "ssn" or "social" in lowered


def redact(text: str) -> str:
    """Mask SSNs completely and long account-like numbers down to the last four."""
    if not text:
        return text
    text = SSN_RE.sub("[REDACTED:SSN]", text)

    def labeled(match: re.Match[str]) -> str:
        label, sep, raw = match.group(1), match.group(2), match.group(3)
        digits = re.sub(r"\D", "", raw)
        if not 6 <= len(digits) <= 17:
            return match.group(0)
        if _is_ssn_label(label):
            return f"{label}{sep}[REDACTED:SSN]"
        return f"{label}{sep}{_mask(digits)}"

    text = LABELED_NUMBER_RE.sub(labeled, text)
    text = LONG_NUMBER_RE.sub(lambda match: _mask(match.group(0)), text)
    return text


def contains_raw_pii(text: str) -> bool:
    if not text:
        return False
    if SSN_RE.search(text) or LONG_NUMBER_RE.search(text):
        return True
    return LABELED_NUMBER_RE.search(text) is not None

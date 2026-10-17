"""Redact SSNs and account numbers from answers and logs."""

from __future__ import annotations

import re

SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
LABELED_NUMBER_RE = re.compile(
    r"(?i)\b(account(?:\s+number)?|acct|routing(?:\s+number)?|social\s+security(?:\s+number)?)"
    r"(\s*[:#-]?\s*)(\d{6,17})"
)
LONG_NUMBER_RE = re.compile(r"\b\d{10,17}\b")


def _mask(digits: str) -> str:
    return f"****{digits[-4:]}"


def redact(text: str) -> str:
    """Mask SSNs completely and long account-like numbers down to the last four."""
    if not text:
        return text
    text = SSN_RE.sub("[REDACTED:SSN]", text)

    def labeled(match: re.Match[str]) -> str:
        label, sep, digits = match.group(1), match.group(2), match.group(3)
        if "social" in label.lower():
            return f"{label}{sep}[REDACTED:SSN]"
        return f"{label}{sep}{_mask(digits)}"

    text = LABELED_NUMBER_RE.sub(labeled, text)
    text = LONG_NUMBER_RE.sub(lambda match: _mask(match.group(0)), text)
    return text


def contains_raw_pii(text: str) -> bool:
    if SSN_RE.search(text):
        return True
    if LONG_NUMBER_RE.search(text):
        return True
    return False

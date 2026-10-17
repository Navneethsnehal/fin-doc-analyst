"""Detect prompt injection in user turns and in retrieved document text.

Document text is untrusted data. A match quarantines that chunk so it is never
shown to the answering model as instructions.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions", re.I),
    ),
    (
        "disregard",
        re.compile(r"disregard\s+(the\s+)?(above|previous|documents|instructions)", re.I),
    ),
    ("you_are_now", re.compile(r"you\s+are\s+now\b", re.I)),
    ("admin_mode", re.compile(r"\badmin\s+mode\b", re.I)),
    ("system_prompt", re.compile(r"\bsystem\s+prompt\b", re.I)),
    ("do_not_cite", re.compile(r"do\s+not\s+cite", re.I)),
    ("jailbreak", re.compile(r"\bjailbreak\b", re.I)),
    ("respond_approved", re.compile(r"(respond|reply)\s+with\s+approved", re.I)),
    (
        "approve_all",
        re.compile(r"approve\s+(every|all)\s+transactions?", re.I),
    ),
    ("reveal_prompt", re.compile(r"reveal\s+(the\s+)?(system\s+)?prompt", re.I)),
    (
        "new_instructions",
        re.compile(r"(new|updated)\s+instructions\s+for\s+the\s+assistant", re.I),
    ),
]


def detect_injection(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]


def neutralize(text: str) -> str:
    """Replace injection phrases so a downstream model sees a marker, not a command."""
    cleaned = text
    for _name, pattern in INJECTION_PATTERNS:
        cleaned = pattern.sub("[filtered-instruction]", cleaned)
    return cleaned

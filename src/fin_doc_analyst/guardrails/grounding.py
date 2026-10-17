"""Check that an answer is supported by retrieved text or a recorded tool result."""

from __future__ import annotations

import json
import re

from fin_doc_analyst.textutil import stem, tokenize

# Template words the agent may add around verbatim excerpts and tool totals.
# Stored stemmed so "calculated" (stem "calculat") is ignored consistently.
_TEMPLATE_WORDS = {
    "calculated",
    "computed",
    "using",
    "result",
    "review",
    "following",
    "source",
    "page",
    "section",
    "clause",
    "severity",
    "high",
    "medium",
    "low",
    "tool",
    "sum",
    "equals",
    "stated",
    "inputs",
}
TEMPLATE_TOKENS = {stem(word) for word in _TEMPLATE_WORDS}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CITATION_LINE = re.compile(r"(?im)^source:.*$")


def answer_body(answer: str) -> str:
    return _CITATION_LINE.sub("", answer).strip()


def _number_forms(text: str) -> set[str]:
    return set(_NUMBER.findall(text.lower().replace(",", "")))


def grounding_score(
    answer: str,
    contexts: list[str],
    tool_results: list[dict[str, object]] | None = None,
) -> float:
    body = answer_body(answer)
    tokens = [token for token in tokenize(body) if token not in TEMPLATE_TOKENS]
    if not tokens:
        return 1.0
    context_blob = " ".join(contexts)
    tool_blob = json.dumps(tool_results or [], default=str)
    supported_tokens = set(tokenize(context_blob)) | set(tokenize(tool_blob))
    context_numbers = _number_forms(context_blob) | _number_forms(tool_blob)
    supported = 0
    for token in tokens:
        if token in supported_tokens or token in context_numbers:
            supported += 1
        elif not token.isalpha() and token in context_numbers:
            supported += 1
    return supported / len(tokens)


def numbers_supported(
    answer: str,
    contexts: list[str],
    tool_results: list[dict[str, object]] | None = None,
) -> bool:
    body = answer_body(answer)
    needed = _number_forms(body)
    if not needed:
        return True
    available = _number_forms(" ".join(contexts)) | _number_forms(
        json.dumps(tool_results or [], default=str)
    )
    return needed <= available


def is_grounded(
    answer: str,
    contexts: list[str],
    tool_results: list[dict[str, object]] | None = None,
    min_score: float = 0.72,
) -> bool:
    if not answer_body(answer):
        return False
    if not numbers_supported(answer, contexts, tool_results):
        return False
    return grounding_score(answer, contexts, tool_results) >= min_score

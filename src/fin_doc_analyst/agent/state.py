"""Graph state. Nodes return partial dicts and LangGraph merges them."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    doc_ids: list[str]
    route: str
    retrieved: list[dict[str, object]]
    graded: list[dict[str, object]]
    quarantined: list[dict[str, object]]
    tool_results: list[dict[str, object]]
    risks: list[dict[str, object]]
    citations: list[dict[str, object]]
    answer: str
    refused: bool
    refusal_reason: str
    injection_detected: bool
    grounded: bool
    used_contexts: list[str]
    retrieved_contexts: list[str]


NOT_FOUND = "Not found in the documents."
INJECTION_REFUSAL = (
    "I can't follow that request. I only answer questions about the uploaded financial documents."
)
BLOCKED_DOCUMENT = (
    "Not found in the trusted documents. Matching text was blocked because it contained "
    "instructions aimed at the assistant."
)
UNGROUNDED = "Not found in the documents."

"""Deterministic client used by tests, CI, and the offline demo.

The graph still calls this through the same JSON interface as OpenAI. Routing,
grading, and answering are lexical so a golden question always yields the same
trace without a network call.
"""

from __future__ import annotations

from fin_doc_analyst.guardrails.injection import detect_injection
from fin_doc_analyst.textutil import overlap_score


class FakeLLM:
    provider = "fake"

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        kind = _task(system)
        if kind == "classify":
            return {"route": classify_question(user)}
        if kind == "grade":
            question, excerpt = _split_grade_user(user)
            return {"relevant": overlap_score(question, excerpt) >= 0.75}
        if kind == "answer":
            return {"answer": "", "citations": [], "refused": True}
        return {}


def classify_question(question: str) -> str:
    if detect_injection(question):
        return "refuse"
    text = question.lower()
    if any(
        phrase in text
        for phrase in (
            "risky",
            "red flag",
            "unusual clause",
            "unusual clauses",
            "what are the risks",
            "flag the",
            "flag any",
            "clauses should",
        )
    ):
        return "risk"
    if any(
        phrase in text
        for phrase in (
            "calculate",
            "compute",
            "sum of",
            "total of",
            "monthly payment",
            "how much altogether",
        )
    ):
        return "compute"
    return "answer"


def _task(system: str) -> str:
    head = system.lower()
    if "classify" in head[:80] or "route" in head[:120]:
        return "classify"
    if "relevant" in head and "excerpt" in head:
        return "grade"
    if "answer only" in head or "financial document analyst" in head:
        return "answer"
    return "other"


def _split_grade_user(user: str) -> tuple[str, str]:
    question = ""
    excerpt = ""
    if "QUESTION:" in user and "EXCERPT:" in user:
        question = user.split("QUESTION:", 1)[1].split("EXCERPT:", 1)[0].strip()
        excerpt = user.split("EXCERPT:", 1)[1].strip()
    return question, excerpt

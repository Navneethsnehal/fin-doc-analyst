"""Clause scanner for fees, penalties, exclusions, variable rates, and renewals."""

from __future__ import annotations

import re

from fin_doc_analyst.schemas import Chunk, RiskFlag

# Posting tables are not clauses. Terms sections are.
SKIP_SECTIONS = {"deposits", "withdrawals", "account summary"}
_DATED_POSTING = re.compile(r"^\d{2}/\d{2}/\d{4}\b")

PATTERNS: list[tuple[str, str, re.Pattern[str], str]] = [
    (
        "variable_rate",
        "high",
        re.compile(r"variable rate|prime rate plus", re.I),
        "The rate can move with the Prime Rate, so the payment is not fixed.",
    ),
    (
        "auto_renewal",
        "medium",
        re.compile(r"automatic(?:ally)? renew", re.I),
        "The contract renews unless the customer cancels before the deadline.",
    ),
    (
        "exclusion",
        "high",
        re.compile(r"\bexcluded\b|not provided under this policy", re.I),
        "This peril is not covered and may require a separate policy.",
    ),
    (
        "acceleration",
        "high",
        re.compile(r"acceleration|entire unpaid balance", re.I),
        "After default the lender can demand the remaining balance at once.",
    ),
    (
        "prepayment",
        "low",
        re.compile(r"prepayment penalty|prepay this loan", re.I),
        "Prepayment terms change the cost of paying the loan off early.",
    ),
    (
        "penalty",
        "high",
        re.compile(r"late fee|denial of a claim", re.I),
        "A fee or a denied claim applies when a condition is missed.",
    ),
    (
        "fee",
        "medium",
        re.compile(r"overdraft fee|maintenance fee", re.I),
        "A bank fee is disclosed here and should be checked against the expected product.",
    ),
]


def scan_risks(chunks: list[Chunk]) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    seen: set[tuple[str, str, int]] = set()
    for chunk in chunks:
        if chunk.section.strip().lower() in SKIP_SECTIONS:
            continue
        for sentence in _sentences(chunk.text):
            if len(sentence) < 18 or _DATED_POSTING.match(sentence):
                continue
            for clause_type, severity, pattern, explanation in PATTERNS:
                if not pattern.search(sentence):
                    continue
                key = (clause_type, sentence, chunk.page)
                if key in seen:
                    break
                seen.add(key)
                flags.append(
                    RiskFlag(
                        clause_type=clause_type,
                        severity=severity,
                        excerpt=sentence,
                        explanation=explanation,
                        doc_id=chunk.doc_id,
                        doc_name=chunk.doc_name,
                        page=chunk.page,
                        section=chunk.section,
                        chunk_id=chunk.chunk_id,
                    )
                )
                break
    severity_order = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda flag: (severity_order.get(flag.severity, 9), flag.doc_name, flag.page))
    return flags


def _sentences(text: str) -> list[str]:
    from fin_doc_analyst.textutil import sentences

    return sentences(text)

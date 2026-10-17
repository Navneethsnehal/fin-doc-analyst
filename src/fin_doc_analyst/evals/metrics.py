"""RAGAS-style metrics that do not call a judge model.

Faithfulness is token and number support against the context actually passed to
the answerer, plus recorded tool outputs. Answer relevance is recall of the
hand-labeled facts. Context precision and recall use the gold evidence strings.
"""

from __future__ import annotations

from fin_doc_analyst.guardrails.grounding import grounding_score
from fin_doc_analyst.guardrails.pii import redact
from fin_doc_analyst.schemas import AnalysisResult
from fin_doc_analyst.textutil import contains_fact


def faithfulness(result: AnalysisResult) -> float:
    if result.refused:
        return 0.0
    contexts = [redact(text) for text in result.used_contexts]
    tools = [tool.model_dump() for tool in result.tool_results]
    return grounding_score(result.answer, contexts, tools)


def fact_recall(answer: str, facts: list[str]) -> float:
    if not facts:
        return 1.0
    hits = sum(1 for fact in facts if contains_fact(answer, fact))
    return hits / len(facts)


def context_precision(result: AnalysisResult, evidence: list[str]) -> float:
    contexts = result.used_contexts
    if not contexts:
        return 0.0
    if not evidence:
        return 1.0
    relevant = sum(1 for text in contexts if any(contains_fact(text, item) for item in evidence))
    return relevant / len(contexts)


def context_recall(result: AnalysisResult, evidence: list[str]) -> float:
    if not evidence:
        return 1.0
    blob = "\n".join(result.used_contexts)
    hits = sum(1 for item in evidence if contains_fact(blob, item))
    return hits / len(evidence)

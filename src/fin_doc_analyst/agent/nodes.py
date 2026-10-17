"""Explicit graph nodes. Each one does a single job and returns a partial state."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fin_doc_analyst.agent.answer import (
    citations_for,
    compose_tool_answer,
    extractive_answer,
    format_sources,
    plan_tools,
)
from fin_doc_analyst.agent.risk import scan_risks
from fin_doc_analyst.agent.state import (
    BLOCKED_DOCUMENT,
    INJECTION_REFUSAL,
    NOT_FOUND,
    UNGROUNDED,
    AgentState,
)
from fin_doc_analyst.guardrails.grounding import is_grounded
from fin_doc_analyst.guardrails.injection import detect_injection, neutralize
from fin_doc_analyst.llm.prompts import ANSWER_SYSTEM, CLASSIFY_SYSTEM, GRADE_SYSTEM
from fin_doc_analyst.logging_config import get_logger
from fin_doc_analyst.schemas import Chunk, RiskFlag
from fin_doc_analyst.textutil import corpus_token_set, overlap_score

GRADE_MIN_SCORE = 0.75

Store = Any
LLM = Any


def build_nodes(store: Store, llm: LLM) -> dict[str, Callable[[AgentState], dict[str, object]]]:
    log = get_logger("agent")

    def route(state: AgentState) -> dict[str, object]:
        question = state["question"]
        hits = detect_injection(question)
        if hits:
            log.info("node.route", route="refuse", injection=hits)
            return {
                "route": "refuse",
                "refusal_reason": "injection",
                "injection_detected": True,
                "refused": True,
            }
        try:
            payload = llm.complete_json(CLASSIFY_SYSTEM, question)
            chosen = str(payload.get("route", "answer"))
        except Exception as exc:  # noqa: BLE001 — fail closed into the refusal path
            log.info("node.route.error", error=str(exc))
            chosen = "refuse"
        if chosen not in {"answer", "compute", "risk", "refuse"}:
            chosen = "answer"
        log.info("node.route", route=chosen)
        update: dict[str, object] = {"route": chosen, "injection_detected": False}
        if chosen == "refuse":
            update["refusal_reason"] = "out_of_scope"
            update["refused"] = True
        return update

    def retrieve(state: AgentState) -> dict[str, object]:
        doc_ids = state.get("doc_ids") or None
        k = 6 if state.get("route") == "risk" else 4
        found = store.search(state["question"], k=k, doc_ids=doc_ids)
        retrieved = [item.chunk.model_dump() for item in found]
        log.info(
            "node.retrieve",
            chunks=[chunk["chunk_id"] for chunk in retrieved],
            count=len(retrieved),
        )
        return {
            "retrieved": retrieved,
            "retrieved_contexts": [str(chunk["text"]) for chunk in retrieved],
        }

    def grade(state: AgentState) -> dict[str, object]:
        kept: list[dict[str, object]] = []
        quarantined: list[dict[str, object]] = []
        corpus = _corpus(store)
        seen_ids: set[str] = set()
        candidates = list(state.get("retrieved") or [])
        # A poisoned document can be about the question and still miss the top-k
        # list. Scan the in-scope corpus so those chunks are quarantined either way.
        for chunk in store.all_chunks(state.get("doc_ids") or None):
            payload = chunk.model_dump()
            if (
                detect_injection(chunk.text)
                and overlap_score(state["question"], chunk.text, corpus_tokens=corpus) >= 0.5
            ):
                candidates.append(payload)
        for chunk in candidates:
            chunk_id = str(chunk["chunk_id"])
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            text = str(chunk["text"])
            if detect_injection(text):
                quarantined.append(chunk)
                continue
            if _is_relevant(llm, state["question"], text, corpus):
                kept.append(chunk)
        reason = ""
        refused = False
        if state.get("route") != "compute" and not kept:
            refused = True
            reason = "blocked_document" if quarantined else "not_found"
        log.info(
            "node.grade",
            kept=[chunk["chunk_id"] for chunk in kept],
            quarantined=[chunk["chunk_id"] for chunk in quarantined],
        )
        detected = bool(state.get("injection_detected") or quarantined)
        return {
            "graded": kept,
            "quarantined": quarantined,
            "used_contexts": [str(chunk["text"]) for chunk in kept],
            "refusal_reason": reason or state.get("refusal_reason", ""),
            "refused": refused or bool(state.get("refused")),
            "injection_detected": detected,
        }

    def compute(state: AgentState) -> dict[str, object]:
        scope = [chunk.model_dump() for chunk in store.all_chunks(state.get("doc_ids") or None)]
        scope = [chunk for chunk in scope if not detect_injection(str(chunk["text"]))]
        tool_results = plan_tools(state["question"], scope)
        log.info("node.compute", tools=[tool["name"] for tool in tool_results])
        if not tool_results:
            return {"tool_results": [], "refusal_reason": "not_found", "refused": True}
        supporting = _supporting_chunks(state["question"], scope, tool_results)
        return {
            "tool_results": tool_results,
            "graded": supporting,
            "used_contexts": [str(chunk["text"]) for chunk in supporting],
            "refused": False,
            "refusal_reason": "",
        }

    def answer(state: AgentState) -> dict[str, object]:
        chunks = list(state.get("graded") or [])
        tool_results = list(state.get("tool_results") or [])
        if tool_results or state.get("route") == "compute":
            text, citations, refused = compose_tool_answer(state["question"], chunks, tool_results)
        elif getattr(llm, "provider", "fake") == "fake":
            text, citations, refused = extractive_answer(
                state["question"], chunks, corpus_tokens=_corpus(store)
            )
        else:
            text, citations, refused = _llm_answer(llm, state["question"], chunks, tool_results)
        log.info("node.answer", refused=refused, citations=len(citations))
        return {
            "answer": text,
            "citations": [citation.model_dump() for citation in citations],
            "refused": refused,
            "refusal_reason": "not_found" if refused else "",
            "used_contexts": [str(chunk["text"]) for chunk in chunks],
        }

    def analyze_risk(state: AgentState) -> dict[str, object]:
        raw_chunks = store.all_chunks(state.get("doc_ids") or None)
        safe = [chunk for chunk in raw_chunks if not detect_injection(chunk.text)]
        flags = scan_risks(safe)
        if not flags:
            return {
                "answer": NOT_FOUND,
                "risks": [],
                "citations": [],
                "refused": True,
                "refusal_reason": "not_found",
            }
        excerpts = [flag.excerpt for flag in flags]
        cited_chunks = _chunks_for_flags(safe, flags)
        citations = citations_for([chunk.model_dump() for chunk in cited_chunks])
        body = " ".join(excerpts)
        sources = format_sources(citations)
        answer = f"{body}\n{sources}" if sources else body
        log.info("node.risk", flags=len(flags))
        return {
            "answer": answer,
            "risks": [flag.model_dump() for flag in flags],
            "citations": [citation.model_dump() for citation in citations],
            "refused": False,
            "refusal_reason": "",
            "used_contexts": [chunk.text for chunk in cited_chunks],
            "graded": [chunk.model_dump() for chunk in cited_chunks],
        }

    def refuse(state: AgentState) -> dict[str, object]:
        reason = state.get("refusal_reason") or "not_found"
        if reason == "injection":
            message = INJECTION_REFUSAL
        elif reason == "blocked_document":
            message = BLOCKED_DOCUMENT
        elif reason == "ungrounded":
            message = UNGROUNDED
        else:
            message = NOT_FOUND
        log.info("node.refuse", reason=reason)
        return {
            "answer": message,
            "citations": [],
            "refused": True,
            "refusal_reason": reason,
            "grounded": True,
        }

    def ground(state: AgentState) -> dict[str, object]:
        if state.get("refused"):
            return {"grounded": True}
        answer = state.get("answer") or ""
        if detect_injection(answer):
            log.info("node.ground", grounded=False, reason="answer_contained_injection")
            return {
                "grounded": False,
                "refusal_reason": "injection",
                "injection_detected": True,
            }
        contexts = list(state.get("used_contexts") or [])
        tools = list(state.get("tool_results") or [])
        ok = is_grounded(answer, contexts, tools)
        log.info("node.ground", grounded=ok)
        if ok:
            return {"grounded": True}
        return {"grounded": False, "refusal_reason": "ungrounded"}

    return {
        "route": route,
        "retrieve": retrieve,
        "grade": grade,
        "compute": compute,
        "answer": answer,
        "analyze_risk": analyze_risk,
        "refuse": refuse,
        "ground": ground,
    }


def _corpus(store: Store) -> set[str]:
    return corpus_token_set(chunk.text for chunk in store.all_chunks())


def _is_relevant(llm: LLM, question: str, text: str, corpus: set[str]) -> bool:
    if getattr(llm, "provider", "fake") == "fake":
        return overlap_score(question, text, corpus_tokens=corpus) >= GRADE_MIN_SCORE
    payload = llm.complete_json(
        GRADE_SYSTEM,
        f"QUESTION:\n{question}\nEXCERPT:\n{neutralize(text)}",
    )
    return bool(payload.get("relevant"))


def _llm_answer(
    llm: LLM,
    question: str,
    chunks: list[dict[str, object]],
    tool_results: list[dict[str, object]],
) -> tuple[str, list, bool]:
    from fin_doc_analyst.agent.state import NOT_FOUND
    from fin_doc_analyst.schemas import Citation

    blocks = []
    for chunk in chunks:
        blocks.append(
            "\n".join(
                [
                    f"[excerpt chunk_id={chunk['chunk_id']} doc={chunk['doc_name']} "
                    f"page={chunk['page']} section={chunk['section']}]",
                    neutralize(str(chunk["text"])),
                    "[/excerpt]",
                ]
            )
        )
    tool_block = ""
    if tool_results:
        tool_block = f"\nTOOL RESULTS:\n{tool_results}"
    user = f"EXCERPTS:\n{chr(10).join(blocks)}{tool_block}\n\nQUESTION:\n{question}"
    try:
        payload = llm.complete_json(ANSWER_SYSTEM, user)
    except Exception:
        return NOT_FOUND, [], True
    refused = bool(payload.get("refused"))
    text = str(payload.get("answer") or "").strip()
    if refused or not text:
        return NOT_FOUND, [], True
    allowed = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    citations: list[Citation] = []
    raw_citations = payload.get("citations") or []
    if isinstance(raw_citations, list):
        for item in raw_citations:
            if not isinstance(item, dict):
                continue
            chunk = allowed.get(str(item.get("chunk_id", "")))
            if chunk is None:
                continue
            citations.append(citations_for([chunk])[0])
    if not citations:
        citations = citations_for(chunks)
    sources = format_sources(citations)
    if "Source:" not in text and sources:
        text = f"{text}\n{sources}"
    return text, citations, False


def _supporting_chunks(
    question: str,
    chunks: list[dict[str, object]],
    tool_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not tool_results:
        return []
    name = str(tool_results[0]["name"])
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = str(chunk["text"])
        useful = False
        if name == "amortized_payment" and any(
            label in text for label in ("Principal amount:", "APR is", "Term:")
        ):
            useful = True
        if name == "sum_currency" and _chunk_has_sum_source(question, text):
            useful = True
        chunk_id = str(chunk["chunk_id"])
        if useful and chunk_id not in seen:
            seen.add(chunk_id)
            selected.append(chunk)
    return selected


def _chunk_has_sum_source(question: str, text: str) -> bool:
    import re

    keyword = "fee" if "fee" in question.lower() else "deposit"
    dated = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text) is not None
    return keyword in text.lower() and "$" in text and dated


def _chunks_for_flags(chunks: list[Chunk], flags: list[RiskFlag]) -> list[Chunk]:
    wanted = {flag.chunk_id for flag in flags}
    return [chunk for chunk in chunks if chunk.chunk_id in wanted]

"""Extractive answers for the offline provider, and tool planning for both providers."""

from __future__ import annotations

import re

from fin_doc_analyst.agent.tools import amortized_payment, sum_currency
from fin_doc_analyst.schemas import Citation
from fin_doc_analyst.textutil import extract_money, overlap_score, sentences

_DATE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
_PRINCIPAL = re.compile(r"Principal amount:\s*\$([0-9,]+\.\d{2})", re.I)
_APR = re.compile(r"APR is\s*([0-9]+(?:\.[0-9]+)?)%", re.I)
_TERM = re.compile(r"Term:\s*(\d+)\s+months", re.I)


def format_sources(citations: list[Citation]) -> str:
    lines: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    for citation in citations:
        key = (citation.doc_name, citation.page, citation.section)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"Source: {citation.doc_name}, page {citation.page}, section {citation.section}."
        )
    return "\n".join(lines)


def citations_for(chunks: list[dict[str, object]]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        text = str(chunk["text"]).strip().replace("\n", " ")
        citations.append(
            Citation(
                chunk_id=chunk_id,
                doc_id=str(chunk["doc_id"]),
                doc_name=str(chunk["doc_name"]),
                page=int(chunk["page"]),
                section=str(chunk["section"]),
                excerpt=text[:240],
            )
        )
    return citations


def extractive_answer(
    question: str,
    chunks: list[dict[str, object]],
    corpus_tokens: set[str] | None = None,
    min_score: float = 0.75,
) -> tuple[str, list[Citation], bool]:
    """Copy supporting sentences. Refuse when no sentence covers the question."""
    from fin_doc_analyst.agent.state import NOT_FOUND

    ranked: list[tuple[float, str, dict[str, object]]] = []
    for chunk in chunks:
        for sentence in sentences(str(chunk["text"])):
            score = overlap_score(question, sentence, corpus_tokens=corpus_tokens)
            if score > 0:
                ranked.append((score, sentence, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] < min_score:
        return NOT_FOUND, [], True
    chosen_sentences: list[str] = []
    chosen_chunks: list[dict[str, object]] = []
    seen: set[str] = set()
    for score, sentence, chunk in ranked:
        if score < min_score or sentence in seen:
            continue
        seen.add(sentence)
        chosen_sentences.append(sentence)
        chosen_chunks.append(chunk)
        if len(chosen_sentences) == 3:
            break
    citations = citations_for(chosen_chunks)
    body = " ".join(chosen_sentences)
    sources = format_sources(citations)
    answer = f"{body}\n{sources}" if sources else body
    return answer, citations, False


def plan_tools(question: str, chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Choose a tool from the question and the figures actually present in chunks."""
    text = question.lower()
    blob = "\n".join(str(chunk["text"]) for chunk in chunks)
    if "monthly payment" in text:
        principal = _PRINCIPAL.search(blob)
        apr = _APR.search(blob)
        term = _TERM.search(blob)
        if principal and apr and term:
            rate = f"{float(apr.group(1)) / 100:.4f}"
            output = amortized_payment(
                principal.group(1).replace(",", ""), rate, int(term.group(1))
            )
            return [
                {
                    "name": "amortized_payment",
                    "arguments": {
                        "principal": output["principal"],
                        "annual_rate": output["annual_rate"],
                        "months": output["months"],
                    },
                    "output": output,
                }
            ]
    if any(phrase in text for phrase in ("calculate", "compute", "total of", "sum of")):
        amounts = _amounts_for_total(question, chunks)
        if len(amounts) >= 2:
            output = sum_currency(amounts)
            return [
                {
                    "name": "sum_currency",
                    "arguments": {"amounts": output["amounts"]},
                    "output": output,
                }
            ]
    return []


def _amounts_for_total(question: str, chunks: list[dict[str, object]]) -> list[str]:
    lowered = question.lower()
    if "fee" in lowered:
        keyword = "fee"
    elif "deposit" in lowered:
        keyword = "deposit"
    elif "withdrawal" in lowered:
        keyword = "withdrawal"
    else:
        keyword = ""
    matched: list[str] = []
    for chunk in chunks:
        for sentence in sentences(str(chunk["text"])):
            if keyword and keyword not in sentence.lower():
                continue
            if re.search(r"\btotal\b", sentence, re.I):
                continue
            if extract_money(sentence):
                matched.append(sentence)
    dated = [sentence for sentence in matched if _DATE.search(sentence)]
    chosen = dated or matched
    amounts: list[str] = []
    for sentence in chosen:
        amounts.extend(extract_money(sentence))
    return amounts


def support_sentences(
    question: str, chunks: list[dict[str, object]]
) -> tuple[list[str], list[dict[str, object]]]:
    """Sentences that carry the figures a tool just used."""
    tool_calls = plan_tools(question, chunks)
    if not tool_calls:
        return [], []
    name = str(tool_calls[0]["name"])
    picked: list[str] = []
    owners: list[dict[str, object]] = []
    for chunk in chunks:
        for sentence in sentences(str(chunk["text"])):
            keep = False
            if name == "sum_currency" and sentence in _source_sentences(question, chunks):
                keep = True
            if name == "amortized_payment" and (
                _PRINCIPAL.search(sentence) or _APR.search(sentence) or _TERM.search(sentence)
            ):
                keep = True
            if keep:
                picked.append(sentence)
                owners.append(chunk)
    return picked, owners


def _source_sentences(question: str, chunks: list[dict[str, object]]) -> set[str]:
    lowered = question.lower()
    keyword = "fee" if "fee" in lowered else "deposit" if "deposit" in lowered else ""
    found: list[str] = []
    for chunk in chunks:
        for sentence in sentences(str(chunk["text"])):
            if keyword and keyword not in sentence.lower():
                continue
            if re.search(r"\btotal\b", sentence, re.I):
                continue
            if extract_money(sentence):
                found.append(sentence)
    dated = [sentence for sentence in found if _DATE.search(sentence)]
    return set(dated or found)


def compose_tool_answer(
    question: str, chunks: list[dict[str, object]], tool_results: list[dict[str, object]]
) -> tuple[str, list[Citation], bool]:
    from fin_doc_analyst.agent.state import NOT_FOUND

    if not tool_results:
        return extractive_answer(question, chunks)
    lines: list[str] = []
    for tool in tool_results:
        output = tool["output"]
        if not isinstance(output, dict):
            continue
        if tool["name"] == "sum_currency":
            rendered = ", ".join(f"${amount}" for amount in output["amounts"])
            lines.append(f"Calculated total is ${output['total']} using amounts {rendered}.")
        elif tool["name"] == "amortized_payment":
            lines.append(
                "Calculated monthly payment is "
                f"${output['monthly_payment']} using principal ${output['principal']}, "
                f"annual rate {output['annual_rate']}, and term {output['months']} months."
            )
    supporting, owners = support_sentences(question, chunks)
    citations = citations_for(owners)
    body_parts = lines + supporting
    if not body_parts:
        return NOT_FOUND, [], True
    body = " ".join(body_parts)
    sources = format_sources(citations)
    answer = f"{body}\n{sources}" if sources else body
    return answer, citations, False

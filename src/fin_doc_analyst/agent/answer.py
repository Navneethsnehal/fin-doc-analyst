"""Extractive answers for the offline provider, and tool planning for both providers."""

from __future__ import annotations

import re

from fin_doc_analyst.agent.tools import amortized_payment, sum_currency
from fin_doc_analyst.schemas import Citation
from fin_doc_analyst.textutil import (
    expand,
    extract_money,
    overlap_score,
    query_tokens,
    sentences,
    tokenize,
)

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


# A bare "total of the fees" is not a list of line names. Those questions still
# sum every posted fee. A name with its own words ("overdraft fee", "ATM
# withdrawal") has to match that line and only that line.
_GENERIC_AMOUNT_WORDS = (
    "fee",
    "fees",
    "deposit",
    "deposits",
    "withdrawal",
    "withdrawals",
    "charge",
    "charges",
    "charged",
    "amount",
    "amounts",
    "payment",
    "payments",
    "transaction",
    "transactions",
)
_GENERIC_AMOUNT_TOKENS = {
    token for phrase in _GENERIC_AMOUNT_WORDS for token in query_tokens(phrase)
}
_TOTAL_OF = re.compile(r"\b(?:total|sum)\s+of\s+(.+)$", re.I)
_TRAILING_SCOPE = re.compile(
    r"\s+\b(?:on|in|from|for)\s+the\s+"
    r"(?:bank|loan|policy|statement|agreement|document|documents)\b.*$",
    re.I,
)
_ITEM_SPLIT = re.compile(r"\s*(?:,|&|\band\b)\s*", re.I)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)
_MONEY_SPAN = re.compile(r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?")


def _amounts_for_total(question: str, chunks: list[dict[str, object]]) -> list[str]:
    return [amount for _sentence, amount in _sum_lines(question, chunks)]


def _sum_lines(question: str, chunks: list[dict[str, object]]) -> list[tuple[str, str]]:
    """(sentence, amount) pairs that belong in the sum.

    Named lines are matched one-for-one. If any name is missing from the
    documents, the whole sum is dropped so the tool cannot return a partial
    or unrelated total. Questions that do not name lines keep the posted-fee
    behavior.
    """
    phrases = _named_sum_phrases(question)
    if phrases:
        return _lines_for_phrases(phrases, chunks)
    return _lines_for_keyword(question, chunks)


def _named_sum_phrases(question: str) -> list[str]:
    cleaned = question.strip().rstrip(".?!")
    match = _TOTAL_OF.search(cleaned)
    if not match:
        return []
    body = _TRAILING_SCOPE.sub("", match.group(1).strip())
    phrases: list[str] = []
    for part in _ITEM_SPLIT.split(body):
        phrase = _LEADING_ARTICLE.sub("", part.strip(" ."))
        if phrase:
            phrases.append(phrase)
    if not phrases:
        return []
    for phrase in phrases:
        tokens = query_tokens(phrase)
        if not tokens or not any(token not in _GENERIC_AMOUNT_TOKENS for token in tokens):
            return []
    return phrases


def _phrase_supported(phrase: str, sentence: str) -> bool:
    tokens = query_tokens(phrase)
    if not tokens:
        return False
    haystack = set(tokenize(sentence))
    return all(expand([token]) & haystack for token in tokens)


def _lines_for_phrases(
    phrases: list[str], chunks: list[dict[str, object]]
) -> list[tuple[str, str]]:
    catalog: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for sentence in sentences(str(chunk["text"])):
            if sentence in seen or re.search(r"\btotal\b", sentence, re.I):
                continue
            if not extract_money(sentence):
                continue
            seen.add(sentence)
            catalog.append(sentence)
    chosen: list[tuple[str, str]] = []
    used: set[str] = set()
    for phrase in phrases:
        best_sentence: str | None = None
        best_rank: tuple[int, int, int] | None = None
        for index, sentence in enumerate(catalog):
            if sentence in used or not _phrase_supported(phrase, sentence):
                continue
            # A dated posting is the amount that hit the account. A clause that
            # also states a balance threshold is the fallback, not a second hit.
            rank = (
                1 if _DATE.search(sentence) else 0,
                -len(extract_money(sentence)),
                -index,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_sentence = sentence
        if best_sentence is None:
            return []
        amount = _amount_near_phrase(best_sentence, phrase)
        if amount is None:
            return []
        used.add(best_sentence)
        chosen.append((best_sentence, amount))
    return chosen


def _amount_near_phrase(sentence: str, phrase: str) -> str | None:
    """The figure that belongs to this line.

    Fee clauses often mention a second number, the balance that triggers the
    fee. The amount next to the fee name is the one that goes into the total.
    """
    found = extract_money(sentence)
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    matches = list(_MONEY_SPAN.finditer(sentence))
    if not matches:
        return found[0]
    anchors = query_tokens(phrase)
    anchor = anchors[-1] if anchors else ""
    origin = sentence.lower().rfind(anchor) if anchor else 0
    if origin < 0:
        origin = 0
    chosen = min(matches, key=lambda match: (abs(match.start() - origin), match.start()))
    dollars = chosen.group(1).replace(",", "")
    cents = chosen.group(2) or "00"
    return f"{int(dollars)}.{cents}"


def _lines_for_keyword(question: str, chunks: list[dict[str, object]]) -> list[tuple[str, str]]:
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
    lines: list[tuple[str, str]] = []
    for sentence in dated or matched:
        for amount in extract_money(sentence):
            lines.append((sentence, amount))
    return lines


def support_sentences(
    question: str, chunks: list[dict[str, object]]
) -> tuple[list[str], list[dict[str, object]]]:
    """Sentences that carry the figures a tool just used."""
    tool_calls = plan_tools(question, chunks)
    if not tool_calls:
        return [], []
    name = str(tool_calls[0]["name"])
    sources = _source_sentences(question, chunks) if name == "sum_currency" else set()
    picked: list[str] = []
    owners: list[dict[str, object]] = []
    for chunk in chunks:
        for sentence in sentences(str(chunk["text"])):
            keep = False
            if name == "sum_currency" and sentence in sources:
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
    return {sentence for sentence, _amount in _sum_lines(question, chunks)}


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

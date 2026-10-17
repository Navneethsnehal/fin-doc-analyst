"""Tokenization and sentence splitting shared by retrieval, grading, and evals."""

from __future__ import annotations

import re
from collections.abc import Iterable

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "and",
    "or",
    "for",
    "on",
    "in",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "do",
    "does",
    "did",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "by",
    "as",
    "at",
    "into",
    "over",
    "under",
    "about",
    "please",
    "tell",
    "me",
    "any",
    "all",
    "can",
    "you",
    "your",
    "my",
    "our",
    "their",
    "should",
    "would",
    "could",
    "if",
    "then",
    "than",
    "not",
    "no",
    "yes",
    "there",
    "here",
    "also",
    "just",
    "have",
    "has",
    "had",
    "i",
    "we",
}

# Words that show up in almost every question but do not identify a passage.
QUERY_IGNORE = STOPWORDS | {
    "harborview",
    "lumenshield",
    "lumen",
    "alex",
    "morgan",
    "northwind",
    "bank",
    "statement",
    "loan",
    "policy",
    "insurance",
    "document",
    "documents",
    "auto",
    "home",
    "homeowner",
    "homeowners",
    "account",
    "customer",
    "show",
    "find",
    "give",
    "list",
    "month",
    "period",
    "september",
    "agreement",
    "contract",
    "community",
    "lending",
    "company",
    "according",
    "uploaded",
    "file",
    "files",
    "doc",
    "docs",
    "synthetic",
}

SYNONYMS: dict[str, set[str]] = {
    "money": {"balance"},
    "balance": {"money"},
    "end": {"ending"},
    "ending": {"end"},
    "covered": {"coverage", "cover", "exclud", "exclusion"},
    "cover": {"coverage", "exclud", "exclusion"},
    "coverage": {"cover", "exclud", "exclusion"},
    "excluded": {"cover", "coverage", "exclud", "exclusion"},
    "exclud": {"cover", "coverage", "exclusion"},
    "exclusion": {"exclud", "cover"},
    "fee": {"fees"},
    "fees": {"fee"},
    "charge": {"charged", "charges"},
    "charged": {"charge"},
    "apr": {"percentage"},
    "instruct": {"instruction"},
    "instruction": {"instruct"},
}

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
_MONEY = re.compile(r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?")


def stem(token: str) -> str:
    """Light suffix trim so 'ending' and 'end' meet without a stemmer dependency."""
    if not token.isalpha():
        return token
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_text(text: str) -> str:
    return text.lower().replace(",", "").replace("$", "")


def tokenize(text: str) -> list[str]:
    cleaned = text.lower().replace(",", "")
    tokens: list[str] = []
    for raw in _TOKEN.findall(cleaned):
        if raw in STOPWORDS or len(raw) <= 1:
            continue
        tokens.append(stem(raw) if raw.isalpha() else raw)
    return tokens


def query_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in QUERY_IGNORE]


def expand(tokens: list[str]) -> set[str]:
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        expanded.update(stem(item) if item.isalpha() else item for item in SYNONYMS.get(token, ()))
        expanded.update(SYNONYMS.get(token, ()))
    return expanded


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def extract_money(text: str) -> list[str]:
    """Return currency amounts as normalized decimal strings (no thousands separators)."""
    found: list[str] = []
    for match in _MONEY.finditer(text):
        dollars = match.group(1).replace(",", "")
        cents = match.group(2) or "00"
        found.append(f"{int(dollars)}.{cents}")
    return found


def contains_fact(text: str, fact: str) -> bool:
    if not fact:
        return True
    return normalize_text(fact) in normalize_text(text)


def corpus_token_set(texts: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        tokens.update(tokenize(text))
    return tokens


def overlap_score(
    query: str,
    text: str,
    corpus_tokens: set[str] | None = None,
) -> float:
    """Share of discriminative query tokens supported by `text`.

    Tokens that never occur in the corpus are dropped first. "Is flood damage
    covered?" can then match "Flood is excluded." even though the documents
    never use the word "damage".
    """
    tokens = query_tokens(query)
    if corpus_tokens is not None:
        present = [token for token in tokens if expand([token]) & corpus_tokens]
        if present:
            tokens = present
    if not tokens:
        return 0.0
    haystack = set(tokenize(text))
    hits = sum(1 for token in tokens if expand([token]) & haystack)
    return hits / len(tokens)

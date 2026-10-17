"""Provider-agnostic chat interface used by the graph."""

from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    """Raised when a provider response cannot be used."""


class LLMClient(Protocol):
    provider: str

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        """Return a JSON object produced at temperature 0."""

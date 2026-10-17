"""Anthropic messages client. The SDK object can be injected in tests."""

from __future__ import annotations

from typing import Any

from fin_doc_analyst.llm.base import LLMError
from fin_doc_analyst.llm.parsing import parse_json_object


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str, client: Any | None = None) -> None:
        if client is None:
            if not api_key:
                raise LLMError("ANTHROPIC_API_KEY is required for the Anthropic provider")
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
        self.model = model
        self._client = client

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1200,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return parse_json_object(text)

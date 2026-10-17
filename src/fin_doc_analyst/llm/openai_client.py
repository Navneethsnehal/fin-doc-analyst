"""OpenAI chat client. The SDK object can be injected in tests."""

from __future__ import annotations

from typing import Any

from fin_doc_analyst.llm.base import LLMError
from fin_doc_analyst.llm.parsing import parse_json_object


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str, api_key: str, client: Any | None = None) -> None:
        if client is None:
            if not api_key:
                raise LLMError("OPENAI_API_KEY is required for the OpenAI provider")
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.model = model
        self._client = client

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — vendor SDK raises a wide tree
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        content = response.choices[0].message.content or ""
        return parse_json_object(content)

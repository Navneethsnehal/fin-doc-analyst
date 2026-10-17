"""Local Ollama chat via HTTP. A transport can be injected in tests."""

from __future__ import annotations

from typing import Any

import httpx

from fin_doc_analyst.llm.base import LLMError
from fin_doc_analyst.llm.parsing import parse_json_object


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=120.0)

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            payload = response.json()
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Ollama request failed: {exc}") from exc
        content = payload.get("message", {}).get("content", "")
        return parse_json_object(str(content))

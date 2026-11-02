"""Free news classifier: a local language model served by Ollama on the owner's computer.

Ollama (https://ollama.com) runs open models locally over HTTP on ``localhost:11434``; nothing is
sent to a paid API. Structured output is requested with the same JSON schema as for Claude, and
the result goes through the same validation and deterministic policy.
"""

from __future__ import annotations

from typing import Any

import httpx

from quant.news.classifier import ClassifierError, Transport

OLLAMA_HOST = "http://localhost:11434"
# small enough for a MacBook Air: ~5 GB RAM for an 8B model, ~3 GB for a 4B one
DEFAULT_LOCAL_MODEL = "qwen3:8b"


def ollama_available(model: str, host: str = OLLAMA_HOST, http: httpx.Client | None = None) -> bool:
    """True if Ollama is running and the model has been downloaded (``ollama pull <model>``)."""
    client = http or httpx.Client(timeout=3.0)
    try:
        resp = client.get(f"{host}/api/tags")
        resp.raise_for_status()
    except httpx.HTTPError:
        return False
    names = {m.get("name", "") for m in resp.json().get("models", [])}
    return model in names or f"{model}:latest" in names


def ollama_transport(
    model: str = DEFAULT_LOCAL_MODEL, host: str = OLLAMA_HOST, http: httpx.Client | None = None
) -> Transport:
    client = http or httpx.Client(timeout=180.0)  # a laptop model may need a minute for a batch

    def call(system: str, user: str, schema: dict[str, Any]) -> str:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,  # Ollama structured outputs: the reply must match the JSON schema
            "stream": False,
            "think": False,  # no long reasoning traces: faster and cooler on a laptop
            "options": {"temperature": 0},
        }
        try:
            resp = client.post(f"{host}/api/chat", json=body)
            resp.raise_for_status()
            text = str(resp.json().get("message", {}).get("content", ""))
        except (httpx.HTTPError, ValueError) as exc:
            raise ClassifierError(f"local model unavailable: {exc!r}"[:300]) from exc
        if not text:
            raise ClassifierError("local model returned an empty response")
        return text

    return call

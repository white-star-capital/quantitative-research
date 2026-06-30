"""
Ollama client for local LLM inference.

Drop-in replacement for OpenRouterClient used by the pricing engines.
Talks to a local `ollama serve` instance running qwen3:8b.

Usage mirrors OpenRouterClient so the AI-analysis step can swap providers
without changing call sites:

    with OllamaClient() as client:
        resp = client.create_chat_completion(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "..."}],
            temperature=0.2,
        )
    content = resp["choices"][0]["message"]["content"]

qwen3 is a reasoning model that emits <think>...</think> blocks. We use
Ollama's native /api/chat endpoint with `think` routed to a separate field
(not the content), and strip any residual tags, so downstream JSON parsing
on the message content stays clean.
"""

import os
import re
import time
import random
import requests
from typing import List, Dict, Optional

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaClient:
    """A Pythonic client for a local Ollama server.

    Exposes a `create_chat_completion` method shaped like the OpenRouter /
    OpenAI chat-completions response so it can be used as a drop-in
    replacement in the pricing engines.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        think: bool = False,
        timeout: float = 300.0,
    ):
        """Initialize the Ollama client.

        Args:
            base_url: Ollama server URL. Defaults to OLLAMA_BASE_URL env var
                      or http://localhost:11434.
            model: Default model to use when none is passed to a call.
                   Defaults to OLLAMA_MODEL env var or qwen3:8b.
            think: Whether to let the model reason. When True, reasoning is
                   routed to a separate field and kept OUT of the message
                   content. Defaults to False for faster, deterministic output.
            timeout: Per-request timeout in seconds. Local inference can be
                     slow, so this is generous by default.
        """
        load_dotenv()

        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
        self.think = think
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def create_chat_completion(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Dict:
        """Create a chat completion via Ollama's native /api/chat endpoint.

        Args:
            model: Model name. Falls back to the client default.
            messages: Conversation history (OpenAI message format).
            **kwargs: Generation params. `temperature`, `top_p`, `seed`,
                      `num_predict`, etc. are forwarded into Ollama `options`.
                      `think` overrides the client default for this call.

        Returns:
            An OpenAI/OpenRouter-shaped dict:
            {"choices": [{"message": {"role": "assistant", "content": "..."}}], ...}
        """
        selected_model = model or self.model
        messages = messages or []

        think = kwargs.pop("think", self.think)

        # Map common OpenAI-style params into Ollama's `options` block.
        options: Dict = {}
        for key in ("temperature", "top_p", "top_k", "seed", "stop", "num_ctx"):
            if key in kwargs and kwargs[key] is not None:
                options[key] = kwargs[key]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            options["num_predict"] = kwargs["max_tokens"]

        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "think": think,
        }
        if options:
            payload["options"] = options

        data = self._post_with_retries("/api/chat", payload)

        message = data.get("message", {}) or {}
        content = message.get("content", "") or ""
        thinking = message.get("thinking", "") or ""

        # Safety net: strip any think tags that leaked into content.
        content = _THINK_TAG_RE.sub("", content).strip()

        return {
            "id": data.get("created_at", ""),
            "model": data.get("model", selected_model),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": content,
                        "reasoning": thinking,
                    },
                    "finish_reason": data.get("done_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        }

    def chat(self, prompt: str, **kwargs) -> str:
        """Convenience helper: send a single user prompt, return the text."""
        resp = self.create_chat_completion(
            messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return resp["choices"][0]["message"]["content"]

    def _post_with_retries(self, path: str, payload: Dict) -> Dict:
        """POST with exponential backoff on transient errors."""
        url = f"{self.base_url}{path}"
        backoff = 1.0
        last_err: Optional[Exception] = None

        for _ in range(4):
            try:
                response = self._session.post(url, json=payload, timeout=self.timeout)
                if response.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(
                        f"Ollama returned {response.status_code}; retrying"
                    )
                    time.sleep(backoff + random.uniform(0, 0.5))
                    backoff = min(backoff * 2, 16)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ConnectionError as e:
                last_err = RuntimeError(
                    f"Could not reach Ollama at {self.base_url}. "
                    f"Is `ollama serve` running? ({e})"
                )
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 2, 16)
            except Exception as e:
                last_err = e
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 2, 16)

        if last_err:
            raise last_err
        raise RuntimeError("Ollama: max retries exceeded.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()


if __name__ == "__main__":
    # Smoke test against the local server.
    print(f"Testing OllamaClient against {DEFAULT_BASE_URL} ({DEFAULT_MODEL})...")
    with OllamaClient() as client:
        resp = client.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": 'Reply with only this JSON: {"status": "ok"}',
                }
            ],
            temperature=0.0,
        )
    content = resp["choices"][0]["message"]["content"]
    print("Response content:", repr(content))
    print("Usage:", resp["usage"])

"""OpenRouter OpenAI-compatible client with cache-safe probes and usage parsing."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class ClientError(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_zero_or_missing(self) -> bool:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return True
        pt = self.prompt_tokens or 0
        ct = self.completion_tokens or 0
        tt = self.total_tokens or (pt + ct)
        return tt == 0


@dataclass
class ChatResult:
    content: str
    usage: Usage
    model: str
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ClientError(
                "OPENROUTER_API_KEY is required (env var or --api-key)"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # OpenRouter attribution headers (optional, harmless)
                "HTTP-Referer": "https://github.com/tokenizer-detector",
                "X-Title": "TokenizerDetector",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def chat(
        self,
        model: str,
        user_content: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system: str | None = None,
        nonce: str | None = None,
    ) -> ChatResult:
        """Send a chat completion and parse usage.

        A unique nonce is appended as a hidden trailing comment when requested
        to defeat OpenRouter response caching (cache hits zero out usage).
        """
        content = user_content
        if nonce is not None:
            # Keep nonce in the prompt so the request body is unique.
            content = f"{user_content}\n\n[nonce:{nonce}]"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 3):
            try:
                resp = self._client.post("/chat/completions", json=payload)
                if resp.status_code == 429:
                    wait = 2.0 * (attempt + 1)
                    time.sleep(min(wait, 15.0))
                    last_err = ClientError(f"rate limited: {resp.text[:200]}")
                    continue
                if resp.status_code >= 400:
                    raise ClientError(
                        f"HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                result = _parse_chat_result(data)
                # Cache hit => usage zeroed. Retry once with a new nonce.
                if result.usage.is_zero_or_missing and nonce is not None:
                    nonce = uuid.uuid4().hex
                    payload["messages"][-1]["content"] = (
                        f"{user_content}\n\n[nonce:{nonce}]"
                    )
                    last_err = ClientError("usage zeroed (cache hit?), retrying")
                    continue
                return result
            except (httpx.HTTPError, ValueError) as exc:
                last_err = exc
                time.sleep(min(0.8 * (attempt + 1), 10.0))
        raise ClientError(f"chat failed after retries: {last_err}")

    def chat_with_nonce(self, model: str, user_content: str, **kwargs: Any) -> ChatResult:
        return self.chat(model, user_content, nonce=uuid.uuid4().hex, **kwargs)


def _parse_chat_result(data: dict[str, Any]) -> ChatResult:
    choices = data.get("choices") or []
    if not choices:
        raise ClientError(f"no choices in response: {str(data)[:300]}")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        # content parts
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )

    usage_raw = data.get("usage") or {}
    # Some providers put counts under nested details or generation stats
    prompt = _as_int(usage_raw.get("prompt_tokens"))
    completion = _as_int(usage_raw.get("completion_tokens"))
    total = _as_int(usage_raw.get("total_tokens"))
    if prompt is None:
        prompt = _as_int(usage_raw.get("prompt_tokens_details", {}).get("tokens") if isinstance(usage_raw.get("prompt_tokens_details"), dict) else None)
    if prompt is None:
        # last resort: OpenRouter generation-style fields if present on the object
        prompt = _as_int(data.get("native_tokens_prompt") or data.get("tokens_prompt"))
    if completion is None:
        completion = _as_int(data.get("native_tokens_completion") or data.get("tokens_completion"))

    usage = Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        raw=usage_raw,
    )
    return ChatResult(
        content=content,
        usage=usage,
        model=data.get("model") or "",
        finish_reason=choice.get("finish_reason"),
        raw=data,
    )


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

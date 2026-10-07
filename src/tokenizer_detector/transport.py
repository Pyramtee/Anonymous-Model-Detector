"""One HTTP attempt per send, with a durable budget and no hidden retries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json as jsonlib
import math
import time
from typing import Any, Callable
from urllib.parse import urlsplit
import uuid

import httpx

from .run_store import RunStore

BASE_URL = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class AttemptResult:
    attempt_id: str
    status_code: int | None = None
    data: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    uncertain: bool = False
    elapsed_seconds: float = 0.0
    method: str = ""


@dataclass
class ParsedChat:
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    usage: dict[str, Any]
    model: str
    provider: str | None
    generation_id: str | None
    finish_reason: str | None
    token_bytes: list[bytes] | None
    token_bytes_valid: bool
    raw: dict[str, Any]


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def parse_chat_result(data: Any) -> ParsedChat:
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError("Chat response is not a successful object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Chat response has no usable choices")
    choice = choices[0]
    if choice.get("error"):
        raise ValueError("Chat choice contains an error")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Chat response has no message")
    raw_content = message.get("content")
    content_known = raw_content is None or isinstance(raw_content, str)
    content = raw_content if isinstance(raw_content, str) else ""
    if isinstance(raw_content, list):
        content_known = all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str) for p in raw_content)
        content = "".join(p["text"] for p in raw_content if isinstance(p, dict) and isinstance(p.get("text"), str))
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    logprobs = choice.get("logprobs")
    items = logprobs.get("content") if isinstance(logprobs, dict) else None
    token_bytes = None
    valid = False
    if isinstance(items, list) and items:
        values = [item.get("bytes") if isinstance(item, dict) else None for item in items]
        if all(isinstance(v, list) and v and all(type(n) is int and 0 <= n <= 255 for n in v) for v in values):
            token_bytes = [bytes(v) for v in values]
            try:
                valid = content_known and b"".join(token_bytes) == content.encode("utf-8")
            except UnicodeError:
                valid = False
    return ParsedChat(
        content=content,
        prompt_tokens=_count(usage.get("prompt_tokens")),
        completion_tokens=_count(usage.get("completion_tokens")),
        reasoning_tokens=_count(details.get("reasoning_tokens")),
        usage=usage,
        model=_string(data.get("model")) or "",
        provider=_string(data.get("provider")),
        generation_id=_string(data.get("id")),
        finish_reason=_string(choice.get("finish_reason")),
        token_bytes=token_bytes,
        token_bytes_valid=valid,
        raw=data,
    )


def _scrub(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {
            _scrub(str(k), secret): "[REDACTED]" if str(k).lower() in {"authorization", "api_key", "api-key", "x-api-key"} else _scrub(v, secret)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(v, secret) for v in value]
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


class BudgetedTransport:
    def __init__(self, store: RunStore, api_key: str, max_requests: int,
                 client: httpx.Client | None = None, min_interval: float = 3.1,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if type(max_requests) is not int or max_requests < 0:
            raise ValueError("max_requests must be a nonnegative integer")
        if not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("min_interval must be finite and nonnegative")
        if not api_key:
            raise ValueError("API key is required for online transport")
        self.store, self.api_key, self.max_requests = store, api_key, max_requests
        self.min_interval, self.sleep, self.clock = min_interval, sleep, clock
        self._last_started: float | None = None
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0, write=10.0, pool=5.0),
            transport=httpx.HTTPTransport(retries=0), follow_redirects=False,
        )

    def send(self, method: str, path: str, *, json: Any = None,
             params: dict[str, Any] | None = None, purpose: str = "",
             probe_id: str | None = None, group_id: str | None = None,
             logical_id: str | None = None) -> AttemptResult:
        method = method.upper()
        parsed = urlsplit(path)
        if method not in {"GET", "POST"} or parsed.scheme or parsed.netloc or not path.startswith("/") or path.startswith("//") or parsed.fragment:
            raise ValueError("Only relative OpenRouter GET/POST paths are allowed")
        if self.store.attempts_used >= self.max_requests:
            raise BudgetExceeded(f"HTTP attempt budget exhausted ({self.store.attempts_used}/{self.max_requests})")
        # Build/serialize before reservation: invalid local requests never reach the wire.
        request = self._client.build_request(method, BASE_URL + path, json=json, params=params,
            headers={"Authorization": f"Bearer {self.api_key}", "X-OpenRouter-Cache": "false"})
        if self._last_started is not None:
            wait = self.min_interval - (self.clock() - self._last_started)
            if wait > 0:
                self.sleep(wait)
        if self.store.attempts_used >= self.max_requests:
            raise BudgetExceeded("HTTP attempt budget exhausted")
        aid = uuid.uuid4().hex
        self.store.append("attempt_started", attempt_id=aid,
            request=_scrub({"method": method, "path": path, "json": json, "params": params,
                            "headers": {"X-OpenRouter-Cache": "false"}}, self.api_key),
            purpose=_scrub(purpose, self.api_key), probe_id=_scrub(probe_id, self.api_key),
            group_id=_scrub(group_id, self.api_key), logical_id=_scrub(logical_id, self.api_key),
            attempts_used=self.store.attempts_used + 1, max_requests=self.max_requests)
        start = self.clock()
        self._last_started = start
        result = AttemptResult(attempt_id=aid, method=method)
        try:
            response = self._client.send(request, follow_redirects=False)
            result.status_code = response.status_code
            result.headers = _scrub({k: v for k, v in response.headers.items()
                if k in {"content-type", "retry-after", "x-generation-id", "x-request-id", "x-openrouter-cache"}
                or k.startswith("x-openrouter-cache-") or k.startswith("x-ratelimit-")}, self.api_key)
            try:
                result.data = _scrub(jsonlib.loads(response.content.decode("utf-8"), parse_constant=_reject_constant), self.api_key)
            except (ValueError, UnicodeError) as exc:
                result.error = f"InvalidJSON: {type(exc).__name__}"
                result.data = {"raw_text": _scrub(response.content.decode("utf-8", errors="replace"), self.api_key)}
            if response.status_code >= 400 and result.error is None:
                result.error = f"HTTP {response.status_code}"
            elif 300 <= response.status_code < 400:
                result.error = f"Redirect refused: HTTP {response.status_code}"
            response.close()
        except httpx.HTTPError as exc:
            result.error = _scrub(f"{type(exc).__name__}: {exc}", self.api_key)
            result.uncertain = method == "POST" and not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
        result.elapsed_seconds = max(0.0, self.clock() - start)
        self.store.append("attempt_finished", attempt_id=aid, status_code=result.status_code,
            data=result.data, headers=result.headers, error=result.error, uncertain=result.uncertain,
            elapsed_seconds=result.elapsed_seconds, method=method)
        return result

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "BudgetedTransport":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def retry_after_seconds(headers: dict[str, str]) -> float | None:
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if value is None:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


def classify_retry(result: AttemptResult) -> bool:
    """Eligibility only: the runner must also enforce per-request and run limits."""
    if result.uncertain:
        return False
    if result.status_code is None:
        return result.method == "GET" and result.error is not None
    if result.status_code in {502, 503, 504}:
        return True
    if result.status_code == 429:
        text = jsonlib.dumps(result.data, ensure_ascii=False).lower()
        return not any(s in text for s in ("daily", "per day", "per-day", "requests/day"))
    if result.status_code == 402 and isinstance(result.data, dict):
        error = result.data.get("error")
        metadata = error.get("metadata") if isinstance(error, dict) else None
        return isinstance(metadata, dict) and metadata.get("limit_source") == "openrouter_in_flight_budget"
    return False

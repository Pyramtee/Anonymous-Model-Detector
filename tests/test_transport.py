import json

import httpx
import pytest

from tokenizer_detector.run_store import RunStore, RunStoreError, read_events
from tokenizer_detector.transport import (
    AttemptResult, BudgetedTransport, BudgetExceeded, classify_retry,
    parse_chat_result, retry_after_seconds,
)


def chat(content="中", **usage):
    return {"id": "gen-1", "model": "actual/model", "provider": "provider-a",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": usage}


@pytest.mark.parametrize("limit", [12, 40])
def test_all_http_attempts_count_and_limit_cannot_be_exceeded(tmp_path, limit):
    calls = []
    def handler(request):
        calls.append(request)
        # Reservation must already exist on disk when the HTTP call begins.
        assert len([e for e in read_events(tmp_path) if e["type"] == "attempt_started"]) == len(calls)
        return httpx.Response(429, json={"error": {"message": "rate limit"}})
    with RunStore(tmp_path, create=True) as store, httpx.Client(transport=httpx.MockTransport(handler)) as client:
        transport = BudgetedTransport(store, "secret", limit, client, min_interval=0)
        for index in range(limit):
            result = transport.send("GET" if index % 2 else "POST", "/chat/completions")
            assert result.status_code == 429
            assert len(calls) == index + 1  # never an implicit retry
        with pytest.raises(BudgetExceeded):
            transport.send("POST", "/chat/completions")
        assert store.attempts_used == limit == len(calls)


def test_resume_12_to_40_keeps_cumulative_budget_including_orphan(tmp_path):
    with RunStore(tmp_path, create=True) as store:
        for index in range(12):
            store.append("attempt_started", attempt_id=str(index))
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=chat(prompt_tokens=3))
    with RunStore(tmp_path) as store, httpx.Client(transport=httpx.MockTransport(handler)) as client:
        transport = BudgetedTransport(store, "secret", 40, client, min_interval=0)
        for _ in range(28):
            transport.send("GET", "/generation", params={"id": "gen-1"})
        with pytest.raises(BudgetExceeded):
            transport.send("GET", "/generation")
        assert store.attempts_used == 40
        assert len(calls) == 28


def test_no_http_after_reservation_fsync_failure(tmp_path, monkeypatch):
    calls = []
    with RunStore(tmp_path, create=True) as store, httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r))) as client:
        def fail(_):
            raise OSError("disk failure")
        monkeypatch.setattr("tokenizer_detector.run_store.os.fsync", fail)
        transport = BudgetedTransport(store, "secret", 12, client, min_interval=0)
        with pytest.raises(RunStoreError):
            transport.send("POST", "/chat/completions")
        with pytest.raises(RunStoreError):
            transport.send("POST", "/chat/completions")
    assert not calls


def test_redirect_is_not_followed_and_secret_reflections_are_scrubbed(tmp_path):
    secret = "sk-secret-token"
    calls = []
    def handler(request):
        calls.append(request)
        assert request.headers["authorization"] == f"Bearer {secret}"
        assert request.headers["x-openrouter-cache"] == "false"
        return httpx.Response(302, headers={"location": "https://elsewhere.example/", "x-request-id": secret},
                              json={"error": {"message": secret, "authorization": "another-secret"}})
    with RunStore(tmp_path, create=True) as store, httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        result = BudgetedTransport(store, secret, 12, client, min_interval=0).send(
            "POST", "/chat/completions", json={"prompt": secret}, purpose=secret)
        assert result.error == "Redirect refused: HTTP 302"
        assert result.headers["x-request-id"] == "[REDACTED]"
        assert len(calls) == 1
    persisted = (tmp_path / "events.jsonl").read_text()
    assert secret not in persisted and "another-secret" not in persisted


@pytest.mark.parametrize("method,error,uncertain,retry", [
    ("POST", httpx.ReadTimeout, True, False),
    ("POST", httpx.WriteTimeout, True, False),
    ("POST", httpx.ConnectTimeout, False, False),
    ("GET", httpx.ReadTimeout, False, True),
])
def test_network_errors_are_counted_and_ambiguous_posts_not_retried(tmp_path, method, error, uncertain, retry):
    def handler(request):
        raise error("reflected secret", request=request)
    with RunStore(tmp_path, create=True) as store, httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = BudgetedTransport(store, "secret", 12, client, min_interval=0).send(method, "/generation")
        assert result.uncertain is uncertain
        assert classify_retry(result) is retry
        assert store.attempts_used == 1
        assert "secret" not in result.error


@pytest.mark.parametrize("body", [b"not json", b"{\"x\": NaN}", b"\xff"])
def test_malformed_responses_remain_logged_and_consume_budget(tmp_path, body):
    with RunStore(tmp_path, create=True) as store, httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))) as client:
        result = BudgetedTransport(store, "secret", 12, client, min_interval=0).send("POST", "/chat/completions")
        assert result.error.startswith("InvalidJSON:")
        assert store.attempts_used == 1
        assert "raw_text" in store.events[-1]["data"]


def test_throttle_between_starts(tmp_path):
    now = [0.0]
    starts, sleeps = [], []
    def handler(request):
        starts.append(now[0])
        now[0] += 0.2
        return httpx.Response(200, json={})
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    with RunStore(tmp_path, create=True) as store, httpx.Client(transport=httpx.MockTransport(handler)) as client:
        transport = BudgetedTransport(store, "secret", 12, client, sleep=sleep, clock=lambda: now[0])
        transport.send("GET", "/generation")
        transport.send("GET", "/generation")
    assert starts == pytest.approx([0, 3.1])
    assert sleeps == pytest.approx([2.9])


@pytest.mark.parametrize("invalid", [True, False, "10", 10.0, -1, None])
def test_usage_requires_actual_nonnegative_integer(invalid):
    parsed = parse_chat_result(chat(prompt_tokens=invalid, completion_tokens=invalid,
                                   completion_tokens_details={"reasoning_tokens": invalid},
                                   native_tokens_prompt=10))
    assert parsed.prompt_tokens is None
    assert parsed.completion_tokens is None
    assert parsed.reasoning_tokens is None


def test_usage_preserves_prompt_cache_and_hidden_reasoning_counts():
    parsed = parse_chat_result(chat(prompt_tokens=12, completion_tokens=8,
                                   prompt_tokens_details={"cached_tokens": 10},
                                   completion_tokens_details={"reasoning_tokens": 7}))
    assert parsed.prompt_tokens == 12 and parsed.completion_tokens == 8
    assert parsed.reasoning_tokens == 7
    assert parsed.usage["prompt_tokens_details"]["cached_tokens"] == 10
    assert (parsed.model, parsed.provider, parsed.generation_id) == ("actual/model", "provider-a", "gen-1")


@pytest.mark.parametrize("values,valid", [
    ([[228], [184, 173]], True),  # UTF-8 may be split across token boundaries
    ([[228, 184, 173]], True),
    ([[228, 184]], False),
    ([[True, 184, 173]], False),
    ([None], False),
    ([[]], False),
])
def test_logprob_bytes_validate_complete_content_not_token_strings(values, valid):
    data = chat()
    data["choices"][0]["logprobs"] = {"content": [{"token": "untrusted display", "bytes": value} for value in values]}
    assert parse_chat_result(data).token_bytes_valid is valid


@pytest.mark.parametrize("status,data,expected", [
    (429, {"error": {"message": "daily limit"}}, False),
    (429, {"error": {"message": "rate limit"}}, True),
    (402, {"error": {"metadata": {"limit_source": "openrouter_in_flight_budget"}}}, True),
    (402, {}, False), (503, {}, True), (400, {}, False), (401, {}, False),
])
def test_retry_classification(status, data, expected):
    assert classify_retry(AttemptResult("x", status_code=status, data=data)) is expected


def test_retry_after_parser():
    assert retry_after_seconds({"Retry-After": "3"}) == 3
    assert retry_after_seconds({"retry-after": "nan"}) is None
    assert retry_after_seconds({"retry-after": "invalid"}) is None
    assert retry_after_seconds({}) is None

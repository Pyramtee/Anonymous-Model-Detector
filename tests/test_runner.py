"""Offline integration fixtures are hand-written count tables, never adapters."""

import json
from dataclasses import replace

import httpx
import pytest

from tokenizer_detector.models import Probe, ProbeBank, Profile
from tokenizer_detector.run_store import RunStore, RunStoreError, read_events
from tokenizer_detector.runner import RunConfig, execute_run, replay_run


def manual_bank(selection_count=3, equivalent_selection=False):
    probes = {"a": Probe("a", "anchor", "fixed anchor", "anchor", "anchor")}
    alpha, beta = {"a": 1}, {"a": 1}
    for index in range(selection_count):
        pid = f"s{index:02d}"
        probes[pid] = Probe(pid, pid, f"fixed selection {index:02d}", "text", "selection")
        alpha[pid] = index + 2
        beta[pid] = alpha[pid] if equivalent_selection else 2 * index + 3
    for index in range(4):
        pid = f"c{index}"
        probes[pid] = Probe(pid, pid, f"fixed holdout {index}", "text", "confirmation", round=index // 2)
        alpha[pid], beta[pid] = index + 5, 2 * index + 9
    profiles = {
        "alpha": Profile("alpha", "alpha-ref", "alpha-family", "A", "Alpha", "ordinary", alpha),
        "beta": Profile("beta", "beta-ref", "beta-family", "B", "Beta", "ordinary", beta),
    }
    return ProbeBank(probes, profiles, {"fixture": "manual-v1", "selection_count": selection_count,
                                       "equivalent_selection": equivalent_selection})


def response_for(bank, request, **updates):
    body = json.loads(request.content)
    text = body["messages"][0]["content"]
    probe = next(p for p in bank.probes.values() if p.text == text)
    data = {"id": f"gen-{probe.probe_id}", "model": "actual-model", "provider": "stable-provider",
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": bank.profiles["alpha"].counts[probe.probe_id] + 10,
                      "completion_tokens": 1}}
    data.update(updates)
    return data


def execute_mock(path, bank, handler, *, mode="quick", resume=False, adapters=None, **config_kwargs):
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return execute_run(path, bank, RunConfig("anonymous-model", mode=mode, min_interval=0,
                                               **config_kwargs), "offline-key",
                           client=client, sleep=lambda _: None, resume=resume, adapters=adapters)


def test_confirmed_run_replays_identically_as_json_v2_and_exact_requests(tmp_path):
    bank, calls = manual_bank(), []
    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        assert body["model"] == "anonymous-model"
        assert body["messages"] in [[{"role": "user", "content": p.text}] for p in bank.probes.values()]
        assert body["max_tokens"] == 32
        assert body["stream"] is False
        assert body["provider"]["allow_fallbacks"] is False
        assert body["provider"]["only"] == ["chosen-provider"]
        assert "nonce" not in body
        assert request.headers["x-openrouter-cache"] == "false"
        return httpx.Response(200, json=response_for(bank, request))
    report = execute_mock(tmp_path, bank, handler, provider="chosen-provider")
    assert report["status"] == "compatible"
    assert report["confirmation"]["complete"] is True
    assert report["candidates"][0]["family"] == "alpha-family"
    assert report["budget"]["used"] == len(calls) == 8
    assert report["schema_version"] == 2
    assert json.loads(json.dumps(report, ensure_ascii=False)) == report
    assert replay_run(tmp_path) == report
    assert "offline-key" not in (tmp_path / "events.jsonl").read_text()


@pytest.mark.parametrize("mode,limit", [("quick", 12), ("deep", 40)])
def test_real_orchestration_counts_retry_and_diagnostic_within_total_budget(tmp_path, mode, limit):
    bank, calls = manual_bank(50, equivalent_selection=True), []
    pending_native = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rate limit"}})
        if request.method == "GET":
            assert request.url.path.endswith("/generation")
            return httpx.Response(200, json={"data": {"native_tokens_prompt": pending_native.pop()}})
        data = response_for(bank, request)
        if len(calls) == 2:
            pending_native.append(data["usage"]["prompt_tokens"])
            data["usage"] = {}
        return httpx.Response(200, json=data)
    report = execute_mock(tmp_path, bank, handler, mode=mode)
    assert report["budget"]["limit"] == limit
    assert report["budget"]["used"] == len(calls) == 10
    assert len(calls) <= limit
    assert [r.method for r in calls[:3]] == ["POST", "POST", "GET"]
    assert report["confirmation"]["complete"] is True
    assert report["status"] == "compatible"
    assert replay_run(tmp_path) == report


def test_partial_confirmation_never_claims_compatible_and_resume_uses_new_holdouts(tmp_path):
    bank, calls = manual_bank(), []
    def first_handler(request):
        calls.append(json.loads(request.content)["messages"][0]["content"])
        if calls[-1] == bank.probes["c1"].text:
            return httpx.Response(400, json={"error": {"message": "unavailable"}})
        return httpx.Response(200, json=response_for(bank, request))
    first = execute_mock(tmp_path, bank, first_handler)
    assert first["status"] == "request_failed"
    assert first["confirmation"]["complete"] is False
    assert first["budget"]["used"] == 8
    assert replay_run(tmp_path) == first
    resumed_calls = []
    def resumed_handler(request):
        resumed_calls.append(json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(200, json=response_for(bank, request))
    resumed = execute_mock(tmp_path, bank, resumed_handler, mode="deep", resume=True)
    assert resumed["status"] == "compatible"
    assert resumed["confirmation"]["rounds"] == 2
    assert resumed["budget"] == {"used": 11, "limit": 40, "outcome_unknown": []}
    assert resumed_calls == [bank.probes[p].text for p in ("a", "c2", "c3")]
    assert replay_run(tmp_path) == resumed


@pytest.mark.parametrize("field,value", [("provider", "different-provider"), ("model", "different-model")])
def test_route_drift_is_unstable_not_a_cross_route_match(tmp_path, field, value):
    bank, calls = manual_bank(), []
    def handler(request):
        calls.append(request)
        updates = {field: value} if len(calls) == 2 else {}
        return httpx.Response(200, json=response_for(bank, request, **updates))
    report = execute_mock(tmp_path, bank, handler)
    assert report["status"] == "unstable_endpoint"
    assert report["confirmation"]["complete"] is False
    assert len(calls) == 2
    assert replay_run(tmp_path) == report


def test_post_timeout_is_recorded_unknown_and_not_retransmitted(tmp_path):
    bank, calls = manual_bank(), []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("response outcome unknown", request=request)
    report = execute_mock(tmp_path, bank, handler)
    assert len(calls) == report["budget"]["used"] == 1
    assert len(report["budget"]["outcome_unknown"]) == 1
    assert report["stop_reason"] == "outcome_unknown"
    assert report["status"] == "request_failed"
    assert replay_run(tmp_path) == report


def test_small_budget_does_not_turn_a_shortlist_into_confirmation(tmp_path):
    bank, calls = manual_bank(), []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response_for(bank, request))
    report = execute_mock(tmp_path, bank, handler, max_requests=7)
    assert report["budget"]["used"] == len(calls) <= 7
    assert report["status"] == "insufficient_evidence"
    assert report["confirmation"]["complete"] is False
    assert replay_run(tmp_path) == report


@pytest.mark.parametrize("status,payload", [
    (500, {"data": {"native_tokens_prompt": 11}}),
    (200, {"data": [11]}),
    (200, {"data": "11"}),
    (200, {"data": {"native_tokens_prompt": True}}),
    (200, {"data": {"tokens_prompt": 11}}),
])
def test_invalid_diagnostic_cannot_supply_native_usage(tmp_path, status, payload):
    bank, calls = manual_bank(), []
    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(status, json=payload)
        return httpx.Response(200, json=response_for(bank, request, usage={}))
    report = execute_mock(tmp_path, bank, handler)
    assert report["status"] == "insufficient_evidence"
    assert report["channels"]["input_count"] == "unavailable"
    assert report["budget"]["used"] == len(calls) == 2
    assert report["stop_reason"] == "missing_native_usage"


def test_resume_rejects_reference_identity_change_without_http(tmp_path):
    bank = manual_bank()
    execute_mock(tmp_path, bank, lambda request: httpx.Response(200, json=response_for(bank, request)))
    changed = replace(bank, identity={"fixture": "changed"})
    def no_http(request):
        pytest.fail("incompatible resume must not send HTTP")
    with pytest.raises(RunStoreError, match="变化"):
        execute_mock(tmp_path, changed, no_http, mode="deep", resume=True)


def test_orphan_reservation_survives_resume_and_budget_accounting(tmp_path):
    bank = manual_bank()
    first = execute_mock(tmp_path, bank, lambda request: httpx.Response(200, json=response_for(bank, request)))
    with RunStore(tmp_path) as store:
        store.append("attempt_started", attempt_id="crash-uncertain", purpose="confirmation")
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response_for(bank, request))
    resumed = execute_mock(tmp_path, bank, handler, mode="deep", resume=True)
    assert resumed["budget"]["used"] == first["budget"]["used"] + 1 + len(calls)
    assert resumed["budget"]["outcome_unknown"] == ["crash-uncertain"]
    assert replay_run(tmp_path) == resumed


def output_bank():
    bank = manual_bank()
    original_beta = bank.profiles["beta"]
    bank.profiles["beta"] = replace(original_beta, counts=dict(bank.profiles["alpha"].counts))
    bank.profiles["gamma"] = replace(original_beta, profile_id="gamma", reference_id="gamma-ref",
                                      family_id="gamma-family", vendor="C", display="Gamma",
                                      counts=dict(original_beta.counts))
    for index in range(3):
        pid = f"o{index}"
        bank.probes[pid] = Probe(pid, pid, f"fixed output {index}", "output", "output")
        for profile in bank.profiles.values():
            profile.counts[pid] = 5
    bank.identity = {"fixture": "manual-output-v1"}
    return bank


class StaticVocabulary:
    """Independent fixture vocabulary; intentionally no tokenizer implementation."""
    def __init__(self, *tokens):
        self.tokens = frozenset(tokens)

    def vocabulary_bytes(self):
        return self.tokens


def output_adapters():
    return {"alpha-ref": StaticVocabulary(b"OK"), "beta-ref": StaticVocabulary(b"NO"),
            "gamma-ref": StaticVocabulary(b"??")}


def test_output_channel_requires_two_fresh_confirmations_and_counts_retry(tmp_path):
    bank, output_calls = output_bank(), []
    def handler(request):
        body = json.loads(request.content)
        data = response_for(bank, request)
        if body.get("logprobs"):
            output_calls.append(request)
            assert body["max_tokens"] == 128
            assert body["provider"]["require_parameters"] is True
            assert body["provider"]["allow_fallbacks"] is False
            if len(output_calls) == 1:
                return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rate limit"}})
            data["choices"][0]["logprobs"] = {"content": [{"token": "OK", "bytes": [79, 75]}]}
        return httpx.Response(200, json=data)
    report = execute_mock(tmp_path, bank, handler, mode="deep", adapters=output_adapters())
    assert report["status"] == "compatible"
    assert report["confirmation"] == {"complete": True, "rounds": 2, "source": "output", "output_complete": True}
    assert report["channels"]["output_bytes"] == "observed"
    assert report["budget"]["used"] == 12
    assert len(output_calls) == 4
    assert replay_run(tmp_path) == report


def test_output_bytes_without_prompt_usage_need_only_three_posts_and_no_diagnostics(tmp_path):
    bank, output_calls = output_bank(), []
    def handler(request):
        assert request.method == "POST", "Reliable output bytes must not trigger usage diagnostics"
        body = json.loads(request.content)
        data = response_for(bank, request)
        if body.get("logprobs"):
            output_calls.append(request)
            data.pop("usage")
            data["choices"][0]["logprobs"] = {"content": [{"token": "OK", "bytes": [79, 75]}]}
        return httpx.Response(200, json=data)
    report = execute_mock(tmp_path, bank, handler, mode="deep", adapters=output_adapters())
    assert report["status"] == "compatible"
    assert report["confirmation"]["output_complete"] is True
    assert report["channels"]["output_bytes"] == "observed"
    assert len(output_calls) == 3
    assert report["budget"]["used"] == 11
    assert replay_run(tmp_path) == report


@pytest.mark.parametrize("failure", ["unsupported", "missing_bytes", "different_route"])
def test_output_capability_failures_preserve_input_ambiguity(tmp_path, failure):
    bank, output_calls = output_bank(), []
    def handler(request):
        body = json.loads(request.content)
        data = response_for(bank, request)
        if body.get("logprobs"):
            output_calls.append(request)
            if failure == "unsupported":
                return httpx.Response(400, json={"error": {"message": "logprobs unsupported"}})
            if failure == "different_route":
                data["provider"] = "another-provider"
                data["choices"][0]["logprobs"] = {"content": [{"bytes": [79, 75]}]}
        return httpx.Response(200, json=data)
    report = execute_mock(tmp_path, bank, handler, mode="deep", adapters=output_adapters())
    assert report["status"] == ("unstable_endpoint" if failure == "different_route" else "ambiguous")
    assert {p["family"] for p in report["candidates"]} == {"alpha-family", "beta-family"}
    assert report["channels"]["output_bytes"] == "not_observed"
    assert len(output_calls) == 1
    assert replay_run(tmp_path) == report


@pytest.mark.parametrize("header", ["x-openrouter-cache-status", "x-openrouter-cache"])
def test_response_cache_hit_is_not_a_fresh_observation(tmp_path, header):
    bank, calls = manual_bank(), []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={header: "HIT"},
                              json=response_for(bank, request))
    report = execute_mock(tmp_path, bank, handler)
    assert report["status"] == "insufficient_evidence"
    assert report["channels"]["input_count"] == "unavailable"
    assert len(calls) == 1
    assert replay_run(tmp_path) == report


@pytest.mark.parametrize("declared_minimum,expected_calls,expected_status", [(64, 9, "compatible"), (256, 1, "request_failed")])
def test_declared_output_minimum_has_finite_cap_and_budgeted_adaptation(tmp_path, declared_minimum, expected_calls, expected_status):
    bank, calls = manual_bank(), []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"message": f"max_tokens minimum {declared_minimum}"}})
        assert body["max_tokens"] == declared_minimum
        return httpx.Response(200, json=response_for(bank, request))
    report = execute_mock(tmp_path, bank, handler)
    assert report["status"] == expected_status
    assert report["budget"]["used"] == len(calls) == expected_calls
    assert replay_run(tmp_path) == report


@pytest.mark.parametrize("corruption", ["missing_bank", "unknown_event", "missing_attempt", "wrong_phase", "wrong_sent_text"])
def test_semantically_corrupt_journal_refuses_replay_and_resume_without_mutation(tmp_path, corruption):
    bank = manual_bank()
    execute_mock(tmp_path, bank, lambda request: httpx.Response(200, json=response_for(bank, request)))
    events = read_events(tmp_path)
    if corruption == "missing_bank":
        del events[0]["bank"]
    elif corruption == "unknown_event":
        events.append({"type": "unknown_semantics", "seq": len(events) + 1,
                       "timestamp": events[-1]["timestamp"]})
    elif corruption == "missing_attempt":
        next(e for e in events if e["type"] == "observation")["observation"]["attempt_id"] = "nonexistent-attempt"
    elif corruption == "wrong_phase":
        next(e for e in events if e["type"] == "observation")["observation"]["phase"] = "selection"
    else:
        first_start = next(e for e in events if e["type"] == "attempt_started")
        first_start["request"]["json"]["messages"][0]["content"] = "different unrecorded probe"
    journal = tmp_path / "events.jsonl"
    journal.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events))
    original_bytes = journal.read_bytes()
    original_attempts = sum(e["type"] == "attempt_started" for e in read_events(tmp_path))
    with pytest.raises(RunStoreError):
        replay_run(tmp_path)
    assert journal.read_bytes() == original_bytes
    calls = []
    def no_http(request):
        calls.append(request)
        pytest.fail("Semantic corruption must be rejected before sending any HTTP")
    with pytest.raises(RunStoreError):
        execute_mock(tmp_path, bank, no_http, mode="deep", resume=True)
    assert calls == []
    assert journal.read_bytes() == original_bytes
    assert sum(e["type"] == "attempt_started" for e in read_events(tmp_path)) == original_attempts

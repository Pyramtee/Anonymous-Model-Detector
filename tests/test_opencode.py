"""Native client doubles and hand-written upstream streams; no external network."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import httpx
import pytest

from tokenizer_detector import cli
from tokenizer_detector.opencode import OpenCodeBackend, OpenCodeError, _client_version, request_context
from tokenizer_detector.run_store import RunStore, RunStoreError, read_events
from tokenizer_detector.runner import RunConfig, execute_run, replay_run
from tokenizer_detector.transport import BudgetedTransport, parse_chat_result, parse_chat_sse
from typer.testing import CliRunner
from test_runner import manual_bank, output_bank, output_adapters


def sse(prompt_tokens=20, *, model="big-pickle", tool=False, cache=0):
    delta = {"tool_calls": [{"index": 0, "id": "call-1", "type": "function",
                            "function": {"name": "write", "arguments": '{"filePath":"unexpected","content":"bad"}'}}]} if tool else {"content": "中"}
    chunks = [{"id": "gen-1", "model": model, "choices": [{"index": 0, "delta": delta}]},
              {"id": "gen-1", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool else "stop"}]},
              {"id": "gen-1", "model": model, "choices": [], "usage": {
                  "prompt_tokens": prompt_tokens, "completion_tokens": 32,
                  "prompt_tokens_details": {"cached_tokens": cache},
                  "completion_tokens_details": {"reasoning_tokens": 31}}}]
    return ("".join("data: " + json.dumps(c, ensure_ascii=False) + "\r\n\r\n" for c in chunks)
            + "data: [DONE]\r\n\r\n").encode("utf-8")


class Pieces(httpx.SyncByteStream):
    def __init__(self, payload):
        self.payload = payload

    def __iter__(self):
        # Includes split UTF-8 sequences and split SSE delimiters.
        for offset in range(0, len(self.payload), 3):
            yield self.payload[offset:offset + 3]


class NativeDouble:
    def __init__(self, *, drift_at=None, extra=False, version="opencode v2.0.28", fail=False):
        self.backend = None
        self.drift_at, self.extra, self.version, self.fail = drift_at, extra, version, fail
        self.calls = []

    def __call__(self, args, **kwargs):
        if args[-1] == "--version":
            return subprocess.CompletedProcess(args, 0, self.version.encode(), b"")
        text = kwargs["input"].decode("utf-8")
        self.calls.append(text)
        config = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
        assert "--standalone" in args and "--title" in args and text not in args
        assert "--pure" not in args and "--dir" not in args
        assert config["share"] == "disabled" and config["snapshots"] is False
        assert config["permissions"] == [{"action": "*", "resource": "*", "effect": "ask"}]
        assert config["policies"] == [{"action": "provider.use", "resource": "*", "effect": "deny"},
                                     {"action": "provider.use", "resource": "opencode", "effect": "allow"}]
        assert config["update"] == "disable" and config["plugins"] == []
        assert config["compaction"] == {"auto": False} and config["warming"] is False
        assert config["agents"]["title"] == {"disabled": True}
        provider = config["providers"]["opencode"]
        assert provider["package"] == "@opencode/ai/providers/openai-compatible"
        assert provider["compatibility"] == {"maxTokensField": "max_tokens"}
        assert kwargs["env"]["PWD"] == str(kwargs["cwd"])
        assert Path(kwargs["env"]["OPENCODE_TEST_HOME"]) == self.backend.runtime / "home"
        assert Path(kwargs["env"]["TMPDIR"]) == self.backend.runtime / "tmp"
        body = {"model": self.backend.model, "max_tokens": self.backend.output_limit,
                "messages": [{"role": "system", "content": "fixed client system"},
                             {"role": "user", "content": text},
                             {"role": "assistant", "content": "fixed step limit"}],
                "tools": [{"type": "function", "function": {"name": "write", "parameters": {"type": "object"}}}],
                "stream": True, "stream_options": {"include_usage": True}, "tool_choice": "auto"}
        if self.drift_at == len(self.calls):
            body["messages"][0]["content"] = "changed client system"
        self.backend.receive(body, {"User-Agent": "opencode/latest/" + _client_version(self.version.encode()) + "/cli",
                                   "x-opencode-session": "ses-offline", "x-opencode-client": "cli"})
        ack = self.backend.acknowledgment()
        assert b"tool_calls" not in ack and b'"prompt_tokens": 0' in ack
        if self.extra:
            self.backend.receive(body, {})
        return subprocess.CompletedProcess(args, 1 if self.fail else 0, b"", b"client error" if self.fail else b"")

    def factory(self, store, key, limit, **kwargs):
        self.backend = OpenCodeBackend(store, key, limit, process_runner=self,
                                       binary="/offline/opencode", **kwargs)
        return self.backend


def native_run(path, bank, *, native=None, handler=None, mode="quick", resume=False, **kwargs):
    native = native or NativeDouble()
    calls = []
    def default(request):
        calls.append(request)
        body = json.loads(request.content)
        text = next(m["content"] for m in body["messages"] if m["role"] == "user")
        probe = next(p for p in bank.probes.values() if p.text == text)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=Pieces(sse(bank.profiles["alpha"].counts[probe.probe_id] + 8000, tool=True, cache=7000)))
    with httpx.Client(transport=httpx.MockTransport(handler or default)) as client:
        report = execute_run(path, bank, RunConfig("opencode/big-pickle", backend="opencode",
            mode=mode, min_interval=0, **kwargs), "", client=client,
            opencode_factory=native.factory, sleep=lambda _: None, resume=resume,
            adapters=output_adapters())
    return report, calls, native


def test_native_run_counts_raw_usage_preserves_text_and_replays(tmp_path):
    bank = manual_bank()
    bank.probes["s00"] = replace(bank.probes["s00"], text='Data:\nहिन्दी "quotes"  \t世界\r\nEnd.')
    report, calls, native = native_run(tmp_path, bank)
    assert report["status"] == "compatible"
    assert report["budget"]["used"] == len(calls) == 8
    assert report["backend"] == "opencode" and report["model"] == "big-pickle"
    assert report["opencode"]["client_version"] == "2.0.28"
    assert report["opencode"]["bridge_version"] == "opencode-bridge-v2"
    assert report["channels"]["output_bytes"] == "unsupported"
    assert bank.probes["s00"].text in native.calls
    assert replay_run(tmp_path) == report
    for request in calls:
        assert str(request.url) == "https://opencode.ai/inference/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer public"
        assert request.headers["user-agent"] == "opencode/latest/2.0.28/cli"
        assert "x-openrouter-cache" not in request.headers
        assert "provider" not in json.loads(request.content)
    events = read_events(tmp_path)
    assert all(e["usage_source"] == "response_usage" and e["observation"]["prompt_tokens"] > 8000
               for e in events if e["type"] == "observation")
    assert any("raw_sse" in e.get("data", {}) for e in events if e["type"] == "attempt_finished")


def test_native_deep_does_not_attempt_output_channel(tmp_path):
    report, calls, _ = native_run(tmp_path, output_bank(), mode="deep")
    assert report["status"] == "ambiguous"
    assert len(calls) == 8
    assert not any(e.get("purpose") in {"output", "output_confirmation"} for e in read_events(tmp_path))


@pytest.mark.parametrize("limit", [1, 6, 12, 40])
def test_native_budget_is_cumulative_across_resume(tmp_path, limit):
    bank = manual_bank()
    report, calls, _ = native_run(tmp_path, bank, max_requests=limit)
    assert report["budget"]["used"] == len(calls) <= limit
    resumed, new_calls, _ = native_run(tmp_path, bank, mode="deep", resume=True, max_requests=50)
    assert resumed["budget"]["used"] == len(calls) + len(new_calls)
    assert replay_run(tmp_path) == resumed


def test_native_context_change_is_blocked_before_upstream(tmp_path):
    report, calls, _ = native_run(tmp_path, manual_bank(), native=NativeDouble(drift_at=2))
    assert report["status"] == "unstable_endpoint"
    assert report["stop_reason"] == "opencode_context_changed"
    assert report["budget"]["used"] == len(calls) == 1
    assert replay_run(tmp_path) == report


def test_native_extra_client_request_is_never_forwarded(tmp_path):
    report, calls, _ = native_run(tmp_path, manual_bank(), native=NativeDouble(extra=True))
    assert report["status"] == "request_failed"
    assert report["stop_reason"] == "unexpected_client_request"
    assert report["budget"]["used"] == len(calls) == 1


def test_native_version_change_and_context_change_on_resume(tmp_path):
    bank = manual_bank()
    original, _, _ = native_run(tmp_path, bank, max_requests=1)
    report, calls, _ = native_run(tmp_path, bank, native=NativeDouble(version="opencode v2.0.29"), resume=True, mode="deep")
    assert report["status"] == "unstable_endpoint" and calls == []
    assert report["budget"]["used"] == original["budget"]["used"]
    report, calls, _ = native_run(tmp_path, bank, native=NativeDouble(drift_at=1), resume=True, mode="deep")
    assert report["status"] == "unstable_endpoint" and calls == []
    assert replay_run(tmp_path) == report


def test_native_missing_usage_never_queries_generation(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        assert request.method == "POST" and request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={"id": "gen", "model": "big-pickle",
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {}})
    report, _, _ = native_run(tmp_path, manual_bank(), handler=handler)
    assert report["stop_reason"] == "missing_native_usage"
    assert report["status"] == "insufficient_evidence"
    assert len(calls) == report["budget"]["used"] == 1
    assert replay_run(tmp_path) == report


def test_native_retry_is_owned_by_runner_and_logged(tmp_path):
    bank, calls = manual_bank(), []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rate limit"}})
        body = json.loads(request.content)
        text = next(m["content"] for m in body["messages"] if m["role"] == "user")
        probe = next(p for p in bank.probes.values() if p.text == text)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=sse(bank.profiles["alpha"].counts[probe.probe_id] + 8000))
    report, _, _ = native_run(tmp_path, bank, handler=handler)
    assert report["status"] == "compatible"
    assert len(calls) == report["budget"]["used"] == 9
    assert replay_run(tmp_path) == report


@pytest.mark.parametrize("corruption", ["probe", "context", "usage", "sse", "backend"])
def test_native_corrupt_evidence_rejects_replay_and_resume_without_http(tmp_path, corruption):
    bank = manual_bank()
    native_run(tmp_path, bank)
    events = read_events(tmp_path)
    start = next(e for e in events if e["type"] == "attempt_started")
    finish = next(e for e in events if e["type"] == "attempt_finished")
    if corruption == "probe":
        start["request"]["json"]["messages"][1]["content"] = "rewritten"
    elif corruption == "context":
        start["request"]["json"]["tools"] = []
    elif corruption == "usage":
        next(e for e in events if e["type"] == "observation")["observation"]["prompt_tokens"] += 1
    elif corruption == "sse":
        finish["data"]["chat"]["usage"]["prompt_tokens"] += 1
    else:
        events[0]["config"]["backend"] = "openrouter"
    journal = tmp_path / "events.jsonl"
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    before = journal.read_bytes()
    with pytest.raises(RunStoreError):
        replay_run(tmp_path)
    with pytest.raises(RunStoreError):
        native_run(tmp_path, bank, resume=True, mode="deep", handler=lambda _: pytest.fail("network on corrupt resume"))
    assert journal.read_bytes() == before


@pytest.mark.parametrize("raw", [b"data: {}\n\n", b"data: not-json\n\ndata: [DONE]\n\n",
                                   b'data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n'])
def test_bad_sse_consumes_one_attempt_and_is_retained(tmp_path, raw):
    with RunStore(tmp_path, create=True) as store, httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Pieces(raw)))) as client:
        result = BudgetedTransport(store, "", 1, client=client, backend="opencode", min_interval=0).send(
            "POST", "/chat/completions", json={})
        assert result.error and result.data["raw_text"] == raw.decode()
        assert store.attempts_used == 1


def test_sse_keeps_full_prompt_and_reasoning_separate():
    parsed = parse_chat_result(parse_chat_sse(sse(8062, cache=7000).decode()))
    assert parsed.prompt_tokens == 8062
    assert parsed.reasoning_tokens == 31 and parsed.completion_tokens == 32
    assert parsed.usage["prompt_tokens_details"]["cached_tokens"] == 7000
    assert parsed.content == "中"


def test_zen_cost_trailer_after_done_is_metadata_not_token_evidence(tmp_path):
    raw = sse(8057, cache=6144) + b'data: {"choices":[],"cost":"0"}\n\n'
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Pieces(raw))
    report, _, _ = native_run(tmp_path, manual_bank(), handler=handler, max_requests=1)
    assert report["channels"]["input_count"] == "observed"
    assert report["budget"]["used"] == len(calls) == 1
    observation = next(e for e in read_events(tmp_path) if e["type"] == "observation")
    assert observation["observation"]["prompt_tokens"] == 8057
    assert replay_run(tmp_path) == report
    for trailer in [b'data: {"usage":{"prompt_tokens":1}}\n\n',
                    b'data: {"error":{"message":"late error"}}\n\n',
                    b'data: {"choices":[{"delta":{"content":"late"}}]}\n\n']:
        with pytest.raises(ValueError):
            parse_chat_sse((sse() + trailer).decode())


def test_backend_cli_validation_and_optional_credentials(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-only")
    assert cli._key(None, "opencode") == ""
    monkeypatch.setenv("OPENCODE_API_KEY", "zen-key")
    assert cli._key(None, "opencode") == "zen-key"
    assert cli._key("explicit", "opencode") == "explicit"
    monkeypatch.setattr(cli, "_references", lambda *_: pytest.fail("invalid arguments loaded references"))
    for args in [["--backend", "bad"], ["--backend", "opencode", "--provider", "route"]]:
        result = CliRunner().invoke(cli.app, ["detect", "--model", "big-pickle", *args])
        assert result.exit_code == 2


def test_historical_openrouter_config_remains_keyless_replayable(tmp_path):
    from test_runner import execute_mock, response_for
    bank = manual_bank()
    report = execute_mock(tmp_path, bank, lambda request: httpx.Response(200, json=response_for(bank, request)))
    events = read_events(tmp_path)
    for event in events:
        if "config" in event:
            event["config"].pop("backend", None)
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    report.pop("backend")
    assert replay_run(tmp_path) == report


def test_cli_detect_resume_and_replay_restore_native_backend(tmp_path, monkeypatch):
    bank, native = manual_bank(), NativeDouble()
    captured = []
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENCODE_API_KEY", "zen-test-secret")
    monkeypatch.setattr(cli, "_references", lambda *_: ({}, bank))
    def execute(path, bank, config, key, **kwargs):
        captured.append((config.backend, config.model, config.mode, key))
        def upstream(request):
            body = json.loads(request.content)
            text = next(m["content"] for m in body["messages"] if m["role"] == "user")
            probe = next(p for p in bank.probes.values() if p.text == text)
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=sse(bank.profiles["alpha"].counts[probe.probe_id] + 8000))
        with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
            return execute_run(path, bank, replace(config, min_interval=0), key, client=client,
                               opencode_factory=native.factory, sleep=lambda _: None, **kwargs)
    monkeypatch.setattr(cli, "execute_run", execute)
    run = tmp_path / "run"
    runner = CliRunner()
    result = runner.invoke(cli.app, ["detect", "--backend", "opencode", "--model", "opencode/big-pickle",
                                    "--max-requests", "1", "--run-dir", str(run)])
    assert result.exit_code == 0, result.exception
    result = runner.invoke(cli.app, ["resume", str(run), "--mode", "deep"])
    assert result.exit_code == 0, result.exception
    assert captured == [("opencode", "big-pickle", "quick", "zen-test-secret"),
                        ("opencode", "big-pickle", "deep", "zen-test-secret")]
    assert "zen-test-secret" not in (run / "events.jsonl").read_text()
    monkeypatch.setattr(cli, "_key", lambda *_: pytest.fail("offline replay loaded key"))
    monkeypatch.setattr(cli, "_references", lambda *_: pytest.fail("offline replay loaded references"))
    monkeypatch.setattr(cli, "execute_run", lambda *_: pytest.fail("offline replay created a client"))
    assert runner.invoke(cli.app, ["replay", str(run)]).exit_code == 0


def test_missing_opencode_is_reported_without_upstream(tmp_path, monkeypatch):
    monkeypatch.setattr("tokenizer_detector.opencode.shutil.which", lambda _: None)
    report = execute_run(tmp_path, manual_bank(), RunConfig("big-pickle", backend="opencode"), "")
    assert report["status"] == "request_failed"
    assert report["stop_reason"] == "opencode_unavailable"
    assert report["budget"]["used"] == 0
    assert replay_run(tmp_path) == report


def test_native_fsync_failure_prevents_forwarding(tmp_path, monkeypatch):
    bank, native = manual_bank(), NativeDouble()
    calls = []
    def fail(*_):
        raise OSError("disk unavailable")
    def factory(store, key, limit, **kwargs):
        backend = native.factory(store, key, limit, **kwargs)
        # Metadata was persisted; fail the first reservation fsync.
        original = store.append
        def append(kind, **fields):
            if kind == "attempt_started":
                monkeypatch.setattr("tokenizer_detector.run_store.os.fsync", fail)
            return original(kind, **fields)
        monkeypatch.setattr(store, "append", append)
        return backend
    with httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r))) as client:
        with pytest.raises(RunStoreError):
            execute_run(tmp_path, bank, RunConfig("big-pickle", backend="opencode", min_interval=0), "",
                        client=client, opencode_factory=factory)
    assert calls == []


def test_native_read_timeout_is_counted_unknown_and_not_retried(tmp_path):
    calls = []
    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout("timeout", request=request)
    report, _, native = native_run(tmp_path, manual_bank(), handler=timeout)
    assert report["stop_reason"] == "outcome_unknown"
    assert report["budget"]["used"] == len(calls) == len(native.calls) == 1
    assert len(report["budget"]["outcome_unknown"]) == 1
    assert replay_run(tmp_path) == report


@pytest.mark.parametrize("raw", [b"opencode v2.0.28\n", b"2.0.28\n", b"v2.0.28\n"])
def test_v2_version_output_is_normalized(raw):
    assert _client_version(raw) == "2.0.28"


@pytest.mark.parametrize("version,reason", [
    ("1.18.15", "opencode_unsupported_version"),
    ("opencode v3.0.0", "opencode_unsupported_version"),
    ("", "opencode_unavailable"),
    ("opencode v2.invalid", "opencode_unavailable"),
])
def test_unsupported_client_stops_before_upstream_and_replays(tmp_path, version, reason):
    native = NativeDouble(version=version)
    report, calls, _ = native_run(tmp_path, manual_bank(), native=native)
    assert report["status"] == "request_failed" and report["stop_reason"] == reason
    assert report["budget"]["used"] == 0 and calls == native.calls == []
    assert replay_run(tmp_path) == report


def test_legacy_bridge_remains_replayable_but_cannot_resume_with_v2(tmp_path, monkeypatch):
    bank = manual_bank()
    report, _, _ = native_run(tmp_path, bank, max_requests=1)
    events = read_events(tmp_path)
    ready = next(e for e in events if e["type"] == "backend_initialized")
    ready.update(bridge_version="opencode-bridge-v1", client_version="1.18.15")
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    report["opencode"].update(bridge_version="opencode-bridge-v1", client_version="1.18.15")
    with monkeypatch.context() as offline:
        offline.setattr("tokenizer_detector.opencode.shutil.which", lambda _: pytest.fail("replay loaded client"))
        assert replay_run(tmp_path) == report
    resumed, calls, _ = native_run(tmp_path, bank, resume=True)
    assert resumed["status"] == "unstable_endpoint"
    assert resumed["stop_reason"] == "opencode_version_changed" and calls == []
    assert resumed["budget"]["used"] == 1
    assert replay_run(tmp_path) == resumed


def test_unknown_bridge_version_is_not_replayable(tmp_path):
    native_run(tmp_path, manual_bank(), max_requests=1)
    events = read_events(tmp_path)
    next(e for e in events if e["type"] == "backend_initialized")["bridge_version"] = "unknown"
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(RunStoreError, match="客户端元数据"):
        replay_run(tmp_path)


@pytest.mark.parametrize("fields", [{"max_tokens": True}, {"max_tokens": 33},
                                     {"max_completion_tokens": 32}])
def test_v2_output_limit_mismatch_is_blocked_before_upstream(tmp_path, fields):
    native = NativeDouble()
    def factory(store, key, limit, **kwargs):
        backend = native.factory(store, key, limit, **kwargs)
        original = backend.receive
        backend.receive = lambda body, headers: original({**body, **fields}, headers)
        return backend
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("invalid limit reached upstream"))) as client:
        report = execute_run(tmp_path, manual_bank(), RunConfig("big-pickle", backend="opencode"), "",
                             client=client, opencode_factory=factory)
    assert report["stop_reason"] == "opencode_context_changed" and report["budget"]["used"] == 0
    assert replay_run(tmp_path) == report


def test_v2_stdout_error_is_saved_and_scrubbed(tmp_path):
    native = NativeDouble()
    secret = "private-test-key"
    def process(args, **kwargs):
        if args[-1] == "--version":
            return native(args, **kwargs)
        return subprocess.CompletedProcess(args, 1, json.dumps({"type": "error", "error": secret}).encode(), b"")
    def factory(store, key, limit, **kwargs):
        return OpenCodeBackend(store, key, limit, process_runner=process, binary="/offline/opencode", **kwargs)
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("client failure reached upstream"))) as client:
        report = execute_run(tmp_path, manual_bank(), RunConfig("big-pickle", backend="opencode"), secret,
                             client=client, opencode_factory=factory)
    assert report["stop_reason"] == "opencode_client_failed"
    assert '"type": "error"' in report["client_errors"][0]["detail"]
    assert "[REDACTED]" in report["client_errors"][0]["detail"]
    assert secret not in (tmp_path / "events.jsonl").read_text()
    assert replay_run(tmp_path) == report


@pytest.fixture
def real_v2_factory():
    binary = shutil.which("opencode")
    if binary is None:
        pytest.skip("OpenCode v2 is not installed")
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, timeout=10)
        if result.returncode:
            pytest.skip("OpenCode version command failed")
        _client_version(result.stdout)
    except (OSError, subprocess.TimeoutExpired, OpenCodeError):
        pytest.skip("OpenCode v2 is not available")
    runtimes = []
    def factory(store, key, limit, **kwargs):
        backend = OpenCodeBackend(store, key, limit, binary=binary, **kwargs)
        runtimes.append(backend.runtime)
        return backend
    yield factory
    for runtime in set(runtimes):
        shutil.rmtree(runtime)


@pytest.mark.opencode_native
@pytest.mark.parametrize("resume", [False, True])
def test_real_v2_client_full_detection_and_resume_with_mocked_upstream(
    tmp_path, monkeypatch, real_v2_factory, resume,
):
    bank = manual_bank()
    bank.probes["s00"] = replace(bank.probes["s00"], text='  Data:\nहिन्दी "quotes"\t世界\r\nEnd.  \n')
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "opencode.json").write_text('{"model":"invalid/ambient-model"}')
    monkeypatch.setenv("PWD", str(outside))
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"model":"invalid/inherited-model"}')
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(outside))
    monkeypatch.setenv("OPENCODE_MODELS_URL", "http://127.0.0.1:9/invalid")
    calls = []
    def upstream(request):
        body = json.loads(request.content)
        assert str(request.url) == "https://opencode.ai/inference/v1/chat/completions"
        assert body["max_tokens"] == 32 and "max_completion_tokens" not in body
        assert request.headers["x-opencode-client"] == "cli"
        assert request.headers["user-agent"].startswith("opencode/")
        assert request.headers["authorization"] == "Bearer public"
        text = next(m["content"] for m in body["messages"] if m["role"] == "user")
        probe = next(p for p in bank.probes.values() if p.text == text)
        calls.append(text)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=sse(bank.profiles["alpha"].counts[probe.probe_id] + 8000,
                                          tool=True, cache=7000))
    path = tmp_path / "run"
    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        report = execute_run(path, bank, RunConfig("big-pickle", backend="opencode",
                             max_requests=1 if resume else 12, min_interval=0), "", client=client,
                             opencode_factory=real_v2_factory, sleep=lambda _: None)
        assert replay_run(path) == report
        if resume:
            assert report["budget"]["used"] == 1
            report = execute_run(path, bank, RunConfig("big-pickle", backend="opencode",
                                 max_requests=12, min_interval=0), "", client=client, resume=True,
                                 opencode_factory=real_v2_factory, sleep=lambda _: None)
    assert report["status"] == "compatible" and report["confirmation"]["complete"] is True
    assert report["budget"]["used"] == len(calls) <= 12
    assert bank.probes["s00"].text in calls
    assert replay_run(path) == report
    events = read_events(path)
    assert sum(e["type"] == "request_context_frozen" for e in events) == 1
    assert not any(e["type"] == "client_request_blocked" for e in events)
    runtime = Path(report["opencode"]["runtime_dir"])
    assert not (runtime / "work" / "unexpected").exists()

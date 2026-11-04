"""Isolated OpenCode client and a budgeted, single-probe loopback bridge."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from typing import Any, Callable
import uuid

from .run_store import RunStore
from .transport import AttemptResult, BudgetedTransport, _scrub

BRIDGE_VERSION = "opencode-bridge-v2"
READABLE_BRIDGE_VERSIONS = frozenset({"opencode-bridge-v1", BRIDGE_VERSION})


class OpenCodeError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _client_version(stdout: bytes) -> str:
    raw = stdout.decode("utf-8", errors="replace").strip()
    match = re.fullmatch(r"(?:opencode\s+v|v)?(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)", raw)
    if len(raw) > 200 or match is None:
        raise OpenCodeError("opencode_unavailable", "opencode --version 未返回有效版本")
    version = match[1]
    if version.split(".", 1)[0] != "2":
        raise OpenCodeError("opencode_unsupported_version", "在线检测需要 OpenCode v2；当前版本为 " + version)
    return version


def request_context(body: dict[str, Any], text: str) -> dict[str, Any]:
    """Only the single user text may vary; all other wire fields are frozen."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise ValueError("OpenCode 请求缺少 messages")
    messages = body["messages"]
    if any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"} for m in messages):
        raise ValueError("OpenCode 请求包含非法角色或工具历史")
    users = [m for m in messages if m.get("role") == "user"]
    if len(users) != 1 or users[0].get("content") != text:
        raise ValueError("OpenCode 改写了探针文本或加入了历史消息")
    context = deepcopy(body)
    next(m for m in context["messages"] if m["role"] == "user")["content"] = {"detector_probe": True}
    return context


def context_hash(context: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _run_process(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Kill the complete client process group on timeout or interruption."""
    payload, timeout = kwargs.pop("input", None), kwargs.pop("timeout", 150)
    kwargs.pop("capture_output", None)
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True, **kwargs)
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


class OpenCodeBackend:
    def __init__(self, store: RunStore, api_key: str, max_requests: int, *,
                 model: str, output_limit: int = 32, client: Any = None,
                 min_interval: float = 3.1, sleep: Callable | None = None,
                 process_runner: Callable = _run_process, binary: str | None = None) -> None:
        self.store, self.api_key, self.model = store, api_key, model
        self.output_limit, self.process_runner = output_limit, process_runner
        self.binary = binary or shutil.which("opencode")
        if not self.binary:
            raise OpenCodeError("opencode_unavailable", "找不到已安装的 opencode 客户端")
        try:
            version = process_runner([self.binary, "--version"], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OpenCodeError("opencode_unavailable", "无法读取 opencode 版本") from exc
        if version.returncode:
            raise OpenCodeError("opencode_unavailable", "opencode --version 未返回有效版本")
        self.client_version = _client_version(version.stdout)
        previous = next((e for e in store.events if e["type"] == "backend_initialized"), None)
        if previous:
            if previous["bridge_version"] != BRIDGE_VERSION or previous["client_version"] != self.client_version:
                raise OpenCodeError("opencode_version_changed", "OpenCode 或桥接版本变化，请创建新实验")
            self.runtime = Path(previous["runtime_dir"])
            if not (self.runtime / "work").is_dir():
                raise OpenCodeError("opencode_runtime_missing", "原 OpenCode 工作目录缺失，请创建新实验")
        else:
            self.runtime = Path(tempfile.mkdtemp(prefix="tokenizer-detector-opencode-"))
            (self.runtime / "work").mkdir()
            store.append("backend_initialized", backend="opencode", bridge_version=BRIDGE_VERSION,
                         client_version=self.client_version, runtime_dir=str(self.runtime))
        frozen = next((e for e in store.events if e["type"] == "request_context_frozen"), None)
        self.context_sha256 = frozen["context_sha256"] if frozen else None
        options = {"client": client, "min_interval": min_interval, "backend": "opencode"}
        if sleep is not None:
            options["sleep"] = sleep
        self.transport = BudgetedTransport(store, api_key, max_requests, **options)
        self._active: dict[str, Any] | None = None
        self._received = False
        self._result: AttemptResult | None = None
        self._failure: BaseException | None = None
        self._gate = threading.Lock()
        self._token = uuid.uuid4().hex
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                try:
                    if self.path != f"/{owner._token}/v1/chat/completions":
                        raise ValueError("Unexpected bridge path")
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 1024 * 1024:
                        raise ValueError("Invalid bridge request size")
                    body = json.loads(self.rfile.read(size).decode("utf-8"))
                    owner.receive(body, dict(self.headers))
                except BaseException as exc:
                    owner._failure = exc
                # The client receives no provider tool calls or error retries.
                # Only the recorded upstream result is evidence for the detector.
                data = owner.acknowledgment()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as exc:
            self.transport.close()
            raise OpenCodeError("opencode_unavailable", "无法启动本地 OpenCode 桥接") from exc
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    def environment(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPENCODE_")}
        for key, directory in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                               ("XDG_CACHE_HOME", "cache"), ("XDG_STATE_HOME", "state")):
            env[key] = str(self.runtime / directory)
        for directory in ("home", "tmp"):
            (self.runtime / directory).mkdir(exist_ok=True)
        # v2 run resolves its project from PWD before consulting process.cwd().
        env["PWD"] = str(self.runtime / "work")
        env["OPENCODE_TEST_HOME"] = str(self.runtime / "home")
        env["TMPDIR"] = str(self.runtime / "tmp")
        for key in ("OPENCODE_CONFIG_PROJECT_DISABLE", "OPENCODE_DISABLE_MODELS_FETCH",
                    "OPENCODE_FILEWATCHER_DISABLE"):
            env[key] = "1"
        env["NO_PROXY"] = ",".join(filter(None, (env.get("NO_PROXY", ""), "127.0.0.1,localhost,::1")))
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps({
            "update": "disable", "snapshots": False, "share": "disabled",
            "plugins": [], "compaction": {"auto": False}, "warming": False,
            "policies": [{"action": "provider.use", "resource": "*", "effect": "deny"},
                         {"action": "provider.use", "resource": "opencode", "effect": "allow"}],
            "permissions": [{"action": "*", "resource": "*", "effect": "ask"}],
            "agents": {"build": {"steps": 1}, "title": {"disabled": True},
                       "summary": {"disabled": True}, "compaction": {"disabled": True}},
            "providers": {"opencode": {"package": "@opencode/ai/providers/openai-compatible",
                "compatibility": {"maxTokensField": "max_tokens"}, "settings": {
                "baseURL": f"http://127.0.0.1:{self._server.server_port}/{self._token}/v1",
                "apiKey": self.api_key or "public"}, "models": {self.model: {
                    "modelID": self.model, "name": self.model, "capabilities": {"tools": True},
                    "limit": {"context": 131072, "output": self.output_limit}}}}},
        }, ensure_ascii=False)
        return env

    def receive(self, body: dict[str, Any], headers: dict[str, str]) -> None:
        """Called by the loopback server, or directly by an offline client double."""
        with self._gate:
            if self._active is None or self._received:
                self.store.append("client_request_blocked", reason="unexpected_client_request")
                self._failure = OpenCodeError("unexpected_client_request", "OpenCode 发起了额外请求，已拦截")
                return
            self._received = True
            try:
                context = request_context(body, self._active["text"])
                if _scrub(body, self.api_key) != body:
                    raise ValueError("探针或上下文包含 API Key，不能保存可回放证据")
                if body.get("model") != self.model or body.get("stream") is not True:
                    raise ValueError("OpenCode 模型或流式协议不符合检测配置")
                if (type(body.get("max_tokens")) is not int or body["max_tokens"] != self.output_limit
                        or "max_completion_tokens" in body):
                    raise ValueError("OpenCode 输出上限不符合检测配置")
                fingerprint = context_hash(context)
                if self.context_sha256 is not None and self.context_sha256 != fingerprint:
                    raise ValueError("OpenCode 固定请求上下文发生变化")
                if self.context_sha256 is None:
                    # Scrubbing must not silently change evidence or its hash.
                    self.store.append("request_context_frozen", context=context, context_sha256=fingerprint)
                    self.context_sha256 = fingerprint
            except (ValueError, TypeError) as exc:
                self.store.append("client_request_blocked", reason="opencode_context_changed", detail=str(exc))
                self._failure = OpenCodeError("opencode_context_changed", str(exc))
                return
            try:
                self._result = self.transport.send("POST", "/chat/completions", json=body,
                    headers=headers, context_sha256=fingerprint,
                    purpose=self._active["purpose"], probe_id=self._active["probe_id"],
                    logical_id=self._active["logical_id"])
            except BaseException as exc:
                self._failure = exc

    def acknowledgment(self) -> bytes:
        common = {"id": "detector-ack", "object": "chat.completion.chunk", "model": self.model}
        chunks = [{**common, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"},
                                         "finish_reason": None}]},
                  {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}]
        return ("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n").encode("utf-8")

    def send(self, method: str, path: str, *, json: dict[str, Any], purpose: str = "",
             probe_id: str | None = None, logical_id: str | None = None, **kwargs: Any) -> AttemptResult:
        if method != "POST" or path != "/chat/completions" or purpose in {"output", "output_confirmation"}:
            raise ValueError("OpenCode 后端仅支持输入计数探针")
        self.output_limit = json["max_tokens"]
        self._active = {"text": json["messages"][0]["content"], "purpose": purpose,
                        "probe_id": probe_id, "logical_id": logical_id}
        self._received, self._result, self._failure = False, None, None
        try:
            process = self.process_runner([self.binary, "run", "--standalone", "--format", "json",
                "--title", "TokenizerDetector probe",
                "--agent", "build", "-m", "opencode/" + self.model],
                input=self._active["text"].encode("utf-8"), env=self.environment(),
                cwd=self.runtime / "work", capture_output=True, timeout=150)
            if self._failure is not None:
                raise self._failure
            if process.returncode or self._result is None:
                stderr = process.stderr.decode("utf-8", errors="replace").strip()
                # v2 can report structured errors on stdout with empty stderr.
                detail = _scrub((stderr or process.stdout.decode("utf-8", errors="replace"))[-2000:], self.api_key)
                raise OpenCodeError("opencode_client_failed", "OpenCode 客户端未完成探针：" + detail)
            return self._result
        except subprocess.TimeoutExpired as exc:
            raise OpenCodeError("opencode_timeout", "OpenCode 客户端超时，已停止") from exc
        except OSError as exc:
            raise OpenCodeError("opencode_client_failed", "无法执行 OpenCode 客户端") from exc
        finally:
            self._active = None

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join()
        # Wait for handlers before the journal can be closed by the runner.
        self._server.server_close()
        self.transport.close()

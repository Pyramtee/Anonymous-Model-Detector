"""Serial experiment orchestration. No inference calls occur on import."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from . import engine
from .models import ALGORITHM_VERSION, SCHEMA_VERSION, Action, Observation, ProbeBank
from .run_store import RunStore, RunStoreError, read_events
from .report import write_reports
from .transport import (
    BudgetExceeded, BudgetedTransport, classify_retry, parse_chat_result,
    retry_after_seconds,
)


@dataclass
class RunConfig:
    model: str
    mode: str = "quick"
    provider: str | None = None
    max_requests: int | None = None
    output_limit: int = 32
    min_interval: float = 3.1

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model 不能为空")
        if self.mode not in {"quick", "deep"}:
            raise ValueError("mode 必须为 quick 或 deep")
        if self.max_requests is None:
            self.max_requests = 12 if self.mode == "quick" else 40
        if self.max_requests < 1:
            raise ValueError("max_requests 必须为正整数")


def _group(config: RunConfig, parsed: Any, output: bool) -> str:
    values = [config.model, config.provider, parsed.model, parsed.provider,
              config.output_limit if not output else 128, output]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()[:20]


def restore(events: list[dict[str, Any]]) -> tuple[Any, ProbeBank, RunConfig, str | None]:
    try:
        return _restore_checked(events)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RunStoreError(f"实验日志语义损坏，拒绝回放或续跑：{exc}") from exc


def _restore_checked(events: list[dict[str, Any]]) -> tuple[Any, ProbeBank, RunConfig, str | None]:
    if not events or events[0]["type"] != "run_started":
        raise RunStoreError("缺少 run_started，不能回放旧版报告")
    first = events[0]
    if first.get("schema_version") != SCHEMA_VERSION:
        raise RunStoreError("只支持 schema_version=2 的完整实验记录")
    if first.get("algorithm_version") != ALGORITHM_VERSION:
        raise RunStoreError("算法版本不兼容；请使用原版本回放或创建新实验")
    bank = ProbeBank.from_dict(first["bank"])
    config = RunConfig(**first["config"])
    state = engine.new_state(bank, mode=config.mode)
    stop = None
    starts, finishes = {}, {}
    known = {"attempt_started", "attempt_finished", "observation", "confirmation_frozen",
             "run_continued", "request_config_changed", "run_stopped", "action_decided",
             "capability_unavailable", "invalid_observation"}
    for event in events[1:]:
        kind = event["type"]
        if kind not in known:
            raise RunStoreError(f"未知实验事件：{kind}")
        if kind == "attempt_started":
            starts[event["attempt_id"]] = event
        elif kind == "attempt_finished":
            finishes[event["attempt_id"]] = event
        if kind == "observation":
            data = dict(event["observation"])
            aid = data["attempt_id"]
            if aid not in starts or aid not in finishes:
                raise RunStoreError("观测缺少已完成的实际请求")
            start, finish = starts[aid], finishes[aid]
            if (start.get("probe_id") != data["probe_id"] or start.get("purpose") != data["phase"]
                    or finish.get("status_code") != 200 or finish.get("error") or finish.get("uncertain")):
                raise RunStoreError("观测与请求结果/探针/阶段不一致")
            request = start["request"]
            if (request["method"] != "POST" or request["json"]["messages"] != [
                    {"role": "user", "content": bank.probes[data["probe_id"]].text}]):
                raise RunStoreError("观测与实际发送文本不一致")
            if type(data["valid"]) is not bool or not isinstance(data["group_id"], str):
                raise RunStoreError("观测的有效性/分组字段无效")
            if data.get("allowed_profiles") is not None:
                if (data["phase"] not in {"output", "output_confirmation"}
                        or not isinstance(data["allowed_profiles"], list)
                        or not set(data["allowed_profiles"]).issubset(bank.profiles)):
                    raise RunStoreError("输出词表证据包含非法假设")
                data["allowed_profiles"] = tuple(data["allowed_profiles"])
            state = engine.observe(state, Observation(**data))
        elif kind == "confirmation_frozen":
            state = engine.freeze_confirmation(state, bank)
        elif kind == "run_continued":
            config = RunConfig(**event["config"])
            state = engine.continue_state(state, mode=config.mode)
            stop = None
        elif kind == "request_config_changed":
            config = RunConfig(**event["config"])
        elif kind == "run_stopped":
            stop = event["stop_reason"]
    return state, bank, config, stop


def _report(state: Any, bank: ProbeBank, config: RunConfig,
            events: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    result = engine.conclude(state, stop_reason=reason).to_dict()
    started = [e for e in events if e["type"] == "attempt_started"]
    finished = {e["attempt_id"]: e for e in events if e["type"] == "attempt_finished"}
    result.update({
        "schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "experiment_id": events[0].get("experiment_id"),
        "model": config.model, "mode": config.mode, "reference_identity": bank.identity,
        "evidence": {"journal": "events.jsonl", "prediction_matrix_event": 1},
        "routes": [dict(provider=provider, model=model) for provider, model in
                   sorted({(e["observation"].get("provider"), e["observation"].get("actual_model"))
                           for e in events if e["type"] == "observation"}, key=str)],
        "budget": {"used": len(started), "limit": config.max_requests,
                   "outcome_unknown": [e["attempt_id"] for e in started
                                       if e["attempt_id"] not in finished or
                                       finished[e["attempt_id"]].get("uncertain")]},
        "channels": {"input_count": "observed" if any(
            e["type"] == "observation" and e["observation"].get("valid")
            and e["observation"].get("phase") in {"anchor", "selection", "confirmation"}
            for e in events) else "unavailable",
            "output_bytes": "observed" if any(e["type"] == "observation" and
                e["observation"].get("allowed_profiles") is not None for e in events)
                else "not_observed"},
        "unavailable_channels": [dict(capability=e["capability"], attempt_id=e.get("attempt_id"))
                                 for e in events if e["type"] == "capability_unavailable"],
        "errors": [{k: e.get(k) for k in ("attempt_id", "status_code", "error", "uncertain")}
                   for e in finished.values() if e.get("error") or
                   (e.get("status_code") or 0) >= 400],
        "validation_scope": "结论限于已记录的观测与编码假设，不提供线上准确率或厂商概率",
    })
    return result


def replay_run(path: Path) -> dict[str, Any]:
    events = read_events(path)
    state, bank, config, stop = restore(events)
    return _report(state, bank, config, events, stop or "interrupted")


def _body(config: RunConfig, text: str, output: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {"model": config.model,
        "messages": [{"role": "user", "content": text}], "stream": False,
        "max_tokens": 128 if output else config.output_limit,
        "provider": {"allow_fallbacks": False}}
    if config.provider:
        body["provider"]["only"] = [config.provider]
    if output:
        body["logprobs"] = True
        body["provider"]["require_parameters"] = True
    return body


def execute_run(
    path: Path, bank: ProbeBank, config: RunConfig, api_key: str,
    *, resume: bool = False, adapters: dict[str, Any] | None = None,
    client: Any = None, sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute via an injected client in tests, or explicit live CLI invocation only."""
    log = progress or (lambda message: None)
    with RunStore(path, create=not resume) as store:
        if resume:
            state, old_bank, old_config, _ = restore(store.events)
            if old_bank.identity != bank.identity:
                raise RunStoreError("参考/探针/运行版本发生变化，不能原地续跑")
            if old_config.model != config.model or old_config.provider != config.provider:
                raise RunStoreError("模型或 provider 设置变化，请创建新实验")
            if config.max_requests <= store.attempts_used:
                raise RunStoreError("新累计预算必须大于已使用尝试数")
            config.output_limit = old_config.output_limit
            store.append("run_continued", config=asdict(config))
            state = engine.continue_state(state, mode=config.mode)
        else:
            store.append("run_started", schema_version=SCHEMA_VERSION,
                         algorithm_version=ALGORITHM_VERSION, config=asdict(config),
                         experiment_id=path.name,
                         bank=bank.to_dict())
            state = engine.new_state(bank, mode=config.mode)
        transport = BudgetedTransport(store, api_key, config.max_requests,
                                      client=client, min_interval=config.min_interval,
                                      sleep=sleep)
        reason = "insufficient_evidence"
        output_attempts = sum(e["type"] == "attempt_started" and
                              e.get("purpose") in {"output", "output_confirmation"}
                              for e in store.events)
        output_started = output_attempts > 0
        try:
            while True:
                remaining = config.max_requests - store.attempts_used
                action = engine.next_action(state, bank, remaining)
                store.append("action_decided", action=asdict(action), remaining=remaining)
                if action.kind == "freeze":
                    store.append("confirmation_frozen")
                    state = engine.freeze_confirmation(state, bank)
                    continue
                if action.kind == "stop":
                    current = engine.conclude(state, stop_reason=action.reason)
                    if (config.mode == "deep" and current.status == "ambiguous"
                            and adapters and not output_started and remaining >= 3):
                        outputs = sorted((p for p in bank.probes.values() if p.split == "output"),
                                         key=lambda p: p.probe_id)
                        if len(outputs) >= 3:
                            action = Action("send", (outputs[0].probe_id,), "output")
                            store.append("action_decided", action=asdict(action), remaining=remaining,
                                         reason="input_channel_exhausted")
                            output_started = True
                        else:
                            reason = "output_confirmation_unavailable"
                            break
                    else:
                        reason = action.reason or "completed"
                        break
                if not action.probe_ids:
                    raise RuntimeError("发送动作缺少 probe ID")
                stopped = False
                for probe_id in action.probe_ids:
                    probe = bank.probes[probe_id]
                    output = action.phase in {"output", "output_confirmation"}
                    if output:
                        if output_attempts >= 4:
                            reason, stopped = "output_budget_exhausted", True
                            break
                        output_attempts += 1
                    log(f"{action.phase}: {probe_id} ({store.attempts_used}/{config.max_requests})")
                    body = _body(config, probe.text, output)
                    result = transport.send("POST", "/chat/completions", json=body,
                        purpose=action.phase, probe_id=probe_id,
                        logical_id=f"{probe_id}:{store.attempts_used}")
                    if (classify_retry(result) and config.max_requests - store.attempts_used > 3
                            and (not output or output_attempts < 4)):
                        delay = retry_after_seconds(result.headers)
                        if delay is not None and delay > 30:
                            reason, stopped = "retry_after_exceeds_wait_limit", True
                            break
                        sleep(2 if delay is None else max(0, delay))
                        if output:
                            output_attempts += 1
                        result = transport.send("POST", "/chat/completions", json=body,
                            purpose=action.phase, probe_id=probe_id,
                            logical_id=f"retry:{result.attempt_id}")
                    # A declared minimum may be honored once, with a finite upper bound.
                    if not output and result.status_code in {400, 422}:
                        error_text = json.dumps(result.data, ensure_ascii=False).lower()
                        minimum = re.search(r"(?:minimum|min value|at least|>=)\D{0,15}(\d+)", error_text)
                        if ("max_tokens" in error_text and minimum and
                                config.output_limit < int(minimum.group(1)) <= 128 and
                                config.max_requests - store.attempts_used > 3):
                            config.output_limit = int(minimum.group(1))
                            store.append("request_config_changed", config=asdict(config),
                                         reason="declared_output_minimum")
                            body = _body(config, probe.text)
                            result = transport.send("POST", "/chat/completions", json=body,
                                purpose=action.phase, probe_id=probe_id,
                                logical_id=f"output_limit:{result.attempt_id}")
                    if result.error or result.status_code is None or result.status_code >= 400:
                        if output and result.status_code in {400, 404, 422}:
                            store.append("capability_unavailable", capability="output_bytes",
                                         attempt_id=result.attempt_id)
                            reason, stopped = "output_capability_unavailable", True
                            break
                        reason = "outcome_unknown" if result.uncertain else "request_failed"
                        stopped = True
                        break
                    try:
                        parsed = parse_chat_result(result.data)
                    except (ValueError, TypeError) as exc:
                        reason, stopped = "invalid_response", True
                        store.append("invalid_observation", attempt_id=result.attempt_id,
                                     reason=str(exc))
                        break
                    # A native-only generation lookup is a budgeted diagnostic, never an estimate.
                    cache_hit = any(k.lower().startswith("x-openrouter-cache") and str(v).lower() == "hit"
                                    for k, v in result.headers.items())
                    usage_source = "response_usage"
                    if (not output and not cache_hit and (parsed.prompt_tokens is None or parsed.prompt_tokens <= 0)
                            and parsed.generation_id):
                        if config.max_requests - store.attempts_used > 2:
                            diag = transport.send("GET", "/generation",
                                params={"id": parsed.generation_id}, purpose="usage_diagnostic")
                            detail = diag.data.get("data") if isinstance(diag.data, dict) else None
                            native = detail.get("native_tokens_prompt") if isinstance(detail, dict) else None
                            if (diag.status_code == 200 and not diag.error and
                                    type(native) is int and native > 0):
                                parsed.prompt_tokens = native
                                usage_source = "generation_native"
                    valid = not cache_hit and type(parsed.prompt_tokens) is int and parsed.prompt_tokens > 0
                    allowed = None
                    if output:
                        previous_routes = {(e["observation"].get("provider"), e["observation"].get("actual_model"))
                                           for e in store.events if e["type"] == "observation" and
                                           e["observation"]["phase"] in {"anchor", "selection", "confirmation"}}
                        if (parsed.provider, parsed.model) not in previous_routes:
                            store.append("capability_unavailable", capability="output_bytes",
                                         attempt_id=result.attempt_id, reason="output_route_changed")
                            reason, stopped = "output_route_changed", True
                            break
                        if cache_hit or not parsed.token_bytes_valid or not parsed.token_bytes:
                            store.append("capability_unavailable", capability="output_bytes",
                                         attempt_id=result.attempt_id,
                                         reason="response_cache_hit" if cache_hit else "unreliable_native_bytes")
                            reason, stopped = "output_capability_unavailable", True
                            break
                        accepted = []
                        for profile_id, profile in bank.profiles.items():
                            vocab = adapters[profile.reference_id].vocabulary_bytes()
                            if vocab is None or all(b in vocab for b in parsed.token_bytes):
                                accepted.append(profile_id)
                        allowed = tuple(accepted)
                        valid = not cache_hit
                    observation = Observation(
                        attempt_id=result.attempt_id, probe_id=probe_id,
                        group_id=_group(config, parsed, output), phase=action.phase,
                        prompt_tokens=parsed.prompt_tokens, provider=parsed.provider,
                        actual_model=parsed.model, valid=valid,
                        reason=None if valid else "missing_native_usage",
                        allowed_profiles=allowed,
                    )
                    store.append("observation", observation=asdict(observation), usage_source=usage_source)
                    state = engine.observe(state, observation)
                    if not valid:
                        reason, stopped = "missing_native_usage", True
                        break
                if stopped:
                    break
        except BudgetExceeded:
            reason = "budget_exhausted"
        except KeyboardInterrupt:
            reason = "interrupted"
        finally:
            transport.close()
        store.append("run_stopped", stop_reason=reason)
        result = _report(state, bank, config, store.events, reason)
        # Keep derived reports under the same lock as their journal snapshot.
        write_reports(path, result)
        return result

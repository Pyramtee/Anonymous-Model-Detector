"""One conclusion rendered as Chinese Markdown, terminal text and JSON."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from rich.console import Console

LABELS = {
    "compatible": "与单一参考家族兼容", "ambiguous": "多个参考家族仍兼容",
    "no_reference_match": "当前编码假设下未匹配参考库",
    "insufficient_evidence": "证据不足", "unstable_endpoint": "端点观测不稳定",
    "reference_unavailable": "参考库不完整", "request_failed": "请求失败",
}

def render_markdown(payload: dict[str, Any]) -> str:
    status = payload.get("status", "insufficient_evidence")
    budget = payload.get("budget", {})
    lines = ["# Tokenizer 检测报告", "", f"**结果：{LABELS.get(status, status)}**", "",
             f"模型：`{payload.get('model', '—')}`", "",
             f"停止原因：`{payload.get('stop_reason', '—')}`", "",
             f"HTTP 尝试：{budget.get('used', 0)} / {budget.get('limit', '—')}", "",
             "候选与编码假设：", ""]
    candidates = payload.get("candidates", [])
    if not candidates:
        lines += ["没有已建立兼容关系的候选；这不自动证明存在新的 tokenizer。", ""]
    for candidate in candidates:
        family = candidate.get("family_id", candidate.get("family", "?"))
        lines.append(f"- {family} {candidate.get('vendor', '')}：`{json.dumps(candidate, ensure_ascii=False)}`")
    lines += ["", "确认状态：", "", "```json",
              json.dumps(payload.get("confirmation", {}), ensure_ascii=False, indent=2),
              "```", "", "观测通道：", "", "```json",
              json.dumps(payload.get("channels", {}), ensure_ascii=False, indent=2), "```", "",
              "结论限于本次参考集、编码假设与实际观测；不证明模型权重或开发商身份相同。",
              "不输出未校准的归属概率。完整残差、排除证据及版本见 JSON 与实验日志。", ""]
    if budget.get("outcome_unknown"):
        lines += ["存在执行结果不确定的请求，已保守计入预算。", ""]
    return "\n".join(lines)

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)

def write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def write_reports(directory: Path, payload: dict[str, Any]) -> None:
    write_json(directory / "report.json", payload)
    _atomic_write(directory / "report.md", render_markdown(payload))

def print_report(payload: dict[str, Any], console: Console | None = None) -> None:
    (console or Console()).print(render_markdown(payload), markup=False, highlight=False)

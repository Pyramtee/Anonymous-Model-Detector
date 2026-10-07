"""Project-local v2 CLI. Offline commands never construct a live client."""
from __future__ import annotations
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from .dotenv import load_dotenv
from .models import SCHEMA_VERSION
from .references import DEFAULT_ARTIFACT_DIR, DEFAULT_MANIFEST, fetch_references, inspect_references, load_adapters, load_manifest
from .probe_bank import build_probe_bank
from .report import print_report, write_json
from .runner import RunConfig, execute_run, replay_run, restore
from .run_store import read_events

PROJECT_ROOT = Path(__file__).resolve().parents[2]
app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="通过可回放的 tokenizer 指纹证据识别匿名模型（v2）。")
console = Console()

def _fail(exc: Exception) -> None:
    console.print(f"错误：{exc}", markup=False)
    raise typer.Exit(2)

def _references(manifest: Path, artifacts: Path):
    data = load_manifest(manifest)
    adapters = load_adapters(data, artifacts)
    bank = build_probe_bank(data, adapters, cache_dir=PROJECT_ROOT / "data/cache/matrices")
    return adapters, bank

@app.command()
def fetch(
    family: Optional[str] = typer.Option(None, "--family", help="逗号分隔的厂商 ID"),
    include_controls: bool = typer.Option(False, "--include-controls", help="同时准备离线外部对照"),
    manifest: Path = typer.Option(DEFAULT_MANIFEST, "--manifest"),
    artifacts: Path = typer.Option(DEFAULT_ARTIFACT_DIR, "--artifacts"),
) -> None:
    """只获取清单中已锁定的公开 tokenizer 文件，不下载权重。"""
    try:
        data = load_manifest(manifest)
        families = [x.strip() for x in family.split(",") if x.strip()] if family else None
        ready, errors = fetch_references(data, artifacts, families=families, include_controls=include_controls)
        for item in ready:
            console.print(f"已准备：{item}", markup=False)
        for error in errors:
            console.print(f"失败：{error}", markup=False)
        if errors:
            raise typer.Exit(1)
    except (ValueError, RuntimeError, OSError) as exc:
        _fail(exc)

@app.command("list")
def list_references(
    manifest: Path = typer.Option(DEFAULT_MANIFEST, "--manifest"),
    artifacts: Path = typer.Option(DEFAULT_ARTIFACT_DIR, "--artifacts"),
) -> None:
    """离线检查参考来源、版本、完整性与编码策略。"""
    try:
        rows = inspect_references(load_manifest(manifest), artifacts)
        table = Table(title="固定 tokenizer 参考")
        for name in ["参考", "角色", "完整性", "后端", "策略", "固定版本"]:
            table.add_column(name)
        for row in rows:
            table.add_row(row["display"], row["role"], row["status"], row["backend"],
                          ", ".join(row["policies"]), row["revision"][:12])
        console.print(table)
        for row in rows:
            for artifact in row["artifacts"]:
                if artifact["status"] != "ready":
                    console.print(f"{row['reference_id']}/{artifact['repo_path']}: {artifact['status']}", markup=False)
    except (ValueError, RuntimeError, OSError) as exc:
        _fail(exc)

@app.command()
def fingerprint(
    manifest: Path = typer.Option(DEFAULT_MANIFEST, "--manifest"),
    artifacts: Path = typer.Option(DEFAULT_ARTIFACT_DIR, "--artifacts"),
    output: Optional[Path] = typer.Option(None, "-o", "--output"),
) -> None:
    """离线构建完整文本计数矩阵、候选对见证与不可分组。"""
    try:
        _, bank = _references(manifest, artifacts)
        console.print(json.dumps(bank.diagnostics, ensure_ascii=False, indent=2), markup=False)
        if output:
            write_json(output, bank.to_dict())
    except (ValueError, RuntimeError, OSError) as exc:
        _fail(exc)

def _key(value: str | None) -> str:
    load_dotenv()
    key = value or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("需要 OPENROUTER_API_KEY；fingerprint 和 replay 不需要凭证")
    return key

def _finish(path: Path, result: dict, output: Path | None) -> None:
    if output:
        write_json(output, result)
    print_report(result, console)
    console.print(f"实验记录：{path}", markup=False)

@app.command()
def detect(
    model: str = typer.Option(..., "--model"),
    mode: str = typer.Option("quick", "--mode", help="quick 或 deep"),
    provider: Optional[str] = typer.Option(None, "--provider", help="精确 endpoint slug"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    max_requests: Optional[int] = typer.Option(None, "--max-requests", min=1),
    run_dir: Optional[Path] = typer.Option(None, "--run-dir"),
    manifest: Path = typer.Option(DEFAULT_MANIFEST, "--manifest"),
    artifacts: Path = typer.Option(DEFAULT_ARTIFACT_DIR, "--artifacts"),
    output: Optional[Path] = typer.Option(None, "-o", "--output"),
    max_calls: Optional[int] = typer.Option(None, "--max-calls", hidden=True),
    unknown_threshold: Optional[float] = typer.Option(None, "--unknown-threshold", hidden=True),
    no_completion: bool = typer.Option(False, "--no-completion", hidden=True),
) -> None:
    """显式发起在线实验；默认快检最多 12 次 HTTP 尝试。"""
    if max_calls is not None or unknown_threshold is not None or no_completion:
        _fail(ValueError("旧评分参数已取消；--max-calls 请改为 --max-requests（包含重试与诊断）。"))
    try:
        config = RunConfig(model=model, mode=mode, provider=provider, max_requests=max_requests)
        adapters, bank = _references(manifest, artifacts)
        key = _key(api_key)
        directory = run_dir or PROJECT_ROOT / "data/runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        result = execute_run(directory, bank, config, key, adapters=adapters,
                             progress=lambda x: console.print(x, markup=False))
        _finish(directory, result, output)
    except (ValueError, RuntimeError, OSError) as exc:
        if output:
            write_json(output, {"schema_version": SCHEMA_VERSION, "status": "reference_unavailable"
                       if "reference" in type(exc).__name__.lower() else "request_failed",
                       "stop_reason": str(exc), "candidates": [], "model": model})
        _fail(exc)

@app.command()
def resume(
    run: Path = typer.Argument(...),
    mode: str = typer.Option("deep", "--mode"),
    max_requests: Optional[int] = typer.Option(None, "--max-requests", min=1),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    manifest: Path = typer.Option(DEFAULT_MANIFEST, "--manifest"),
    artifacts: Path = typer.Option(DEFAULT_ARTIFACT_DIR, "--artifacts"),
    output: Optional[Path] = typer.Option(None, "-o", "--output"),
) -> None:
    """继续同一实验；默认深检将累计预算提高至 40，而非追加 40。"""
    try:
        _, _, old, _ = restore(read_events(run))
        adapters, bank = _references(manifest, artifacts)
        config = RunConfig(model=old.model, provider=old.provider, mode=mode, max_requests=max_requests, output_limit=old.output_limit)
        result = execute_run(run, bank, config, _key(api_key), resume=True,
                             adapters=adapters, progress=lambda x: console.print(x, markup=False))
        _finish(run, result, output)
    except (ValueError, RuntimeError, OSError) as exc:
        _fail(exc)

@app.command()
def replay(
    run: Path = typer.Argument(...),
    output: Optional[Path] = typer.Option(None, "-o", "--output"),
) -> None:
    """仅根据完整事件日志离线重算，不加载凭证或网络客户端。"""
    try:
        payload = replay_run(run)
        if output:
            write_json(output, payload)
        print_report(payload, console)
    except (ValueError, RuntimeError, OSError) as exc:
        _fail(exc)

if __name__ == "__main__":
    app()

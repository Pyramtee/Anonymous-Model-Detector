"""CLI entry: fetch / list / detect."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .client import DEFAULT_BASE_URL, OpenRouterClient
from .detect import DetectConfig, run_detect
from .dotenv import load_dotenv
from .fetch import FetchError, fetch_families
from .registry import DEFAULT_TOKENIZER_DIR, DEFAULT_VENDORS_YAML, load_registry
from .report import print_verdict, verdict_to_dict, write_json

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Detect which known vendor tokenizer an anonymous OpenRouter model uses.",
)
console = Console()

# Load API keys from project .env early
load_dotenv()


def _load(vendors: Path | None, tokenizer_dir: Path | None):
    return load_registry(
        vendors_path=vendors or DEFAULT_VENDORS_YAML,
        tokenizer_dir=tokenizer_dir or DEFAULT_TOKENIZER_DIR,
    )


@app.command()
def fetch(
    family: Optional[str] = typer.Option(
        None, "--family", help="Comma-separated family ids (default: all)"
    ),
    vendors: Optional[Path] = typer.Option(None, "--vendors", help="vendors.yaml path"),
    tokenizer_dir: Optional[Path] = typer.Option(
        None, "--tokenizer-dir", help="Where to store tokenizer.json files"
    ),
    timeout: float = typer.Option(60.0, help="HTTP timeout seconds"),
) -> None:
    """Download tokenizer.json files from HuggingFace for known families."""
    registry = _load(vendors, tokenizer_dir)
    ids = [x.strip() for x in family.split(",") if x.strip()] if family else None
    if ids:
        unknown = [i for i in ids if i not in registry.families]
        if unknown:
            console.print(f"[red]Unknown families:[/red] {', '.join(unknown)}")
            console.print(f"Known: {', '.join(registry.family_ids())}")
            raise typer.Exit(2)

    dest_root = tokenizer_dir or DEFAULT_TOKENIZER_DIR
    console.print(f"Fetching tokenizers into [cyan]{dest_root}[/cyan] ...")
    results, errors = fetch_families(
        registry,
        family_ids=ids,
        tokenizer_dir=dest_root,
        timeout=timeout,
    )

    for fid, path in results.items():
        fam = registry.get(fid)
        console.print(
            f"  [green]✓[/green] {fid:12} {fam.display:10} "
            f"sha256={(fam.tokenizer_sha256 or '')[:12]}…  {path.name}"
        )
    for err in errors:
        console.print(f"  [red]✗[/red] {err}")

    if results and not errors:
        console.print(f"[bold]Done.[/bold] {len(results)} family tokenizer(s) ready.")
    elif results:
        console.print(
            f"[yellow]Partial.[/yellow] {len(results)} ok, {len(errors)} failed."
        )
        raise typer.Exit(1)
    else:
        console.print("[red]No tokenizers downloaded.[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    vendors: Optional[Path] = typer.Option(None, "--vendors"),
    tokenizer_dir: Optional[Path] = typer.Option(None, "--tokenizer-dir"),
) -> None:
    """List known tokenizer families and local availability."""
    registry = _load(vendors, tokenizer_dir)
    table = Table(title="Tokenizer families")
    table.add_column("Family", style="cyan")
    table.add_column("Vendor")
    table.add_column("Display")
    table.add_column("Local", justify="center")
    table.add_column("SHA256")
    table.add_column("HF repos")
    table.add_column("Shared models")

    for fam in registry.families.values():
        table.add_row(
            fam.id,
            fam.vendor,
            fam.display,
            "[green]yes[/green]" if fam.available else "[red]no[/red]",
            (fam.tokenizer_sha256 or "")[:12] or "-",
            ", ".join(fam.hf_repos[:2]),
            ", ".join(fam.models[:3]),
        )
    console.print(table)
    n = len(registry.available_families())
    console.print(f"[dim]{n}/{len(registry.families)} families have local tokenizer files[/dim]")


@app.command()
def detect(
    model: str = typer.Option(..., "--model", help="OpenRouter model id to probe"),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="OpenRouter API key (default: $OPENROUTER_API_KEY)"
    ),
    max_calls: int = typer.Option(12, "--max-calls", min=3, max=30),
    no_completion: bool = typer.Option(
        False, "--no-completion", help="Skip echo/completion probes"
    ),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Write JSON report"),
    vendors: Optional[Path] = typer.Option(None, "--vendors"),
    tokenizer_dir: Optional[Path] = typer.Option(None, "--tokenizer-dir"),
    unknown_threshold: float = typer.Option(0.55, help="Score below this => unknown"),
) -> None:
    """Probe an anonymous OpenRouter model and rank known tokenizer families."""
    registry = _load(vendors, tokenizer_dir)
    if not registry.available_families():
        console.print("[red]No local tokenizers.[/red] Run: tokenizer-detector fetch")
        raise typer.Exit(1)

    config = DetectConfig(
        model=model,
        max_calls=max_calls,
        use_completion=not no_completion,
        unknown_threshold=unknown_threshold,
    )

    def progress(msg: str) -> None:
        console.print(f"[dim]… {msg}[/dim]")

    try:
        with OpenRouterClient(api_key=api_key, base_url=DEFAULT_BASE_URL) as client:
            result = run_detect(registry, client, config, progress=progress)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]detect failed:[/red] {exc}")
        raise typer.Exit(1)

    print_verdict(result.verdict, console=console)
    console.print(f"[dim]API calls used: {result.calls_used}[/dim]")

    payload = verdict_to_dict(
        result.verdict,
        extra={
            "model": model,
            "endpoint": DEFAULT_BASE_URL,
            "calls_used": result.calls_used,
            "probe_set_version": __import__(
                "tokenizer_detector.probes", fromlist=["PROBE_SET_VERSION"]
            ).PROBE_SET_VERSION,
            "raw": result.raw,
        },
    )
    if output:
        write_json(output, payload)
        console.print(f"[green]Wrote[/green] {output}")


if __name__ == "__main__":
    app()

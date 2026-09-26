"""Terminal and JSON reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .fuse import Verdict


def print_verdict(verdict: Verdict, console: Console | None = None) -> None:
    console = console or Console()
    console.print()
    title = f"[bold]Verdict:[/bold] {verdict.verdict}"
    if verdict.verdict not in ("unknown", "ambiguous"):
        title += f"  (confidence {verdict.top_confidence:.2f})"
    console.print(title)
    console.print(f"[dim]{verdict.explanation}[/dim]")
    console.print(f"unknown_score: {verdict.unknown_score:.2f}")
    console.print()

    table = Table(title="Top candidates", show_lines=False)
    table.add_column("Family", style="cyan")
    table.add_column("Vendor")
    table.add_column("Confidence", justify="right")
    table.add_column("Combined", justify="right")
    table.add_column("S1", justify="right")
    table.add_column("S2", justify="right")
    table.add_column("S3", justify="right")
    table.add_column("Shared models")

    for c in verdict.candidates[:8]:
        eq = ", ".join(c.equivalent_families) if c.equivalent_families else ""
        family_label = f"{c.display or c.family_id}"
        if eq:
            family_label += f" ≡ {eq}"
        table.add_row(
            family_label,
            c.vendor or "-",
            f"{c.confidence:.2f}",
            f"{c.combined_score:.2f}",
            f"{c.signals.get('prompt_count', 0):.2f}",
            f"{c.signals.get('completion', 0):.2f}",
            f"{c.signals.get('text_stats', 0):.2f}",
            ", ".join(c.shared_models[:4]) or "-",
        )
    console.print(table)


def verdict_to_dict(verdict: Verdict, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "verdict": verdict.verdict,
        "unknown_score": round(verdict.unknown_score, 4),
        "top_confidence": round(verdict.top_confidence, 4),
        "explanation": verdict.explanation,
        "candidates": [
            {
                "family": c.family_id,
                "display": c.display,
                "vendor": c.vendor,
                "confidence": round(c.confidence, 4),
                "combined_score": round(c.combined_score, 4),
                "signals": {k: round(v, 4) for k, v in c.signals.items()},
                "shared_models": c.shared_models,
                "equivalent_families": c.equivalent_families,
            }
            for c in verdict.candidates
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

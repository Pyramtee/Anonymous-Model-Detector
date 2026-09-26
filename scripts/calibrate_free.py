"""Calibrate on free OpenRouter models only (no balance required)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tokenizer_detector.client import OpenRouterClient
from tokenizer_detector.detect import DetectConfig, run_detect
from tokenizer_detector.dotenv import load_dotenv
from tokenizer_detector.registry import load_registry

# Free models. expected=None means "should be unknown / out of library"
# expected=family means that family should win (if tokenizer still matches).
CASES = [
    ("qwen", "qwen/qwen3.8-27b:free"),
    (None, "google/gemma-4-31b-it:free"),
    (None, "nvidia/nemotron-3-super-120b-a12b:free"),
    (None, "inclusionai/ling-3.0-flash-sante:free"),
    (None, "dots-studio/dots-3-note-preview:free"),
]


def main() -> None:
    load_dotenv()
    reg = load_registry()
    results = []
    # Smaller budget to survive free-tier rate limits
    with OpenRouterClient(max_retries=6) as client:
        for expected, model in CASES:
            print(f"\n===== expected={expected}  model={model} =====", flush=True)
            try:
                r = run_detect(
                    reg,
                    client,
                    DetectConfig(model=model, max_calls=12, use_completion=False),
                    progress=lambda m: print(f"  {m}", flush=True),
                )
                top = r.verdict.candidates[0]
                if expected is None:
                    hit = r.verdict.verdict in ("unknown", "ambiguous")
                else:
                    hit = (
                        expected in (top.family_id, *top.equivalent_families)
                        or expected in r.verdict.verdict
                    )
                print(
                    f"  VERDICT={r.verdict.verdict} top={top.family_id} "
                    f"conf={top.confidence:.2f} hit={hit}"
                )
                for c in r.verdict.candidates[:4]:
                    print(
                        f"    {c.family_id:10} conf={c.confidence:.2f} "
                        f"score={c.combined_score:.2f} s1={c.signals.get('prompt_count', 0):.2f}"
                    )
                results.append(
                    {
                        "expected": expected,
                        "model": model,
                        "verdict": r.verdict.verdict,
                        "top": top.family_id,
                        "confidence": top.confidence,
                        "hit": hit,
                        "calls": r.calls_used,
                        "raw_signals": r.raw.get("signals"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {exc}")
                results.append({"expected": expected, "model": model, "error": str(exc)})
            time.sleep(8)

    out = Path("data/calibration_free.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    n_ok = sum(1 for r in results if r.get("hit"))
    print(f"\n=== {n_ok}/{len(results)} hits; wrote {out} ===")


if __name__ == "__main__":
    main()

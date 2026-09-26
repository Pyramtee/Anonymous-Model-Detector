"""Calibrate detector against known OpenRouter models (one per family)."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizer_detector.client import OpenRouterClient
from tokenizer_detector.detect import DetectConfig, run_detect
from tokenizer_detector.dotenv import load_dotenv
from tokenizer_detector.registry import load_registry

# Models closest to the tokenizers we vendored.
CASES = [
    ("qwen", "qwen/qwen3-8b"),
    ("deepseek", "deepseek/deepseek-chat"),
    ("kimi", "moonshotai/kimi-k2"),
    ("glm", "z-ai/glm-4.5"),
    ("mimo", "xiaomi/mimo-v2.5"),
    ("step", "stepfun/step-3.5-flash"),
    ("minimax", "minimax/minimax-m1"),
]


def main() -> None:
    load_dotenv()
    reg = load_registry()
    results = []
    with OpenRouterClient() as client:
        for expected, model in CASES:
            print(f"\n===== {expected}  <=  {model} =====", flush=True)
            try:
                r = run_detect(
                    reg,
                    client,
                    DetectConfig(model=model, max_calls=16, use_completion=True),
                    progress=lambda m: print(f"  {m}", flush=True),
                )
                top = r.verdict.candidates[0]
                hit = expected in (top.family_id, *top.equivalent_families) or expected in r.verdict.verdict
                print(f"  VERDICT={r.verdict.verdict}  top={top.family_id} conf={top.confidence:.2f} hit={hit}")
                for c in r.verdict.candidates[:4]:
                    print(f"    {c.family_id:10} conf={c.confidence:.2f} score={c.combined_score:.2f} s1={c.signals.get('prompt_count',0):.2f}")
                results.append(
                    {
                        "expected": expected,
                        "model": model,
                        "verdict": r.verdict.verdict,
                        "top": top.family_id,
                        "confidence": top.confidence,
                        "hit": hit,
                        "calls": r.calls_used,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {exc}")
                results.append({"expected": expected, "model": model, "error": str(exc)})

    out = Path("data/calibration_real.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    n_ok = sum(1 for r in results if r.get("hit"))
    print(f"\n=== {n_ok}/{len(results)} hits; wrote {out} ===")


if __name__ == "__main__":
    main()

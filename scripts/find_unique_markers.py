"""Find exact unique special-token probes per family (dev helper)."""
from __future__ import annotations

import json
from pathlib import Path

from tokenizer_detector.registry import DEFAULT_TOKENIZER_DIR, load_registry
from tokenizer_detector.tokenizer_backends import load_backend


def main() -> None:
    reg = load_registry()
    bs = {f.id: load_backend(f.id, DEFAULT_TOKENIZER_DIR) for f in reg.available_families()}
    ids = list(bs)

    markers: set[str] = set()
    for fid in ids:
        p = DEFAULT_TOKENIZER_DIR / fid / "tokenizer.json"
        if p.exists():
            data = json.loads(p.read_text())
            for t in data.get("added_tokens", []):
                markers.add(t["content"])

    # Print markers that encode as 1 for exactly one family
    print("=== unique single-token markers ===")
    for m in sorted(markers):
        vec = {fid: bs[fid].encode_len(m) for fid in ids}
        ones = [fid for fid, v in vec.items() if v == 1]
        if len(ones) == 1:
            cps = " ".join(f"U+{ord(c):04X}" for c in m[:12])
            print(f"{ones[0]:10} {m!r:40} {cps}")

    print("\n=== pairwise still-difficult? ===")
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            diffs = [
                m
                for m in markers
                if bs[a].encode_len(m) != bs[b].encode_len(m)
            ]
            if not diffs:
                print(f"NO marker difference: {a} vs {b}")
            else:
                print(f"{a} vs {b}: {len(diffs)} differing markers")


if __name__ == "__main__":
    main()

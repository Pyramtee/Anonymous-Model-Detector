"""Quick diagnostic: raw usage counts for one model."""

from __future__ import annotations

import os

from tokenizer_detector.client import OpenRouterClient
from tokenizer_detector.dotenv import load_dotenv
from tokenizer_detector.probes import LONG_PROBES, SPECIAL_PROBES
from tokenizer_detector.registry import DEFAULT_TOKENIZER_DIR, load_registry
from tokenizer_detector.tokenizer_backends import load_backend


def main() -> None:
    load_dotenv()
    model = os.environ.get("MODEL", "qwen/qwen3.8-27b:free")
    reg = load_registry()
    bs = {
        f.id: load_backend(f.id, DEFAULT_TOKENIZER_DIR)
        for f in reg.available_families()
    }
    probes = LONG_PROBES + SPECIAL_PROBES[:8]
    with OpenRouterClient() as client:
        for p in probes:
            r = client.chat_with_nonce(model, p.text, max_tokens=4, temperature=0)
            api = r.usage.prompt_tokens
            locals_ = {fid: b.encode_len(p.text) for fid, b in bs.items()}
            residuals = {fid: api - n for fid, n in locals_.items()} if api else {}
            best = sorted(residuals.items(), key=lambda kv: abs(kv[1] - 15))[:3]
            print(f"{p.id:22} api={api} best={best}")


if __name__ == "__main__":
    main()

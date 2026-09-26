"""Tokenizer loading backends: HF tokenizers.json and tiktoken.model."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TokenizerBackend(Protocol):
    def encode_len(self, text: str) -> int: ...


class HFTokenizerBackend:
    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(path))

    def encode_len(self, text: str) -> int:
        return len(self._tok.encode(text).ids)


class TiktokenBackend:
    """Count tokens using a tiktoken.model BPE file (Kimi / OpenAI style)."""

    def __init__(self, path: Path) -> None:
        import tiktoken

        self._enc = tiktoken.Encoding(
            name=path.parent.name,
            pat_str=(
                r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            ),
            mergeable_ranks=_load_tiktoken_ranks(path),
            special_tokens={},
        )

    def encode_len(self, text: str) -> int:
        return len(self._enc.encode(text, allowed_special=set()))


def _load_tiktoken_ranks(path: Path) -> dict[bytes, int]:
    """Load mergeable ranks from tiktoken.model.

    Supports two layouts:
    1. Text lines: ``<base64-token> <rank>`` (Kimi style)
    2. Binary: repeated (uint32 len | token bytes | uint32 rank)
    """
    import base64
    import struct

    data = path.read_bytes()

    # Text form: first non-empty line looks like "IQ== 0"
    head = data[:256]
    first_line = head.split(b"\n", 1)[0]
    if b" " in first_line and not first_line.startswith(b"\x00"):
        try:
            left, right = first_line.rsplit(b" ", 1)
            int(right.strip())
            base64.b64decode(left.strip(), validate=True)
            ranks: dict[bytes, int] = {}
            for line in data.decode("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                left, right = line.rsplit(" ", 1)
                token = base64.b64decode(left.strip())
                ranks[token] = int(right.strip())
            return ranks
        except Exception:
            pass  # fall through to binary

    # Binary form
    offset = 0
    nl = data.find(b"\n")
    if 0 < nl < 200 and all(32 <= b < 127 for b in data[:nl]):
        offset = nl + 1

    ranks = {}
    i = offset
    n = len(data)
    while i + 8 <= n:
        (tlen,) = struct.unpack_from("<I", data, i)
        i += 4
        if tlen > 4096 or i + tlen + 4 > n:
            raise ValueError(f"invalid tiktoken.model at offset {i}")
        token = data[i : i + tlen]
        i += tlen
        (rank,) = struct.unpack_from("<I", data, i)
        i += 4
        ranks[token] = rank
    return ranks


def load_backend(family_id: str, tokenizer_dir: Path) -> TokenizerBackend:
    """Load the appropriate backend for a family directory."""
    json_path = tokenizer_dir / family_id / "tokenizer.json"
    if json_path.exists():
        return HFTokenizerBackend(json_path)

    bpe_path = tokenizer_dir / family_id / "tiktoken.model"
    if bpe_path.exists():
        return TiktokenBackend(bpe_path)

    raise FileNotFoundError(
        f"No tokenizer.json or tiktoken.model for family '{family_id}' in {tokenizer_dir}"
    )

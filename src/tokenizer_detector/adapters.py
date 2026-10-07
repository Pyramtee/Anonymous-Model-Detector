"""Reviewed local tokenizer adapters with explicit added-token policies."""
from __future__ import annotations

import ast
import base64
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .references import artifact_path, canonical_hash

ADAPTER_VERSION = "2026-09-27.1"
POLICIES = ("plain", "literal_special")
KIMI_PATTERN = "|".join([
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
])


@dataclass(frozen=True)
class Encoding:
    ids: tuple[int, ...]
    token_bytes: tuple[bytes, ...] | None


def _bytelevel_inverse() -> dict[str, int]:
    values = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    chars = values.copy()
    n = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + n)
            n += 1
    return {chr(char): value for value, char in zip(values, chars)}


class HFAdapter:
    supported_policies = POLICIES
    implementation_version = ADAPTER_VERSION

    def __init__(self, reference: dict, paths: dict[str, Path]):
        from tokenizers import Tokenizer
        self.reference = reference
        source = json.loads(paths["tokenizer.json"].read_text(encoding="utf-8"))
        self._source = source
        self._literal = Tokenizer.from_str(json.dumps(source))
        # This is explicitly the base-pipeline hypothesis, NOT a claim about any
        # provider's sanitization. No added token (including special=False) remains.
        plain = {**source, "added_tokens": []}
        self._plain = Tokenizer.from_str(json.dumps(plain))
        self._bytes: dict[int, bytes] | None = None
        self._vocabulary_cache: frozenset[bytes] | None = None
        # A byte decoder requires the reversible ByteLevel alphabet. Other
        # decoders are intentionally unsupported for hard output exclusion.
        if (source.get("decoder") or {}).get("type") == "ByteLevel":
            inverse = _bytelevel_inverse()
            try:
                added_by_id = {item["id"]: item["content"].encode("utf-8") for item in source.get("added_tokens", [])}
                self._bytes = {idx: (added_by_id[idx] if idx in added_by_id else bytes(inverse[ch] for ch in token))
                               for token, idx in source["model"]["vocab"].items()}
                for item in source.get("added_tokens", []):
                    self._bytes[item["id"]] = item["content"].encode("utf-8")
            except (KeyError, TypeError):
                self._bytes = None

    def encode(self, text: str, policy: str = "literal_special") -> Encoding:
        if policy not in self.supported_policies:
            raise ValueError(f"Unsupported encoding policy: {policy}")
        tokenizer = self._plain if policy == "plain" else self._literal
        ids = tuple(tokenizer.encode(text, add_special_tokens=False).ids)
        parts = tuple(self._bytes[i] for i in ids) if self._bytes is not None else None
        return Encoding(ids, parts)

    def vocabulary_bytes(self) -> frozenset[bytes] | None:
        if self._bytes is None:
            return None
        if self._vocabulary_cache is None:
            self._vocabulary_cache = frozenset(self._bytes.values())
        return self._vocabulary_cache

    def component_summary(self) -> dict[str, Any]:
        source = self._source
        model = source["model"]
        merges = [" ".join(x) if isinstance(x, list) else x for x in model.get("merges", [])]
        ids = set(model["vocab"].values()) | {x["id"] for x in source.get("added_tokens", [])}
        return {"base_vocab_count": len(model["vocab"]), "merges_count": len(merges),
                "added_count": len(source.get("added_tokens", [])), "unique_id_count": len(ids),
                "vocab_hash": canonical_hash(model["vocab"]), "merges_hash": canonical_hash(merges),
                **{name + "_hash": canonical_hash(source.get(name)) for name in
                   ("normalizer", "pre_tokenizer", "post_processor", "decoder", "added_tokens")}}


class KimiAdapter:
    supported_policies = POLICIES
    implementation_version = ADAPTER_VERSION

    def __init__(self, reference: dict, paths: dict[str, Path]):
        import tiktoken
        self.reference = reference
        # Parse constants for parity verification only; never execute remote code.
        tree = ast.parse(paths["tokenization_kimi.py"].read_text(encoding="utf-8"))
        patterns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pat_str" for t in node.targets):
                patterns.append("|".join(ast.literal_eval(node.value.args[0])))
        if patterns != [KIMI_PATTERN]:
            raise ValueError("Kimi official regex no longer matches reviewed adapter")
        self._ranks = {}
        for line in paths["tiktoken.model"].read_bytes().splitlines():
            if line.strip():
                token, rank = line.split()
                self._ranks[base64.b64decode(token, validate=True)] = int(rank)
        if len(self._ranks) != 163584 or set(self._ranks.values()) != set(range(163584)):
            raise ValueError("Unexpected Kimi K3 rank layout")
        config = json.loads(paths["tokenizer_config.json"].read_text(encoding="utf-8"))
        named = {int(key): value["content"] for key, value in config["added_tokens_decoder"].items()}
        self._special = {named.get(i, f"<|reserved_token_{i}|>"): i for i in range(163584, 163840)}
        self._enc = tiktoken.Encoding(name="kimi-k3-locked", pat_str=KIMI_PATTERN,
                                      mergeable_ranks=self._ranks, special_tokens=self._special)
        self._vocab_bytes = frozenset(self._ranks) | frozenset(x.encode("utf-8") for x in self._special)

    def encode(self, text: str, policy: str = "literal_special") -> Encoding:
        if policy not in self.supported_policies:
            raise ValueError(f"Unsupported encoding policy: {policy}")
        ids = tuple(self._enc.encode(text, allowed_special="all") if policy == "literal_special"
                    else self._enc.encode(text, disallowed_special=()))
        return Encoding(ids, tuple(self._enc.decode_single_token_bytes(i) for i in ids))

    def vocabulary_bytes(self) -> frozenset[bytes]:
        return self._vocab_bytes

    def component_summary(self) -> dict:
        return {"base_vocab_count": len(self._ranks), "special_count": len(self._special),
                "unique_id_count": 163840, "regex_hash": canonical_hash(KIMI_PATTERN),
                "special_hash": canonical_hash(self._special)}


def load_adapter(reference: dict, artifact_dir: Path):
    paths = {Path(item["repo_path"]).name: artifact_path(item, artifact_dir) for item in reference["artifacts"]}
    if reference["backend"] == "hf_json_v1":
        return HFAdapter(reference, paths)
    if reference["backend"] == "kimi_k3_v1":
        return KimiAdapter(reference, paths)
    raise ValueError(f"Unknown tokenizer backend: {reference['backend']}")

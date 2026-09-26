"""Vendor/family registry and local tokenizer loading."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_VENDORS_YAML = PROJECT_ROOT / "data" / "vendors.yaml"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "data" / "tokenizers"


@dataclass
class Family:
    id: str
    display: str
    vendor: str
    hf_repos: list[str]
    models: list[str]
    tokenizer_path: Path | None = None
    tokenizer_sha256: str | None = None

    @property
    def available(self) -> bool:
        return self.tokenizer_path is not None and self.tokenizer_path.exists()


@dataclass
class Registry:
    families: dict[str, Family] = field(default_factory=dict)

    def get(self, family_id: str) -> Family:
        return self.families[family_id]

    def available_families(self) -> list[Family]:
        return [f for f in self.families.values() if f.available]

    def family_ids(self) -> list[str]:
        return list(self.families.keys())


def _parse_families(data: dict) -> dict[str, Family]:
    raw = data.get("families") or {}
    families: dict[str, Family] = {}
    for fid, meta in raw.items():
        families[fid] = Family(
            id=fid,
            display=meta.get("display", fid),
            vendor=meta.get("vendor", fid),
            hf_repos=list(meta.get("hf_repos") or []),
            models=list(meta.get("models") or []),
        )
    return families


def load_registry(
    vendors_path: Path | None = None,
    tokenizer_dir: Path | None = None,
) -> Registry:
    vendors_path = vendors_path or DEFAULT_VENDORS_YAML
    tokenizer_dir = tokenizer_dir or DEFAULT_TOKENIZER_DIR

    with open(vendors_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    families = _parse_families(data)

    for fid, family in families.items():
        json_path = tokenizer_dir / fid / "tokenizer.json"
        bpe_path = tokenizer_dir / fid / "tiktoken.model"
        for tok_path in (json_path, bpe_path):
            if tok_path.exists():
                family.tokenizer_path = tok_path
                family.tokenizer_sha256 = sha256_file(tok_path)
                break

    return Registry(families=families)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))

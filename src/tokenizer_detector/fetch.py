"""Fetch tokenizer files from HuggingFace for known families."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from .registry import Family, Registry, sha256_file

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class FetchError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    if HF_TOKEN:
        return {"Authorization": f"Bearer {HF_TOKEN}"}
    return {}


def _download(url: str, dest: Path, timeout: float = 60.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with httpx.stream(
            "GET",
            url,
            headers=_headers(),
            timeout=timeout,
            follow_redirects=True,
        ) as resp:
            if resp.status_code != 200:
                raise FetchError(f"HTTP {resp.status_code} for {url}")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
            head = tmp.read_bytes()[:200]
            if head.startswith(b"version https://") or head.lstrip().startswith(b"<"):
                raise FetchError(f"Non-file content at {url}")
        tmp.replace(dest)
        return dest
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def fetch_family(
    family: Family,
    tokenizer_dir: Path,
    timeout: float = 60.0,
) -> Path:
    """Download tokenizer.json or tiktoken.model for a family."""
    dest_dir = tokenizer_dir / family.id
    json_path = dest_dir / "tokenizer.json"
    bpe_path = dest_dir / "tiktoken.model"

    if json_path.exists():
        family.tokenizer_path = json_path
        family.tokenizer_sha256 = sha256_file(json_path)
        return json_path
    if bpe_path.exists():
        family.tokenizer_path = bpe_path
        family.tokenizer_sha256 = sha256_file(bpe_path)
        return bpe_path

    errors: list[str] = []
    for repo in family.hf_repos:
        # Prefer native tokenizer.json
        try:
            path = _download(
                f"https://huggingface.co/{repo}/resolve/main/tokenizer.json",
                json_path,
                timeout=timeout,
            )
            family.tokenizer_path = path
            family.tokenizer_sha256 = sha256_file(path)
            return path
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{repo}/tokenizer.json: {exc}")

        # Fallback: tiktoken.model (Kimi / OpenAI-style)
        try:
            path = _download(
                f"https://huggingface.co/{repo}/resolve/main/tiktoken.model",
                bpe_path,
                timeout=timeout,
            )
            family.tokenizer_path = path
            family.tokenizer_sha256 = sha256_file(path)
            return path
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{repo}/tiktoken.model: {exc}")

    raise FetchError(
        f"Failed to fetch tokenizer for family '{family.id}'. Attempts:\n  "
        + "\n  ".join(errors)
    )


def fetch_families(
    registry: Registry,
    family_ids: list[str] | None = None,
    tokenizer_dir: Path | None = None,
    timeout: float = 60.0,
) -> tuple[dict[str, Path], list[str]]:
    """Fetch tokenizers; continue past per-family failures.

    Returns (successes, error_messages).
    """
    from .registry import DEFAULT_TOKENIZER_DIR

    tokenizer_dir = tokenizer_dir or DEFAULT_TOKENIZER_DIR
    targets = family_ids or registry.family_ids()
    results: dict[str, Path] = {}
    errors: list[str] = []
    for fid in targets:
        family = registry.get(fid)
        try:
            results[fid] = fetch_family(family, tokenizer_dir, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{fid}: {exc}")

    return results, errors

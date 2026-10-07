"""Pinned, hash-verified reference artifacts. No model inference or remote code."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "reference-manifest.json"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"


class ReferenceError(RuntimeError):
    """A required reference is missing, invalid, or cannot be loaded."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load_manifest(path: Path | str | None = None) -> dict:
    with Path(path or DEFAULT_MANIFEST).open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("references"), list):
        raise ReferenceError("Unsupported reference manifest schema")
    seen: set[str] = set()
    for ref in manifest["references"]:
        rid = ref.get("reference_id")
        if not isinstance(rid, str) or rid in seen:
            raise ReferenceError(f"Invalid/duplicate reference ID: {rid}")
        seen.add(rid)
        if ref.get("role") not in ("target", "external_control"):
            raise ReferenceError(f"Invalid reference role: {rid}")
        if ref.get("backend") not in ("hf_json_v1", "kimi_k3_v1"):
            raise ReferenceError(f"Unknown backend: {rid}")
        if not ref.get("artifacts"):
            raise ReferenceError(f"Empty artifact list: {rid}")
        for item in ref["artifacts"]:
            if not re.fullmatch(r"[a-f0-9]{40}", item.get("revision", "")):
                raise ReferenceError(f"Reference revision must be a fixed commit: {rid}")
            if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
                raise ReferenceError(f"Invalid artifact digest: {rid}")
            if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] <= 0:
                raise ReferenceError(f"Invalid artifact size: {rid}")
            if ".." in Path(item["repo_path"]).parts or Path(item["repo_path"]).is_absolute():
                raise ReferenceError(f"Unsafe artifact path: {rid}")
    return manifest


def artifact_path(artifact: dict, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> Path:
    return Path(artifact_dir) / artifact["sha256"] / Path(artifact["repo_path"]).name


def _file_status(artifact: dict, artifact_dir: Path | str) -> str:
    path = artifact_path(artifact, artifact_dir)
    if not path.is_file():
        return "missing"
    if path.stat().st_size != artifact["size_bytes"]:
        return "size_mismatch"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "ready" if digest == artifact["sha256"] else "hash_mismatch"


def inspect_references(manifest: dict, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> list[dict]:
    result = []
    for ref in manifest["references"]:
        artifacts = [{"repo_path": a["repo_path"], "sha256": a["sha256"],
                      "status": _file_status(a, artifact_dir)} for a in ref["artifacts"]]
        result.append({"reference_id": ref["reference_id"], "family_id": ref["family_id"],
                       "display": ref["display"], "vendor": ref["vendor"], "role": ref["role"],
                       "backend": ref["backend"], "policies": ref["policies"],
                       "revision": ref["artifacts"][0]["revision"],
                       "status": "ready" if all(a["status"] == "ready" for a in artifacts) else "unavailable",
                       "artifacts": artifacts})
    return result


def fetch_references(manifest: dict, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
                     families: list[str] | None = None, include_controls: bool = False) -> tuple[list, list]:
    """Fetch only fixed tokenizer artifacts; retries/updates are explicitly manual."""
    requested = set(families or [])
    eligible = [r for r in manifest["references"] if include_controls or r["role"] == "target"]
    known = {r[k] for r in eligible for k in ("reference_id", "family_id")}
    if requested - known:
        raise ReferenceError(f"Unknown selected references: {sorted(requested - known)}")
    targets = [r for r in eligible if not requested or requested & {r["family_id"], r["reference_id"]}]
    ready, errors = [], []
    with httpx.Client(timeout=90.0, follow_redirects=True) as client:
        for ref in targets:
            try:
                for item in ref["artifacts"]:
                    if _file_status(item, artifact_dir) == "ready":
                        continue
                    url = f"https://huggingface.co/{item['repo_id']}/resolve/{item['revision']}/{item['repo_path']}"
                    path = artifact_path(item, artifact_dir)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(path.name + ".download")
                    try:
                        with client.stream("GET", url) as response:
                            response.raise_for_status()
                            digest = hashlib.sha256()
                            size = 0
                            with tmp.open("wb") as fh:
                                for chunk in response.iter_bytes():
                                    size += len(chunk)
                                    if size > item["size_bytes"]:
                                        raise ReferenceError(f"Unexpected download size: {ref['reference_id']}/{item['repo_path']}")
                                    digest.update(chunk)
                                    fh.write(chunk)
                        if size != item["size_bytes"] or digest.hexdigest() != item["sha256"]:
                            raise ReferenceError(f"Artifact verification failed: {ref['reference_id']}/{item['repo_path']}")
                        tmp.replace(path)
                    finally:
                        tmp.unlink(missing_ok=True)
                # A successful fetch includes validating the adapter, not just HTTP 200.
                from .adapters import load_adapter
                load_adapter(ref, Path(artifact_dir))
                ready.append(ref["reference_id"])
            except Exception as exc:
                errors.append(f"{ref['reference_id']}: {exc}")
    return ready, errors


def load_adapters(manifest: dict, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
                  include_controls: bool = False) -> dict:
    from .adapters import load_adapter
    selected = [r for r in manifest["references"] if include_controls or r["role"] == "target"]
    statuses = {r["reference_id"]: r for r in inspect_references(manifest, artifact_dir)}
    unavailable = [r["reference_id"] for r in selected if statuses[r["reference_id"]]["status"] != "ready"]
    if unavailable:
        raise ReferenceError("Required references unavailable: " + ", ".join(unavailable))
    adapters = {}
    for ref in selected:
        try:
            adapters[ref["reference_id"]] = load_adapter(ref, Path(artifact_dir))
        except Exception as exc:
            raise ReferenceError(f"Unable to load {ref['reference_id']}: {exc}") from exc
    return adapters

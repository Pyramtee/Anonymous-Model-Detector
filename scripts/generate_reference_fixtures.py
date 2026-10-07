"""Manually regenerate OFFLINE reference fixtures after reviewed reference changes.

Run: uv run python scripts/generate_reference_fixtures.py --update-fixtures
This overwrites both static fixtures. Tests NEVER invoke this script. No network,
model API, production adapter, or inference engine is used to generate truth.
The frozen probe texts come from the probe bank; token IDs come directly from
pinned official tokenizer files and reviewed Kimi configuration constants.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def artifact_paths(reference):
    paths = {}
    for item in reference["artifacts"]:
        path = ROOT / "data/artifacts" / item["sha256"] / Path(item["repo_path"]).name
        raw = path.read_bytes()
        if len(raw) != item["size_bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Pinned artifact verification failed: {path}")
        paths[Path(item["repo_path"]).name] = path
    return paths


def official_oracle(reference):
    """Construct the reference oracle without importing production adapters."""
    from tokenizers import Tokenizer
    import tiktoken
    paths = artifact_paths(reference)
    if reference["backend"] == "hf_json_v1":
        literal = Tokenizer.from_file(str(paths["tokenizer.json"]))
        base = json.loads(paths["tokenizer.json"].read_text(encoding="utf-8"))
        base["added_tokens"] = []  # Defined base-only hypothesis, not all providers.
        plain = Tokenizer.from_str(json.dumps(base))
        return lambda text, policy: (plain if policy == "plain" else literal).encode(
            text, add_special_tokens=False).ids
    tree = ast.parse(paths["tokenization_kimi.py"].read_text(encoding="utf-8"))
    patterns = ["|".join(ast.literal_eval(node.value.args[0])) for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "pat_str" for target in node.targets)]
    if len(patterns) != 1:
        raise ValueError("Expected exactly one official Kimi regex constant")
    ranks = {base64.b64decode(token, validate=True): int(rank)
             for token, rank in (line.split() for line in paths["tiktoken.model"].read_bytes().splitlines() if line.strip())}
    config = json.loads(paths["tokenizer_config.json"].read_text(encoding="utf-8"))
    named = {int(idx): value["content"] for idx, value in config["added_tokens_decoder"].items()}
    reserved = [ast.literal_eval(node.value) for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "num_reserved_special_tokens" for target in node.targets)]
    if reserved != [256]:
        raise ValueError("Unexpected official Kimi reserved special-token count")
    special = {named.get(i, f"<|reserved_token_{i}|>"): i
               for i in range(len(ranks), len(ranks) + reserved[0])}
    encoding = tiktoken.Encoding(name="pinned-official-oracle", pat_str=patterns[0],
                                 mergeable_ranks=ranks, special_tokens=special)
    return lambda text, policy: (encoding.encode(text, disallowed_special=()) if policy == "plain"
                                  else encoding.encode(text, allowed_special="all"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-fixtures", action="store_true",
                        help="explicitly overwrite the reviewed static fixtures")
    args = parser.parse_args()
    if not args.update_fixtures:
        parser.error("No files changed. Pass --update-fixtures to regenerate both fixtures.")
    from tokenizer_detector.probe_bank import _probes, PROBE_VERSION
    manifest = json.loads((ROOT / "data/reference-manifest.json").read_text(encoding="utf-8"))
    probes = _probes()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {
        "kind": "synthetic_offline_official_tokenizer_oracle",
        "method": "Direct official HF JSON; official Kimi constants extracted through AST and used by tiktoken. No production adapter calls. Plain is a defined base-pipeline hypothesis. HTTP counts add synthetic fixed offset 17, not real API evidence.",
        "source_script": "scripts/generate_reference_fixtures.py",
        "source_script_sha256": source_sha,
        "manifest_sha256": digest(manifest), "probe_version": PROBE_VERSION,
        "probe_sha256": digest({k: v.__dict__ for k, v in probes.items()}),
        **{name + "_version": importlib.metadata.version(name) for name in ("tokenizers", "tiktoken")},
        "references": {r["reference_id"]: {"revision": r["artifacts"][0]["revision"],
                       "artifact_sha256": {a["repo_path"]: a["sha256"] for a in r["artifacts"]}}
                       for r in manifest["references"]},
    }
    profiles, cases = {}, []
    samples = ["Hello 世界 123456789", "नमस्ते दुनिया", "é e\u0301 a\u0301\u0327",
               "a\t b\r\nc", "<im_start>", "<｜place▁holder▁no▁0｜>", "[BOS]", "<|reserved_token_163592|>"]
    for reference in manifest["references"]:
        encode = official_oracle(reference)
        rid = reference["reference_id"]
        for text in samples:
            cases.append({"reference_id": rid, "policy": "literal_special", "text": text,
                          "ids": encode(text, "literal_special")})
        if reference["backend"] == "kimi_k3_v1":
            for text in ("[BOS]", "<|reserved_token_163592|>", "<|open|>"):
                cases.append({"reference_id": rid, "policy": "plain", "text": text,
                              "ids": encode(text, "plain")})
        for policy in ("plain", "literal_special"):
            counts = {pid: len(encode(probe.text, policy)) for pid, probe in probes.items()}
            profiles[rid + ":" + policy] = {
                "reference_id": rid, "family_id": reference["family_id"], "role": reference["role"],
                "policy": policy, "offset": 17, "counts": counts,
                "prompt_counts": {pid: count + 17 for pid, count in counts.items()},
            }
    outputs = {
        "reference_goldens.json": {"schema_version": 1, "provenance": provenance, "cases": cases},
        "observations_pinned.json": {"schema_version": 1, "provenance": provenance,
                                     "probes": {k: v.__dict__ for k, v in probes.items()}, "profiles": profiles},
    }
    for name, payload in outputs.items():
        path = ROOT / "tests/fixtures" / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {path.relative_to(ROOT)}")
    print(f"Frozen {len(cases)} ID cases; {len(profiles)} profiles x {len(probes)} probes. No network used.")


if __name__ == "__main__":
    main()

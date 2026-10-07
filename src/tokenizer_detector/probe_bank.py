"""Frozen probe pools and reproducible, target-only local count matrices."""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path

from .adapters import ADAPTER_VERSION
from .models import Probe, Profile, ProbeBank
from .probes import LONG_PROBES, BOUNDARY_PROBES
from .references import ReferenceError, canonical_hash

PROBE_VERSION = "2026-09-27.3"
WRAPPER_PREFIX = "Data:\n"
WRAPPER_SUFFIX = "\nEnd."


def render(payload: str) -> str:
    return WRAPPER_PREFIX + payload + WRAPPER_SUFFIX


def _probes() -> dict[str, Probe]:
    result: dict[str, Probe] = {}
    def add(pid, payload, category="ordinary", split="selection", bundle=None, protocol="ordinary", round=0):
        result[pid] = Probe(pid, payload, render(payload), category, split, bundle, protocol, round)
    add("anchor", "A quiet river. 平静的河流。", split="anchor")
    for probe in LONG_PROBES + BOUNDARY_PROBES:
        add(probe.id, probe.text, category=probe.category)
    # Ordinary script and boundary examples are frozen independently of controls.
    ordinary = ["नमस्ते दुनिया", "हिन्दी भाषा", "किंकर्तव्यविमूढ़", "abc123456789XYZ",
                "é e\u0301 a\u0301\u0327", "甲\t乙  丙\r\n丁", "a = [1, 2, 3]\nprint(a[::-1])",
                "日本語 한국어 ภาษาไทย العربية עברית", "000001234567890123456789", "👩🏽‍💻🇨🇳🏳️‍🌈"]
    for idx, text in enumerate(ordinary):
        add(f"ordinary_{idx}", text)
        add(f"ordinary_expanded_{idx}", (text + "\n") * 8)
    markers = ["<im_start>", "<patch_start>", "<tool_call>",
               "<｜place▁holder▁no▁0｜>", "<｜begin▁of▁repo▁name｜>", "｜DSML｜"]
    for idx, marker in enumerate(markers):
        bundle = f"marker_bundle_{idx}"
        add(f"marker_{idx}_raw", marker, "marker", bundle=bundle, protocol="markers")
        damaged = marker[:1] + " " + marker[1:]
        add(f"marker_{idx}_damaged", damaged, "damaged_control", bundle=bundle, protocol="markers")
    # Confirmation content/order is fixed. Neither external controls nor live
    # candidate scores select or rewrite these held-out samples.
    confirms = [
        ("春天的早晨，研究人员记录温度、风速与河流水位。汉字边界和标点需要精确保存。" * 5,
         "def validate_record(x):\n    assert x['count'] >= 0\n    return sorted(x.items())\n" * 5),
        ("अलग-अलग भाषाओं में शब्द सीमाएँ बदलती हैं। 日本語の文章と한국어 문장。" * 4,
         "Numbers 000123456789; repeated_spaces    and tabs\t\t; words don't disappear.\n" * 5),
        ("边界测试：纤维、数据库、分词规则、词元长度。结合字符 e\u0301 与 emoji 👩‍🚀。" * 5,
         "const verify = (items) => items.map((v, i) => ({value:v, index:i}));\n" * 5),
    ]
    directions = [("<patch_start>", "<｜begin▁of▁repo▁name｜>"),
                  ("<tool_call>", "｜DSML｜"),
                  ("<im_start>", "<｜place▁holder▁no▁0｜>")]
    # Every held-out round independently covers Unicode normalization and
    # boundary behavior as well as ordinary prose/code. Fixed categories and
    # text; no control-tokenizer scores or online responses enter this pool.
    confirmation_boundaries = [
        ("\nBoundary review: ﬁnal ﬂower ﬀix; q\u0308 r\u0327 e\u0301; １２３４５; 👩🏽‍🔬.\n" * 3,
         "\nMixed boundaries: ﬃ ﬄ ﬁ; z\u0301 x\u0308 n\u0327; ७८९०; 𠀁𪛗.\n" * 3),
        ("\nReview spaces: ﬁeld\tﬂuent   ﬀrame; v\u0327 u\u0308; ４５６.\n" * 3,
         "\nReview scripts: ﬄow ﬁt ﬃ; q\u0301 b\u0327; कृतज्ञता 👨🏿‍🚀.\n" * 3),
        ("\nFinal review: ﬂight ﬁnish ﬀ; w\u0308 p\u0327; ８９０１２; 𫠞.\n" * 3,
         "\nFinal boundaries: ﬃ ﬄ; y\u0301 t\u0327; ＡＸＺ; 👩🏼‍🎨.\n" * 3),
    ]
    for round, pair in enumerate(confirms):
        for j, payload in enumerate(pair):
            payload += confirmation_boundaries[round][j]
            add(f"confirm_ordinary_{round}_{j}", payload, split="confirmation", round=round)
        for j, marker in enumerate(directions[round]):
            payload = f"固定复核上下文 {round}:\n{marker}\n复核结束。"
            add(f"confirm_markers_{round}_{j}", payload, "marker", "confirmation", protocol="markers", round=round)
    for i, text in enumerate(["山川与河流。", "def square(x): return x*x", "नमस्ते दुनिया 🎉"]):
        payload = "Copy the following text exactly, with no explanation:\n" + text
        # Output requests have their own protocol; they are never counted as
        # ordinary confirmation by the engine.
        add(f"output_{i}", payload, "output", "output", protocol="output", round=i)
    return result


def _groups(profiles: dict[str, Profile], probe_ids: list[str], anchor: str) -> list[list[str]]:
    groups: dict[tuple, list[str]] = {}
    for pid, profile in sorted(profiles.items()):
        signature = tuple(profile.counts[p] - profile.counts[anchor] for p in probe_ids)
        groups.setdefault(signature, []).append(pid)
    return list(groups.values())


def _diagnostics(probes: dict[str, Probe], profiles: dict[str, Profile]) -> dict:
    ordinary = [p.probe_id for p in probes.values() if p.split == "selection" and p.protocol == "ordinary"]
    markers = [p.probe_id for p in probes.values() if p.split == "selection" and p.protocol == "markers"]
    witnesses = []
    for (aid, a), (bid, b) in itertools.combinations(sorted(profiles.items()), 2):
        if a.family_id == b.family_id:
            continue
        found = None
        for channel, pool in (("ordinary", ordinary), ("markers", markers)):
            candidates = [(abs((a.counts[p] - a.counts['anchor']) - (b.counts[p] - b.counts['anchor'])), p)
                          for p in pool]
            gap, probe_id = max(candidates, default=(0, ""))
            if gap:
                found = {"profiles": [aid, bid], "channel": channel, "anchor_probe_id": "anchor",
                         "probe_id": probe_id, "delta_gap": gap, "kind": "finite_probe_witness"}
                break
        witnesses.append(found or {"profiles": [aid, bid], "kind": "none_found_in_pool"})
    # Profile-pair witnesses do not assert all policies of two families separable.
    return {"ordinary_groups": _groups(profiles, ordinary, "anchor"),
            "all_selection_groups": _groups(profiles, ordinary + markers, "anchor"),
            "profile_pair_witnesses": witnesses,
            "interpretation": "Finite-pool signatures, not proof of complete tokenizer equivalence; all policies must survive family exclusion."}


def build_probe_bank(manifest: dict, adapters: dict, cache_dir: Path | str | None = None) -> ProbeBank:
    probes = _probes()
    targets = [r for r in manifest['references'] if r['role'] == 'target']
    missing = [r['reference_id'] for r in targets if r['reference_id'] not in adapters]
    if missing:
        raise ReferenceError("Probe matrix requires every target reference: " + ", ".join(missing))
    identity = {"schema_version": 2, "manifest_sha256": canonical_hash(manifest),
                "reference_manifest": manifest,
                "probe_version": PROBE_VERSION,
                "probe_sha256": canonical_hash({k: p.__dict__ for k, p in probes.items()}),
                "adapter_version": ADAPTER_VERSION,
                "adapter_source_sha256": hashlib.sha256(Path(__file__).with_name("adapters.py").read_bytes()).hexdigest(),
                "matrix_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "runtime_versions": {"python": platform.python_version(),
                                     **{name: importlib.metadata.version(name) for name in ('tokenizers', 'tiktoken')}},
                "adapter_versions": {r['reference_id']: adapters[r['reference_id']].implementation_version for r in targets}}
    identity['cache_key'] = canonical_hash(identity)
    cache_path = Path(cache_dir) / (identity['cache_key'] + '.json') if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        try:
            saved = json.loads(cache_path.read_text(encoding='utf-8'))
            payload = saved['bank']
            if saved['payload_sha256'] == canonical_hash(payload) and payload['identity'] == identity:
                bank = ProbeBank.from_dict(payload)
                expected = {r['reference_id'] + ':' + p for r in targets for p in adapters[r['reference_id']].supported_policies}
                if set(bank.profiles) == expected and set(bank.probes) == set(probes):
                    return bank
        except (ValueError, KeyError, TypeError):
            pass
    profiles = {}
    for ref in targets:
        adapter = adapters[ref['reference_id']]
        for policy in adapter.supported_policies:
            pid = ref['reference_id'] + ':' + policy
            counts = {key: len(adapter.encode(probe.text, policy).ids) for key, probe in probes.items()}
            profiles[pid] = Profile(pid, ref['reference_id'], ref['family_id'], ref['vendor'], ref['display'], policy, counts)
    diagnostics = _diagnostics(probes, profiles)
    summaries = {r['reference_id']: adapters[r['reference_id']].component_summary()
                 for r in targets if hasattr(adapters[r['reference_id']], 'component_summary')}
    diagnostics['component_summaries'] = summaries
    diagnostics['component_differences'] = [
        {'references': [left, right],
         'different_components': sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))}
        for (left, a), (right, b) in itertools.combinations(sorted(summaries.items()), 2)
    ]
    bank = ProbeBank(probes, profiles, identity, diagnostics)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = bank.to_dict()
        temp = cache_path.with_suffix('.tmp')
        temp.write_text(json.dumps({'payload_sha256': canonical_hash(payload), 'bank': payload}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temp.replace(cache_path)
    return bank

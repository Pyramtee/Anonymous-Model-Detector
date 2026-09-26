from tokenizer_detector.fuse import SignalScores, fuse, group_equivalent_families
from tokenizer_detector.probes import ALL_PROBES, LONG_PROBES, BOUNDARY_PROBES, ECHO_PROBES
from tokenizer_detector.registry import load_registry, DEFAULT_TOKENIZER_DIR
from tokenizer_detector.tokenizer_backends import load_backend
from tokenizer_detector.signals.prompt_count import CountSample, score_prompt_counts


def test_probes_are_nonempty_and_categorized():
    assert LONG_PROBES and BOUNDARY_PROBES and ECHO_PROBES
    ids = [p.id for p in ALL_PROBES]
    assert len(ids) == len(set(ids))
    for p in LONG_PROBES:
        assert p.category == "long"
        assert len(p.text) > 200


def test_registry_loads_families():
    reg = load_registry()
    assert "qwen" in reg.families
    assert "deepseek" in reg.families
    assert "kimi" in reg.families
    assert "glm" in reg.families
    assert "mimo" in reg.families
    assert "step" in reg.families
    assert "minimax" in reg.families
    for fam in reg.families.values():
        assert fam.hf_repos
        assert fam.models


def test_prompt_count_perfect_match_scores_high():
    samples = [
        CountSample(
            probe_id="p1",
            api_prompt_tokens=1000,
            local_counts={"a": 1000, "b": 1400},
            api_prompt_len_chars=3000,
        ),
        CountSample(
            probe_id="p2",
            api_prompt_tokens=800,
            local_counts={"a": 800, "b": 1200},
            api_prompt_len_chars=2500,
        ),
    ]
    scores = score_prompt_counts(samples, ["a", "b"])
    by = {s.family_id: s.score for s in scores}
    assert by["a"] > 0.95
    assert by["a"] > by["b"]


def test_fuse_prefers_matching_family():
    signals = SignalScores(
        prompt_count={"qwen": 0.95, "deepseek": 0.4, "kimi": 0.3},
        completion={"qwen": 0.9, "deepseek": 0.2, "kimi": 0.2},
        text_stats={"qwen": 0.85, "deepseek": 0.3, "kimi": 0.3},
    )
    meta = {
        "qwen": {"vendor": "Alibaba", "display": "Qwen", "models": ["Qwen2.5-*"]},
        "deepseek": {"vendor": "DeepSeek", "display": "DeepSeek", "models": ["DeepSeek-V3"]},
        "kimi": {"vendor": "Moonshot", "display": "Kimi", "models": ["Kimi-K2*"]},
    }
    v = fuse(["qwen", "deepseek", "kimi"], signals, meta=meta)
    assert v.verdict == "qwen"
    assert v.candidates[0].family_id == "qwen"
    assert v.candidates[0].confidence > 0.5
    assert v.candidates[0].vendor == "Alibaba"


def test_fuse_unknown_when_all_low():
    signals = SignalScores(
        prompt_count={"qwen": 0.1, "deepseek": 0.1},
        completion={},
        text_stats={},
    )
    v = fuse(["qwen", "deepseek"], signals, weights={"prompt_count": 1.0})
    assert v.verdict in ("unknown", "ambiguous")


def test_equivalence_groups_identical_fingerprints():
    fp = {
        "qwen": (1, 2, 3),
        "mimo": (1, 2, 3),
        "deepseek": (9, 9, 9),
    }
    eq = group_equivalent_families(["qwen", "mimo", "deepseek"], fp)
    assert set(eq["qwen"]) == {"qwen", "mimo"}
    assert eq["deepseek"] == ["deepseek"]


def test_fuse_marks_equivalent_families():
    signals = SignalScores(
        prompt_count={"qwen": 0.95, "mimo": 0.95, "deepseek": 0.4},
    )
    meta = {
        "qwen": {"vendor": "Alibaba", "display": "Qwen", "models": ["Qwen2.5-*"]},
        "mimo": {"vendor": "Xiaomi", "display": "MiMo", "models": ["MiMo*"]},
        "deepseek": {"vendor": "DeepSeek", "display": "DeepSeek", "models": ["DeepSeek-V3"]},
    }
    eq = {
        "qwen": ["qwen", "mimo"],
        "mimo": ["qwen", "mimo"],
        "deepseek": ["deepseek"],
    }
    v = fuse(
        ["qwen", "mimo", "deepseek"],
        signals,
        weights={"prompt_count": 1.0},
        meta=meta,
        equivalence=eq,
    )
    assert "qwen" in v.verdict and "mimo" in v.verdict
    assert v.candidates[0].equivalent_families


def test_local_tokenizers_load_and_count():
    reg = load_registry()
    available = reg.available_families()
    if len(available) < 7:
        return  # skip if fetch not run
    backends = {
        f.id: load_backend(f.id, DEFAULT_TOKENIZER_DIR) for f in available
    }
    sample = LONG_PROBES[0].text
    counts = {fid: b.encode_len(sample) for fid, b in backends.items()}
    assert all(v > 50 for v in counts.values())
    # Known observed: qwen/mimo and deepseek/step share behavior
    assert counts["qwen"] == counts["mimo"]
    assert counts["deepseek"] == counts["step"]

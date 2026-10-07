import hashlib
import json

import pytest
from tokenizers import Tokenizer, AddedToken, models, decoders, processors

from tokenizer_detector.adapters import HFAdapter, KIMI_PATTERN


def tiny_adapter(tmp_path):
    tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate('ab<>x')}, merges=[]))
    tok.decoder = decoders.ByteLevel()
    tok.add_tokens([AddedToken('<x>', special=False, normalized=False)])
    tok.add_special_tokens(['<bos>'])
    tok.post_processor = processors.TemplateProcessing(single='<bos> $A', special_tokens=[('<bos>', 6)])
    path = tmp_path / 'tokenizer.json'
    tok.save(str(path))
    return HFAdapter({'reference_id': 'tiny'}, {'tokenizer.json': path})


def test_plain_bypasses_even_non_special_added_tokens_and_both_omit_bos(tmp_path):
    adapter = tiny_adapter(tmp_path)
    assert adapter.encode('<x>', 'literal_special').ids == (5,)
    assert adapter.encode('<x>', 'plain').ids == (2, 4, 3)
    assert adapter.encode('ab', 'plain').ids == (0, 1)
    assert adapter.encode('ab', 'literal_special').ids == (0, 1)
    assert adapter.encode('<x>', 'plain').token_bytes == (b'<', b'x', b'>')
    assert b'<x>' in adapter.vocabulary_bytes()


def test_bad_policy_is_explicit_error(tmp_path):
    with pytest.raises(ValueError, match='Unsupported encoding policy'):
        tiny_adapter(tmp_path).encode('a', 'automatic_guess')


def test_kimi_regex_has_script_and_digit_rules():
    assert r'[\p{Han}]+' in KIMI_PATTERN
    assert r'\p{N}{1,3}' in KIMI_PATTERN
    assert r'&&[^\p{Han}]' in KIMI_PATTERN

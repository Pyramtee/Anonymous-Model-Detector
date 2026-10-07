"""Built-in fixed probe set for tokenizer fingerprinting.

Probes are versioned and deterministic for reproducibility.
Categories:
  - long: long discriminative texts (primary prompt_tokens signal)
  - boundary: short strings that expose segmentation differences
  - echo: fixed strings for completion re-tokenization
"""

from __future__ import annotations

from dataclasses import dataclass


PROBE_SET_VERSION = "1"


@dataclass(frozen=True)
class Probe:
    id: str
    category: str  # long | boundary | echo
    text: str
    note: str = ""


# Long probes: template overhead (~10-20 tokens) is relatively small here.
LONG_PROBES: list[Probe] = [
    Probe(
        id="long_zh_article",
        category="long",
        text=(
            "自然语言处理的发展史可以追溯到二十世纪五十年代。"
            "早期的机器翻译系统依赖于大规模的双语词典与手工编写的语法规则，"
            "效果十分有限。随着统计方法的兴起，研究者开始利用平行语料自动学习翻译概率，"
            "显著提升了质量。进入深度学习时代之后，循环神经网络、注意力机制与Transformer"
            "结构相继成为主流。近年来，大语言模型通过在海量文本上预训练，"
            "展现出跨任务的泛化能力。今天的挑战包括幻觉、对齐、推理效率以及多语言公平性。"
            "在中国，通义千问、深度求索、智谱、月之暗面、小米MiMo、阶跃星辰与MiniMax"
            "等团队都发布了具有竞争力的开源权重模型，推动了整个生态的发展。"
            "本段文本专门用于探测分词器在中文、标点、数字混排场景下的切分行为。"
            "请注意：的、了、是、在、和、与、或、而、但、因为、所以、虽然、然而、因此。"
            "数字混排：2024年3月15日，OpenRouter平台上线了usage统计接口，支持prompt_tokens与completion_tokens。"
            "再补一段关于工程实践的内容：当我们需要判断一个匿名接口背后究竟是哪一家的模型时，"
            "可以利用其原生分词器对相同文本给出的token计数作为指纹。"
            "如果两个模型共享词表，它们对同一段文本的计数应当完全一致；"
            "如果词表不同，则在精心设计的边界样本上会出现可测量的差异。"
            "因此，固定探针集的质量直接决定了检测的区分能力。"
        ),
        note="中文长文 + 数字/标点混排",
    ),
    Probe(
        id="long_en_tech",
        category="long",
        text=(
            "The history of natural language processing stretches back to the 1950s, "
            "when researchers attempted machine translation with hand-crafted rules and "
            "bilingual dictionaries. Statistical methods later replaced many brittle rules "
            "by learning probabilities from parallel corpora. Deep learning introduced "
            "recurrent networks, attention, and eventually the Transformer architecture, "
            "which scaled remarkably well. Large language models pretrained on web-scale "
            "text now transfer across tasks with minimal fine-tuning. Remaining challenges "
            "include hallucination, alignment, inference cost, multilingual fairness, and "
            "evaluation. This paragraph is intentionally dense with punctuation: commas, "
            "semicolons; colons: parentheses (like this), brackets [and these], quotes "
            "'single' and \"double\", dashes - em dashes — and ellipses... "
            "CamelCase identifiers such as promptTokens, completion_tokens, totalTokens "
            "appear frequently. Numbers: 1024, 65536, 151643, 200064, 160000. "
            "URLs like https://openrouter.ai/api/v1/chat/completions and email-like tokens "
            "user@example.com stress byte-level BPE boundaries. "
            "Repeated English function words: the, of, and, to, in, a, is, that, for, it, "
            "as, was, with, be, by, on, not, he, this, are, or, his, from, at, which, but. "
            "Code snippet: def encode(text: str) -> list[int]: return tokenizer.encode(text).ids  "
            "SQL fragment: SELECT id, name FROM models WHERE vendor IN ('Qwen','DeepSeek'); "
            "JSON fragment: {\"model\": \"anonymous\", \"usage\": {\"prompt_tokens\": 42}} "
        ),
        note="英文技术长文 + 代码/符号",
    ),
    Probe(
        id="long_mixed_code",
        category="long",
        text=(
            "#include <stdio.h>\n"
            "int main() {\n"
            "  printf(\"hello 世界 emoji 🚀\\n\");\n"
            "  return 0;\n"
            "}\n"
            "def fingerprint(tokenizer, probes):\n"
            "    return {p.id: len(tokenizer.encode(p.text).ids) for p in probes}\n"
            "const USAGE = { prompt_tokens: 12, completion_tokens: 34 };\n"
            "SELECT COUNT(*) FROM tokens WHERE vocab_size = 151643;\n"
            "print(f\"{'qwen':>10} {'deepseek':>10} {'kimi':>10}\")\n"
            "while (i < n) { sum += arr[i++]; }\n"
            "// TODO: handle multi-byte UTF-8 boundaries around 你好世界\n"
            "error: expected ';' before '}' token\n"
            "path/to/file.py::ClassName.method(arg=1, kw=[2, 3], nested={'a': True})\n"
            "regex: ^[a-z]+\\d{2,4}-[A-Z]{2}$\n"
            "base64: SGVsbG8gV29ybGQhCg==\n"
            "hex: 0xDEADBEEF 0x00 0xFF\n"
            "unicode escapes: \\u0041 \\u4e2d \\U0001F600\n"
            "whitespace stress:    tabs\t\tand\nnewlines\n\nand trailing spaces   \n"
            "mixed: The function encodeText() 将字符串转换为 token 序列，返回 list[int]。\n"
        ),
        note="多语言代码 + 转义 + 空白边界",
    ),
    Probe(
        id="long_zh_en_mix",
        category="long",
        text=(
            "Tokenizer fingerprinting works because each vendor trains a distinct vocabulary. "
            "分词器指纹检测的原理是：不同厂商训练得到的词表并不相同。 "
            "Even when two models share the same BPE merges, a third model may diverge on rare characters. "
            "即使两个模型共享相同的 BPE 合并规则，第三个模型仍可能在罕见字符上产生分歧。 "
            "We probe with: emoji 🎉🇨🇳👨‍👩‍👧‍👦, rare CJK 𠀀𪛖𫝀, ligatures ﬁ ﬂ, "
            "fullwidth ＡＢＣ１２３, halfwidth ｱｲｳ, combining é (e + ́), "
            "zero-width ​joiner, RTL עברית عربي, "
            "math ∑∫√≠≤≥, currency ¥€$₹, arrows →←↑↓, "
            "box drawing ┌─┐│└┘, braille ⠁⠂⠃, musical 𝄞, "
            "and ordinary Chinese punctuation：《》「」『』【】……——、。！？；："
            " plus ordinary English punctuation: . , ! ? ; : ' \" ( ) [ ] { } < > / \\ | @ # $ % ^ & * - _ + = ~ ` "
            "Final mixed sentence: 使用 OpenRouter 的 usage 字段可以拿到 native tokenizer 的 token 数。 "
        ),
        note="超长边界字符清单 + 中英混排",
    ),
]


# Short boundary probes: high differential power across vocabularies.
# Absolute prompt_tokens is noisy due to chat template; relative patterns still help.
BOUNDARY_PROBES: list[Probe] = [
    Probe(id="bnd_emoji_zwj", category="boundary", text="👨‍👩‍👧‍👦", note="ZWJ emoji family"),
    Probe(id="bnd_flag", category="boundary", text="🇨🇳🇯🇵🇺🇸", note="regional flags"),
    Probe(id="bnd_cjk_rare", category="boundary", text="𠀀𪛖𫝀𫠝𫡈", note="rare CJK ext"),
    Probe(id="bnd_fullwidth", category="boundary", text="ＡＢＣ１２３", note="fullwidth"),
    Probe(id="bnd_hiragana", category="boundary", text="あいうえおアイウエオ", note="kana"),
    Probe(id="bnd_hangul", category="boundary", text="한국어문자열", note="hangul"),
    Probe(id="bnd_math", category="boundary", text="∑∫√≠≤≥±×÷", note="math symbols"),
    Probe(id="bnd_currency", category="boundary", text="¥€$₹₿", note="currency"),
    Probe(id="bnd_ligature", category="boundary", text="ﬁﬂﬀ", note="ligatures"),
    Probe(id="bnd_combining", category="boundary", text="éā̈", note="combining marks"),
    Probe(id="bnd_zh_short", category="boundary", text="中华人民共和国", note="common CJK"),
    Probe(id="bnd_zh_punct", category="boundary", text="《》「」【】……——", note="CJK punct"),
    Probe(id="bnd_digits_units", category="boundary", text="151643tokens", note="digit+alpha"),
    Probe(id="bnd_camel", category="boundary", text="promptTokensCompletion", note="camelCase"),
    Probe(id="bnd_snake", category="boundary", text="prompt_tokens_completion", note="snake_case"),
    Probe(id="bnd_path", category="boundary", text="src/tokenizer_detector/cli.py", note="path"),
    Probe(id="bnd_url", category="boundary", text="https://openrouter.ai/api/v1", note="url"),
    Probe(id="bnd_spaces", category="boundary", text="a  b   c    d", note="multi spaces"),
    Probe(id="bnd_newlines", category="boundary", text="a\nb\r\nc", note="newlines"),
    Probe(id="bnd_mixed1", category="boundary", text="Hello世界123", note="script mix"),
]

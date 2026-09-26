# TokenizerDetector

Detect which **known Chinese open-source vendor tokenizer** an anonymous OpenRouter model uses.

## Idea

OpenRouter reports `usage.prompt_tokens` / `usage.completion_tokens` using the **model's native tokenizer**.  
This tool sends fixed probe texts, compares those counts against locally downloaded `tokenizer.json` files, and ranks vendor families (Qwen, DeepSeek, Kimi, GLM, MiMo, Step, MiniMax) with multi-signal confidence.

## Install

```bash
uv sync
uv run tokenizer-detector --help
```

## Usage

```bash
export OPENROUTER_API_KEY=sk-or-...

# 1) Download known tokenizers (once)
uv run tokenizer-detector fetch

# 2) Inspect the library
uv run tokenizer-detector list

# 3) Detect an anonymous model
uv run tokenizer-detector detect --model "some/anonymous" -o report.json
```

## Notes

- First version is **text-only** and **OpenRouter-only**.
- Probes defeat response caching (cache hits zero out `usage`).
- Same-vocabulary models are reported at **family/vendor** level with a shared-model list.
- `prompt_tokens` includes chat-template overhead; long probes use **cross-probe deltas** so the template offset cancels.
- Kimi ships `tiktoken.model` (not `tokenizer.json`); the tool loads it via a tiktoken backend.
- Special-token probes distinguish families that share BPE vocab but differ in added tokens:
  - `qwen` vs `mimo` (MiMo-unique `</think>` / `<|mimo_…|>`)
  - `deepseek` vs `step` (DeepSeek tool markers vs Step `<im_start>` / `<|BOT|>`)
- API key: put `OPENROUTER_API_KEY=...` in `.env` (gitignored). CLI loads it automatically.
- Free-tier OpenRouter is heavily rate-limited (~50 req/day without credits). Paid `usage` counts work on normal models when quota is available.

## Layout

```
src/tokenizer_detector/   # library + CLI
data/vendors.yaml         # family metadata + HF repos
data/tokenizers/<family>/tokenizer.json   # created by fetch
tests/
```

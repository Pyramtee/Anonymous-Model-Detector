"""Legacy v0.1 entry point, retained as an explicit migration notice."""
if __name__ == "__main__":
    raise SystemExit(
        "v0.1 script retired. Use 'uv run --extra dev tokenizer-detector fingerprint' "
        "for offline analysis, and 'uv run --extra dev pytest -q' for offline acceptance. "
        "This script never sends model requests."
    )

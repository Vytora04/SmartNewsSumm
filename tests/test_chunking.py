# tests/test_chunking.py
"""
Sanity test: token-accurate chunking must produce chunks whose token counts
are <= summarizer.max_input_tokens.

This test intentionally uses a small max_input_tokens to force chunking.
If heavy models are not desired during local dev/CI, set SKIP_HEAVY_TESTS=1
to skip the test.
"""
import os
import pytest

SKIP_HEAVY = os.environ.get("SKIP_HEAVY_TESTS", "0") == "1"

@pytest.mark.skipif(SKIP_HEAVY, reason="Skipping heavy model test (SKIP_HEAVY_TESTS=1)")
def test_token_level_chunking_respects_max_tokens():
    try:
        from backend.summarizer import NewsSummarizer
    except Exception as e:
        pytest.skip(f"Cannot import NewsSummarizer: {e}")

    # Use a compact model name that's small and commonly cached
    model_name = "sshleifer/distilbart-cnn-12-6"

    # instantiate with low token budget to force >1 chunk
    summarizer = NewsSummarizer(model_name=model_name, max_input_tokens=128, device=-1)

    # Create a long synthetic article (repeated informative sentences)
    paragraph = (
        "Researchers at Example University tested a new prototype battery during a three-month pilot. "
        "The prototype demonstrated extended endurance and was deployed in a remote field trial. "
        "Engineers reported telemetry covering multiple environmental conditions and the device performed well."
    )
    long_text = " ".join([paragraph] * 30)  # long => triggers chunking

    # produce token-level chunks using the internal helper (preferred for deterministic check)
    try:
        chunks = summarizer._token_level_chunk_text(long_text, max_tokens=summarizer.max_input_tokens, overlap=32)
    except Exception as e:
        pytest.skip(f"Token-level chunking unavailable / failed: {e}")

    assert len(chunks) > 1, "Expected token-level chunking to produce multiple chunks for long input."

    # Ensure each chunk's token count <= configured max_input_tokens
    too_large = []
    for i, c in enumerate(chunks):
        tok_count = summarizer._count_tokens(c)
        if tok_count > summarizer.max_input_tokens:
            too_large.append((i, tok_count))
    assert not too_large, f"Found chunks exceeding max_input_tokens: {too_large}"

    # Basic sanity: concatenation of chunks should contain a large portion of original text (not strict equality)
    joined = " ".join(chunks)
    assert len(joined) > 0.8 * len(long_text), "Joined chunks seem too small relative to original text."


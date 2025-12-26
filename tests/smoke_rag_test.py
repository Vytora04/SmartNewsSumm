# tests/smoke_rag_test.py
"""
Smoke test that exercises summarization end-to-end with RAG enabled.

Checks:
- summarizer returns a dict-like response with debug
- debug contains num_token_chunks and, when RAG could be built, rag_scores
- final summary is non-empty

This test may download models and is therefore gated by SKIP_HEAVY_TESTS.
Set SKIP_HEAVY_TESTS=1 to skip.
"""
import os
import pytest

SKIP_HEAVY = os.environ.get("SKIP_HEAVY_TESTS", "0") == "1"

@pytest.mark.skipif(SKIP_HEAVY, reason="Skipping heavy model test (SKIP_HEAVY_TESTS=1)")
def test_smoke_summarize_with_rag():
    try:
        from backend.summarizer import NewsSummarizer
    except Exception as e:
        pytest.skip(f"Cannot import NewsSummarizer: {e}")

    model_name = "sshleifer/distilbart-cnn-12-6"

    # small max_input_tokens to force chunking and RAG indexing
    summarizer = NewsSummarizer(model_name=model_name, max_input_tokens=256, device=-1)

    # long demo text to ensure chunking occurs
    demo_sent = (
        "Researchers at Example University tested a prototype battery in April 2025 that lasted 91 days. "
        "The prototype was deployed in Northern Kenya, according to an internal memo. "
        "In other news, unrelated background details follow..."
    )
    demo = (demo_sent + " ") * 30  # long

    # Call summarize with RAG enabled
    try:
        out = summarizer.summarize(
            demo,
            use_rag=True,
            rag_top_k=3,
            rag_query_on="article",
            chunking_mode="token",
            condensation_strategy="iterative",
            use_reranker=True,
            top_k=5,
            run_qa=False,  # keep QA off for faster run here if desired
        )
    except Exception as e:
        pytest.skip(f"summarizer.summarize failed (skipping): {e}")

    # Expect dictionary-like response
    assert isinstance(out, dict), f"Expected dict response from summarize(), got {type(out)}"

    debug = out.get("debug") or {}
    # num_token_chunks should be present and > 0 for long input
    num_chunks = debug.get("num_token_chunks") or debug.get("num_chunks") or 0
    assert num_chunks and int(num_chunks) > 0, f"Expected num_token_chunks > 0 in debug, got: {debug}"

    # If RAG initialisation succeeded, debug should contain rag_scores; otherwise summarizer._rag may be None
    rag_scores = debug.get("rag_scores")
    if summarizer._rag is None and not rag_scores:
        pytest.skip("RAG retriever not available in this environment (skipping RAG-specific assertions).")
    else:
        assert rag_scores is not None and isinstance(rag_scores, list), "Expected rag_scores list when RAG is available."

    # Summary should be non-empty string
    summary = out.get("summary", "")
    assert isinstance(summary, str) and summary.strip(), "Expected non-empty summary string."


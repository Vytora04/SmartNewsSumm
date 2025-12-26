# tests/test_length_control.py
"""
Ensures that summary length control actually works.

This test verifies:
- short < medium < long (by word count)
- summaries are not identical
"""

import os
import pytest

SKIP_HEAVY = os.environ.get("SKIP_HEAVY_TESTS", "0") == "1"

@pytest.mark.skipif(SKIP_HEAVY, reason="Skipping heavy model test")
def test_summary_length_levels():
    from backend.summarizer import NewsSummarizer

    summarizer = NewsSummarizer(
        model_name="sshleifer/distilbart-cnn-12-6",
        device=-1,
        max_input_tokens=512,
    )

    article = (
        "Researchers at Example University announced a major breakthrough in battery technology. "
        "The prototype battery was tested during a three-month pilot program conducted in Northern Kenya "
        "between April and June 2025. The device demonstrated consistent performance across varying "
        "environmental conditions including heat, humidity, and dust exposure. "
        "Experts believe this advancement could significantly impact renewable energy storage, "
        "medical devices, and remote sensing infrastructure. "
    ) * 8  # long enough to matter

    configs = {
        "short": (25, 60),
        "medium": (55, 120),
        "long": (110, 220),
    }

    results = {}

    for label, (min_len, max_len) in configs.items():
        out = summarizer.summarize(
            article,
            min_length=min_len,
            max_length=max_len,
            run_qa=False,
            use_rag=False,
            condensation_strategy="iterative",
        )
        summary = out["summary"]
        results[label] = len(summary.split())

    # Assertions
    assert results["short"] < results["medium"], f"Short >= Medium: {results}"
    assert results["medium"] < results["long"], f"Medium >= Long: {results}"

    # Ensure summaries are not identical
    assert len(set(results.values())) == 3, f"All summaries have same length: {results}"

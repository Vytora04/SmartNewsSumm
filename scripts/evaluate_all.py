# scripts/evaluate_all.py
"""
Combined evaluation for SmartNewsSumm.

Includes:
1) Extractive baselines (TF-IDF, TextRank)
2) Pure BART single-pass baseline (FORCED single generate call by truncating input)
3) Full pipeline BART (chunking + iterative condensation + reranker)
4) Optional RAG variant
5) Hybrid variant
6) LoRA variants (only if adapter exists)

Metrics:
- ROUGE-1 / ROUGE-2 / ROUGE-L
- BLEU
- BERTScore (F1)

Output:
- results/evaluate_all.json
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable

import torch
from tqdm import tqdm
from peft import PeftModel

from backend.metrics import evaluate_summary
from backend.summarizer import NewsSummarizer


# =========================
# CONFIG
# =========================
BBC_ROOT = "data/bbc-news-summary"
OUT_PATH = Path("results/evaluate_all.json")

LORA_ADAPTER = "results/lora_kfold/fold_0/adapter"  # verified path

MAX_SAMPLES = 10
MIN_LEN, MAX_LEN = 55, 120

INCLUDE_BERTSCORE = True  # set False if too slow


# -------------------------
# Full pipeline knobs ("best system mode")
# -------------------------
FULL_PIPELINE_KWARGS = dict(
    extractive_method=None,
    extractive_prefilter="none",
    hybrid_mode=False,
    use_rag=False,               # enabled in a dedicated variant
    rag_top_k=3,
    rag_query_on="article",
    use_reranker=True,
    top_k=5,
    condensation_strategy="iterative",
    chunking_mode="token",
    token_chunk_overlap=50,
    max_condensation_iters=5,
)


# =========================
# DATASET LOADER
# =========================
def load_bbc_dataset(root_dir: str, limit: Optional[int] = None) -> List[Dict]:
    root = Path(root_dir)
    news_dir = root / "news"
    summaries_dir = root / "summaries"

    if not news_dir.exists() or not summaries_dir.exists():
        raise FileNotFoundError(
            f"BBC dataset not found at: {root.resolve()}\n"
            f"Expected: {news_dir} and {summaries_dir}"
        )

    samples: List[Dict] = []
    for category_dir in news_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        for article_path in category_dir.glob("*.txt"):
            summary_path = summaries_dir / category / article_path.name
            if not summary_path.exists():
                continue

            article = article_path.read_text(encoding="utf-8", errors="ignore").strip()
            reference = summary_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not article or not reference:
                continue

            samples.append(
                {
                    "id": article_path.stem,
                    "category": category,
                    "article": article,
                    "reference": reference,
                }
            )

            if limit and len(samples) >= limit:
                return samples

    return samples


# =========================
# INPUT CONTROL HELPERS
# =========================
def truncate_to_model_max_tokens(text: str, summarizer: NewsSummarizer) -> str:
    """
    Force input <= summarizer.max_input_tokens so NewsSummarizer MUST take single-pass path.
    This is what makes "Pure BART single-pass" actually pure.
    """
    if not text or not text.strip():
        return ""

    max_toks = int(getattr(summarizer, "max_input_tokens", 512))
    tok = summarizer.tokenizer

    # Tokenize + truncate, then decode back to text
    enc = tok(text, truncation=True, max_length=max_toks, return_tensors=None)
    ids = enc.get("input_ids")
    if not ids:
        return text

    # ids is List[int] in non-tensor mode
    if isinstance(ids[0], list):
        ids = ids[0]
    return tok.decode(ids, skip_special_tokens=True).strip()


# =========================
# EVALUATION HELPER
# =========================
@torch.no_grad()
def evaluate_variant(
    summarizer: NewsSummarizer,
    samples: List[Dict],
    label: str,
    preprocess: Optional[Callable[[str, NewsSummarizer], str]] = None,
    **kwargs,
) -> Dict[str, float]:
    metrics_list: List[Dict[str, float]] = []

    for s in tqdm(samples, desc=f"{label:34}"):
        article = s["article"]
        if preprocess is not None:
            article = preprocess(article, summarizer)

        out = summarizer.summarize(
            article,
            min_length=MIN_LEN,
            max_length=MAX_LEN,
            run_qa=False,  # keep eval stable + faster
            **kwargs,
        )
        summary = out["summary"] if isinstance(out, dict) else str(out)

        m = evaluate_summary(
            generated=summary,
            reference=s["reference"],
            include_bertscore=INCLUDE_BERTSCORE,
        )
        metrics_list.append(m)

    if not metrics_list:
        raise RuntimeError("No metrics computed. Check dataset loading and evaluation loop.")

    avg = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}

    print(f"\n→ {label}")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")

    return avg


# =========================
# MAIN
# =========================
def main():
    hf_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {'CUDA' if hf_device == 0 else 'CPU'}")

    samples = load_bbc_dataset(BBC_ROOT, limit=MAX_SAMPLES)
    print(f"Loaded {len(samples)} BBC samples from {BBC_ROOT}")

    results: Dict[str, Dict[str, float]] = {}

    # -------------------------
    # 1) Extractive baselines
    # -------------------------
    summarizer_extractive = NewsSummarizer(
        model_name="sshleifer/distilbart-cnn-12-6",
        device=hf_device,
    )

    results["Simple Baseline (TF-IDF)"] = evaluate_variant(
        summarizer_extractive,
        samples,
        "Simple Baseline (TF-IDF)",
        extractive_method="tfidf",
    )

    # results["Stronger Baseline (TextRank)"] = evaluate_variant(
    #     summarizer_extractive,
    #     samples,
    #     "Stronger Baseline (TextRank)",
    #     extractive_method="textrank",
    # )

    # -------------------------
    # 2) Pure BART single-pass baseline
    #    ✅ forced single-pass by truncating input first
    # -------------------------
    summarizer_pure = NewsSummarizer(
        model_name="sshleifer/distilbart-cnn-12-6",
        device=hf_device,
    )

    results["BART (Pure, single-pass)"] = evaluate_variant(
        summarizer_pure,
        samples,
        "BART (Pure, single-pass)",
        preprocess=truncate_to_model_max_tokens,
        extractive_method=None,
        extractive_prefilter="none",
        hybrid_mode=False,
        use_rag=False,
        use_reranker=False,
        # condensation_strategy irrelevant here because truncation forces single-pass path
        condensation_strategy="iterative",
        chunking_mode="token",
    )

    # -------------------------
    # 3) Full pipeline (no RAG)
    # -------------------------
    summarizer_full = NewsSummarizer(
        model_name="sshleifer/distilbart-cnn-12-6",
        device=hf_device,
    )

    results["BART (Full Pipeline, no RAG)"] = evaluate_variant(
        summarizer_full,
        samples,
        "BART (Full Pipeline, no RAG)",
        **FULL_PIPELINE_KWARGS,
    )

    # -------------------------
    # 4) Full pipeline + RAG
    # -------------------------
    summarizer_full_rag = NewsSummarizer(
        model_name="sshleifer/distilbart-cnn-12-6",
        device=hf_device,
    )

    results["BART (Full Pipeline + RAG)"] = evaluate_variant(
        summarizer_full_rag,
        samples,
        "BART (Full Pipeline + RAG)",
        **{**FULL_PIPELINE_KWARGS, "use_rag": True},
    )

    # -------------------------
    # 5) Full pipeline + Hybrid
    # -------------------------
    summarizer_full_hybrid = NewsSummarizer(
        model_name="sshleifer/distilbart-cnn-12-6",
        device=hf_device,
    )

    results["BART (Full Pipeline + Hybrid)"] = evaluate_variant(
        summarizer_full_hybrid,
        samples,
        "BART (Full Pipeline + Hybrid)",
        **{
            **FULL_PIPELINE_KWARGS,
            "hybrid_mode": True,
            "hybrid_extractive_ratio": 0.3,
        },
    )

    # -------------------------
    # 6) LoRA variants (if adapter exists)
    # -------------------------
    adapter_path = Path(LORA_ADAPTER)
    if adapter_path.exists():
        # Full pipeline + LoRA
        summarizer_full_lora = NewsSummarizer(
            model_name="sshleifer/distilbart-cnn-12-6",
            device=hf_device,
        )
        summarizer_full_lora.model = PeftModel.from_pretrained(
            summarizer_full_lora.model,
            str(adapter_path),
        ).merge_and_unload()
        summarizer_full_lora.model.to(torch_device)
        summarizer_full_lora.model.eval()

        results["BART (Full Pipeline + LoRA)"] = evaluate_variant(
            summarizer_full_lora,
            samples,
            "BART (Full Pipeline + LoRA)",
            **FULL_PIPELINE_KWARGS,
        )

        # Full pipeline + Hybrid + LoRA
        summarizer_full_hybrid_lora = NewsSummarizer(
            model_name="sshleifer/distilbart-cnn-12-6",
            device=hf_device,
        )
        summarizer_full_hybrid_lora.model = PeftModel.from_pretrained(
            summarizer_full_hybrid_lora.model,
            str(adapter_path),
        ).merge_and_unload()
        summarizer_full_hybrid_lora.model.to(torch_device)
        summarizer_full_hybrid_lora.model.eval()

        results["BART (Full Pipeline + Hybrid + LoRA)"] = evaluate_variant(
            summarizer_full_hybrid_lora,
            samples,
            "BART (Full Pipeline + Hybrid + LoRA)",
            **{
                **FULL_PIPELINE_KWARGS,
                "hybrid_mode": True,
                "hybrid_extractive_ratio": 0.3,
            },
        )
    else:
        print(f"\n[WARN] LoRA adapter not found at: {adapter_path}. Skipping LoRA variants.")

    # -------------------------
    # SAVE RESULTS
    # -------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS SUMMARY ===")
    for model, metrics in results.items():
        print(f"\n{model}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print(f"\nSaved to: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

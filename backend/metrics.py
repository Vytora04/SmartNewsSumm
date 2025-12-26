"""
Evaluation metrics for summarization quality (used only for evaluation on scripts/evaluate.py).

Provides:
- ROUGE (rouge1, rouge2, rougeL)
- BLEU
- BERTScore
"""

from typing import Dict, List, Union
import warnings

try:
    from rouge_score import rouge_scorer  
    _HAS_ROUGE = True
except Exception:
    _HAS_ROUGE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction 
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    _HAS_BLEU = True
except Exception:
    _HAS_BLEU = False

try:
    import bert_score 
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False


def calculate_rouge(generated: str, reference: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    Returns dictionary with keys: rouge1, rouge2, rougeL (F1 scores)
    """
    if not _HAS_ROUGE:
        warnings.warn("rouge-score not installed; returning zeroes for ROUGE.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return {
            "rouge1": float(scores['rouge1'].fmeasure),
            "rouge2": float(scores['rouge2'].fmeasure),
            "rougeL": float(scores['rougeL'].fmeasure),
        }
    except Exception as e:
        warnings.warn(f"ROUGE calculation failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def calculate_bleu(generated: str, reference: str) -> float:
    """
    Calculate a sentence-level BLEU score (0-1).
    If NLTK not available, returns 0.0 and warns.
    """
    if not _HAS_BLEU:
        warnings.warn("nltk or punkt not installed; returning BLEU=0.0.")
        return 0.0

    try:
        from nltk.tokenize import word_tokenize  # type: ignore
        ref_tokens = [word_tokenize(reference.lower())]
        gen_tokens = word_tokenize(generated.lower())
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing)
        return float(score)
    except Exception as e:
        warnings.warn(f"BLEU calculation failed: {e}")
        return 0.0


def calculate_bertscore(
    generated: Union[str, List[str]],
    reference: Union[str, List[str]],
    lang: str = "en",
    model_type: str = "microsoft/deberta-base-mnli",
) -> Dict[str, float]:
    """
    Calculate BERTScore (precision / recall / f1). Returns mean values in a dict.
    If bert-score is unavailable returns zeros.
    """
    if not _HAS_BERTSCORE:
        warnings.warn("bert-score not installed; returning zeros for BERTScore.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        if isinstance(generated, str):
            generated = [generated]
        if isinstance(reference, str):
            reference = [reference]

        P, R, F1 = bert_score.score(
            generated,
            reference,
            lang=lang,
            model_type=model_type,
            verbose=False,
        )
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item()),
        }
    except Exception as e:
        warnings.warn(f"BERTScore calculation failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def evaluate_summary(
    generated: str,
    reference: str,
    include_bertscore: bool = False,
    **kwargs,
) -> Dict[str, float]:
    """
    Comprehensive evaluation wrapper.

    Args:
      generated: generated summary (string)
      reference: reference/gold summary (string)
      include_bertscore: whether to compute BERTScore (may be slow)
      **kwargs: accepted and ignored (keeps backward compatibility with external callers/tests)

    Returns:
      Dictionary with keys: rouge1, rouge2, rougeL, bleu, optionally bertscore_precision/recall/f1
    """
    results: Dict[str, float] = {}

    # ROUGE
    rouge_scores = calculate_rouge(generated, reference)
    results.update(rouge_scores)

    # BLEU
    results["bleu"] = calculate_bleu(generated, reference)

    # Optional BERTScore
    if include_bertscore:
        bert_scores = calculate_bertscore(generated, reference)
        results["bertscore_precision"] = bert_scores["precision"]
        results["bertscore_recall"] = bert_scores["recall"]
        results["bertscore_f1"] = bert_scores["f1"]

    return results
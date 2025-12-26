# backend/preprocess.py
"""
Plain-text preprocessing utilities for News Summarizer (Batch 2).

Simplified: we assume user input is plain article text (not full HTML pages).
Features:
- normalize_whitespace(text)
- keep_long_paragraphs(text, min_words)
- split_into_sentences(text)
- preprocess_for_model(raw_input, return_sentences=False, min_para_words=3)
  - returns cleaned text and optional sentence list
- batch_preprocess_texts(texts)
"""

from typing import List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Sentence splitting (spaCy preferred, NLTK fallback)
try:
    import spacy
    _HAS_SPACY = True
    try:
        # prefer blank pipeline with sentencizer to minimize model weight requirements
        _SPACY_NLP = spacy.blank("en")
        if "sentencizer" not in _SPACY_NLP.pipe_names:
            _SPACY_NLP.add_pipe("sentencizer")
    except Exception:
        _SPACY_NLP = None
except Exception:
    _HAS_SPACY = False
    _SPACY_NLP = None

try:
    import nltk
    _HAS_NLTK = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except Exception:
    _HAS_NLTK = False


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and newlines; collapse long sequences."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")
    # collapse 3+ newlines into two (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    # trim
    return text.strip()


def keep_long_paragraphs(text: str, min_words: int = 5) -> str:
    """
    Keep paragraphs that have at least `min_words` words.
    This removes small noise lines while preserving substantive content.
    """
    if not text:
        return ""
    paras = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    if not paras:
        return ""
    # If total content is small, keep everything
    total_words = sum(len(p.split()) for p in paras)
    if total_words <= (min_words * 2):
        return "\n\n".join(paras)
    filtered = [p for p in paras if len(p.split()) >= min_words]
    if not filtered:
        # if filtering would remove everything, return original cleaned text
        return "\n\n".join(paras)
    return "\n\n".join(filtered)


def split_into_sentences(text: str) -> List[str]:
    """
    Sentence segmentation using spaCy (preferred) or NLTK fallback or naive regex.
    """
    if not text:
        return []
    text = text.strip()

    # spaCy (lightweight sentencizer)
    if _HAS_SPACY and _SPACY_NLP is not None:
        try:
            doc = _SPACY_NLP(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception as e:
            logger.debug(f"spaCy sentencizer failed: {e}")

    # NLTK
    if _HAS_NLTK:
        try:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(text)
            return [s.strip() for s in sents if s.strip()]
        except Exception as e:
            logger.debug(f"NLTK sentence split failed: {e}")

    # Fallback regex
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def preprocess_for_model(
    raw_input: str,
    min_para_words: int = 3,
    return_sentences: bool = False,
    sentence_split: bool = True,
) -> Tuple[str, Optional[List[str]]]:
    """
    Preprocess assumed plain-text input for model consumption.

    Steps:
      - normalize whitespace
      - remove trivial lines / small nav-like lines
      - keep paragraphs with at least `min_para_words` words (but preserve small docs)
      - optionally return sentence list

    Returns:
      (cleaned_text, sentences or None)
    """
    if not raw_input:
        return ("", [] if return_sentences else None)

    text = raw_input

    # Normalize whitespace
    text = normalize_whitespace(text)

    # Remove trivial short lines ("Read more", "Follow", copyright lines, etc)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned_lines = []
    for ln in lines:
        lower = ln.lower()
        # heuristics to drop short nav-like lines
        if len(ln.split()) < 4 and (lower.startswith("read") or lower.startswith("follow") or "copyright" in lower or "subscribe" in lower):
            continue
        cleaned_lines.append(ln)
    text = "\n\n".join(cleaned_lines)

    # Keep long paragraphs (filter very short paras)
    text = keep_long_paragraphs(text, min_words=min_para_words)
    text = normalize_whitespace(text)

    sentences = None
    if return_sentences or sentence_split:
        sentences = split_into_sentences(text) if sentence_split else None

    return (text, sentences) if return_sentences else (text, None)





# Minimal self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = (
        "First line.\n\nRead more at example.com\n\n"
        "Second paragraph with more info. This should be kept.\n\n"
        "Footer copyright 2025"
    )
    clean, sents = preprocess_for_model(sample, return_sentences=True)
    print("CLEANED:", clean)
    print("SENTS:", sents)

# backend/extractive.py
"""
Extractive summarization baselines for comparison.

Methods:
- TF-IDF based extraction
- TextRank (graph-based) extraction
"""

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import re

try:
    import spacy
    _HAS_SPACY = True
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    except Exception:
        _SPACY_NLP = spacy.blank("en")
        _SPACY_NLP.add_pipe("sentencizer")
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


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy or NLTK."""
    if not text:
        return []
    
    if _HAS_SPACY and _SPACY_NLP is not None:
        try:
            doc = _SPACY_NLP(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception:
            pass
    
    if _HAS_NLTK:
        try:
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    
    # Fallback: simple regex
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def extractive_tfidf(text: str, num_sentences: int = 3) -> str:
    """
    TF-IDF based extractive summarization.
    
    Args:
        text: Input article text
        num_sentences: Number of sentences to extract
    
    Returns:
        Summary containing top-ranked sentences
    """
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return text
    
    # Create TF-IDF matrix
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores (sum of TF-IDF values)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get top N sentences (preserve original order)
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_indices_sorted = sorted(top_indices)
        
        summary_sentences = [sentences[i] for i in top_indices_sorted]
        return " ".join(summary_sentences)
    
    except Exception as e:
        # Fallback: return first N sentences
        return " ".join(sentences[:num_sentences])


def extractive_textrank(text: str, num_sentences: int = 3, damping: float = 0.85) -> str:
    """
    TextRank (graph-based) extractive summarization.
    
    Args:
        text: Input article text
        num_sentences: Number of sentences to extract
        damping: Damping factor for PageRank (default 0.85)
    
    Returns:
        Summary containing top-ranked sentences
    """
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return text
    
    try:
        # Create TF-IDF vectors for similarity calculation
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity matrix
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        # Create graph
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        # Calculate PageRank scores
        scores = nx.pagerank(nx_graph, alpha=damping)
        
        # Get top N sentences (preserve original order)
        ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
        top_indices = sorted([i for _, i in ranked_sentences[:num_sentences]])
        
        summary_sentences = [sentences[i] for i in top_indices]
        return " ".join(summary_sentences)
    
    except Exception as e:
        # Fallback to TF-IDF if TextRank fails
        return extractive_tfidf(text, num_sentences)


def extractive_lead(text: str, num_sentences: int = 3) -> str:
    """
    Lead baseline: Simply take the first N sentences.
    
    Args:
        text: Input article text
        num_sentences: Number of sentences to extract
    
    Returns:
        First N sentences
    """
    sentences = split_sentences(text)
    return " ".join(sentences[:num_sentences])

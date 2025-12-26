# backend/reranker.py
"""
SBERT-based reranker for chunk (or chunk-summary) selection.

Improvements for Batch 2:
- Batched encoding that honors the `device` parameter.
- Exposes `score_candidates` (returns (idx, score) list) and `select_top_k` (returns (idx, score, text) tuples).
- Deterministic tie-breaking: sort by (score desc, index asc).
- Robust logging and defensive behavior for empty inputs.

Dependencies:
- sentence-transformers (SentenceTransformer, util)
- numpy
- torch (available implicitly via sentence-transformers)
"""

from typing import List, Tuple, Optional
import logging

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    raise ImportError("sentence-transformers is required for reranker.py (pip install sentence-transformers)") from e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SBERTReranker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None, encode_batch_size: int = 64):
        """
        Args:
            model_name: SentenceTransformer model id.
            device: 'cpu' or 'cuda' or None (None -> let SentenceTransformer decide).
            encode_batch_size: batch size for encoding texts (controls memory/perf tradeoff).
        """
        self.model_name = model_name
        self.device = device
        self.encode_batch_size = encode_batch_size or 64

        logger.info(f"Initializing SBERTReranker(model_name={model_name}, device={device}, batch_size={self.encode_batch_size})")
        try:
            # SentenceTransformer accepts device argument (e.g., 'cuda' or 'cpu' or None)
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logger.exception(f"Failed to load SentenceTransformer '{model_name}': {e}")
            raise

    # --------------------
    # Encoding helpers
    # --------------------
    def encode(self, texts: List[str], batch_size: Optional[int] = None, convert_to_numpy: bool = True):
        if not texts:
            # Always return a consistent empty object
            if convert_to_numpy:
                return np.zeros((0, 384), dtype=np.float32)  # 384 is MiniLM default; safe placeholder
            else:
                import torch
                return torch.empty((0, 384))

        bsize = batch_size or self.encode_batch_size
        try:
            embs = self.model.encode(
                texts,
                batch_size=bsize,
                convert_to_tensor=not convert_to_numpy,
                convert_to_numpy=convert_to_numpy,
                show_progress_bar=False,
            )
            return embs
        except Exception as e:
            logger.exception(f"Encoding failed for {len(texts)} texts: {e}")
            if convert_to_numpy:
                return np.zeros((0, 384), dtype=np.float32)
            else:
                import torch
                return torch.empty((0, 384))


    # --------------------
    # Scoring helpers
    # --------------------
    def score_candidates(self, reference: str, candidates: List[str], normalize: bool = True) -> List[Tuple[int, float]]:
        """
        Compute cosine similarity scores between a reference text and each candidate.

        Args:
            reference: reference string (e.g., full article or query)
            candidates: list of candidate strings
            normalize: if True, ensure scores in [-1,1] and convert to float

        Returns:
            List of tuples (original_index, score) in the original order.
        """
        if not candidates:
            return []

        try:
            # Encode reference and candidates
            ref_emb = self.encode([reference], convert_to_numpy=False)
            cand_embs = self.encode(candidates, convert_to_numpy=False)

            if ref_emb is None or cand_embs is None:
                logger.warning("Embedding returned None; returning zero scores.")
                return [(i, 0.0) for i in range(len(candidates))]

            # util.cos_sim returns a torch tensor
            sims = util.cos_sim(ref_emb, cand_embs)

            # Ensure sims is 2D: (1, N)
            if sims.dim() == 1:
                sims = sims.unsqueeze(0)

            # Convert safely
            sims_np = sims.detach().cpu().numpy().reshape(-1)

            # normalize safety: clip to [-1,1]
            if normalize:
                sims_np = np.clip(sims_np.astype(float), -1.0, 1.0)

            scores = [(int(i), float(s)) for i, s in enumerate(sims_np)]
            return scores
        except Exception as e:
            logger.exception(f"Scoring candidates failed: {e}")
            # Fallback: zero scores
            return [(i, 0.0) for i in range(len(candidates))]

    

    def select_top_k(self, reference: str, candidates: List[str], k: int = 3, min_score: Optional[float] = None) -> List[Tuple[int, float, str]]:
        """
        Rank candidate texts by cosine similarity to reference and return top-k list of tuples:
        (original_index, score, candidate_text)

        Deterministic tie-breaking:
          - primary: descending score
          - secondary: ascending original index

        Args:
            reference: text to compare against
            candidates: list of candidate texts
            k: number to return
            min_score: optional threshold; candidates with score < min_score are filtered out

        Returns:
            list of (index, score, text) tuples ordered by descending score
        """
        if not candidates:
            return []

        if len(candidates) == 1:
            scores = self.score_candidates(reference, candidates)
            score = scores[0][1] if scores else 0.0
            return [(0, float(score), candidates[0])]
    
        scores = self.score_candidates(reference, candidates)
        # Filter by min_score if provided
        if min_score is not None:
            scores = [(i, s) for (i, s) in scores if s >= float(min_score)]

        # Create list of (score, index) for sorting. Tie-break by index asc.
        # Sorting key: (-score, index)
        # Keep mapping to original text
        scored_with_text = [(i, s, candidates[i]) for (i, s) in scores]

        # Defensive: if no scored items after filtering, return empty
        if not scored_with_text:
            logger.debug("No candidates passed min_score filter (or no candidates at all).")
            return []

        # Sort deterministically: by score desc, index asc
        scored_with_text.sort(key=lambda tup: (-tup[1], tup[0]))

        topk = scored_with_text[:min(k, len(scored_with_text))]
        logger.info(f"select_top_k: chosen indices {[t[0] for t in topk]} with scores {[t[1] for t in topk]}")
        return [(int(idx), float(score), text) for (idx, score, text) in topk]

    # --------------------
    # Convenience / utilities
    # --------------------
    def batch_score_topk(self, references: List[str], candidates_list: List[List[str]], k: int = 3) -> List[List[Tuple[int, float, str]]]:
        """
        Score multiple reference/candidate groups in a batchy fashion.

        Args:
            references: list of reference strings (len = N)
            candidates_list: list of candidate-lists (len = N)
            k: top-k to return per group

        Returns:
            list (len N) of lists of (idx, score, text) tuples
        """
        results = []
        for ref, cands in zip(references, candidates_list):
            results.append(self.select_top_k(ref, cands, k=k))
        return results

    def __repr__(self):
        return f"SBERTReranker(model_name={self.model_name}, device={self.device}, batch_size={self.encode_batch_size})"

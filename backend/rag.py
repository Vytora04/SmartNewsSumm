# backend/rag.py
"""
Retriever abstraction for RAG-style retrieval over article passages.

Features:
- Build an index over passages (sentence-level windows recommended).
- Uses sentence-transformers to create embeddings.
- Uses FAISS when available, otherwise sklearn NearestNeighbors / brute-force numpy fallback.
- API:
    r = Retriever(model_name="all-MiniLM-L6-v2", device="cpu")
    r.build_index(passages)           # passages: List[str]
    results = r.search("query", top_k=5)  # returns list of (idx, score, passage)
"""

from typing import List, Tuple, Optional, Any, Dict
import logging
import numpy as np

logger = logging.getLogger("backend.rag")
logger.setLevel(logging.INFO)

# Attempt to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBT = True
except Exception as e:
    logger.warning(f"sentence-transformers import failed: {e}")
    _HAS_SBT = False

# Try FAISS
_HAS_FAISS = False
_faiss = None
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
    _faiss = faiss
except Exception:
    _HAS_FAISS = False

# sklearn fallback
_HAS_SK = False
try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    _HAS_SK = True
except Exception:
    _HAS_SK = False


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Args:
            model_name: sentence-transformers model id
            device: 'cpu' or 'cuda' (or None to autodetect inside SentenceTransformer)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.passages: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[Any] = None
        self.index_type: Optional[str] = None

        if not _HAS_SBT:
            logger.warning("sentence-transformers not available. Retriever will be disabled.")
            return

        # initialize model lazily (don't download until used)
        try:
            logger.info(f"Initializing SentenceTransformer '{model_name}' on device={device}")
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer '{model_name}': {e}")
            self.model = None

    # -----------------------
    # Index building
    # -----------------------
    def build_index(self, passages: List[str], use_faiss: bool = True, normalize: bool = True) -> None:
        """
        Build embedding index for the given passages.

        Args:
            passages: list of text passages to index
            use_faiss: prefer faiss if available
            normalize: whether to l2-normalize embeddings (recommended for cosine sim)
        """
        if not passages:
            self.passages = []
            self.embeddings = None
            self.index = None
            return

        if self.model is None:
            logger.warning("Retriever model unavailable; cannot build index.")
            self.passages = passages
            self.embeddings = None
            self.index = None
            return

        logger.info(f"Computing embeddings for {len(passages)} passages...")
        try:
            embs = self.model.encode(passages, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            self.passages = passages
            self.embeddings = None
            self.index = None
            return

        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            embs = embs / norms

        self.passages = passages
        self.embeddings = embs.astype(np.float32)

        # Build index
        if _HAS_FAISS and use_faiss:
            try:
                d = self.embeddings.shape[1]
                # IndexFlatIP assumes normalized vectors -> inner product equals cosine
                index = _faiss.IndexFlatIP(d)
                index.add(self.embeddings)
                self.index = index
                self.index_type = "faiss"
                logger.info("FAISS index built (IndexFlatIP).")
                return
            except Exception as e:
                logger.warning(f"FAISS index build failed: {e}")

        # sklearn fallback (brute force / kd-tree depending on availability)
        if _HAS_SK:
            try:
                # We keep the embeddings and use NearestNeighbors for search
                # use metric='cosine' (which computes 1 - cosine_similarity)
                nn = NearestNeighbors(n_neighbors=min(10, len(passages)), metric="cosine", algorithm="brute")
                nn.fit(self.embeddings)
                self.index = nn
                self.index_type = "sklearn"
                logger.info("Sklearn NearestNeighbors index built (cosine, brute).")
                return
            except Exception as e:
                logger.warning(f"Sklearn index build failed: {e}")

        # Last resort: no index; keep embeddings and do brute-force numpy
        self.index = None
        self.index_type = "bruteforce"
        logger.info("No vector index built; using brute-force search on stored embeddings.")

    # -----------------------
    # Searching
    # -----------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Search for top_k passages for the query.

        Returns list of tuples: (index, score, passage) where score is cosine-similarity (0..1 approx).
        """
        if not self.passages:
            logger.debug("No passages indexed - returning empty search.")
            return []
        
        top_k = min(int(top_k), len(self.passages))

        if self.model is None:
            logger.warning("Retriever model unavailable - returning empty results.")
            return []

        try:
            q_emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        # normalize for cosine if embeddings created normalized
        if self.embeddings is not None:
            # ensure shapes
            if self.embeddings.ndim == 2:
                q_norm = np.linalg.norm(q_emb)
                if q_norm == 0:
                    q_norm = 1.0
                q_emb_n = q_emb / q_norm
            else:
                q_emb_n = q_emb
        else:
            q_emb_n = q_emb

        results: List[Tuple[int, float, str]] = []
        try:
            if self.index_type == "faiss" and _HAS_FAISS and self.index is not None:
                # we assume embeddings were normalized => use inner product
                D, I = self.index.search(np.expand_dims(q_emb_n, axis=0), top_k)
                # D are inner products
                for score, idx in zip(D[0], I[0]):
                    if idx < 0 or idx >= len(self.passages):
                        continue
                    results.append((int(idx), float(score), self.passages[int(idx)]))
                return results

            if self.index_type == "sklearn" and _HAS_SK and self.index is not None:
                # sklearn returns distances (cosine), convert to similarity = 1 - dist
                distances, indices = self.index.kneighbors(np.expand_dims(q_emb_n, axis=0), n_neighbors=min(top_k, len(self.passages)))
                for dist, idx in zip(distances[0], indices[0]):
                    score = 1.0 - float(dist)
                    results.append((int(idx), score, self.passages[int(idx)]))
                return results

            # brute-force numpy similarity
            if self.embeddings is not None:
                sims = np.dot(self.embeddings, q_emb_n)
                # If embeddings are not normalized, normalize by norms (safe fallback)
                if sims.max() > 1.1 or sims.min() < -1.1:
                    # compute cosine
                    emb_norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(q_emb_n) + 1e-12)
                    sims = np.dot(self.embeddings, q_emb_n) / (emb_norms + 1e-12)
                # get top_k
                order = np.argsort(-sims)[:top_k]
                for idx in order:
                    results.append((int(idx), float(sims[int(idx)]), self.passages[int(idx)]))
                return results
        except Exception as e:
            logger.warning(f"Indexed search failed: {e}")

        # ultimate fallback: naive substring match scoring
        logger.info("Falling back to substring-similarity scoring.")
        q_low = (query or "").lower().strip()
        scores = []
        for i, p in enumerate(self.passages):
            p_low = (p or "").lower()
            score = float(p_low.count(q_low)) if q_low else 0.0
            scores.append((i, score, p))

        scores_sorted = sorted(scores, key=lambda x: -x[1])[:top_k]
        max_score = max((s for _, s, _ in scores_sorted), default=1.0)
        return [(i, (s / max_score) if max_score else 0.0, p) for i, s, p in scores_sorted]



# -----------------------
# Utilities for passage grouping
# -----------------------
def make_sentence_windows(sentences: List[str], window_size: int = 1, stride: int = 1) -> List[str]:
    """
    Group sentences into overlapping windows for indexing.
    Example: sentences = [s0, s1, s2, s3], window_size=2, stride=1 =>
      [s0+s1, s1+s2, s2+s3]
    """
    if not sentences:
        return []
    if window_size <= 1:
        return [s.strip() for s in sentences if s.strip()]
    windows = []
    n = len(sentences)
    for start in range(0, n, stride):
        end = start + window_size
        if start >= n:
            break
        w = " ".join(sentences[start:min(end, n)])
        if w.strip():
            windows.append(w.strip())
        if end >= n:
            break
    return windows

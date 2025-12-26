# backend/summarizer.py
"""
SmartNewsSumm - Core summarization engine.

What this module supports:
- Abstractive summarization with HF Seq2Seq models (DistilBART/BART/T5/Pegasus).
- Long-document handling:
  - Token-accurate chunking (fast tokenizer offsets)
  - Map-Reduce style fusion (iterative condensation)
  - Optional SBERT reranking of chunk summaries
  - Optional RAG (sentence-window indexing + retrieval)
- Hybrid mode (extract-then-abstract):
  - Uses a TextRank-style sentence scoring step to pre-filter the input text.
- Optional extractive-only mode:
  - tfidf / textrank / lead (bypasses transformer generation)

Important design decisions:
- UI typically thinks in "words" for summary length sliders.
  HF generation uses "tokens" for min_length/max_length.
  This implementation:
    1) Converts requested word bounds -> approximate token bounds for generation.
    2) Enforces the final summary to the requested *word* bounds deterministically.

LoRA:
- LoRA is intentionally NOT auto-loaded here to avoid "sticky" adapters across requests.
- Apply LoRA at the API layer (backend/app.py) per request.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
import logging
import math
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase

from .extractive import (
    extractive_tfidf,
    extractive_textrank,
    extractive_lead,
    split_sentences,
)

from .reranker import SBERTReranker
from .qa_checker import check_summary_against_source
from .rag import Retriever, make_sentence_windows

# sentence splitting fallback libs (optional)
try:
    import spacy  # type: ignore
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

try:
    import nltk  # type: ignore
    _HAS_NLTK = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except Exception:
    _HAS_NLTK = False

logger = logging.getLogger("backend.summarizer")
logger.setLevel(logging.INFO)

# Lazy, cached sentencizer (spacy) to avoid recreating each call
_SPACY_SENTENCIZER = None


class NewsSummarizer:
    """
    Abstractive summarizer with long-document pipeline and optional extractive modes.
    """

    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "sshleifer/distilbart-cnn-12-6": {"min_length": 40, "max_length": 160, "length_penalty": 1.0},
        "facebook/bart-large-cnn": {"min_length": 55, "max_length": 200, "length_penalty": 1.0},
        "google/flan-t5-base": {"min_length": 50, "max_length": 150, "length_penalty": 1.0},
        "google/pegasus-cnn_dailymail": {"min_length": 64, "max_length": 128, "length_penalty": 0.8},
    }

    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        device: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        chunk_overlap_tokens: int = 50,
        sentence_window_for_rag: int = 2,
    ):
        self.model_name = model_name
        self.chunk_overlap_tokens = int(chunk_overlap_tokens)
        self.sentence_window_for_rag = int(sentence_window_for_rag)

        # Device selection
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = int(device)

        logger.info(f"Loading tokenizer & model: {model_name} (device={self.device})")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Determine usable max input tokens based on tokenizer.model_max_length
        tm = getattr(self.tokenizer, "model_max_length", None)
        if tm is None or tm <= 0 or tm > 1_000_000_000:
            tm = 1024
        safety_margin = 8
        default_max_input = max(64, int(tm) - safety_margin)
        if max_input_tokens is None:
            self.max_input_tokens = default_max_input
        else:
            self.max_input_tokens = int(min(int(max_input_tokens), default_max_input))

        logger.info(f"Tokenizer.model_max_length={tm}; using max_input_tokens={self.max_input_tokens}")

        # Move model to target device once
        self._model_device = self._resolve_torch_device(self.device)
        try:
            self.model.to(self._model_device)
        except Exception as e:
            logger.warning(f"Could not move model to {self._model_device}; falling back to CPU. Error: {e}")
            self._model_device = torch.device("cpu")
            self.model.to(self._model_device)

        self.model.eval()

        # Reranker (SBERT) - CPU by default
        try:
            self.reranker = SBERTReranker(model_name="all-MiniLM-L6-v2", device="cpu")
        except Exception as e:
            logger.warning(f"Could not initialize SBERTReranker: {e}")
            self.reranker = None

        # RAG retriever (lazy init)
        self._rag: Optional[Retriever] = None
        self.rag_model_name = "all-MiniLM-L6-v2"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def summarize(
        self,
        text: str,
        *,
        extractive_method: Optional[str] = None,
        hybrid_mode: bool = False,
        hybrid_extractive_ratio: float = 0.3,
        min_length: Optional[int] = None,   # interpreted as "word target" from UI (legacy name)
        max_length: Optional[int] = None,   # interpreted as "word target" from UI (legacy name)
        min_words: Optional[int] = None,    # preferred explicit naming
        max_words: Optional[int] = None,    # preferred explicit naming
        num_beams: int = 4,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        extractive_prefilter: str = "none",
        use_reranker: bool = True,
        top_k: int = 5,
        run_qa: bool = True,
        use_rag: bool = False,
        rag_top_k: int = 3,
        rag_query_on: str = "article",
        condensation_strategy: str = "iterative",
        chunking_mode: str = "token",
        token_chunk_overlap: Optional[int] = None,
        max_condensation_iters: int = 5,
    ) -> Dict[str, Any]:
        t0 = time.time()

        debug: Dict[str, Any] = {
            "token_count": None,
            "num_token_chunks": 0,
            "num_generate_calls": 0,
            "reranker_scores": None,
            "rag_scores": None,
            "method": None,
            "hybrid_filtered_sentences": None,
        }

        if not text or not text.strip():
            return {"summary": "", "qa_report": None, "debug": debug}

        # -------------------------
        # ✅ Word-limit normalization
        # Treat min_length/max_length as word bounds (UI sliders),
        # but allow explicit min_words/max_words override.
        # -------------------------
        if min_words is None:
            min_words = min_length
        if max_words is None:
            max_words = max_length

        # Convert requested word bounds -> approximate token bounds for generation
        gen_min_len, gen_max_len, gen_len_penalty = self._resolve_generation_lengths(
            min_words=min_words,
            max_words=max_words,
            length_penalty=length_penalty,
        )

        # --------------------------------------------------
        # 1) Hybrid mode: pre-filter input text by sentence scoring
        # --------------------------------------------------
        if hybrid_mode:
            try:
                filtered_text, kept, total = self._hybrid_filter_textrank(
                    text,
                    keep_ratio=float(hybrid_extractive_ratio),
                    min_keep=3,
                )
                if kept is not None and total is not None:
                    debug["hybrid_filtered_sentences"] = f"{kept}/{total}"
                text = filtered_text
                logger.info(f"Hybrid mode applied: kept {debug['hybrid_filtered_sentences']} sentences")
            except Exception as e:
                logger.warning(f"Hybrid filtering failed: {e}. Using full text.")

        # --------------------------------------------------
        # 2) Extractive-only fast path
        # --------------------------------------------------
        if extractive_method:
            method = str(extractive_method).lower().strip()
            try:
                if method == "tfidf":
                    summ = extractive_tfidf(text)
                    summ = self._enforce_word_length(summ, min_words=min_words, max_words=max_words)
                    debug["method"] = "extractive_tfidf"
                    debug["elapsed_s"] = time.time() - t0
                    return {"summary": summ, "qa_report": None, "debug": debug}

                if method == "textrank":
                    summ = extractive_textrank(text)
                    summ = self._enforce_word_length(summ, min_words=min_words, max_words=max_words)
                    debug["method"] = "extractive_textrank"
                    debug["elapsed_s"] = time.time() - t0
                    return {"summary": summ, "qa_report": None, "debug": debug}

                if method == "lead":
                    summ = extractive_lead(text)
                    summ = self._enforce_word_length(summ, min_words=min_words, max_words=max_words)
                    debug["method"] = "extractive_lead"
                    debug["elapsed_s"] = time.time() - t0
                    return {"summary": summ, "qa_report": None, "debug": debug}

                logger.warning(f"Unknown extractive_method='{extractive_method}', falling back to abstractive pipeline.")
            except Exception as e:
                logger.warning(f"Extractive summarization failed ({extractive_method}): {e}. Falling back to abstractive.")

        # --------------------------------------------------
        # 3) Optional prefilter (lead sentences) BEFORE chunking/abstractive
        # --------------------------------------------------
        sentences = self._split_into_sentences(text)
        if extractive_prefilter == "lead":
            N = 8
            working_text = " ".join(sentences[:N]).strip()
        else:
            working_text = text.strip()

        # Token count estimate
        token_count = self._count_tokens(working_text)
        debug["token_count"] = token_count

        if chunking_mode not in ("token", "sentence"):
            chunking_mode = "token"

        overlap_tokens = int(token_chunk_overlap) if token_chunk_overlap is not None else self.chunk_overlap_tokens

        # We'll keep evidences here so QA can use them (especially when RAG is on)
        evidences: Optional[List[Tuple[int, float, str]]] = None

        # --------------------------------------------------
        # 4) Short input: single-pass generation
        # --------------------------------------------------
        if token_count <= self.max_input_tokens:
            # Optional: build RAG index + retrieve evidences (for QA only OR for context injection)
            if use_rag:
                try:
                    sent_list = sentences if sentences else self._split_into_sentences(working_text)
                    rag_passages = make_sentence_windows(
                        sent_list,
                        window_size=self.sentence_window_for_rag,
                        stride=1,
                    )
                    self._maybe_build_rag_index(rag_passages)
                    if self._rag is not None:
                        query_text = working_text if rag_query_on == "article" else working_text[:500] 
                        # Note: 'summary' not avail yet, so query on article is safer here
                        
                        evidences = self._rag.search(query_text, top_k=int(rag_top_k))
                        debug["rag_scores"] = [{"idx": e[0], "score": e[1]} for e in evidences]
                        
                        # ✅ FIX: Actually USE the evidence by prepending it
                        if evidences:
                            evidence_text = "\n".join([e[2] for e in evidences])
                            working_text = f"Context:\n{evidence_text}\n\nArticle:\n{working_text}"
                            
                except Exception as e:
                    logger.warning(f"RAG (short doc) failed (ignored): {e}")

            summary = self._generate_summary_from_text(
                working_text,
                min_length=gen_min_len,
                max_length=gen_max_len,
                num_beams=num_beams,
                length_penalty=gen_len_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
            )

            qa_report = None
            if run_qa:
                try:
                    qa_report = check_summary_against_source(summary, working_text, evidences=evidences)
                except Exception as e:
                    logger.warning(f"QA checker error: {e}")
                    qa_report = {"error": str(e)}

            debug["num_generate_calls"] = 1
            debug["num_token_chunks"] = 1
            debug["elapsed_s"] = time.time() - t0
            debug["method"] = "abstractive_single_pass"
            return {"summary": summary, "qa_report": qa_report, "debug": debug}

        # --------------------------------------------------
        # 5) Long input: chunking
        # --------------------------------------------------
        try:
            if chunking_mode == "token":
                token_chunks = self._token_level_chunk_text(
                    working_text,
                    max_tokens=self.max_input_tokens,
                    overlap=overlap_tokens,
                )
            else:
                sents = sentences if sentences else self._split_into_sentences(working_text)
                token_chunks = self._chunk_sentences_into_texts(sents, self.max_input_tokens)
        except Exception as e:
            logger.warning(f"Chunking failed; falling back to sentence chunking: {e}")
            sents = sentences if sentences else self._split_into_sentences(working_text)
            token_chunks = self._chunk_sentences_into_texts(sents, self.max_input_tokens)

        debug["num_token_chunks"] = len(token_chunks)

        # --------------------------------------------------
        # 6) Optional RAG indexing
        # --------------------------------------------------
        if use_rag:
            try:
                sent_list = sentences if sentences else self._split_into_sentences(working_text)
                rag_passages = make_sentence_windows(
                    sent_list,
                    window_size=self.sentence_window_for_rag,
                    stride=1,
                )
                self._maybe_build_rag_index(rag_passages)
            except Exception as e:
                logger.warning(f"RAG index build failed (ignored): {e}")

        # --------------------------------------------------
        # 7) MAP: summarize each chunk
        # --------------------------------------------------
        per_chunk_summaries: List[str] = []
        gen_calls = 0

        # Use shorter per-chunk targets (tokens), but derived from requested bounds
        per_chunk_min = max(12, int(gen_min_len / 4))
        per_chunk_max = max(30, int(gen_max_len / 3))

        for i, chunk in enumerate(token_chunks):
            try:
                s = self._generate_summary_from_text(
                    chunk,
                    min_length=per_chunk_min,
                    max_length=per_chunk_max,
                    num_beams=num_beams,
                    length_penalty=gen_len_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping,
                )
                per_chunk_summaries.append(s)
                gen_calls += 1
            except Exception as e:
                logger.warning(f"Chunk summarization failed for chunk {i}: {e}")

        debug["num_generate_calls"] = gen_calls

        if not per_chunk_summaries:
            # Fallback: truncate input and do single generation
            logger.warning("All chunk summarizations failed; fallback to truncation.")
            trunc = working_text[: int(len(working_text) * 0.6)]
            final_summary = self._generate_summary_from_text(
                trunc,
                min_length=gen_min_len,
                max_length=gen_max_len,
                num_beams=num_beams,
                length_penalty=gen_len_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
            )
            final_summary = self._enforce_word_length(final_summary, min_words=min_words, max_words=max_words)

            qa_report = None
            if run_qa:
                try:
                    qa_report = check_summary_against_source(final_summary, working_text, evidences=evidences)
                except Exception as e:
                    logger.warning(f"QA checker failed: {e}")
                    qa_report = {"error": str(e)}

            debug["elapsed_s"] = time.time() - t0
            debug["method"] = "abstractive_fallback_trunc"
            return {"summary": final_summary, "qa_report": qa_report, "debug": debug}

        # --------------------------------------------------
        # 8) Optional rerank
        # --------------------------------------------------
        selected_texts = per_chunk_summaries
        if use_reranker and self.reranker is not None:
            try:
                topk = self.reranker.select_top_k(
                    working_text,
                    per_chunk_summaries,
                    k=min(int(top_k), len(per_chunk_summaries)),
                )
                selected_texts = [t[2] for t in topk]
                debug["reranker_scores"] = [{"idx": t[0], "score": t[1]} for t in topk]
            except Exception as e:
                logger.warning(f"Reranker failed: {e}")
                selected_texts = per_chunk_summaries

        # --------------------------------------------------
        # 9) Optional RAG retrieval (prepend evidence)
        #     ✅ store evidences and pass to QA later
        # --------------------------------------------------
        if use_rag and self._rag is not None:
            try:
                query_text = working_text if rag_query_on == "article" else " ".join(per_chunk_summaries)
                evidences = self._rag.search(query_text, top_k=int(rag_top_k))
                evidence_texts = [e[2] for e in evidences]
                selected_texts = evidence_texts + selected_texts
                debug["rag_scores"] = [{"idx": e[0], "score": e[1]} for e in evidences]
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
                evidences = None

        # --------------------------------------------------
        # 10) REDUCE: fuse/condense
        # --------------------------------------------------
        final_summary = ""
        if condensation_strategy == "onepass":
            fused_input = " ".join(selected_texts)
            final_summary = self._generate_summary_from_text(
                fused_input,
                min_length=gen_min_len,
                max_length=gen_max_len,
                num_beams=num_beams,
                length_penalty=gen_len_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
            )
            gen_calls += 1
        else:
            current_candidates = selected_texts[:]
            iter_count = 0

            while True:
                iter_count += 1
                concat = " ".join(current_candidates)

                if self._count_tokens(concat) <= self.max_input_tokens or iter_count >= int(max_condensation_iters):
                    final_summary = self._generate_summary_from_text(
                        concat,
                        min_length=gen_min_len,
                        max_length=gen_max_len,
                        num_beams=num_beams,
                        length_penalty=gen_len_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        early_stopping=early_stopping,
                    )
                    gen_calls += 1
                    break

                n = len(current_candidates)
                window_size = max(2, int(math.ceil(math.sqrt(n))))
                stride = window_size

                new_candidates: List[str] = []
                for i in range(0, n, stride):
                    group = current_candidates[i : i + window_size]
                    group_text = " ".join(group)
                    try:
                        grp_summary = self._generate_summary_from_text(
                            group_text,
                            min_length=max(8, int(gen_min_len / 6)),
                            max_length=max(20, int(gen_max_len / 6)),
                            num_beams=num_beams,
                            length_penalty=gen_len_penalty,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            early_stopping=early_stopping,
                        )
                        new_candidates.append(grp_summary)
                        gen_calls += 1
                    except Exception as e:
                        logger.warning(f"Condensation group generation failed: {e}")
                        new_candidates.append(group_text)

                current_candidates = new_candidates

                if use_reranker and self.reranker is not None:
                    try:
                        topk2 = self.reranker.select_top_k(
                            working_text,
                            current_candidates,
                            k=min(int(top_k), len(current_candidates)),
                        )
                        current_candidates = [t[2] for t in topk2]
                    except Exception:
                        pass

        # Enforce final word constraints (consistent everywhere)
        final_summary = self._enforce_word_length(final_summary, min_words=min_words, max_words=max_words)

        # QA (pass evidences if available)
        qa_report = None
        if run_qa:
            try:
                qa_report = check_summary_against_source(final_summary, working_text, evidences=evidences)
            except Exception as e:
                logger.warning(f"QA checker failed: {e}")
                qa_report = {"error": str(e)}

        debug["num_generate_calls"] = gen_calls
        debug["elapsed_s"] = time.time() - t0
        debug["method"] = "abstractive_map_reduce"
        return {"summary": final_summary, "qa_report": qa_report, "debug": debug}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_torch_device(device: int) -> torch.device:
        if device != -1 and torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")

    def _get_model_device(self) -> torch.device:
        """Prefer actual param device (works even if model was swapped/LoRA merged)."""
        try:
            return next(self.model.parameters()).device
        except Exception:
            return self._model_device

    def _resolve_generation_lengths(
        self,
        *,
        min_words: Optional[int],
        max_words: Optional[int],
        length_penalty: Optional[float],
    ) -> tuple[int, int, float]:
        """
        Convert UI word bounds into approximate generation token bounds.
        """
        cfg = self.MODEL_CONFIGS.get(self.model_name, {})
        default_min = int(cfg.get("min_length", 40))
        default_max = int(cfg.get("max_length", 160))
        default_lp = float(cfg.get("length_penalty", 1.0))

        # English: ~1.2-1.4 tokens per word for BPE models.
        TOK_PER_WORD = 1.35

        if max_words is not None and max_words > 0:
            gen_max = max(20, int(max_words * TOK_PER_WORD))
        else:
            gen_max = default_max

        if min_words is not None and min_words > 0:
            gen_min = max(5, int(min_words * TOK_PER_WORD * 0.8))
        else:
            gen_min = default_min

        if gen_min >= gen_max:
            gen_min = max(5, gen_max - 10)

        lp = float(length_penalty) if length_penalty is not None else default_lp
        return gen_min, gen_max, lp

    def _maybe_build_rag_index(self, chunks: List[str], device: str = "cpu") -> None:
        if not chunks:
            return
        if self._rag is None:
            try:
                self._rag = Retriever(model_name=self.rag_model_name, device=device)
            except Exception as e:
                logger.warning(f"Could not initialize Retriever: {e}; RAG disabled.")
                self._rag = None
                return
        try:
            self._rag.build_index(chunks)
        except Exception as e:
            logger.warning(f"RAG index build failed: {e}")
            self._rag = None

    def _generate_summary_from_text(
        self,
        text: str,
        *,
        min_length: int,
        max_length: int,
        num_beams: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
        early_stopping: bool,
    ) -> str:
        if not text:
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        model_device = self._get_model_device()
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        gen_kwargs = dict(
            max_length=int(max_length),
            min_length=int(min_length),
            num_beams=int(num_beams),
            length_penalty=float(length_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            early_stopping=bool(early_stopping),
            use_cache=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded[0].strip() if decoded else ""

    def _split_into_sentences(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        # spaCy sentencizer (local fallback to ensure identical behavior to before)
        if _HAS_SPACY:
            global _SPACY_SENTENCIZER
            try:
                if _SPACY_SENTENCIZER is None:
                    nlp = spacy.blank("en")
                    if "sentencizer" not in nlp.pipe_names:
                        nlp.add_pipe("sentencizer")
                    _SPACY_SENTENCIZER = nlp
                doc = _SPACY_SENTENCIZER(text)
                sents = [s.text.strip() for s in doc.sents if s.text.strip()]
                if sents:
                    return sents
            except Exception:
                pass

        # nltk punkt
        if _HAS_NLTK:
            try:
                from nltk.tokenize import sent_tokenize  # type: ignore
                sents = sent_tokenize(text)
                return [s.strip() for s in sents if s.strip()]
            except Exception:
                pass

        # regex fallback
        import re
        sents = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sents if s.strip()]

    def _count_tokens(self, text: str) -> int:
        try:
            toks = self.tokenizer.encode(text, add_special_tokens=False)
            return len(toks)
        except Exception:
            return max(0, len((text or "").split()))

    def _token_level_chunk_text(self, text: str, max_tokens: int, overlap: int = 50) -> List[str]:
        """
        Chunk using offset mapping so chunks align to original characters.
        Requires a fast tokenizer.
        """
        text = (text or "").strip()
        if not text:
            return []

        try:
            encoded = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoded.get("offset_mapping")
            ids = encoded.get("input_ids")
            if offsets is None or ids is None:
                raise ValueError("tokenizer missing offsets or ids (fast tokenizer required)")

            token_count = len(ids)
            if token_count <= max_tokens:
                return [text]

            chunks: List[str] = []
            i = 0
            step = max(1, int(max_tokens) - int(overlap))

            while i < token_count:
                j = min(i + int(max_tokens), token_count)
                start_char = offsets[i][0]
                end_char = offsets[j - 1][1] if (j - 1) < len(offsets) else offsets[-1][1]
                chunk_text = text[start_char:end_char].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                if j >= token_count:
                    break
                i += step

            return chunks

        except Exception as e:
            logger.warning(f"Fast-tokenizer chunking failed: {e}; falling back to sentence chunking.")
            sents = self._split_into_sentences(text)
            return self._chunk_sentences_into_texts(sents, max_tokens)

    def _chunk_sentences_into_texts(self, sentences: List[str], max_tokens: int) -> List[str]:
        if not sentences:
            return []

        chunks: List[str] = []
        cur: List[str] = []
        cur_tokens = 0

        for sent in sentences:
            s_toks = self._count_tokens(sent)
            if cur and (cur_tokens + s_toks > max_tokens):
                chunks.append(" ".join(cur).strip())
                cur = [sent]
                cur_tokens = s_toks
            else:
                cur.append(sent)
                cur_tokens += s_toks

        if cur:
            chunks.append(" ".join(cur).strip())

        # Safety: if any chunk still exceeds max_tokens, split by words (fallback)
        safe: List[str] = []
        for c in chunks:
            if self._count_tokens(c) <= max_tokens:
                safe.append(c)
                continue

            words = c.split()
            approx_words_per_chunk = max(80, int(max_tokens / 1.3))
            for i in range(0, len(words), approx_words_per_chunk):
                safe.append(" ".join(words[i : i + approx_words_per_chunk]).strip())

        return [s for s in safe if s]

    def _enforce_word_length(self, summary: str, *, min_words: Optional[int], max_words: Optional[int]) -> str:
        """
        Deterministic "word" constraint enforcement:
        - if max_words is set and the summary exceeds it, truncate at sentence boundary if possible.
        - min_words is advisory only (we do not pad / expand output).
        """
        summary = (summary or "").strip()
        if not summary:
            return ""

        if max_words is None:
            return summary

        try:
            max_w = int(max_words)
            if max_w <= 0:
                return summary
        except Exception:
            return summary

        words = summary.split()
        if len(words) <= max_w:
            return summary

        import re
        sents = re.split(r"(?<=[.!?])\s+", summary)
        kept: List[str] = []
        total = 0
        for s in sents:
            sw = len(s.split())
            if total + sw <= max_w:
                kept.append(s.strip())
                total += sw
            else:
                break

        if kept:
            return " ".join(kept).strip()

        return " ".join(words[:max_w]).rstrip() + "..."

    def _hybrid_filter_textrank(
        self,
        text: str,
        *,
        keep_ratio: float,
        min_keep: int = 3,
    ) -> tuple[str, Optional[int], Optional[int]]:
        """
        Filter the input by selecting top-ranked sentences (TextRank-style) then
        returning them in original order.

        Returns: (filtered_text, kept_count, total_sentences)
        """
        sents = self._split_into_sentences(text)
        total = len(sents)
        if total <= 5:
            return text, None, None

        keep_ratio = float(keep_ratio)
        keep_ratio = max(0.1, min(0.8, keep_ratio))
        keep_count = max(int(min_keep), int(round(total * keep_ratio)))
        keep_count = min(keep_count, total)

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            import networkx as nx  # type: ignore

            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf = vectorizer.fit_transform(sents)
            sim = (tfidf * tfidf.T).toarray()

            graph = nx.from_numpy_array(sim)
            scores = nx.pagerank(graph, alpha=0.85)

            ranked = sorted(((scores[i], i) for i in range(total)), reverse=True)
            top_indices = sorted([i for _, i in ranked[:keep_count]])
            filtered = " ".join(sents[i] for i in top_indices).strip()
            return filtered if filtered else text, keep_count, total

        except Exception as e:
            logger.warning(f"Hybrid TextRank scoring failed: {e}. Falling back to lead sentences.")
            filtered = " ".join(sents[:keep_count]).strip()
            return filtered if filtered else text, keep_count, total

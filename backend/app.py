# backend/app.py
"""
SmartNewsSumm FastAPI backend.

This API wraps `backend.summarizer.NewsSummarizer` and exposes a single `/summarize` endpoint.

Goals:
- Keep the backend stable even if the UI sends legacy/extra params.
- Load/cache the base summarizer by model_name (lazy, reused across requests).
- Optionally apply a LoRA adapter per request (and ensure it does NOT "stick" for later requests).
- Support:
  - Abstractive (transformer) summarization
  - Extractive-only methods (tfidf / textrank / lead) via `extractive_method`
  - Hybrid mode (extract-then-abstract) via `hybrid_mode`
  - Optional RAG + reranker + condensation + chunking options (passed through)
  - Optional QA checker
- Return debug fields so you can confirm what happened.

Important:
- Metrics are NOT computed here because the API doesn’t receive a gold/reference summary.
  Dataset-level evaluation should be done in scripts/evaluate.py.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Literal
import logging
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from peft import PeftModel

# local modules
from .preprocess import preprocess_for_model
from .summarizer import NewsSummarizer

logger = logging.getLogger("smartnewssumm_api")
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

app = FastAPI(title="SmartNewsSumm API", version="0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------
# Request models
# ----------------
class SummarizeRequest(BaseModel):
    # Input
    text: str = Field(..., description="Raw article text")

    # Model selection
    model_name: str = Field(
        "sshleifer/distilbart-cnn-12-6",
        description="HF model name for abstractive summarization",
    )
    device: Optional[int] = Field(
        None,
        description="GPU index (e.g., 0) or -1 for CPU. If None, auto-detect.",
    )

    # LoRA toggle
    use_lora: bool = Field(True, description="If True, try to load LoRA adapter (if configured via env var).")

    # Extractive-only mode (bypasses transformer summarization)
    extractive_method: Optional[Literal["tfidf", "textrank", "lead"]] = Field(
        None, description="If set, return extractive summary only."
    )

    # Hybrid mode (extract-then-abstract) - implemented inside NewsSummarizer
    hybrid_mode: bool = Field(False, description="Enable hybrid extract-then-abstract mode")
    hybrid_extractive_ratio: float = Field(
        0.3, description="Hybrid mode: keep top X% sentences before abstractive stage"
    )

    # Generation controls
    min_length: Optional[int] = Field(None, description="Min length (tokens/words, depending on model & enforcement)")
    max_length: Optional[int] = Field(None, description="Max length (tokens/words, depending on model & enforcement)")
    num_beams: int = Field(4, description="Beam search width")
    length_penalty: Optional[float] = Field(None, description="Length penalty for generation")
    no_repeat_ngram_size: int = Field(3, description="No repeat ngram size")
    early_stopping: bool = Field(True, description="Generation early stopping")

    # Optional prefilter (lead sentences) BEFORE chunking/abstraction (summarizer supports `extractive_prefilter`)
    extractive_prefilter: Optional[Literal["none", "lead"]] = Field(
        None, description="Optional prefilter before abstractive pipeline (e.g., 'lead')"
    )

    # Long-document pipeline knobs
    use_reranker: bool = Field(True, description="Use SBERT reranker for long inputs (if available)")
    top_k: int = Field(5, description="Top-k items kept after reranking")
    use_rag: bool = Field(False, description="Enable RAG retrieval")
    rag_top_k: int = Field(3, description="RAG top_k passages")
    rag_query_on: Literal["article", "summaries"] = Field("article", description="RAG query source")
    condensation_strategy: Literal["iterative", "onepass"] = Field("iterative", description="Reduce step strategy")
    chunking_mode: Literal["token", "sentence"] = Field("token", description="Chunking granularity")
    token_chunk_overlap: Optional[int] = Field(None, description="Token chunk overlap; defaults to summarizer setting")
    max_condensation_iters: int = Field(5, description="Max condensation iterations")

    # QA checker
    run_qa: bool = Field(True, description="Run factuality/entity QA checks")

    # LoRA (per-request)
    use_lora: bool = Field(False, description="Enable LoRA adapter")
    lora_adapter_path: Optional[str] = Field(None, description="Path to LoRA adapter folder")
    merge_lora: bool = Field(True, description="Merge LoRA weights into base model for faster inference")

    # Optional extra generation kwargs (advanced / future-proof)
    extra_gen_kwargs: Optional[Dict[str, Any]] = Field(
        None, description="Extra kwargs forwarded to summarizer.summarize()"
    )


class SummarizeResponse(BaseModel):
    summary: str
    qa_report: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None  # not computed by this API (no reference)
    debug: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------
# Global cache (single-process). For multi-worker hosting, each worker
# gets its own cache — that's okay.
# ---------------------------------------------------------------------
summarizer: Optional[NewsSummarizer] = None
_current_model_name: Optional[str] = None

# LoRA state for the currently loaded `summarizer.model`
_current_lora_path: Optional[str] = None
_current_lora_merged: Optional[bool] = None


def _load_summarizer(model_name: str, device: Optional[int], force_reload: bool = False) -> NewsSummarizer:
    """
    Load or reload the global summarizer.

    We reload when:
    - first request
    - model_name changes
    - force_reload=True (used to reset to base model when LoRA was previously applied)
    """
    global summarizer, _current_model_name

    # Normalize device:
    #   None -> auto inside NewsSummarizer
    #   -1   -> CPU
    #   >=0  -> GPU index
    if summarizer is None or force_reload or _current_model_name != model_name:
        logger.info(f"Loading summarizer: model={model_name} device={device} force_reload={force_reload}")
        # If reloading, clear old memory first
        if summarizer is not None:
             del summarizer
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
        
        summarizer = NewsSummarizer(model_name=model_name, device=device)
        _current_model_name = model_name
    return summarizer


def _reset_lora_state() -> None:
    """Reset LoRA cache keys."""
    global _current_lora_path, _current_lora_merged
    _current_lora_path = None
    _current_lora_merged = None


def _ensure_base_model_if_lora_was_applied(req: SummarizeRequest) -> None:
    """
    Critical correctness guard:
    If any previous request applied LoRA, `summarizer.model` has been mutated.
    For a request with use_lora=False, we must reload the base summarizer.
    """
    global _current_lora_path
    if (not req.use_lora) and (_current_lora_path is not None):
        # Reload base (same model_name) and clear LoRA cache.
        _load_summarizer(req.model_name, req.device, force_reload=True)
        _reset_lora_state()


def _apply_lora_if_needed(req: SummarizeRequest) -> None:
    """
    Apply LoRA adapter to the currently loaded summarizer model.

    This mutates `summarizer.model`. Therefore, we also maintain cache keys and
    explicitly reset to base model on later requests if use_lora=False.
    """
    global summarizer, _current_lora_path, _current_lora_merged

    if not req.use_lora:
        return

    if not req.lora_adapter_path:
        raise ValueError("use_lora=True but lora_adapter_path is empty")

    # Basic compatibility guard: LoRA trained for DistilBART must match base.
    # (Remove this check if you later train LoRA for other base models.)
    if req.model_name != "sshleifer/distilbart-cnn-12-6":
        raise ValueError("This LoRA adapter is only compatible with sshleifer/distilbart-cnn-12-6")

    # Skip re-applying if already active and same merge mode.
    if _current_lora_path == req.lora_adapter_path and _current_lora_merged == req.merge_lora:
        return

    if summarizer is None:
        raise RuntimeError("summarizer is not initialized")

    base_model = summarizer.model
    peft_model = PeftModel.from_pretrained(base_model, req.lora_adapter_path)

    if req.merge_lora:
        peft_model = peft_model.merge_and_unload()

    # Place on same device as base_model currently lives
    try:
        target_device = next(base_model.parameters()).device
    except Exception:
        target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    peft_model.to(target_device)
    peft_model.eval()

    summarizer.model = peft_model
    _current_lora_path = req.lora_adapter_path
    _current_lora_merged = req.merge_lora

    logger.info(f"LoRA applied: path={req.lora_adapter_path} merged={req.merge_lora} device={target_device}")


def _call_summarizer(req: SummarizeRequest) -> Dict[str, Any]:
    """
    End-to-end orchestration:
    - preprocess input
    - load summarizer
    - ensure base model if LoRA was previously applied but not requested now
    - apply LoRA if requested
    - forward supported kwargs into NewsSummarizer.summarize()
    - normalize response to a dict
    """
    # 1) Preprocess input
    cleaned_text, _sentences = preprocess_for_model(req.text, return_sentences=True)
    if not cleaned_text or len(cleaned_text.strip()) < 5:
        raise ValueError("Input too short after preprocessing.")

    # 2) Load/reuse summarizer for this model_name
    _load_summarizer(req.model_name, req.device, force_reload=False)

    # If use_lora is True but path is missing, try env var or local fallback
    if req.use_lora and not req.lora_adapter_path:
        import os
        # 1. Try Env Var
        path_from_env = os.getenv("LORA_ADAPTER_PATH")
        if path_from_env:
            req.lora_adapter_path = path_from_env
        else:
            # 2. Try known local path (hardcoded for this project structure)
            # Assuming running from project root
            fallback = os.path.join("results", "lora_kfold", "fold_0", "adapter")
            if os.path.exists(fallback):
                req.lora_adapter_path = fallback
            else:
                # Try absolute path based on known user structure if relative fails
                fallback_abs = r"d:\Code\GitHub\SmartNewsSumm\results\lora_kfold\fold_0\adapter"
                if os.path.exists(fallback_abs):
                    req.lora_adapter_path = fallback_abs

    # 3) Correctness: if LoRA was applied before, but now use_lora=False -> reset to base
    _ensure_base_model_if_lora_was_applied(req)

    # 4) Apply LoRA if requested
    _apply_lora_if_needed(req)

    # 5) Build kwargs for NewsSummarizer.summarize()
    # NOTE: backend/summarizer.py expects `extractive_method` and supports tfidf/textrank/lead.
    gen_kwargs: Dict[str, Any] = {
        "extractive_method": req.extractive_method,
        "hybrid_mode": req.hybrid_mode,
        "hybrid_extractive_ratio": req.hybrid_extractive_ratio,
        "min_length": req.min_length,
        "max_length": req.max_length,
        "num_beams": req.num_beams,
        "length_penalty": req.length_penalty,
        "no_repeat_ngram_size": req.no_repeat_ngram_size,
        "early_stopping": req.early_stopping,
        "use_reranker": req.use_reranker,
        "top_k": req.top_k,
        "run_qa": req.run_qa,
        "use_rag": req.use_rag,
        "rag_top_k": req.rag_top_k,
        "rag_query_on": req.rag_query_on,
        "condensation_strategy": req.condensation_strategy,
        "chunking_mode": req.chunking_mode,
        "token_chunk_overlap": req.token_chunk_overlap,
        "max_condensation_iters": req.max_condensation_iters,
    }

    # Optional prefilter if provided (summarizer supports `extractive_prefilter`)
    if req.extractive_prefilter is not None:
        gen_kwargs["extractive_prefilter"] = req.extractive_prefilter

    # Forward any extra kwargs (advanced). This is useful to avoid breaking UI changes.
    if req.extra_gen_kwargs:
        gen_kwargs.update(req.extra_gen_kwargs)

    # 6) Call summarizer
    try:
        raw = summarizer.summarize(cleaned_text, **gen_kwargs)  # type: ignore[union-attr]
    except TypeError as e:
        # If a mismatched kwarg sneaks in, surface a clear error
        raise ValueError(f"Invalid summarizer arguments: {e}")

    # 7) Normalize output
    out: Dict[str, Any] = {"summary": "", "qa_report": None, "metrics": None, "debug": {}}
    if isinstance(raw, str):
        out["summary"] = raw
    elif isinstance(raw, dict):
        out["summary"] = raw.get("summary") or raw.get("final_summary") or ""
        out["qa_report"] = raw.get("qa_report")
        out["debug"] = raw.get("debug") or {}
        if "metrics" in raw:
            out["metrics"] = raw.get("metrics")
    else:
        out["summary"] = str(raw)

    # 8) Add common debug fields
    dbg = out.get("debug") or {}
    dbg.setdefault("model_name", req.model_name)
    dbg.setdefault("device", "cuda" if torch.cuda.is_available() and (getattr(summarizer, "device", -1) != -1) else "cpu")
    dbg.setdefault("extractive_method", req.extractive_method)
    dbg.setdefault("hybrid_mode", req.hybrid_mode)
    dbg.setdefault("use_rag", req.use_rag)
    dbg.setdefault("use_reranker", req.use_reranker)
    dbg.setdefault("use_lora", req.use_lora)
    dbg.setdefault("lora_adapter_path", req.lora_adapter_path if req.use_lora else None)
    dbg.setdefault("lora_merged", req.merge_lora if req.use_lora else None)
    out["debug"] = dbg

    # API does not compute metrics
    out["metrics"] = None
    return out


# ----------
# Endpoints
# ----------
@app.on_event("startup")
def _startup():
    logger.info("SmartNewsSumm API starting up. Models will be loaded lazily on first request.")


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    """
    Summarize a piece of text.
    Notes:
    - If `extractive_method` is set, the summarizer returns extractive-only output.
    - If `hybrid_mode=True`, the summarizer runs a TextRank filter before abstractive generation.
    - If `use_lora=True`, the server applies the adapter for this request; subsequent requests
      without LoRA automatically reload the base model to avoid "sticky" LoRA.
    """
    try:
        start = time.time()

        # Basic validation
        if req.min_length is not None and req.max_length is not None and req.min_length > req.max_length:
            raise ValueError("min_length cannot be greater than max_length")

        out = _call_summarizer(req)
        elapsed = time.time() - start
        logger.info(f"/summarize done in {elapsed:.2f}s | tokens~{out.get('debug', {}).get('token_count')}")

        return SummarizeResponse(
            summary=out["summary"],
            qa_report=out.get("qa_report"),
            metrics=out.get("metrics"),
            debug=out.get("debug"),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Internal summarization error: {e}")
        raise HTTPException(status_code=500, detail="Internal summarization error")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

# backend/qa_checker.py
"""
QA-based factuality checker.

What it does:
- Splits summary into sentences
- Extracts entities per sentence (spaCy if available; fallback heuristic)
- Verifies entity support using:
  1) QA pipeline (if available)
  2) semantic similarity (SentenceTransformer cosine similarity)
- Optionally (default ON): generates sentence-level "claims" and verifies them too.

Notes:
- If RAG provides evidences, we prefer evidence passages as context (smaller + more relevant).
- Claim generation is kept lightweight (sentence split) and runtime-limited via guards.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------
# spaCy setup (optional)
# -------------------------
_HAS_SPACY = False
_nlp = None
try:
    import spacy  # type: ignore
    try:
        _nlp = spacy.load("en_core_web_sm")
        _HAS_SPACY = True
        logger.info("spaCy model loaded (en_core_web_sm). NER enabled.")
    except Exception as e:
        logger.warning(f"spaCy model unavailable: {e}. Falling back to blank pipeline.")
        try:
            _nlp = spacy.blank("en")
            if "sentencizer" not in _nlp.pipe_names:
                _nlp.add_pipe("sentencizer")
            _HAS_SPACY = True
        except Exception as e_blank:
            logger.warning(f"spaCy blank pipeline failed: {e_blank}. spaCy disabled.")
            _nlp = None
            _HAS_SPACY = False
except Exception as e_import:
    logger.info(f"spaCy import failed: {e_import}. spaCy disabled.")
    _nlp = None
    _HAS_SPACY = False

# -------------------------
# QA pipeline setup (optional)
# -------------------------
_QA_MODEL = "distilbert-base-uncased-distilled-squad"
_qa_pipe = None
try:
    from transformers import pipeline  # type: ignore
    try:
        _qa_pipe = pipeline("question-answering", model=_QA_MODEL, tokenizer=_QA_MODEL)
        logger.info(f"QA pipeline loaded: {_QA_MODEL}")
    except Exception as e_qa:
        logger.warning(f"QA pipeline init failed ({_QA_MODEL}): {e_qa}")
        _qa_pipe = None
except Exception as e_trf:
    logger.warning(f"transformers import failed; QA disabled: {e_trf}")
    _qa_pipe = None

# -------------------------
# Semantic similarity model (lazy load)
# -------------------------
_EMBED_MODEL = None

def _get_embedder():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


# -------------------------
# Helpers
# -------------------------
def _sentence_split(text: str) -> List[str]:
    if not text:
        return []
    if _HAS_SPACY and _nlp is not None:
        try:
            doc = _nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception as e:
            logger.debug(f"spaCy sentence split failed: {e}")
    # fallback
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract entities from `text`.
    Prefer spaCy NER; fallback to capitalization heuristic.
    Returns list of dicts: {"text": str, "label": Optional[str]}.
    """
    if not text:
        return []

    if _HAS_SPACY and _nlp is not None:
        try:
            doc = _nlp(text)
            ents = []
            for ent in getattr(doc, "ents", []):
                t = ent.text.strip()
                if t:
                    ents.append({"text": t, "label": getattr(ent, "label_", None)})
            # dedupe preserving order
            seen = set()
            uniq = []
            for e in ents:
                if e["text"] not in seen:
                    seen.add(e["text"])
                    uniq.append(e)
            if uniq:
                return uniq
        except Exception as e:
            logger.warning(f"spaCy NER failed: {e}")

    # fallback heuristic: consecutive Capitalized Tokens
    tokens = [t.strip() for t in text.replace("\n", " ").split() if t.strip()]
    candidates = []
    cur = []
    for tok in tokens:
        if tok and tok[0].isupper() and not tok.isdigit():
            cur.append(tok.strip(".,;:()[]\"'"))
        else:
            if cur:
                cand = " ".join(cur)
                if len(cand) > 1:
                    candidates.append({"text": cand, "label": "UNK"})
                cur = []
    if cur:
        cand = " ".join(cur)
        if len(cand) > 1:
            candidates.append({"text": cand, "label": "UNK"})

    seen = set()
    uniq = []
    for e in candidates:
        if e["text"] not in seen:
            seen.add(e["text"])
            uniq.append(e)
    return uniq


def _in_context(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    return needle.lower() in haystack.lower()


def _choose_best_evidence_for_text(text: str, evidences: List[Tuple[int, float, str]]) -> Optional[Tuple[int, float, str]]:
    """
    Prefer evidence passages containing the text (substring).
    Fallback: top scoring evidence.
    """
    if not evidences:
        return None

    contains = []
    for idx, score, passage in evidences:
        if _in_context(text, passage):
            contains.append((idx, score, passage))

    if contains:
        contains.sort(key=lambda t: -t[1])
        return contains[0]

    evidences_sorted = sorted(evidences, key=lambda t: -t[1])
    return evidences_sorted[0]


def _semantic_support(claim: str, evidence: str, threshold: float = 0.55) -> Dict[str, Any]:
    if not claim or not evidence:
        return {"supported": False, "score": 0.0}
    try:
        model = _get_embedder()
        emb_claim = model.encode(claim, convert_to_tensor=True)
        emb_evid = model.encode(evidence, convert_to_tensor=True)
        score = float(util.cos_sim(emb_claim, emb_evid).item())
        return {"supported": score >= threshold, "score": score}
    except Exception:
        return {"supported": False, "score": 0.0}


def _qa_check(entity_text: str, context: str, min_score: float = 0.25, top_k: int = 1) -> Dict[str, Any]:
    """
    Run QA pipeline on (entity_text, context).
    If QA unavailable: return error.
    """
    if not entity_text or not context:
        return {"supported": False, "answer": "", "score": 0.0, "error": "no_input", "raw": None}

    if _qa_pipe is None:
        return {"supported": False, "answer": "", "score": 0.0, "error": "qa_pipeline_unavailable", "raw": None}

    # short heuristic question formation
    q = entity_text
    words = entity_text.split()
    if len(words) <= 3:
        if any(ch.isdigit() for ch in entity_text):
            q = f"When did {entity_text} happen?"
        else:
            q = f"What is {entity_text}?"

    try:
        resp = _qa_pipe(question=q, context=context, top_k=top_k)

        if isinstance(resp, list):
            best = resp[0] if resp else {"answer": "", "score": 0.0}
        elif isinstance(resp, dict):
            best = resp
        else:
            return {"supported": False, "answer": "", "score": 0.0, "error": "unexpected_qa_response_type", "raw": resp}

        answer = best.get("answer", "").strip() if isinstance(best, dict) else ""
        score = float(best.get("score", 0.0)) if isinstance(best, dict) else 0.0
        supported = bool(answer) and score >= float(min_score)
        return {"supported": supported, "answer": answer, "score": score, "error": None, "raw": resp}
    except Exception as e:
        logger.warning(f"QA failed for entity='{entity_text}': {e}")
        return {"supported": False, "answer": "", "score": 0.0, "error": str(e), "raw": None}


def _generate_claims(summary: str, max_claims: int = 10) -> List[Dict[str, str]]:
    """
    Claim generator:
    - sentence-split summary
    - keep non-trivial sentences
    - attach a primary entity (first extracted) if available

    Returns list of:
      {"claim": "...", "entity": "..."}
    """
    if not summary or not summary.strip():
        return []

    text = re.sub(r"\s+", " ", summary.strip())
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) >= 10]

    sents = sents[: max(0, int(max_claims))]

    claims: List[Dict[str, str]] = []
    for s in sents:
        ents = extract_entities(s)
        primary_ent = ents[0]["text"] if ents else ""
        claims.append({"claim": s, "entity": primary_ent})

    return claims


# -------------------------
# Main exported function
# -------------------------
def check_summary_against_source(
    summary: str,
    source: str,
    evidences: Optional[List[Tuple[int, float, str]]] = None,
    max_entities_per_sentence: int = 5,
    min_score: float = 0.25,
    use_evidence: bool = True,
    run_claim_generation: bool = True,
    max_claims: int = 10,
) -> Dict[str, Any]:
    """
    Verify summary against source (and optional evidences).
    Returns:
      {
        "sentence_checks": [...],
        "overall_flagged_entities": int,
        "claims": [... or None],
        "evidences_summary": [... or None],
        "errors": [...]
      }
    """
    results: Dict[str, Any] = {
        "sentence_checks": [],
        "overall_flagged_entities": 0,
        "errors": [],
        "evidences_summary": None,
        "claims": None,
    }

    try:
        if not summary or not summary.strip() or not source or not source.strip():
            return results

        # -------------------------
        # ✅ GUARD RINGAN (setelah validasi)
        # -------------------------
        if run_claim_generation:
            # kalau summary terlalu pendek, claim-gen tidak berguna & buang waktu
            if len(summary.split()) < 20:
                run_claim_generation = False

            # hard cap supaya runtime stabil
            max_claims = int(max_claims) if max_claims is not None else 10
            max_claims = max(0, min(max_claims, 8))  # <= 8 claims saja (recommended)

        # debug-friendly evidence preview
        if evidences:
            try:
                results["evidences_summary"] = [
                    {
                        "idx": int(e[0]),
                        "score": float(e[1]),
                        "text_preview": (e[2][:200] + "..." if len(e[2]) > 200 else e[2]),
                    }
                    for e in evidences
                ]
            except Exception as e:
                results["evidences_summary"] = None
                results["errors"].append(f"evidence_format_error: {e}")

        # -------------------------
        # Sentence-level entity checks
        # -------------------------
        sentences = _sentence_split(summary)
        for s in sentences:
            sent_record: Dict[str, Any] = {"sentence": s, "entities": [], "flagged": False}

            try:
                entities = extract_entities(s)
            except Exception as e_ent:
                results["errors"].append(f"entity_extraction_failed: {e_ent}")
                entities = []

            for ent in entities[: max_entities_per_sentence]:
                ent_text = ent.get("text", "").strip()
                ent_label = ent.get("label")

                try:
                    in_src = _in_context(ent_text, source)

                    evidence_idx = None
                    evidence_score = None
                    evidence_text = None

                    if use_evidence and evidences:
                        chosen = _choose_best_evidence_for_text(ent_text, evidences)
                        if chosen:
                            evidence_idx, evidence_score, evidence_text = chosen

                    # choose QA context
                    if evidence_text:
                        qa_context = evidence_text
                    else:
                        # try paragraph that contains entity
                        para_match = None
                        try:
                            paras = [p.strip() for p in re.split(r"\n{1,}", source) if p.strip()]
                            for p in paras:
                                if _in_context(ent_text, p):
                                    para_match = p
                                    break
                        except Exception:
                            para_match = None
                        qa_context = para_match if para_match else source

                    qa_res = _qa_check(ent_text, qa_context, min_score=min_score)
                    semantic_res = _semantic_support(claim=s, evidence=qa_context)

                    flagged = not (qa_res.get("supported", False) or semantic_res["supported"])

                    ent_entry = {
                        "text": ent_text,
                        "label": ent_label,
                        "in_source": in_src,
                        "qa_supported": qa_res.get("supported", False),
                        "qa_answer": qa_res.get("answer", ""),
                        "qa_score": qa_res.get("score", 0.0),
                        "qa_error": qa_res.get("error"),
                        "semantic_supported": semantic_res["supported"],
                        "semantic_score": semantic_res["score"],
                        "flagged": flagged,
                        "evidence_idx": int(evidence_idx) if evidence_idx is not None else None,
                        "evidence_score": float(evidence_score) if evidence_score is not None else None,
                        "evidence_text": evidence_text,
                    }

                    if flagged:
                        sent_record["flagged"] = True
                        results["overall_flagged_entities"] += 1

                    sent_record["entities"].append(ent_entry)

                except Exception as e_check:
                    results["errors"].append(f"entity_check_failed for '{ent_text}': {e_check}")
                    sent_record["entities"].append(
                        {
                            "text": ent_text,
                            "label": ent_label,
                            "in_source": False,
                            "qa_supported": False,
                            "qa_answer": "",
                            "qa_score": 0.0,
                            "qa_error": str(e_check),
                            "semantic_supported": False,
                            "semantic_score": 0.0,
                            "flagged": True,
                            "evidence_idx": None,
                            "evidence_score": None,
                            "evidence_text": None,
                        }
                    )

            results["sentence_checks"].append(sent_record)

        # -------------------------
        # Claim generation + verification (optional but default ON)
        # -------------------------
        if run_claim_generation and max_claims > 0:
            try:
                claims = _generate_claims(summary, max_claims=max_claims)
                claim_checks = []

                for c in claims:
                    claim_text = c.get("claim", "").strip()
                    ent_text = c.get("entity", "").strip()

                    evidence_idx = None
                    evidence_score = None
                    evidence_text = None

                    # choose evidence by entity if possible, else by claim text
                    if use_evidence and evidences:
                        key = ent_text if ent_text else claim_text
                        chosen = _choose_best_evidence_for_text(key, evidences)
                        if chosen:
                            evidence_idx, evidence_score, evidence_text = chosen

                    if evidence_text:
                        ctx = evidence_text
                    else:
                        ctx = source  # keep it simple for claims

                    # For claims:
                    # - if entity exists: QA that entity
                    # - else: skip QA and rely on semantic support
                    if ent_text:
                        qa_res = _qa_check(ent_text, ctx, min_score=min_score)
                    else:
                        qa_res = {"supported": False, "answer": "", "score": 0.0, "error": "no_entity_for_claim", "raw": None}

                    sem_res = _semantic_support(claim_text, ctx)

                    supported = qa_res.get("supported", False) or sem_res["supported"]
                    claim_checks.append(
                        {
                            "claim": claim_text,
                            "entity": ent_text,
                            "supported": supported,
                            "qa_supported": qa_res.get("supported", False),
                            "qa_answer": qa_res.get("answer", ""),
                            "qa_score": qa_res.get("score", 0.0),
                            "qa_error": qa_res.get("error"),
                            "semantic_supported": sem_res["supported"],
                            "semantic_score": sem_res["score"],
                            "evidence_idx": int(evidence_idx) if evidence_idx is not None else None,
                            "evidence_score": float(evidence_score) if evidence_score is not None else None,
                            "evidence_text": evidence_text,
                        }
                    )

                results["claims"] = claim_checks

            except Exception as e_claim:
                logger.warning(f"Claim generation/verification failed: {e_claim}")
                results["errors"].append(f"claimgen_failed: {e_claim}")
                results["claims"] = None

        return results

    except Exception as e_top:
        msg = f"qa_checker_top_level_exception: {e_top}"
        logger.exception(msg)
        results["errors"].append(msg)
        return results

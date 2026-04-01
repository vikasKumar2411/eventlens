# eventlens_v25/adapters.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# make repo root importable if needed
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase3_search_qdrant import (
    ollama_embed,
    build_filter,
    post_filter_hits_by_date,
    rerank_hits,
)

from scripts.phase4_rag_answer import (
    ollama_generate,
    build_evidence_context,
    build_prompt,
)

ACQUISITION_TERMS = [
    "acquisition",
    "acquire",
    "acquired",
    "acquiring",
    "merger",
    "merge",
    "business combination",
    "buyout",
    "takeover",
    "purchase",
]


# -----------------------------
# Planning / rewrite
# -----------------------------
def plan_query(query: str) -> Dict[str, Any]:
    q = query.lower()

    if any(term in q for term in ACQUISITION_TERMS):
        return {
            "intent": "acquisition",
            "plan": "find_mna_events",
        }

    return {
        "intent": "general",
        "plan": "general_sec_search",
    }


def get_plan_settings(plan: str) -> Dict[str, Any]:
    if plan == "find_mna_events":
        return {
            "retrieval_k": 30,
            "final_k": 5,
            "dedupe_key": "symbol",
        }

    return {
        "retrieval_k": 20,
        "final_k": 5,
        "dedupe_key": "accession",
    }


def get_default_max_retries(plan: str) -> int:
    if plan == "find_mna_events":
        return 2
    return 1


def rewrite_query_for_plan(query: str, plan: str) -> str:
    if plan == "find_mna_events":
        return f"{query} mergers acquisitions business combinations definitive agreements"
    return query


def rewrite_query_for_retry(query: str, plan: str, retry_index: int) -> str:
    if plan == "find_mna_events":
        if retry_index == 0:
            return f"{query} acquisition merger business combination definitive agreement"
        elif retry_index == 1:
            return f"{query} announced acquisition merge transaction will acquire"
        elif retry_index == 2:
            return f"{query} SEC 8-K Item 1.01 acquisition merger agreement"
        return rewrite_query_for_plan(query, plan)

    if retry_index == 0:
        return f"{query} SEC 8-K material event"
    elif retry_index == 1:
        return f"{query} filing announcement transaction"
    return rewrite_query_for_plan(query, plan)


def tool_search_sec_filings(
    query: str,
    *,
    collection: str = "sec_8k_chunks",
    qdrant_url: str = "http://localhost:6333",
    ollama_host: str = "http://127.0.0.1:11434",
    embed_model: str = "nomic-embed-text:latest",
    top_k: int = 20,
    symbol: str | None = None,
    exchange: str | None = None,
    accession: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> List[qm.ScoredPoint]:
    qdrant = QdrantClient(url=qdrant_url)

    q_vec = ollama_embed([query], model=embed_model, host=ollama_host)[0]

    q_filter = build_filter(
        symbol=symbol,
        exchange=exchange,
        accession=accession,
        date_from=date_from,
        date_to=date_to,
    )

    res = qdrant.query_points(
        collection_name=collection,
        query=q_vec,
        limit=top_k,
        query_filter=q_filter,
        with_payload=True,
        with_vectors=False,
    )

    hits = res.points
    hits = post_filter_hits_by_date(hits, date_from, date_to)
    return hits

def tool_rerank_candidates(query: str, hits: List[qm.ScoredPoint], verbose: bool = False):
    reranked = rerank_hits(query, hits)
    return reranked


def tool_deduplicate_candidates(hits, key: str = "accession"):
    seen = set()
    out = []

    for h in hits:
        payload = h.payload or {}

        if key == "symbol":
            dedupe_value = payload.get("symbol")
        elif key == "accession":
            dedupe_value = payload.get("sec_accession_number")
        else:
            dedupe_value = payload.get(key)

        if not dedupe_value:
            out.append(h)
            continue

        if dedupe_value in seen:
            continue

        seen.add(dedupe_value)
        out.append(h)

    return out

def tool_build_documents(hits, top_k: int = 5):
    return hits[:top_k]


def tool_build_context(docs, text_chars: int = 1200) -> str:
    return build_evidence_context(docs, text_chars=text_chars)

def tool_answer_from_context(
    question: str,
    context: str,
    *,
    llm_model: str = "qwen2.5:7b-instruct",
    ollama_host: str = "http://127.0.0.1:11434",
) -> str:
    prompt = build_prompt(question, context)
    return ollama_generate(prompt, model=llm_model, host=ollama_host)

def get_result_score(item) -> float:
    score = getattr(item, "score", 0.0)
    return float(score or 0.0)

def summarize_evidence(results) -> Dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "top_score": 0.0,
            "second_score": 0.0,
            "score_spread": 0.0,
            "strong_candidates": 0,
            "unique_titles": 0,
            "unique_symbols": 0,
            "unique_accessions": 0,
            "avg_top5_score": 0.0,
        }

    scores = [get_result_score(r) for r in results]
    top_score = scores[0]
    second_score = scores[1] if len(scores) > 1 else 0.0
    score_spread = top_score - second_score
    strong_candidates = sum(1 for s in scores if s >= 0.9)

    titles = {
        str((r.payload or {}).get("title") or "").strip().lower()
        for r in results
        if (r.payload or {}).get("title")
    }
    symbols = {
        str((r.payload or {}).get("symbol") or "").strip().upper()
        for r in results
        if (r.payload or {}).get("symbol")
    }
    accessions = {
        str((r.payload or {}).get("sec_accession_number") or "").strip()
        for r in results
        if (r.payload or {}).get("sec_accession_number")
    }

    avg_top5_score = sum(scores[:5]) / min(len(scores), 5)

    return {
        "count": len(results),
        "top_score": top_score,
        "second_score": second_score,
        "score_spread": score_spread,
        "strong_candidates": strong_candidates,
        "unique_titles": len(titles),
        "unique_symbols": len(symbols),
        "unique_accessions": len(accessions),
        "avg_top5_score": avg_top5_score,
    }


def evaluate_confidence(
    evidence_summary: Dict[str, Any],
    retry_count: int = 0,
    improved_on_retry: bool = False,
) -> Dict[str, Any]:
    top_score = evidence_summary["top_score"]
    strong_candidates = evidence_summary["strong_candidates"]
    unique_titles = evidence_summary["unique_titles"]
    unique_symbols = evidence_summary["unique_symbols"]
    avg_top5_score = evidence_summary["avg_top5_score"]

    reasons = []

    strong_signal = top_score >= 1.2
    multi_candidate_signal = strong_candidates >= 3
    diversity_signal = unique_titles >= 3 and unique_symbols >= 3
    consistency_signal = avg_top5_score >= 1.0

    if strong_signal:
        reasons.append("high_top_score")
    if multi_candidate_signal:
        reasons.append("multiple_strong_candidates")
    if diversity_signal:
        reasons.append("diverse_evidence")
    if consistency_signal:
        reasons.append("strong_top5_average")
    if retry_count > 0 and improved_on_retry:
        reasons.append("improved_after_retry")

    if strong_signal and multi_candidate_signal and diversity_signal and consistency_signal:
        decision = "confident"
    elif top_score >= 1.0 and strong_candidates >= 2 and (unique_titles >= 2 or unique_symbols >= 2):
        decision = "cautious"
    else:
        decision = "weak_evidence"

    return {
        "decision": decision,
        "reasons": reasons,
        "retry_count": retry_count,
        "improved_on_retry": improved_on_retry,
        "metrics": evidence_summary,
    }


def apply_answer_policy(answer: str, confidence_eval: Dict[str, Any]) -> str:
    decision = confidence_eval.get("decision", "weak_evidence")

    if decision == "confident":
        return answer

    if decision == "cautious":
        return (
            "The evidence appears directionally strong, but some uncertainty remains.\n\n"
            + answer
        )

    return (
        "Evidence is limited or mixed, so this answer should be treated cautiously.\n\n"
        + answer
    )
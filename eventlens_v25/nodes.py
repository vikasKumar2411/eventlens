# eventlens_v25/nodes.py
from __future__ import annotations

from typing import Any, Dict

from eventlens_v25.state import EventLensState
from eventlens_v25.adapters import (
    plan_query,
    get_plan_settings,
    get_default_max_retries,
    rewrite_query_for_plan,
    rewrite_query_for_retry,
    tool_search_sec_filings,
    tool_rerank_candidates,
    tool_deduplicate_candidates,
    tool_build_documents,
    tool_build_context,
    tool_answer_from_context,
    summarize_evidence,
    evaluate_confidence,
    apply_answer_policy,
)


def planner_node(state: EventLensState) -> Dict[str, Any]:
    query = state["query"]

    plan_info = plan_query(query)
    intent = plan_info["intent"]
    plan = plan_info["plan"]
    settings = get_plan_settings(plan)
    max_retries = state.get("max_retries", get_default_max_retries(plan))

    trace_entry = {
        "node": "planner_node",
        "intent": intent,
        "plan": plan,
        "max_retries": max_retries,
    }

    return {
        "intent": intent,
        "plan": plan,
        "settings": settings,
        "max_retries": max_retries,
        "retrieval_attempt": 0,
        "trace": state.get("trace", []) + [trace_entry],
    }


def rewrite_query_node(state: EventLensState) -> Dict[str, Any]:
    query = state["query"]
    plan = state["plan"]
    retrieval_attempt = state.get("retrieval_attempt", 0)

    if retrieval_attempt == 0:
        rewritten_query = rewrite_query_for_plan(query, plan)
    else:
        rewritten_query = rewrite_query_for_retry(
            query=query,
            plan=plan,
            retry_index=retrieval_attempt - 1,
        )

    trace_entry = {
        "node": "rewrite_query_node",
        "retrieval_attempt": retrieval_attempt,
        "rewritten_query": rewritten_query,
    }

    return {
        "rewritten_query": rewritten_query,
        "trace": state.get("trace", []) + [trace_entry],
    }


def retrieve_node(state: EventLensState) -> Dict[str, Any]:
    rewritten_query = state["rewritten_query"]
    settings = state["settings"]

    hits = tool_search_sec_filings(
        query=rewritten_query,
        top_k=settings["retrieval_k"],
    )

    trace_entry = {
        "node": "retrieve_node",
        "rewritten_query": rewritten_query,
        "num_hits": len(hits),
    }

    return {
        "hits": hits,
        "trace": state.get("trace", []) + [trace_entry],
    }


def rerank_node(state: EventLensState) -> Dict[str, Any]:
    query = state["query"]
    hits = state.get("hits", [])

    reranked = tool_rerank_candidates(
        query=query,
        hits=hits,
        verbose=False,
    )

    trace_entry = {
        "node": "rerank_node",
        "num_reranked": len(reranked),
    }

    return {
        "reranked": reranked,
        "trace": state.get("trace", []) + [trace_entry],
    }


def dedupe_node(state: EventLensState) -> Dict[str, Any]:
    reranked = state.get("reranked", [])
    settings = state["settings"]

    deduped = tool_deduplicate_candidates(
        hits=reranked,
        key=settings["dedupe_key"],
    )
    docs = tool_build_documents(
        hits=deduped,
        top_k=settings["final_k"],
    )

    trace_entry = {
        "node": "dedupe_node",
        "dedupe_key": settings["dedupe_key"],
        "num_deduped": len(deduped),
        "num_docs": len(docs),
    }

    return {
        "deduped": deduped,
        "docs": docs,
        "trace": state.get("trace", []) + [trace_entry],
    }


def evaluate_evidence_node(state: EventLensState) -> Dict[str, Any]:
    deduped = state.get("deduped", [])
    retrieval_attempt = state.get("retrieval_attempt", 0)

    evidence_summary = summarize_evidence(deduped)
    confidence_eval = evaluate_confidence(
        evidence_summary=evidence_summary,
        retry_count=retrieval_attempt,
        improved_on_retry=False,  # keep v2.5 simple for now
    )

    trace_entry = {
        "node": "evaluate_evidence_node",
        "decision": confidence_eval.get("decision"),
        "top_score": evidence_summary.get("top_score"),
        "strong_candidates": evidence_summary.get("strong_candidates"),
        "unique_titles": evidence_summary.get("unique_titles"),
        "unique_symbols": evidence_summary.get("unique_symbols"),
    }

    return {
        "evidence_summary": evidence_summary,
        "confidence_eval": confidence_eval,
        "trace": state.get("trace", []) + [trace_entry],
    }


def answer_node(state: EventLensState) -> Dict[str, Any]:
    query = state["query"]
    docs = state.get("docs", [])
    confidence_eval = state.get("confidence_eval", {})

    context = tool_build_context(
        docs=docs,
        text_chars=1200,
    )
    raw_answer = tool_answer_from_context(
        question=query,
        context=context,
    )
    final_answer = apply_answer_policy(
        answer=raw_answer,
        confidence_eval=confidence_eval,
    )

    trace_entry = {
        "node": "answer_node",
        "final_status": "answered",
    }

    return {
        "final_answer": final_answer,
        "final_status": "answered",
        "done": True,
        "trace": state.get("trace", []) + [trace_entry],
    }


def escalate_node(state: EventLensState) -> Dict[str, Any]:
    confidence_eval = state.get("confidence_eval", {})
    decision = confidence_eval.get("decision", "weak_evidence")

    final_answer = (
        "Evidence is limited or mixed, so EventLens cannot answer confidently for this query."
    )

    trace_entry = {
        "node": "escalate_node",
        "decision": decision,
        "final_status": "escalated",
    }

    return {
        "final_answer": final_answer,
        "final_status": "escalated",
        "done": True,
        "trace": state.get("trace", []) + [trace_entry],
    }
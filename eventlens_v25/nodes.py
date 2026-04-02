# eventlens_v25/nodes.py
from __future__ import annotations

import json
import re
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

def _extract_json_block(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction from LLM output.
    Falls back to empty dict if parsing fails.
    """
    if not text:
        return {}

    text = text.strip()

    # Try full-string parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try fenced code block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # Try first JSON object substring
    obj_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(1))
        except Exception:
            pass

    return {}


def _normalize_decision(decision: str) -> str:
    allowed = {"answer", "retry", "escalate"}
    if decision in allowed:
        return decision
    return "retry"


def _normalize_retry_strategy(strategy: str) -> str:
    allowed = {
        "broaden_search",
        "narrow_search",
        "event_keyword_bias",
        "diversify_results",
    }
    if strategy in allowed:
        return strategy
    return "broaden_search"


def _build_evidence_context(state: EventLensState, text_chars: int = 900) -> str:
    docs = state.get("docs", [])
    return tool_build_context(
        docs=docs,
        text_chars=text_chars,
    )


def _llm_evaluate_evidence(
    *,
    query: str,
    context: str,
    evidence_summary: Dict[str, Any],
    retrieval_attempt: int,
    max_retries: int,
) -> Dict[str, Any]:
    prompt = f"""
You are evaluating retrieval evidence for an SEC 8-K event-detection workflow.

User query:
{query}

Retrieval attempt:
{retrieval_attempt} of {max_retries}

Heuristic evidence summary:
{json.dumps(evidence_summary, indent=2)}

Retrieved evidence context:
{context}

Your job:
Decide whether the current evidence is sufficient to answer confidently, whether another retrieval retry is likely useful,
or whether the workflow should escalate because evidence is too weak or mixed.

Return STRICT JSON only with this schema:
{{
  "decision": "answer" | "retry" | "escalate",
  "confidence_band": "high" | "medium" | "low",
  "reasoning": "<short explanation>",
  "evidence_quality": "strong" | "mixed" | "weak",
  "query_match": "high" | "medium" | "low",
  "suggested_retry_focus": "broaden_search" | "narrow_search" | "event_keyword_bias" | "diversify_results" | "none"
}}

Rules:
- Choose "answer" only if the evidence clearly matches the query and is not mixed.
- Choose "retry" if there is some signal but retrieval likely needs refinement.
- Choose "escalate" if the evidence is too weak, too mixed, or retries are unlikely to help.
- Output JSON only.
""".strip()

    raw = tool_answer_from_context(
        question=prompt,
        context=context,
    )
    parsed = _extract_json_block(raw)

    return {
        "decision": _normalize_decision(parsed.get("decision", "retry")),
        "confidence_band": parsed.get("confidence_band", "low"),
        "reasoning": parsed.get("reasoning", ""),
        "evidence_quality": parsed.get("evidence_quality", "weak"),
        "query_match": parsed.get("query_match", "low"),
        "suggested_retry_focus": parsed.get("suggested_retry_focus", "none"),
        "raw": raw,
    }


def _llm_select_retry_strategy(
    *,
    query: str,
    context: str,
    evidence_summary: Dict[str, Any],
    confidence_eval: Dict[str, Any],
    retrieval_attempt: int,
    max_retries: int,
) -> Dict[str, Any]:
    prompt = f"""
You are selecting the best retry strategy for an SEC 8-K retrieval workflow.

User query:
{query}

Retrieval attempt:
{retrieval_attempt} of {max_retries}

Evidence summary:
{json.dumps(evidence_summary, indent=2)}

Current evaluation:
{json.dumps(confidence_eval, indent=2)}

Retrieved evidence context:
{context}

Choose the SINGLE best retry strategy from:
- broaden_search
- narrow_search
- event_keyword_bias
- diversify_results

Guidance:
- broaden_search: too few useful hits or query may be too restrictive
- narrow_search: too many mixed/off-target candidates
- event_keyword_bias: event/entity signal exists but event wording is weak
- diversify_results: repeated similar low-separation results across retries

Return STRICT JSON only:
{{
  "retry_strategy": "broaden_search" | "narrow_search" | "event_keyword_bias" | "diversify_results",
  "retry_reason": "<short reason>"
}}

Output JSON only.
""".strip()

    raw = tool_answer_from_context(
        question=prompt,
        context=context,
    )
    parsed = _extract_json_block(raw)

    return {
        "retry_strategy": _normalize_retry_strategy(parsed.get("retry_strategy", "broaden_search")),
        "retry_reason": parsed.get("retry_reason", "llm_selected_strategy"),
        "raw": raw,
    }


def _heuristic_retry_strategy(
    *,
    evidence_summary: Dict[str, Any],
    confidence_eval: Dict[str, Any],
    retrieval_attempt: int,
) -> Dict[str, str]:
    count = evidence_summary.get("count", 0)
    top_score = evidence_summary.get("top_score", 0.0) or 0.0
    score_spread = evidence_summary.get("score_spread", 0.0) or 0.0
    strong_candidates = evidence_summary.get("strong_candidates", 0)
    unique_titles = evidence_summary.get("unique_titles", 0)
    unique_symbols = evidence_summary.get("unique_symbols", 0)
    decision = confidence_eval.get("decision", "weak_evidence")

    retry_strategy = "broaden_search"
    retry_reason = "default_recovery"

    if count <= 3 or top_score < 0.60:
        retry_strategy = "broaden_search"
        retry_reason = "low_hit_count_or_low_top_score"
    elif count >= 8 and strong_candidates == 0 and unique_titles >= 6:
        retry_strategy = "narrow_search"
        retry_reason = "many_mixed_candidates"
    elif retrieval_attempt >= 1 and abs(score_spread) < 0.02:
        retry_strategy = "diversify_results"
        retry_reason = "low_score_separation_across_retries"
    elif decision == "weak_evidence" and unique_symbols >= 3:
        retry_strategy = "event_keyword_bias"
        retry_reason = "entity_candidates_found_but_event_signal_weak"

    return {
        "retry_strategy": retry_strategy,
        "retry_reason": retry_reason,
    }


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
    retry_strategy = state.get("retry_strategy", "")

    if retrieval_attempt == 0:
        rewritten_query = rewrite_query_for_plan(query, plan)
    else:
        base_retry_query = rewrite_query_for_retry(
            query=query,
            plan=plan,
            retry_index=retrieval_attempt - 1,
        )

        if retry_strategy == "broaden_search":
            rewritten_query = f"{base_retry_query} disclosure announcement transaction"
        elif retry_strategy == "narrow_search":
            rewritten_query = f"{query} acquisition merger announced"
        elif retry_strategy == "event_keyword_bias":
            rewritten_query = f"{query} acquisition acquisition agreement merger transaction"
        elif retry_strategy == "diversify_results":
            rewritten_query = f"{query} corporate action strategic transaction filing"
        else:
            rewritten_query = base_retry_query

    trace_entry = {
        "node": "rewrite_query_node",
        "retrieval_attempt": retrieval_attempt,
        "retry_strategy": retry_strategy,
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
    query = state["query"]
    deduped = state.get("deduped", [])
    retrieval_attempt = state.get("retrieval_attempt", 0)
    max_retries = state.get("max_retries", 2)

    evidence_summary = summarize_evidence(deduped)
    
    # Existing heuristic layer stays
    heuristic_confidence = evaluate_confidence(
        evidence_summary=evidence_summary,
        retry_count=retrieval_attempt,
        improved_on_retry=False,
    )

    # LLM reasoning layer
    context = _build_evidence_context(state, text_chars=900)
    llm_eval = _llm_evaluate_evidence(
        query=query,
        context=context,
        evidence_summary=evidence_summary,
        retrieval_attempt=retrieval_attempt,
        max_retries=max_retries,
    )

    # Conservative merge policy
    heuristic_decision = heuristic_confidence.get("decision", "weak_evidence")
    llm_decision = llm_eval.get("decision", "retry")

    if heuristic_decision == "strong_evidence" and llm_decision == "answer":
        final_decision = "answer"
    elif retrieval_attempt >= max_retries:
        # At max retries, do not keep looping
        final_decision = "escalate" if llm_decision != "answer" else "answer"
    elif llm_decision == "escalate":
        final_decision = "escalate"
    else:
        final_decision = "retry"

    confidence_eval = {
        **heuristic_confidence,
        "decision": final_decision,
        "heuristic_decision": heuristic_decision,
        "llm_decision": llm_decision,
        "llm_confidence_band": llm_eval.get("confidence_band"),
        "llm_reasoning": llm_eval.get("reasoning"),
        "evidence_quality": llm_eval.get("evidence_quality"),
        "query_match": llm_eval.get("query_match"),
        "suggested_retry_focus": llm_eval.get("suggested_retry_focus"),
    }

    trace_entry = {
        "node": "evaluate_evidence_node",
        "heuristic_decision": heuristic_decision,
        "llm_decision": llm_decision,
        "final_decision": final_decision,
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


def select_retry_strategy_node(state: EventLensState) -> Dict[str, Any]:
    query = state["query"]
    evidence_summary = state.get("evidence_summary", {})
    confidence_eval = state.get("confidence_eval", {})
    retrieval_attempt = state.get("retrieval_attempt", 0)
    max_retries = state.get("max_retries", 2)

    heuristic_choice = _heuristic_retry_strategy(
        evidence_summary=evidence_summary,
        confidence_eval=confidence_eval,
        retrieval_attempt=retrieval_attempt,
    )

    context = _build_evidence_context(state, text_chars=700)
    llm_choice = _llm_select_retry_strategy(
        query=query,
        context=context,
        evidence_summary=evidence_summary,
        confidence_eval=confidence_eval,
        retrieval_attempt=retrieval_attempt,
        max_retries=max_retries,
    )

    # Merge policy:
    # Prefer LLM choice when it exists, but keep heuristic as fallback/trace.
    retry_strategy = llm_choice.get("retry_strategy") or heuristic_choice["retry_strategy"]
    retry_reason = llm_choice.get("retry_reason") or heuristic_choice["retry_reason"]

    trace_entry = {
        "node": "select_retry_strategy_node",
        "retrieval_attempt": retrieval_attempt,
        "heuristic_retry_strategy": heuristic_choice["retry_strategy"],
        "heuristic_retry_reason": heuristic_choice["retry_reason"],
        "llm_retry_strategy": llm_choice.get("retry_strategy"),
        "llm_retry_reason": llm_choice.get("retry_reason"),
        "final_retry_strategy": retry_strategy,
        "final_retry_reason": retry_reason,
        "count": evidence_summary.get("count", 0),
        "top_score": evidence_summary.get("top_score", 0.0),
        "score_spread": evidence_summary.get("score_spread", 0.0),
        "strong_candidates": evidence_summary.get("strong_candidates", 0),
        "unique_titles": evidence_summary.get("unique_titles", 0),
        "unique_symbols": evidence_summary.get("unique_symbols", 0),
    }

    failure_reasons = state.get("failure_reasons", [])
    tool_history = state.get("tool_history", [])

    return {
        "retry_strategy": retry_strategy,
        "retry_reason": retry_reason,
        "failure_reasons": failure_reasons + [retry_reason],
        "tool_history": tool_history + [
            {
                "tool": "select_retry_strategy",
                "retry_strategy": retry_strategy,
                "retry_reason": retry_reason,
                "heuristic_retry_strategy": heuristic_choice["retry_strategy"],
                "llm_retry_strategy": llm_choice.get("retry_strategy"),
            }
        ],
        "trace": state.get("trace", []) + [trace_entry],
    }
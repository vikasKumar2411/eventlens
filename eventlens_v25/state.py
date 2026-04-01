# eventlens_v25/state.py
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class EventLensState(TypedDict, total=False):
    # user input
    query: str

    # planning
    intent: str
    plan: str
    settings: Dict[str, Any]

    # retrieval control
    rewritten_query: str
    retrieval_attempt: int
    max_retries: int

    # intermediate results
    hits: List[Dict[str, Any]]
    reranked: List[Dict[str, Any]]
    deduped: List[Dict[str, Any]]
    docs: List[Any]

    # evaluation
    evidence_summary: Dict[str, Any]
    confidence_eval: Dict[str, Any]

    # final output
    final_answer: str
    final_status: str

    # observability
    trace: List[Dict[str, Any]]

    # lifecycle
    done: bool
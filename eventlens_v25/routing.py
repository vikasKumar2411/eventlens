# eventlens_v25/routing.py
from __future__ import annotations

from eventlens_v25.state import EventLensState


def route_after_evaluation(state: EventLensState) -> str:
    """
    Decide what happens after evaluate_evidence_node.

    Returns one of:
    - "retry"
    - "answer"
    - "escalate"
    """
    confidence_eval = state.get("confidence_eval", {})
    decision = confidence_eval.get("decision", "weak_evidence")

    retrieval_attempt = state.get("retrieval_attempt", 0)
    max_retries = state.get("max_retries", 1)

    # strong enough -> answer
    if decision in {"confident", "cautious"}:
        return "answer"

    # weak evidence but still have retry budget -> retry
    if decision == "weak_evidence" and retrieval_attempt < max_retries:
        return "retry"

    # otherwise escalate
    return "escalate"
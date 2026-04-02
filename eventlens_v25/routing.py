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
    decision = confidence_eval.get("decision", "escalate")

    if decision == "answer":
        return "answer"
    if decision == "retry":
        return "retry"
    return "escalate"
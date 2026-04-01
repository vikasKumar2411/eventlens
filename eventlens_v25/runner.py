# eventlens_v25/runner.py
from __future__ import annotations

from typing import Any, Dict

from eventlens_v25.graph import build_eventlens_graph


def run_eventlens_v25(
    query: str,
    *,
    max_retries: int | None = None,
) -> Dict[str, Any]:
    graph = build_eventlens_graph()

    initial_state: Dict[str, Any] = {
        "query": query,
        "trace": [],
        "done": False,
    }

    # optional override if caller wants to force retry budget
    if max_retries is not None:
        initial_state["max_retries"] = max_retries

    final_state = graph.invoke(initial_state)

    return {
        "query": final_state.get("query"),
        "intent": final_state.get("intent"),
        "plan": final_state.get("plan"),
        "rewritten_query": final_state.get("rewritten_query"),
        "retrieval_attempt": final_state.get("retrieval_attempt"),
        "max_retries": final_state.get("max_retries"),
        "evidence_summary": final_state.get("evidence_summary"),
        "confidence_eval": final_state.get("confidence_eval"),
        "final_status": final_state.get("final_status"),
        "final_answer": final_state.get("final_answer"),
        "trace": final_state.get("trace", []),
        "raw_state": final_state,
    }
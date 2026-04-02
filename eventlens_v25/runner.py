# eventlens_v25/runner.py
from __future__ import annotations

from typing import Any, Dict

from eventlens_v25.graph import build_eventlens_graph
from eventlens_v25.run_store import summarize_run, save_run_summary


def run_eventlens_v25(
    query: str,
    *,
    max_retries: int | None = None,
    persist_run: bool = True,
) -> Dict[str, Any]:
    graph = build_eventlens_graph()

    initial_state: Dict[str, Any] = {
        "query": query,
        "trace": [],
        "done": False,
    }

    if max_retries is not None:
        initial_state["max_retries"] = max_retries

    final_state = graph.invoke(initial_state)

    saved_run_path = None
    run_summary = None

    if persist_run:
        run_summary = summarize_run(final_state)
        saved_run_path = str(save_run_summary(run_summary))

    return {
        "query": final_state.get("query"),
        "intent": final_state.get("intent"),
        "plan": final_state.get("plan"),
        "rewritten_query": final_state.get("rewritten_query"),
        "retrieval_attempt": final_state.get("retrieval_attempt"),
        "max_retries": final_state.get("max_retries"),
        "retry_strategy": final_state.get("retry_strategy"),
        "retry_reason": final_state.get("retry_reason"),
        "failure_reasons": final_state.get("failure_reasons", []),
        "evidence_summary": final_state.get("evidence_summary"),
        "confidence_eval": final_state.get("confidence_eval"),
        "final_status": final_state.get("final_status"),
        "final_answer": final_state.get("final_answer"),
        "trace": final_state.get("trace", []),
        "saved_run_path": saved_run_path,
        "run_summary": run_summary,
        "raw_state": final_state,
    }
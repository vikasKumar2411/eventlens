# eventlens_v25/graph.py
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from eventlens_v25.state import EventLensState
from eventlens_v25.nodes import (
    planner_node,
    rewrite_query_node,
    retrieve_node,
    rerank_node,
    dedupe_node,
    evaluate_evidence_node,
    select_retry_strategy_node,
    answer_node,
    escalate_node,
)
from eventlens_v25.routing import route_after_evaluation


def increment_retry_node(state: EventLensState) -> Dict[str, Any]:
    current_attempt = state.get("retrieval_attempt", 0)
    next_attempt = current_attempt + 1

    trace_entry = {
        "node": "increment_retry_node",
        "previous_attempt": current_attempt,
        "new_attempt": next_attempt,
    }

    return {
        "retrieval_attempt": next_attempt,
        "trace": state.get("trace", []) + [trace_entry],
    }


def build_eventlens_graph():
    graph = StateGraph(EventLensState)

    # Nodes
    graph.add_node("planner", planner_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("dedupe", dedupe_node)
    graph.add_node("evaluate", evaluate_evidence_node)
    graph.add_node("select_retry_strategy", select_retry_strategy_node)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("answer", answer_node)
    graph.add_node("escalate", escalate_node)

    # Linear flow
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "dedupe")
    graph.add_edge("dedupe", "evaluate")

    # Conditional routing after evaluation
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "retry": "select_retry_strategy",
            "answer": "answer",
            "escalate": "escalate",
        },
    )

    # Retry loop
    graph.add_edge("select_retry_strategy", "increment_retry")
    graph.add_edge("increment_retry", "rewrite_query")

    # Terminal nodes
    graph.add_edge("answer", END)
    graph.add_edge("escalate", END)

    return graph.compile()
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eventlens_v25.runner import run_eventlens_v25


QUERIES = [
    "Which companies announced acquisitions?",
    "Which companies announced merger deals?",
    "Which companies entered merger agreements?",
    "Which companies announced all-stock deals?",
    "Which firms entered definitive transaction agreements?",
    "Which companies acquired Sportech?",
    "Which companies acquired ElectraMeccanica?",
    "Which company acquired Southwestern Energy?",
]


def main():
    for i, query in enumerate(QUERIES, 1):
        print("\n" + "=" * 80)
        print(f"QUERY {i}: {query}")

        result = run_eventlens_v25(query, max_retries=2)

        print("Intent:", result.get("intent"))
        print("Plan:", result.get("plan"))
        print("Final status:", result.get("final_status"))
        print("Saved run path:", result.get("saved_run_path"))

        trace = result.get("trace", [])
        retry_steps = [
            step for step in trace
            if step.get("node") == "select_retry_strategy_node"
        ]

        if retry_steps:
            print("\nRetry decisions:")
            for step in retry_steps:
                print(
                    {
                        "retrieval_attempt": step.get("retrieval_attempt"),
                        "heuristic": step.get("heuristic_retry_strategy"),
                        "llm": step.get("llm_retry_strategy"),
                        "memory_best": step.get("memory_best_strategy"),
                        "decision_source": step.get("decision_source"),
                        "final": step.get("final_retry_strategy"),
                        "reason": step.get("final_retry_reason"),
                    }
                )
        else:
            print("\nRetry decisions: none")


if __name__ == "__main__":
    main()
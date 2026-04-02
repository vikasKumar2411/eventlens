#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# ensure repo root is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eventlens_v25.runner import run_eventlens_v25


def main():
    query = "Which companies announced acquisitions?"

    result = run_eventlens_v25(
        query,
        max_retries=2,
    )

    print("\n=== QUERY ===")
    print(query)

    print("\n=== PLAN ===")
    print("Intent:", result["intent"])
    print("Plan:", result["plan"])

    print("\n=== STATUS ===")
    print("Final status:", result["final_status"])

    print("\n=== CONFIDENCE ===")
    print(result["confidence_eval"])

    print("\n=== ANSWER ===")
    print(result["final_answer"])

    print("\n=== MEMORY ===")
    print("Saved run path:", result["saved_run_path"])

    if result.get("run_summary"):
        print("\n--- Retry History ---")
        for r in result["run_summary"].get("retry_history", []):
            print(r)

        print("\n--- Rewrite History ---")
        for r in result["run_summary"].get("rewrite_history", []):
            print(r)

    print("\n=== TRACE ===")
    for step in result["trace"]:
        print(step)


if __name__ == "__main__":
    main()
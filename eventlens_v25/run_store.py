# eventlens_v25/run_store.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _extract_retry_history(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []

    for entry in trace:
        if entry.get("node") == "select_retry_strategy_node":
            history.append(
                {
                    "retrieval_attempt": entry.get("retrieval_attempt"),
                    "heuristic_retry_strategy": entry.get("heuristic_retry_strategy"),
                    "heuristic_retry_reason": entry.get("heuristic_retry_reason"),
                    "llm_retry_strategy": entry.get("llm_retry_strategy"),
                    "llm_retry_reason": entry.get("llm_retry_reason"),
                    "final_retry_strategy": entry.get("final_retry_strategy"),
                    "final_retry_reason": entry.get("final_retry_reason"),
                }
            )

    return history


def _extract_rewrite_history(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []

    for entry in trace:
        if entry.get("node") == "rewrite_query_node":
            history.append(
                {
                    "retrieval_attempt": entry.get("retrieval_attempt"),
                    "retry_strategy": entry.get("retry_strategy"),
                    "retry_reason": entry.get("retry_reason"),
                    "heuristic_query": entry.get("heuristic_query"),
                    "rewritten_query": entry.get("rewritten_query"),
                }
            )

    return history


def summarize_run(final_state: Dict[str, Any]) -> Dict[str, Any]:
    trace = final_state.get("trace", []) or []

    run_summary = {
        "run_id": str(uuid4()),
        "timestamp": _utc_now_iso(),
        "query": _safe_str(final_state.get("query")),
        "intent": _safe_str(final_state.get("intent")),
        "plan": _safe_str(final_state.get("plan")),
        "max_retries": final_state.get("max_retries"),
        "retrieval_attempt": final_state.get("retrieval_attempt", 0),
        "final_status": _safe_str(final_state.get("final_status")),
        "final_answer": _safe_str(final_state.get("final_answer")),
        "confidence_eval": final_state.get("confidence_eval", {}),
        "evidence_summary": final_state.get("evidence_summary", {}),
        "failure_reasons": final_state.get("failure_reasons", []),
        "retry_history": _extract_retry_history(trace),
        "rewrite_history": _extract_rewrite_history(trace),
        "trace": trace,
    }

    return run_summary


def save_run_summary(run_summary: Dict[str, Any], runs_dir: Path | None = None) -> Path:
    target_dir = runs_dir or RUNS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_summary.get("run_id", uuid4().hex)
    filename = f"{timestamp}_{run_id}.json"
    path = target_dir / filename

    with path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    return path


def load_all_run_summaries(runs_dir: Path | None = None) -> List[Dict[str, Any]]:
    target_dir = runs_dir or RUNS_DIR
    if not target_dir.exists():
        return []

    runs: List[Dict[str, Any]] = []
    for path in sorted(target_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception:
            continue

    return runs